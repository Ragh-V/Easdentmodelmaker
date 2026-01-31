import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree

# ==========================================
#   INTERACTOR (Corrected VTK Method Names)
# ==========================================
class BorderInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent_tool):
        self.tool = parent_tool
        self.is_dragging = False
        super().__init__()
        # We do NOT add observers here because we are overriding the methods directly below.

    # --- CORRECT OVERRIDES (Standard VTK Naming) ---
    def OnLeftButtonDown(self):
        # 1. Try to pick a handle
        if self.tool.try_pick_handle():
            self.is_dragging = True
            # Stop propagation (Do not call super)
            return 

        # 2. Else, Rotate Camera
        super().OnLeftButtonDown()

    def OnLeftButtonUp(self):
        if self.is_dragging:
            self.tool.end_drag()
        
        # Always call super to reset camera state
        super().OnLeftButtonUp()

    def OnMouseMove(self):
        if self.is_dragging:
            self.tool.drag_active_handle()
            self.GetInteractor().Render()
        else:
            super().OnMouseMove()


# ==========================================
#       THE TOOL LOGIC
# ==========================================
class BorderDeformTool:
    def __init__(self, plotter, actor):
        self.plotter = plotter
        self.actor = actor
        self.mesh = actor.mapper.dataset
        
        self.tree = cKDTree(self.mesh.points)
        self.control_points = []
        self.handle_actors = [] 
        self.border_tube_actor = None
        
        # FIX: Removed SetPickTolerance (Not supported by PropPicker)
        self.picker = vtk.vtkPropPicker()
        
        self.active_handle = None
        self.active_idx = -1
        self.is_dragging = False
        self.drag_depth = 0.0
        
        self.radius = 15.0 
        self.sigma = 7.0   
        
        self._prev_style = None
        self._style = None

    def start(self, explicit_path=None):
        if explicit_path is None or len(explicit_path) < 3:
            print("Error: No valid border path provided.")
            return

        ideal_points = self._resample_polyline(explicit_path, 16)
        dists, ids = self.tree.query(ideal_points)
        self.control_points = [self.mesh.points[i].copy() for i in ids] 

        self.handle_actors = []
        for pt in self.control_points:
            sphere = pv.Sphere(radius=0.8, center=pt)
            actor = self.plotter.add_mesh(
                sphere, color="#e74c3c", 
                pickable=True, 
                render_points_as_spheres=False,
                lighting=False, 
                name=f"BorderHandle_{len(self.handle_actors)}"
            )
            self.handle_actors.append(actor)
            
        self._update_tube_visual()

        # Swap Interactor
        self.plotter.enable_trackball_style()
        self._prev_style = self.plotter.iren.interactor.GetInteractorStyle()
        self._style = BorderInteractorStyle(self)
        self._style.SetDefaultRenderer(self.plotter.renderer)
        self.plotter.iren.interactor.SetInteractorStyle(self._style)
        
    def stop(self):
        if self._prev_style:
            self.plotter.iren.interactor.SetInteractorStyle(self._prev_style)
        else:
            self.plotter.enable_trackball_style()
            
        for actor in self.handle_actors:
            self.plotter.remove_actor(actor)
        if self.border_tube_actor:
            self.plotter.remove_actor(self.border_tube_actor)
            
        self.handle_actors = []
        self.control_points = []
        self.is_dragging = False

    def try_pick_handle(self):
        x, y = self.plotter.iren.get_event_position()
        
        prev_pickable = self.actor.GetPickable()
        self.actor.SetPickable(False) # Occlusion fix
        
        self.picker.Pick(x, y, 0, self.plotter.renderer)
        actor = self.picker.GetActor()
        
        self.actor.SetPickable(prev_pickable)
        
        if actor in self.handle_actors:
            self.active_handle = actor
            self.active_idx = self.handle_actors.index(actor)
            
            prop = actor.GetProperty()
            prop.SetColor(1.0, 1.0, 0.0)
            
            center = np.array(actor.GetCenter())
            self.drag_depth = self._get_depth(center)
            return True
        return False

    def drag_active_handle(self):
        if self.active_idx == -1 or not self.active_handle: return

        x, y = self.plotter.iren.get_event_position()
        new_pos = self._display_to_world(x, y, self.drag_depth)
        
        old_pos = self.control_points[self.active_idx]
        delta = new_pos - old_pos
        self.control_points[self.active_idx] = new_pos
        
        src = self.active_handle.mapper.dataset
        src.points += delta

        # Apply Deformation
        indices = self.tree.query_ball_point(old_pos, self.radius)
        if indices:
            pts = self.mesh.points[indices]
            dists = np.linalg.norm(pts - old_pos, axis=1)
            weights = np.exp(-(dists**2) / (2 * self.sigma**2))
            
            self.mesh.points[indices] += delta * weights[:, np.newaxis]
            
            # FIX: Correct PyVista/VTK update call
            self.mesh.modified() 

        self._update_tube_visual()

    def end_drag(self):
        self.is_dragging = False
        if self.active_handle:
            prop = self.active_handle.GetProperty()
            prop.SetColor(0.905, 0.298, 0.235)
        self.active_handle = None
        self.active_idx = -1

    def set_radius(self, radius):
        self.radius = radius
        self.sigma = radius / 2.5

    def _get_depth(self, world_pos):
        renderer = self.plotter.renderer
        renderer.SetWorldPoint(*world_pos, 1.0)
        renderer.WorldToDisplay()
        return renderer.GetDisplayPoint()[2]

    def _display_to_world(self, x, y, depth):
        renderer = self.plotter.renderer
        renderer.SetDisplayPoint(x, y, depth)
        renderer.DisplayToWorld()
        coords = renderer.GetWorldPoint()
        if coords[3] == 0: return np.array(coords[:3])
        return np.array(coords[:3]) / coords[3]

    def _update_tube_visual(self):
        if self.border_tube_actor: 
            self.plotter.remove_actor(self.border_tube_actor)
        if len(self.control_points) > 2:
            pts = np.vstack([self.control_points, self.control_points[0]])
            tube = pv.Spline(pts, n_points=100).tube(radius=0.2)
            self.border_tube_actor = self.plotter.add_mesh(
                tube, color="#f1c40f", pickable=False, name="BorderPreview"
            )

    def _resample_polyline(self, points, n_samples):
        if len(points) < 2: return np.array(points)
        dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cum_dist = np.insert(np.cumsum(dists), 0, 0)
        targets = np.linspace(0, cum_dist[-1], n_samples + 1)[:-1]
        resampled = []
        for t in targets:
            idx = np.searchsorted(cum_dist, t) - 1
            idx = max(0, min(idx, len(cum_dist)-2))
            t0, t1 = cum_dist[idx], cum_dist[idx+1]
            ratio = (t - t0) / (t1 - t0) if (t1-t0) > 1e-9 else 0
            pt = points[idx] * (1-ratio) + points[idx+1] * ratio
            resampled.append(pt)
        return np.array(resampled)