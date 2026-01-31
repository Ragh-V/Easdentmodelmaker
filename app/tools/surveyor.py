import vtk
import numpy as np
import pyvista as pv
import math
from matplotlib.colors import LinearSegmentedColormap

class SurveyorInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent):
        self.parent = parent
        self.is_dragging = False
        self.AddObserver("LeftButtonPressEvent", self.OnLeftDown)
        self.AddObserver("LeftButtonReleaseEvent", self.OnLeftUp)
        self.AddObserver("MouseMoveEvent", self.OnMouseMove)

    def OnLeftDown(self, obj, event):
        click_pos = self.parent.plotter.iren.interactor.GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.parent.plotter.renderer)
        if picker.GetActor() == self.parent.handle_actor: 
            self.is_dragging = True
            obj.SetAbortFlag(1)
        else:
            self.is_dragging = False
            super().OnLeftButtonDown()

    def OnLeftUp(self, obj, event):
        if self.is_dragging:
            self.is_dragging = False
            self.parent.calculate_undercut()
        super().OnLeftButtonUp()

    def OnMouseMove(self, obj, event):
        if self.is_dragging: 
            self.parent.update_gizmo_from_mouse()
            obj.SetAbortFlag(1)
        else: 
            super().OnMouseMove()

class UndercutSurveyor:
    def __init__(self, plotter, target_actor):
        self.plotter = plotter
        self.actor = target_actor
        self.original_mesh = target_actor.mapper.dataset
        self.analysis_mesh = None 
        
        # Setup Color Map
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
        self.cmap = LinearSegmentedColormap.from_list("dental_depth", colors)
        self.scalar_name = "Undercut_Depth_mm"
        self.bar_title = "Undercut Depth (mm)"
        
        # State
        self.default_vector = np.array([0, 0, -1], dtype=np.float64) 
        self.current_vector = self.default_vector.copy()
        
        self.centroid = np.array(self.original_mesh.center)
        self.arrow_actor = None
        self.handle_actor = None
        
        # Store original visual state
        self.original_texture = self.actor.GetTexture()
        self.original_scalars = self.original_mesh.active_scalars_name
        self.original_color = self.actor.prop.color
        self._style = None

    def start(self):
        if isinstance(self.original_mesh, pv.UnstructuredGrid):
            self.analysis_mesh = self.original_mesh.extract_surface()
        else:
            self.analysis_mesh = self.original_mesh.copy()
            
        try:
            self.analysis_mesh.clean(inplace=True)
            self.analysis_mesh.compute_normals(inplace=True)
        except: pass
        
        self.actor.mapper.SetInputData(self.analysis_mesh)
        self.actor.SetTexture(None)
        self.analysis_mesh.set_active_scalars(None)
        self.actor.prop.color = "white"
        
        self.current_vector = self.default_vector.copy()
        self._update_gizmo_visuals()
        
        self._style = SurveyorInteractorStyle(self)
        self.plotter.iren.interactor.SetInteractorStyle(self._style)
        
        self.calculate_undercut()

    def stop(self):
        if self.arrow_actor: self.plotter.remove_actor(self.arrow_actor)
        if self.handle_actor: self.plotter.remove_actor(self.handle_actor)
        
        try:
            self.plotter.remove_scalar_bar(self.bar_title)
        except (KeyError, ValueError, AttributeError):
            pass
        
        self.actor.mapper.SetInputData(self.original_mesh)
        if self.original_texture:
            self.actor.SetTexture(self.original_texture)
            self.actor.mapper.ScalarVisibilityOff()
        elif self.original_scalars:
            self.original_mesh.set_active_scalars(self.original_scalars)
            self.actor.mapper.ScalarVisibilityOn()
        else:
            self.actor.mapper.ScalarVisibilityOff()
            self.actor.prop.color = self.original_color
            
        self.analysis_mesh = None 
        self.plotter.enable_trackball_style()
        self.plotter.render()

    def _update_gizmo_visuals(self):
        if self.arrow_actor: self.plotter.remove_actor(self.arrow_actor)
        if self.handle_actor: self.plotter.remove_actor(self.handle_actor)
        
        scale = 30.0
        tip_pos = self.centroid + (self.current_vector * scale)
        
        arrow = pv.Arrow(start=self.centroid, direction=self.current_vector, scale=scale)
        self.arrow_actor = self.plotter.add_mesh(arrow, color="#3498db", pickable=False, reset_camera=False)
        
        handle = pv.Sphere(radius=2.5, center=tip_pos)
        self.handle_actor = self.plotter.add_mesh(handle, color="#2980b9", pickable=True, name="GizmoHandle", reset_camera=False)
        
        self.plotter.render()

    def reset_to_original_vector(self):
        self.current_vector = self.default_vector.copy()
        self._update_gizmo_visuals()
        self.calculate_undercut()

    def update_gizmo_from_mouse(self):
        iren = self.plotter.iren
        x, y = iren.interactor.GetEventPosition()
        
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(x, y, 0, self.plotter.renderer)
        pick_pos = np.array(picker.GetPickPosition())
        
        new_vec = pick_pos - self.centroid
        norm = np.linalg.norm(new_vec)
        if norm > 0.001:
            self.current_vector = new_vec / norm
            self._update_gizmo_visuals()

    def update_direction(self, theta_deg, phi_deg):
        theta = np.radians(theta_deg)
        phi = np.radians(phi_deg)
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        self.current_vector = np.array([x, y, z])
        self._update_gizmo_visuals()
        self.calculate_undercut() 

    def calculate_undercut(self):
        if self.analysis_mesh is None: return
        mesh = self.analysis_mesh 
        
        world_ins = self.current_vector
        if self.actor.GetMatrix():
            mat = self.actor.GetMatrix()
            vtk_mat = vtk.vtkMatrix4x4()
            vtk_mat.DeepCopy(mat)
            vtk_mat.Invert()
            vec_homo = [*world_ins, 0.0]
            obj_vec = [0.0]*4
            vtk_mat.MultiplyPoint(vec_homo, obj_vec)
            obj_ins_vec = np.array(obj_vec[:3])
            norm = np.linalg.norm(obj_ins_vec)
            if norm > 0: obj_ins_vec /= norm
        else:
            obj_ins_vec = world_ins
            
        removal_vec = -obj_ins_vec
        
        normals = mesh.point_data["Normals"]
        dots = np.dot(normals, removal_vec)
        candidate_indices = np.where(dots < -0.05)[0]
        
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        
        undercut_depths = np.zeros(mesh.n_points)
        ray_len = mesh.length * 2.0
        points = mesh.points
        
        ray_start = [0.0]*3; ray_end = [0.0]*3
        t = vtk.mutable(0.0)
        x = [0.0]*3; pcoords = [0.0]*3
        subId = vtk.mutable(0); cellId = vtk.mutable(0)
        
        for idx in candidate_indices:
            pt = points[idx]
            for k in range(3):
                ray_start[k] = pt[k] + removal_vec[k] * 0.05
                ray_end[k] = pt[k] + removal_vec[k] * ray_len
            
            if locator.IntersectWithLine(ray_start, ray_end, 0.001, t, x, pcoords, subId, cellId):
                dist = math.sqrt((x[0] - pt[0])**2 + (x[1] - pt[1])**2 + (x[2] - pt[2])**2)
                undercut_depths[idx] = dist
        
        max_depth = np.max(undercut_depths)
        if max_depth == 0: max_depth = 0.1 
        
        mesh.point_data[self.scalar_name] = undercut_depths
        mesh.set_active_scalars(self.scalar_name)
        
        lut = pv.LookupTable(cmap=self.cmap, scalar_range=[0, max_depth], above_range_color="#e74c3c")
        mapper = self.actor.mapper
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0, max_depth) 
        mapper.ScalarVisibilityOn()
        
        try: self.plotter.remove_scalar_bar(self.bar_title)
        except: pass
            
        self.plotter.add_scalar_bar(
            title=self.bar_title, 
            interactive=False, 
            n_labels=5, 
            fmt="%.1f"
        )
        self.plotter.render()