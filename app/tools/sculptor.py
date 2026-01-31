import vtk
import numpy as np
import pyvista as pv

# --- Import C++ Engine ---
try:
    import sculpt_tore
    HAS_SCULPT_CORE = True
    print("Sculpt Core (C++) Loaded Successfully")
except ImportError:
    HAS_SCULPT_CORE = False
    print(" WARNING: sculpt_tore C++ module not found. Performance will be degraded.")

class SculptInteractor(vtk.vtkInteractorStyleTrackballCamera):
    """
    Robust interactor that prioritizes:
    1. Left Drag on Mesh -> Sculpt (Camera Locked)
    2. Left Drag on Background -> Rotate Camera
    3. Right Drag -> Pan Camera
    """
    def __init__(self, tool):
        super().__init__()
        self.tool = tool
        self._interaction_state = "NONE" # NONE, SCULPTING, ROTATING, PANNING
        self._is_hovering_mesh = False 

        # Observers
        self.AddObserver("LeftButtonPressEvent", self.OnLeftDown)
        self.AddObserver("LeftButtonReleaseEvent", self.OnLeftUp)
        self.AddObserver("MouseMoveEvent", self.OnMouseMove)
        self.AddObserver("RightButtonPressEvent", self.OnRightDown)
        self.AddObserver("RightButtonReleaseEvent", self.OnRightUp)
        self.AddObserver("TimerEvent", self.OnTimer) 

    def OnTimer(self, obj, event):
        pass

    def OnRightDown(self, obj, event):
        self._interaction_state = "PANNING"
        self.StartPan()
    
    def OnRightUp(self, obj, event):
        if self._interaction_state == "PANNING":
            self.EndPan()
            self._interaction_state = "NONE"
        
    def OnLeftDown(self, obj, event):
        iren = obj.GetInteractor()
        click_pos = iren.GetEventPosition()
        
        # 1. CHECK WHAT WE CLICKED
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(click_pos[0], click_pos[1], 0, self.tool.plotter.renderer)
        picked_actor = picker.GetActor()
        
        is_target = (picked_actor is not None) and \
                    (picked_actor == self.tool.app.active_actor) and \
                    (picked_actor != self.tool.cursor_actor)

        # 2. DECIDE ACTION
        if is_target:
            self._interaction_state = "SCULPTING"
            self.tool.begin_stroke()
            
            # Apply first stamp immediately
            pos = np.array(picker.GetPickPosition())
            normal = np.array(picker.GetPickNormal())
            self.tool.apply_brush_step(pos, normal)
            
        else:
            self._interaction_state = "ROTATING"
            self.StartRotate() 
            
    def OnLeftUp(self, obj, event):
        if self._interaction_state == "SCULPTING":
            self.tool.end_stroke()
            self._interaction_state = "NONE"
            
        elif self._interaction_state == "ROTATING":
            self.EndRotate()
            self._interaction_state = "NONE"
            
        if self.GetInteractor():
            self.GetInteractor().DestroyTimer()

    def OnMouseMove(self, obj, event):
        self.UpdateHoverState(obj)
        
        if self._interaction_state == "SCULPTING":
            self.ApplyBrush(obj)
            
        elif self._interaction_state == "ROTATING":
            super().OnMouseMove() 
            
        elif self._interaction_state == "PANNING":
            super().OnMouseMove()
            
        else:
            super().OnMouseMove()

    def UpdateHoverState(self, obj):
        iren = obj.GetInteractor()
        x, y = iren.GetEventPosition()
        
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(x, y, 0, self.tool.plotter.renderer)
        picked = picker.GetActor()
        
        hit_valid = (picked is not None) and \
                    (picked == self.tool.app.active_actor) and \
                    (picked != self.tool.cursor_actor)
        
        self._is_hovering_mesh = hit_valid

        if hit_valid:
            pos = np.array(picker.GetPickPosition())
            self.tool.update_cursor_visual(pos)
        else:
            self.tool.hide_cursor()

    def ApplyBrush(self, obj):
        iren = obj.GetInteractor()
        x, y = iren.GetEventPosition()
        
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(x, y, 0, self.tool.plotter.renderer)
        
        # We can sculpt even if the mouse slips off the mesh slightly, 
        # using the last known position or projecting
        if picker.GetPickPosition():
            pos = np.array(picker.GetPickPosition())
            normal = np.array(picker.GetPickNormal())
            self.tool.apply_brush_step(pos, normal)


class SculptTool:
    def __init__(self, app, plotter):
        self.app = app
        self.plotter = plotter
        self.active = False
        self.mesh = None
        self.last_cursor_pos = None 
        
        # --- C++ ENGINE ---
        self.engine = None
        if HAS_SCULPT_CORE:
            try:
                # Initialize the C++ Dynamic Sculptor
                self.engine = sculpt_tore.DynamicSculptor()
            except AttributeError:
                print("Error: sculpt_tore found but 'DynamicSculptor' class missing.")

        # Params
        self.mode = 0  # 0=SMOOTH, 1=REMOVE, 2=ADD
        self.radius = 2.0
        self.strength = 0.5
        
        # Dyntopo
        self.dyntopo_enabled = False
        self.detail_size = 0.5 
        
        # Internals
        self.cursor_actor = None
        self._style = None
        self._prev_style = None

    def set_params(self, mode_str, radius, strength, dyntopo=False, detail_size=0.5):
        mode_map = {"SMOOTH": 0, "REMOVE": 1, "ADD": 2}
        self.mode = mode_map.get(mode_str, 0)
        
        self.radius = float(radius)
        self.strength = float(strength)
        self.dyntopo_enabled = dyntopo
        self.detail_size = float(detail_size)
        
        self._update_cursor_color(mode_str)
        
        if self.active and self.last_cursor_pos is not None:
            self.update_cursor_visual(self.last_cursor_pos)
            self.plotter.render()

    def start(self):
        if not self.app.active_actor: return
        self.active = True
        
        self.app.active_actor.SetPickable(True)
        for actor in self.plotter.actors.values():
            if actor != self.app.active_actor:
                try: actor.SetPickable(False)
                except AttributeError: pass

        self.mesh = self.app.active_actor.mapper.dataset
        

        if isinstance(self.mesh, pv.UnstructuredGrid):
            self.mesh = self.mesh.extract_surface()
            self.app.active_actor.mapper.SetInputData(self.mesh)
        

        if not self.mesh.is_all_triangles:
            self.mesh.triangulate(inplace=True)
        
        # Clean to merge duplicate points (important for topology)
        self.mesh.clean(inplace=True)


        if self.mesh.points.dtype != np.float64:
             self.mesh.points = self.mesh.points.astype(np.float64)

        self.mesh.compute_normals(
            cell_normals=False, 
            point_normals=True, 
            inplace=True, 
            auto_orient_normals=False
        )


        if self.engine and self.mesh.n_faces > 0:
            try:
                # VTK faces are [3, v1, v2, v3, 3, v1, v2, v3...]
                # We need to strip the '3' padding for C++
                raw_faces = self.mesh.faces.reshape(-1, 4)[:, 1:].flatten()
                self.engine.set_mesh(self.mesh.points, raw_faces)
            except Exception as e:
                print(f"Error loading mesh into C++ engine: {e}")

        self._create_cursor()
        
        self._prev_style = self.plotter.iren.interactor.GetInteractorStyle()
        self._style = SculptInteractor(self)
        self.plotter.iren.interactor.SetInteractorStyle(self._style)
        
        self.plotter.disable_picking()
        
        if self.mesh.n_points > 0:
            center = self.mesh.center
            self.update_cursor_visual(center)
            self.plotter.render()

    def stop(self):
        self.active = False
        if self._prev_style:
            self.plotter.iren.interactor.SetInteractorStyle(self._prev_style)
        else:
            self.plotter.enable_trackball_style()
            
        if self.cursor_actor:
            self.plotter.remove_actor(self.cursor_actor)
            self.cursor_actor = None
            
        self.mesh = None
        
        # Clear engine memory
        if self.engine:

            self.engine.set_mesh(np.array([]), np.array([]))

        for actor in self.plotter.actors.values():
            try: actor.SetPickable(True)
            except AttributeError: pass

        self.app.enable_object_selection_mode()
        self.plotter.render()

    def begin_stroke(self):
        pass

    def apply_brush_step(self, center, normal):
        """
        Deforms the mesh. Uses C++ for both topology refinement (Dyntopo) and movement.
        """
        if not self.mesh or not self.engine: return

        eff_detail = self.detail_size if self.dyntopo_enabled else 999999.0


        new_verts, new_faces_flat = self.engine.sculpt(
            center.tolist(), 
            normal.tolist(), 
            self.radius,
            self.strength,
            self.mode,
            eff_detail
        )
        

        self.mesh.points = new_verts
        
 
        if len(new_faces_flat) > 0:
            n_faces = len(new_faces_flat) // 3
            

            padding = np.full((n_faces, 1), 3, dtype=np.int32)

            faces_reshaped = new_faces_flat.reshape(-1, 3)

            vtk_faces = np.hstack((padding, faces_reshaped)).flatten()
            
            self.mesh.faces = vtk_faces
            
 
            self.engine.set_mesh(new_verts, new_faces_flat)


        self.mesh.compute_normals(point_normals=True, cell_normals=False, inplace=True)
        
        self.mesh.GetPoints().Modified()
        self.mesh.Modified()
        self.plotter.render()

    def end_stroke(self):
        pass

    def _create_cursor(self):
        if self.cursor_actor: return
        geo = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
        self.cursor_actor = self.plotter.add_mesh(
            geo, color="cyan", opacity=0.3, lighting=False, 
            pickable=False, name="SculptCursor"
        )
        
    def _update_cursor_color(self, mode_str):
        if not self.cursor_actor: return
        prop = self.cursor_actor.GetProperty()
        if mode_str == "ADD": prop.SetColor(0.2, 0.8, 1.0)
        elif mode_str == "REMOVE": prop.SetColor(1.0, 0.4, 0.4)
        elif mode_str == "SMOOTH": prop.SetColor(0.4, 1.0, 0.4)

    def update_cursor_visual(self, pos):
        if not self.cursor_actor: return
        self.last_cursor_pos = pos 
        self.cursor_actor.SetVisibility(True)
        self.cursor_actor.SetPosition(pos)
        self.cursor_actor.SetScale(self.radius, self.radius, self.radius)

    def hide_cursor(self):
        if self.cursor_actor: self.cursor_actor.SetVisibility(False)