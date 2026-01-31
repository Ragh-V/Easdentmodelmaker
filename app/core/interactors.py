import vtk

class BrushInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent_app):
        self.parent = parent_app
        self.AddObserver("LeftButtonPressEvent", self.on_left_down)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_up)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move)
    
    def _is_cursor_on_mesh(self):
        """
        Robust check: Returns True if mouse is hovering over the ACTIVE MESH.
        """
        x, y = self.GetInteractor().GetEventPosition()
        
        # Use CellPicker with "PickFromList" to ignore everything except the target mesh
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.InitializePickList()
        picker.AddPickList(self.parent.active_actor)
        picker.PickFromListOn()
        
        picker.Pick(x, y, 0, self.parent.plotter.renderer)
        return picker.GetActor() == self.parent.active_actor

    def on_left_down(self, obj, event):
        if self._is_cursor_on_mesh():
            self.parent.is_brushing_now = True
            self.parent.on_brush_action(self.GetInteractor(), event)
        else:
            self.parent.is_brushing_now = False
            super().OnLeftButtonDown()

    def on_left_up(self, obj, event):
        self.parent.is_brushing_now = False
        super().OnLeftButtonUp()

    def on_mouse_move(self, obj, event):
        self.parent.on_brush_hover(self.GetInteractor(), event)

        if self.parent.is_brushing_now:
            self.parent.on_brush_action(self.GetInteractor(), event)
        else:
            super().OnMouseMove()