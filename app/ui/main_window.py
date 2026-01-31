import os
import numpy as np
import pyvista as pv
import vtk
from weakref import WeakKeyDictionary

from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                               QToolBar, QMessageBox, QWidget, QHBoxLayout,
                               QDockWidget, QCheckBox, QSlider, QLabel, QMenu, QInputDialog, 
                               QAbstractItemView, QDialog, QPushButton, QGroupBox,
                               QColorDialog, QToolButton)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import Qt, QTimer
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QSpinBox 

# Local Module Imports
from app.core.commands import (CommandManager, AddMeshCommand, DeleteMeshCommand, 
                               MaterialChangeCommand, TransformCommand, 
                               MultiDeleteCommand, DeleteCellsCommand, ReplaceGeometryCommand)
from app.core.interactors import BrushInteractorStyle
from app.ui.hierarchy import HierarchyPanel
from app.ui.dialogs import HoleFillDialog

# Project Imports / Mock
try:
    from app.ui.dental_wizard import DentalWizardSidebar
except ImportError:
    class DentalWizardSidebar(QWidget):
        def __init__(self, parent=None): super().__init__(parent)

try:
    from Gizmotool import GizmoTool
    from GizmoRot import GizmoRot
except ImportError:
    GizmoTool = None
    GizmoRot = None


class MedicalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EasDent Model Maker")
        self.resize(1200, 800)

        # 1. Core Systems
        self.command_manager = CommandManager(self)
        self.actors = {}
        self.original_textures = WeakKeyDictionary() 
        
        self.active_actor = None
        self.highlight_actor = None
        self.gizmo = None
        self._snapshot_matrix = None
        self.updating_selection = False
        self.is_processing = False

        # 3-Point Plane State
        self.picked_points = []
        self.point_markers = []
        self.plane_actor = None

        # Brush State
        self.brush_active = False
        self.brush_radius = 5.0 # mm
        self.brush_indices = set()
        self.is_brushing_now = False
        self.brush_cursor = None
        self.brush_locator = None
        self.last_brush_pos = None

        # 2. Viewport
        self.plotter = QtInteractor(self)
        
        # 3. UI
        self.setup_ui()
        self.setup_lights()
        self.add_infinite_grid()
        self.setup_picking() 

        # 4. Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.sync_highlight_motion)
        self.timer.start(30)

        self.load_dummy_data()
        
    def setup_ui(self):
        file_menu = self.menuBar().addMenu("File")
        load_action = file_menu.addAction("Import Mesh")
        load_action.triggered.connect(self.open_file_dialog)

        # Toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self.material_btn = QToolButton()
        self.material_btn.setText("Material")
        self.material_btn.setPopupMode(QToolButton.InstantPopup)
        
        material_menu = QMenu(self.material_btn)
        
        act_mauve = material_menu.addAction("Light Mauve (Default)")
        act_mauve.triggered.connect(lambda: self.apply_material_preset("mauve"))
        
        act_custom = material_menu.addAction("Choose Color...")
        act_custom.triggered.connect(self.open_color_picker)
        
        material_menu.addSeparator()

        act_fresnel = material_menu.addAction("Fresnel (Sculpt)")
        act_fresnel.triggered.connect(lambda: self.apply_material_preset("fresnel"))

        act_ref = material_menu.addAction("Reflective")
        act_ref.triggered.connect(lambda: self.apply_material_preset("reflective"))
        
        act_rough = material_menu.addAction("Rough")
        act_rough.triggered.connect(lambda: self.apply_material_preset("rough"))
        
        act_orig = material_menu.addAction("Original")
        act_orig.triggered.connect(lambda: self.apply_material_preset("original"))
        
        self.material_btn.setMenu(material_menu)
        self.toolbar.addWidget(self.material_btn)
        
        self.toolbar.addSeparator()
        
        self.inspect_action = QAction("Inspect", self)
        self.inspect_action.setCheckable(True)
        self.inspect_action.triggered.connect(self.toggle_inspect_mode)
        self.toolbar.addAction(self.inspect_action)

        self.toolbar.addSeparator()

        self.move_action = QAction("Move", self)
        self.move_action.setCheckable(True)
        self.move_action.triggered.connect(self.toggle_move_mode)
        self.toolbar.addAction(self.move_action)

        self.rotate_action = QAction("Rotate", self)
        self.rotate_action.setCheckable(True)
        self.rotate_action.triggered.connect(self.toggle_rotate_mode)
        self.toolbar.addAction(self.rotate_action)
        
        self.fill_holes_action = QAction("Fill Holes (Dialog)", self)
        self.fill_holes_action.triggered.connect(self.open_fill_holes_dialog)
        self.toolbar.addAction(self.fill_holes_action)

        self.toolbar.addSeparator()

        self.plane_mode_action = QAction("3-Point Plane", self)
        self.plane_mode_action.setCheckable(True)
        self.plane_mode_action.triggered.connect(self.toggle_plane_mode)
        self.toolbar.addAction(self.plane_mode_action)
        
        self.check_holes_action = QAction("Auto Repair & Fill", self)
        self.check_holes_action.setToolTip("Fix cracks and fill holes with high-density mesh")
        self.check_holes_action.triggered.connect(lambda: self.repair_and_fill_holes(1000.0))
        self.toolbar.addAction(self.check_holes_action)

        self.toolbar.addSeparator()

        self.brush_action = QAction("Delete Brush", self)
        self.brush_action.setCheckable(True)
        self.brush_action.triggered.connect(self.toggle_brush_mode)
        self.toolbar.addAction(self.brush_action)
        
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QLabel(" Size: "))
        
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(10, 500) 
        self.brush_size_spin.setSingleStep(5)
        self.brush_size_spin.setValue(50)      
        self.brush_size_spin.setSuffix(" px")
        self.brush_size_spin.valueChanged.connect(self.on_brush_size_changed)
        self.toolbar.addWidget(self.brush_size_spin)
        
        self.delete_action = QAction("Delete", self)
        self.delete_action.setShortcut(Qt.Key_Delete)
        self.delete_action.triggered.connect(self.delete_selected_mesh_wrapper)
        self.toolbar.addAction(self.delete_action)

        self.toolbar.addSeparator()

        self.del_floating_action = QAction("Delete Floating", self)
        self.del_floating_action.setToolTip("Remove all loose parts, keeping only the largest connected mesh.")
        # Optional: Set an icon if you have one
        # self.del_floating_action.setIcon(QIcon("path/to/icon.png")) 
        self.del_floating_action.triggered.connect(self.remove_floating_islands)
        self.toolbar.addAction(self.del_floating_action)

        self.toolbar.addSeparator()

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(self.command_manager.undo)
        self.undo_action.setEnabled(False)
        self.toolbar.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.triggered.connect(self.command_manager.redo)
        self.redo_action.setEnabled(False)
        self.toolbar.addAction(self.redo_action)

        # Layout
        self.dental_wizard = DentalWizardSidebar(self)
        container = QWidget()
        main_layout = QHBoxLayout(container)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(self.dental_wizard, 0)
        main_layout.addWidget(self.plotter, 1)
        self.setCentralWidget(container)

        # Hierarchy Dock
        self.hierarchy_dock = QDockWidget("Scene Hierarchy", self)
        self.hierarchy_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.hierarchy_panel = HierarchyPanel(self, self.plotter)
        
        self.hierarchy_panel.item_selected.connect(self.on_hierarchy_selection)
        self.hierarchy_panel.item_renamed.connect(self.on_mesh_renamed)
        self.hierarchy_panel.delete_requested.connect(self.batch_delete_meshes)
        
        self.hierarchy_dock.setWidget(self.hierarchy_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.hierarchy_dock)

    # --- Material Handler Methods ---
    def apply_material_preset(self, preset_name):
        if not self.active_actor:
            QMessageBox.warning(self, "No Selection", "Please select an object to change its material.")
            return

        props = {}
        original_texture = self.original_textures.get(self.active_actor)

        if preset_name == "mauve":
            props = {
                'color': (0.91, 0.76, 0.90),
                'diffuse': 0.9, 'specular': 0.05, 'specular_power': 10,
                'ambient': 0.4, 'metallic': 0.0, 'roughness': 0.9,
                'texture': None
            }
        
        elif preset_name == "fresnel":
            props = {
                'color': (0.7, 0.4, 0.3),
                'diffuse': 0.8, 'specular': 0.5, 'specular_power': 60,
                'ambient': 0.2, 'metallic': 0.1, 'roughness': 0.3,
                'texture': None
            }

        elif preset_name == "reflective":
            props = {
                'color': (0.95, 0.95, 0.95),
                'diffuse': 0.6, 'specular': 0.9, 'specular_power': 100,
                'ambient': 0.2, 'metallic': 0.8, 'roughness': 0.1,
                'texture': None
            }
            
        elif preset_name == "rough":
            props = {
                'color': (0.8, 0.8, 0.8),
                'diffuse': 0.9, 'specular': 0.0, 'specular_power': 0,
                'ambient': 0.5, 'metallic': 0.0, 'roughness': 1.0,
                'texture': None
            }
            
        elif preset_name == "original":
            target_color = (1.0, 1.0, 1.0) if original_texture else (1.0, 1.0, 1.0)
            props = {
                'color': target_color,
                'diffuse': 0.7, 'specular': 0.2, 'specular_power': 30,
                'ambient': 0.3, 'metallic': 0.0, 'roughness': 0.5,
                'texture': original_texture 
            }

        cmd = MaterialChangeCommand(self, self.active_actor, props)
        self.command_manager.execute(cmd)

    # ==========================================
    #            SCREEN-SPACE BRUSH LOGIC
    # ==========================================

    def on_brush_size_changed(self, value):
        self.brush_radius_px = value
        if self.brush_active and self.last_brush_pos is not None:
            self.update_brush_visuals(self.last_brush_pos)

    def open_brush_size_dialog(self):
        current_px = getattr(self, 'brush_radius_px', 50)
        
        val, ok = QInputDialog.getInt(
            self, 
            "Brush Settings", 
            "Brush Size (Screen Pixels):", 
            value=int(current_px), 
            minValue=5, 
            maxValue=500, 
            step=5
        )
        if ok:
            self.brush_radius_px = val
            if self.brush_active and self.last_brush_pos is not None:
                self.update_brush_visuals(self.last_brush_pos)

    def toggle_brush_mode(self, checked):
        self.brush_active = checked
        if checked:
            # DISABLE other tools
            self.move_action.setChecked(False)
            self.rotate_action.setChecked(False)
            self.plane_mode_action.setChecked(False)
            self.destroy_gizmo()
            
            # Switch to Sculpt Visuals
            self.apply_sculpt_visuals()
            
            if hasattr(self, 'brush_size_spin'):
                self.brush_radius_px = self.brush_size_spin.value()
            else:
                self.brush_radius_px = 50
            
            self.enable_brush_interaction()
        else:
            # Revert to Clinical Visuals
            self.reset_clinical_visuals()
            
            self.disable_brush_interaction()

    def enable_brush_interaction(self):
        if not self.active_actor: return
        mesh = self.active_actor.mapper.dataset
        
        if "_brush_mask" not in mesh.cell_data:
            mesh.cell_data["_brush_mask"] = np.zeros(mesh.n_cells, dtype=int)
        
        self.brush_indices.clear()
        if self.highlight_actor: self.highlight_actor.SetVisibility(False)
        
        self.brush_locator = vtk.vtkPointLocator()
        self.brush_locator.SetDataSet(mesh)
        self.brush_locator.BuildLocator()

        # ==========================================
        #    VISUALS
        # ==========================================
        
        self.ring_source = vtk.vtkDiskSource()
        self.ring_source.SetCircumferentialResolution(64)
        self.ring_source.SetNormal(0, 0, 1) 
        
        self.ring_transform = vtk.vtkTransform()
        self.ring_transform_filter = vtk.vtkTransformPolyDataFilter()
        self.ring_transform_filter.SetInputConnection(self.ring_source.GetOutputPort())
        self.ring_transform_filter.SetTransform(self.ring_transform)
        
        mapper_ring = vtk.vtkPolyDataMapper()
        mapper_ring.SetInputConnection(self.ring_transform_filter.GetOutputPort())
        mapper_ring.SetResolveCoincidentTopologyToPolygonOffset()
        mapper_ring.SetRelativeCoincidentTopologyPolygonOffsetParameters(0, -68) 
        
        self.cursor_ring_actor = vtk.vtkActor()
        self.cursor_ring_actor.SetMapper(mapper_ring)
        self.cursor_ring_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.cursor_ring_actor.GetProperty().SetLighting(False)
        self.cursor_ring_actor.SetPickable(False)

        self.fill_source = vtk.vtkDiskSource()
        self.fill_source.SetCircumferentialResolution(64)
        self.fill_source.SetInnerRadius(0) 
        self.fill_source.SetNormal(0, 0, 1)

        self.fill_transform_filter = vtk.vtkTransformPolyDataFilter()
        self.fill_transform_filter.SetInputConnection(self.fill_source.GetOutputPort())
        self.fill_transform_filter.SetTransform(self.ring_transform) 
        
        mapper_fill = vtk.vtkPolyDataMapper()
        mapper_fill.SetInputConnection(self.fill_transform_filter.GetOutputPort())
        mapper_fill.SetResolveCoincidentTopologyToPolygonOffset()
        mapper_fill.SetRelativeCoincidentTopologyPolygonOffsetParameters(0, -66)

        self.cursor_fill_actor = vtk.vtkActor()
        self.cursor_fill_actor.SetMapper(mapper_fill)
        self.cursor_fill_actor.GetProperty().SetColor(0.0, 0.0, 0.0) 
        self.cursor_fill_actor.GetProperty().SetOpacity(0.7)
        self.cursor_fill_actor.GetProperty().SetLighting(False)
        self.cursor_fill_actor.SetPickable(False)

        self.cursor_patch_actor = vtk.vtkActor()
        patch_mapper = vtk.vtkDataSetMapper()
        
        patch_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        patch_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1.0, -100)
        
        self.cursor_patch_actor.SetMapper(patch_mapper)
        self.cursor_patch_actor.GetProperty().SetColor(1.0, 0.0, 0.0) 
        self.cursor_patch_actor.GetProperty().SetLighting(False)
        self.cursor_patch_actor.SetPickable(False)
        self.cursor_patch_actor.SetVisibility(False)

        self.plotter.add_actor(self.cursor_ring_actor)
        self.plotter.add_actor(self.cursor_fill_actor)
        self.plotter.add_actor(self.cursor_patch_actor)
        
        self.brush_radius_px = self.brush_size_spin.value()
        self.last_brush_pos = None
        self.is_brushing_now = False

        self.style_brush = BrushInteractorStyle(self)
        self.plotter.iren.interactor.SetInteractorStyle(self.style_brush)
        
        self.reset_picker()
        self.plotter.render()

    def disable_brush_interaction(self):
        self.brush_active = False 
        self.is_brushing_now = False
        
        self.plotter.enable_trackball_style()
        self.apply_right_click_pan() 
        
        if getattr(self, 'cursor_ring_actor', None):
            self.cursor_ring_actor.SetVisibility(False)
            self.plotter.remove_actor(self.cursor_ring_actor)
            self.cursor_ring_actor = None
            
        if getattr(self, 'cursor_fill_actor', None):
            self.cursor_fill_actor.SetVisibility(False)
            self.plotter.remove_actor(self.cursor_fill_actor)
            self.cursor_fill_actor = None
            
        if getattr(self, 'cursor_patch_actor', None):
            self.cursor_patch_actor.SetVisibility(False)
            self.plotter.remove_actor(self.cursor_patch_actor)
            self.cursor_patch_actor = None

        self.ring_source = None
        self.fill_source = None
        self.ring_transform = None
        self.brush_locator = None
        self.brush_indices.clear()
        
        if self.active_actor:
            mesh = self.active_actor.mapper.dataset
            if "_brush_mask" in mesh.cell_data:
                del mesh.cell_data["_brush_mask"]
        
        if self.highlight_actor:
            self.highlight_actor.SetVisibility(True)
            
        self.enable_object_selection_mode()
        self.plotter.render()

    def get_dynamic_world_radius(self, world_center_pos, screen_radius_px):
        renderer = self.plotter.renderer
        
        renderer.SetWorldPoint(world_center_pos + (1.0,))
        renderer.WorldToDisplay()
        disp_pt = renderer.GetDisplayPoint()
        
        x_offset = disp_pt[0] + screen_radius_px
        
        renderer.SetDisplayPoint(x_offset, disp_pt[1], disp_pt[2])
        renderer.DisplayToWorld()
        world_pt_edge = renderer.GetWorldPoint()
        
        if world_pt_edge[3] != 0:
            world_pt_edge = np.array(world_pt_edge[:3]) / world_pt_edge[3]
        else:
            world_pt_edge = np.array(world_pt_edge[:3])
            
        return np.linalg.norm(np.array(world_center_pos) - world_pt_edge)

    def update_brush_visuals(self, pos):
        if not self.brush_active: return
        if self.is_processing: return
        if not self.active_actor or not hasattr(self, 'cursor_ring_actor'): return
        
        x, y = self.plotter.iren.interactor.GetEventPosition()
        
        picker = vtk.vtkCellPicker()
        picker.InitializePickList()
        picker.AddPickList(self.active_actor)
        picker.PickFromListOn()
        picker.Pick(x, y, 0, self.plotter.renderer)
        
        if picker.GetActor() != self.active_actor:
            self.cursor_ring_actor.SetVisibility(False)
            self.cursor_fill_actor.SetVisibility(False)
            self.plotter.render()
            return
            
        world_pos = np.array(picker.GetPickPosition())
        normal = np.array(picker.GetPickNormal())

        offset_pos = world_pos + (normal * 0.1)

        outer_radius = self.get_dynamic_world_radius(tuple(world_pos), self.brush_radius_px)
        inner_radius = outer_radius * 0.8 
        
        self.ring_source.SetOuterRadius(outer_radius)
        self.ring_source.SetInnerRadius(inner_radius)
        
        self.fill_source.SetOuterRadius(inner_radius)
        
        self.ring_transform.Identity()
        self.ring_transform.Translate(offset_pos)
        
        z_axis = np.array([0.0, 0.0, 1.0])
        rotation_axis = np.cross(z_axis, normal)
        rotation_angle = np.degrees(np.arccos(np.dot(z_axis, normal)))
        
        self.ring_transform.RotateWXYZ(rotation_angle, rotation_axis)
        
        self.cursor_ring_actor.SetVisibility(True)
        self.cursor_fill_actor.SetVisibility(True)
        self.plotter.render()

    def on_brush_hover(self, obj, event):
        if not self.brush_active: return
        
        x, y = self.plotter.iren.interactor.GetEventPosition()
        
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.InitializePickList()
        if self.active_actor:
            picker.AddPickList(self.active_actor)
            picker.PickFromListOn()
        
        picker.Pick(x, y, 0, self.plotter.renderer)
        
        if picker.GetActor() != self.active_actor:
            if hasattr(self, 'cursor_ring_actor') and self.cursor_ring_actor:
                self.cursor_ring_actor.SetVisibility(False)
            
            if hasattr(self, 'cursor_fill_actor') and self.cursor_fill_actor:
                self.cursor_fill_actor.SetVisibility(False)

            self.plotter.render()
            return

        pos = picker.GetPickPosition()
        self.last_brush_pos = pos
        self.update_brush_visuals(pos)

    def on_brush_action(self, obj, event):
        if not self.active_actor or self.last_brush_pos is None: return
        
        if not (QApplication.mouseButtons() & Qt.LeftButton):
            self.is_brushing_now = False
            return

        pos = self.last_brush_pos
        mesh = self.active_actor.mapper.dataset
        
        current_world_radius = self.get_dynamic_world_radius(pos, self.brush_radius_px)
        
        result = vtk.vtkIdList()
        self.brush_locator.FindPointsWithinRadius(current_world_radius, pos, result)
        
        num_points = result.GetNumberOfIds()
        is_modified = False

        if num_points > 0:
            cell_ids = vtk.vtkIdList()
            current_mask = mesh.cell_data["_brush_mask"]
            
            for i in range(num_points):
                pid = result.GetId(i)
                mesh.GetPointCells(pid, cell_ids)
                for j in range(cell_ids.GetNumberOfIds()):
                    cid = cell_ids.GetId(j)
                    if current_mask[cid] == 0:
                        current_mask[cid] = 1
                        self.brush_indices.add(cid)
                        is_modified = True
            
            if is_modified:
                thresh = vtk.vtkThreshold()
                thresh.SetInputData(mesh)
                thresh.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "_brush_mask")
                thresh.SetLowerThreshold(0.5)
                thresh.SetUpperThreshold(1.5)
                thresh.Update()
                
                selection_mesh = vtk.vtkUnstructuredGrid()
                selection_mesh.DeepCopy(thresh.GetOutput())
                
                self.cursor_patch_actor.GetMapper().SetInputData(selection_mesh)
                self.cursor_patch_actor.SetVisibility(True)
                
                self.plotter.render()
            
    def open_color_picker(self):
        if not self.active_actor:
            QMessageBox.warning(self, "No Selection", "Please select an object to paint.")
            return
            
        color = QColorDialog.getColor()
        if color.isValid():
            rgb = (color.red()/255, color.green()/255, color.blue()/255)
            
            props = {
                'color': rgb,
                'diffuse': 0.9,
                'specular': 0.1,
                'specular_power': 10,
                'ambient': 0.4,
                'metallic': 0.0,
                'roughness': 0.8,
                'texture': None 
            }
            
            cmd = MaterialChangeCommand(self, self.active_actor, props)
            self.command_manager.execute(cmd)

    def setup_lights(self):
        self.plotter.remove_all_lights()
        
        # 1. CLEANUP
        if hasattr(self.plotter.renderer, 'disable_ssao'):
            try: self.plotter.renderer.disable_ssao()
            except: pass

        # 2. HEADLIGHT
        light = pv.Light(light_type='headlight') 
        light.intensity = 0.8
        self.plotter.add_light(light)

        # 3. SPECULARITY
        if self.active_actor:
            prop = self.active_actor.GetProperty()
            prop.SetSpecular(0.6)       
            prop.SetSpecularPower(40)   
            prop.SetAmbient(0.3)

    def _clear_render_passes(self):
        renderer = self.plotter.renderer
        
        if hasattr(renderer, 'disable_ssao'):
            try: renderer.disable_ssao()
            except: pass
            
        try: self.plotter.disable_shadows()
        except: pass
        
        try: self.plotter.disable_depth_peeling()
        except: pass
        
        if hasattr(renderer, 'disable_eye_dome_lighting'):
            try: renderer.disable_eye_dome_lighting()
            except: pass
     
    def apply_sculpt_visuals(self):
        if not self.active_actor: return
        
        self._clear_render_passes()

        try:
            self.plotter.enable_ssao(radius=5.0, bias=0.5)
        except: pass

        self.plotter.remove_all_lights()
        
        key = pv.Light(position=(-100, 100, 100), focal_point=(0, 0, 0))
        key.intensity = 0.8
        key.set_camera_light() 
        self.plotter.add_light(key)
        
        fill = pv.Light(position=(100, 0, 50), focal_point=(0, 0, 0))
        fill.intensity = 0.4
        fill.diffuse_color = (0.9, 0.9, 1.0) 
        fill.set_camera_light()
        self.plotter.add_light(fill)

        self.active_actor.SetTexture(None)
        if self.active_actor.mapper:
            self.active_actor.mapper.SetScalarVisibility(False)

        prop = self.active_actor.GetProperty()
        prop.SetInterpolationToPBR()
        prop.SetColor(0.8, 0.4, 0.3) 
        prop.SetMetallic(0.0)
        prop.SetRoughness(0.5)        

        self.plotter.render()

    def reset_clinical_visuals(self):
        if not self.active_actor: return

        self._clear_render_passes()

        mapper = self.active_actor.GetMapper()
        prop = self.active_actor.GetProperty()
        
        if mapper: mapper.SetScalarVisibility(True)
        
        prop.SetColor(0.91, 0.76, 0.90) 
        prop.SetInterpolationToPhong() 
        
        self.setup_lights()
        
        self.plotter.render()

    def add_infinite_grid(self):
        grid_size = 20000
        step_size = 1000     
        x_vals = np.arange(-grid_size, grid_size + step_size, step_size)
        y_vals = np.arange(-grid_size, grid_size + step_size, step_size)
        x, y = np.meshgrid(x_vals, y_vals)
        z = np.zeros_like(x)
        grid = pv.StructuredGrid(x, y, z)
        self.plotter.add_mesh(grid, style='wireframe', color='gray', opacity=0.5, line_width=1, pickable=False, reset_camera=False)

    def update_undo_redo_ui(self):
        self.undo_action.setEnabled(len(self.command_manager.undo_stack) > 0)
        self.redo_action.setEnabled(len(self.command_manager.redo_stack) > 0)

    # --- PICKING SYSTEM ---
    def reset_picker(self):
        try: self.plotter.disable_picking()
        except: pass

    def setup_picking(self):
        self.enable_object_selection_mode()

    def enable_object_selection_mode(self):
        self.reset_picker()
        
        self.plotter.enable_trackball_style()
        self.apply_right_click_pan()
        
        self._pick_start_pos = None

        def _on_pick_down(obj, event):
            style = self.plotter.iren.interactor.GetInteractorStyle()
            style_name = style.__class__.__name__
            if "Sculpt" in style_name or "Bezier" in style_name:
                return 

            try:
                self._pick_start_pos = self.plotter.iren.interactor.GetEventPosition()
            except AttributeError:
                self._pick_start_pos = self.plotter.iren.get_event_position()
                

        def _on_pick_up(obj, event):
            style = self.plotter.iren.interactor.GetInteractorStyle()
            style_name = style.__class__.__name__
            if "Sculpt" in style_name or "Bezier" in style_name:
                return 

            if self._pick_start_pos is None: return
            
            try:
                end_pos = self.plotter.iren.interactor.GetEventPosition()
            except AttributeError:
                end_pos = self.plotter.iren.get_event_position()
            
            dist = np.sqrt((end_pos[0] - self._pick_start_pos[0])**2 + (end_pos[1] - self._pick_start_pos[1])**2)
            
            if dist < 3.0:
                picker = vtk.vtkPropPicker()
                picker.Pick(end_pos[0], end_pos[1], 0, self.plotter.renderer)
                actor = picker.GetActor()
                
                self.on_object_picked(actor)
            
            self._pick_start_pos = None
            
        self.plotter.iren.remove_observers("LeftButtonPressEvent")
        self.plotter.iren.remove_observers("LeftButtonReleaseEvent")
        
        self.plotter.iren.add_observer("LeftButtonPressEvent", _on_pick_down)
        self.plotter.iren.add_observer("LeftButtonReleaseEvent", _on_pick_up)

    def on_object_picked(self, actor):
        if self.updating_selection: return
        if self.brush_active: return 
        if actor is None: return
        
        is_node = (actor.name and "BezierNode" in actor.name)
        
        if is_node or actor in self.actors.values():
            self.set_active_actor(actor, update_list=True)

    def on_hierarchy_selection(self, name):
        if self.updating_selection: return
        if name in self.actors:
            self.set_active_actor(self.actors[name], update_list=False)

    def on_mesh_renamed(self, old_name, new_name):
        if old_name in self.actors:
            self.actors[new_name] = self.actors.pop(old_name)
            self.hierarchy_panel.update_item_name(old_name, new_name)

    # --- 3-POINT PLANE ---
    def toggle_plane_mode(self, checked):
        if checked:
            self.move_action.setChecked(False)
            self.rotate_action.setChecked(False)
            self.brush_action.setChecked(False)
            self.destroy_gizmo()
            self.picked_points.clear()
            self._clear_plane_markers()
            if self.plane_actor:
                self.plotter.remove_actor(self.plane_actor)
                self.plane_actor = None
            self.enable_point_picking_mode()
        else:
            self.enable_object_selection_mode()
            self.update_visuals()

    def enable_point_picking_mode(self):
        self.reset_picker()
        self.plotter.enable_point_picking(
            callback=self.on_point_picked_for_plane,
            show_message=False, show_point=False, left_clicking=True, use_mesh=True 
        )
        self.apply_right_click_pan()

    def on_point_picked_for_plane(self, *args):
        point = self.plotter.picked_point
        if point is None: return
        if self.plane_actor is not None:
            self.plotter.remove_actor(self.plane_actor)
            self.plane_actor = None
            self._clear_plane_markers()
            self.picked_points.clear()
        self.picked_points.append(point)
        marker_mesh = pv.Sphere(radius=2.0, center=point)
        actor = self.plotter.add_mesh(marker_mesh, color='yellow', pickable=False, reset_camera=False)
        self.point_markers.append(actor)
        if len(self.picked_points) == 3:
            self.draw_plane_from_points()
        self.plotter.render()

    def _clear_plane_markers(self):
        for a in self.point_markers: self.plotter.remove_actor(a)
        self.point_markers.clear()

    def draw_plane_from_points(self):
        p1, p2, p3 = np.array(self.picked_points[0]), np.array(self.picked_points[1]), np.array(self.picked_points[2])
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-6:
            self._clear_plane_markers()
            self.picked_points.clear()
            return
        normal = normal / norm_mag
        center = (p1 + p2 + p3) / 3.0
        radius = max(np.linalg.norm(p1-center), np.linalg.norm(p2-center), np.linalg.norm(p3-center)) * 1.5 
        disk = pv.Disc(center=center, inner=0, outer=radius, normal=normal)
        self.plane_actor = self.plotter.add_mesh(disk, color='cyan', opacity=0.5, name="ThreePointPlane", pickable=False)

    # --- MODES ---
    def toggle_move_mode(self, checked):
        if checked:
            self.plane_mode_action.setChecked(False)
            self.rotate_action.setChecked(False)
            self.brush_action.setChecked(False)
            self.enable_object_selection_mode()
            self.update_gizmo_target()
        else:
            self.destroy_gizmo()
            self.enable_object_selection_mode()
        self.update_visuals()

    def toggle_rotate_mode(self, checked):
        if checked:
            self.plane_mode_action.setChecked(False)
            self.move_action.setChecked(False)
            self.brush_action.setChecked(False)
            self.enable_object_selection_mode()
            self.update_gizmo_target()
        else:
            self.destroy_gizmo()
            self.enable_object_selection_mode()
        self.update_visuals()

    def open_fill_holes_dialog(self):
        if not self.active_actor:
            QMessageBox.warning(self, "Warning", "Please select a mesh first.")
            return
        
        dlg = HoleFillDialog(self, self.active_actor)
        dlg.show()
     
    def update_gizmo_target(self):
        if GizmoTool is None or GizmoRot is None: return 
        self.destroy_gizmo()
        if not self.active_actor: return
        if self.move_action.isChecked():
            self.gizmo = GizmoTool(self.plotter, self.active_actor, on_drag_start=self._snapshot_state, on_drag_end=self._commit_state)
        elif self.rotate_action.isChecked():
            self.gizmo = GizmoRot(self.plotter, self.active_actor, on_drag_start=self._snapshot_state, on_drag_end=self._commit_state)
        if self.gizmo: self.gizmo.update_positions()
        self.plotter.render()  

    def destroy_gizmo(self):
        if self.gizmo:
            self.gizmo.destroy()
            self.gizmo = None

    def set_active_actor(self, actor, update_list=True):
        if actor == self.active_actor: return
        self.updating_selection = True 
        
        if self.brush_active:
             self.disable_brush_interaction()
             self.brush_action.setChecked(False)

        self.active_actor = actor
        self.update_selection_highlight()
        self.update_visuals()
        if self.move_action.isChecked() or self.rotate_action.isChecked():
            self.update_gizmo_target()
        else:
            self.destroy_gizmo()
        if update_list:
            actor_name = next((k for k, v in self.actors.items() if v == actor), None)
            if actor_name: self.hierarchy_panel.select_item(actor_name)
        self.plotter.render()
        self.updating_selection = False
        self.brush_locator = None

    def update_visuals(self):
        inspect_on = self.inspect_action.isChecked()
        for name, actor in self.actors.items():
            if not actor or not hasattr(actor, 'prop'): continue
            prop = actor.prop
            if (actor == self.active_actor) and inspect_on:
                prop.show_edges = True
                prop.edge_color = 'teal'
                prop.line_width = 2.0
            else:
                prop.show_edges = False
     
    def apply_right_click_pan(self):
        style = self.plotter.iren.interactor.GetInteractorStyle()
        
        if isinstance(style, vtk.vtkInteractorStyleTrackballCamera):
            
            def _on_right_down(obj, event):
                obj.StartPan()
                if hasattr(obj, 'SetAbortFlag'):
                    obj.SetAbortFlag(1)

            def _on_right_up(obj, event):
                obj.EndPan()
                if hasattr(obj, 'SetAbortFlag'):
                    obj.SetAbortFlag(1)

            style.RemoveObservers("RightButtonPressEvent")
            style.RemoveObservers("RightButtonReleaseEvent")
            style.AddObserver("RightButtonPressEvent", _on_right_down, 10.0)
            style.AddObserver("RightButtonReleaseEvent", _on_right_up, 10.0)

    def update_selection_highlight(self):
        if self.highlight_actor:
            self.plotter.remove_actor(self.highlight_actor)
            self.highlight_actor = None
        
        if not self.active_actor: return
        
        is_node = (self.active_actor.name and "BezierNode" in self.active_actor.name)
        if is_node: return

        try:
            original_mesh = self.active_actor.mapper.dataset
            
            target_points = 3000  
            current_points = original_mesh.n_points
            
            if current_points > target_points:
                reduction = 1.0 - (target_points / current_points)
                try:
                    highlight_mesh = original_mesh.decimate(reduction)
                except:
                    highlight_mesh = original_mesh
            else:
                highlight_mesh = original_mesh

            silhouette = vtk.vtkPolyDataSilhouette()
            silhouette.SetInputData(highlight_mesh)
            silhouette.SetCamera(self.plotter.camera)
            
            silhouette.SetProp3D(self.active_actor)
            silhouette.SetEnableFeatureAngle(0)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(silhouette.GetOutputPort())
            
            self.highlight_actor = vtk.vtkActor()
            self.highlight_actor.SetMapper(mapper)
            
            prop = self.highlight_actor.GetProperty()
            prop.SetColor(0.9, 0.1, 0.1)  
            prop.SetLineWidth(3.0)       
            prop.SetLighting(False)      
            prop.SetOpacity(1.0)
            
            self.highlight_actor.SetPickable(False)
            self.plotter.add_actor(self.highlight_actor)
            
        except Exception as e:
            print(f"Error creating outline: {e}")

    def sync_highlight_motion(self):
        if not (self.active_actor and self.highlight_actor): return
        
        is_visible = self.active_actor.GetVisibility()
        if self.highlight_actor.GetVisibility() != is_visible:
             self.highlight_actor.SetVisibility(is_visible)

    # --- STATE MANAGEMENT ---
    def _get_matrix_as_array(self, actor):
        if not hasattr(actor, "GetMatrix"): return np.eye(4)
        vtk_matrix = actor.GetMatrix()
        mat = np.eye(4)
        for i in range(4):
            for j in range(4):
                mat[i, j] = vtk_matrix.GetElement(i, j)
        return mat

    def _snapshot_state(self):
        if self.active_actor: self._snapshot_matrix = self._get_matrix_as_array(self.active_actor)

    def _commit_state(self):
        if self.active_actor and self._snapshot_matrix is not None:
            current_matrix = self._get_matrix_as_array(self.active_actor)
            if not np.allclose(self._snapshot_matrix, current_matrix, atol=1e-5):
                cmd = TransformCommand(self, self.active_actor, self._snapshot_matrix, current_matrix)
                self.command_manager.push_existing(cmd)
            self._snapshot_matrix = None

    def toggle_inspect_mode(self, checked):
        if checked:
            self.move_action.setChecked(False)
            self.rotate_action.setChecked(False)
            self.brush_action.setChecked(False)
            self.toggle_move_mode(False)
        self.update_visuals()

    # ==========================================
    #     REPLACED: REPAIR & FILL LOGIC
    # ==========================================
    
    def repair_and_fill_holes(self, hole_size=1000000.0):
        """
        Replaces the old 'debug_mesh_holes'.
        This function now performs a high-quality repair by:
        1. Welding cracks (fixing import issues).
        2. Calculating the target mesh density.
        3. Filling holes.
        4. Subdividing the filled areas to match the target density.
        5. Smoothing the new geometry.
        """
        if not self.active_actor:
            print("No active mesh to check.")
            return

        print("Checking and repairing mesh...")
        
        # 1. Get current mesh
        original_mesh = self.active_actor.mapper.dataset
        if isinstance(original_mesh, pv.UnstructuredGrid):
            original_mesh = original_mesh.extract_surface()
        
        # 2. Fix Import Cracks (Weld duplicate points)
        try:
            # FIXED: 'merge_tolerance' replaced with 'tolerance' + 'absolute=True'
            cleaned_mesh = original_mesh.clean(point_merging=True, tolerance=1e-4, absolute=True)
        except Exception as e:
            print(f"Cleaning failed: {e}")
            cleaned_mesh = original_mesh

        # 3. CALCULATE DENSITY (Average Edge Length)
        try:
            # Approximate density based on cell area
            sizes = cleaned_mesh.compute_cell_sizes(length=False, area=True, volume=False)
            areas = sizes.cell_data["Area"]
            median_area = np.median(areas)
            target_edge_length = np.sqrt(median_area * 4.0 / 1.732)
            print(f"Target Geometric Resolution (Edge Length): {target_edge_length:.4f} units")
        except:
            target_edge_length = 0.5

        # 4. Fill Holes (Creates large patches)
        try:
            filled_mesh = cleaned_mesh.fill_holes(hole_size=hole_size)
        except Exception as e:
            print(f"Hole filling failed: {e}")
            return

        # 5. ADAPTIVE SUBDIVISION
        # Splits edges to match the target density of the rest of the mesh.
        subdivider = vtk.vtkAdaptiveSubdivisionFilter()
        subdivider.SetInputData(filled_mesh)
        subdivider.SetMaximumEdgeLength(target_edge_length)
        subdivider.SetEdgeLengthCriterion(1) 
        subdivider.Update()
        
        high_res_filled = pv.wrap(subdivider.GetOutput())

        # 6. Smooth/Fairing
        # Relax new vertices to follow curvature
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(high_res_filled)
        smoother.SetNumberOfIterations(20)
        smoother.SetPassBand(0.1) 
        smoother.FeatureEdgeSmoothingOff() 
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        
        final_mesh = pv.wrap(smoother.GetOutput())
        final_mesh.compute_normals(auto_orient_normals=True, inplace=True)

        # 7. Visualization
        self.plotter.remove_actor("Debug_Hole_Fill")
        self.plotter.remove_actor("Debug_Original_Overlay")
        
        self.plotter.add_mesh(
            final_mesh, 
            name="Debug_Hole_Fill",
            color="red", 
            lighting=False, 
            pickable=False
        )
        
        display_original = original_mesh.copy()
        if "Normals" in display_original.point_data:
            display_original.points += display_original.point_data["Normals"] * 0.02
        
        self.plotter.add_mesh(
            display_original, 
            name="Debug_Original_Overlay", 
            color="white", 
            opacity=0.3,
            pickable=False
        )
        
        msg = f"Points: {original_mesh.n_points} -> {final_mesh.n_points}\nResolution matched to {target_edge_length:.3f}mm"
        QMessageBox.information(self, "Auto Repair Complete", msg)
        self.plotter.render()

    def load_dummy_data(self):
        mesh1 = pv.Sphere(center=(0,0,0), radius=50)
        self._internal_add_mesh(mesh1, "Dummy_Sphere_1")
        mesh2 = pv.Sphere(center=(120,0,0), radius=50)
        self._internal_add_mesh(mesh2, "Dummy_Sphere_2")

    def _internal_add_mesh(self, mesh, name):
        cmd = AddMeshCommand(self, mesh, name)
        cmd.execute()

    def open_file_dialog(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Meshes", "", "Mesh Files (*.stl *.obj *.ply)")
        if file_paths:
            for path in file_paths:
                self.load_mesh(path)

    def load_mesh(self, file_path):
        try:
            mesh = pv.read(file_path)
            
            # --- FIX FOR CRACKS ON IMPORT ---
            if hasattr(mesh, 'clean'):
                # FIXED: 'merge_tolerance' replaced with 'tolerance' + 'absolute=True'
                mesh = mesh.clean(point_merging=True, tolerance=1e-4, absolute=True)
            # --------------------------------
            
            if hasattr(mesh, 'triangulate'): mesh = mesh.triangulate()
            
            if mesh.points.dtype != np.float32:
                mesh.points = mesh.points.astype(np.float32)
            
            mesh.compute_normals(auto_orient_normals=True, inplace=True)
            
            texture = self._smart_load_texture(file_path)
            
            unique_name = f"{os.path.basename(file_path)}_{len(self.actors)}"
            cmd = AddMeshCommand(self, mesh, unique_name, texture=texture)
            self.command_manager.execute(cmd)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _smart_load_texture(self, mesh_path):
        folder = os.path.dirname(mesh_path)
        filename = os.path.basename(mesh_path)
        base_name_no_ext = os.path.splitext(filename)[0]
        
        if mesh_path.lower().endswith('.obj'):
            mtl_tex = self._try_load_from_mtl(mesh_path)
            if mtl_tex: return mtl_tex

        suffixes = ["", "_diffuse", "_color", "_tex", "_albedo", ".0"] 
        extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

        for suffix in suffixes:
            for ext in extensions:
                candidate = os.path.join(folder, f"{base_name_no_ext}{suffix}{ext}")
                if os.path.exists(candidate):
                    try:
                        return pv.read_texture(candidate)
                    except: pass

        keywords = ["lower", "upper", "mandible", "maxilla"]
        mesh_name_lower = base_name_no_ext.lower()
        
        target_keyword = None
        for k in keywords:
            if k in mesh_name_lower:
                target_keyword = k
                break
        
        if target_keyword:
            try:
                for file in os.listdir(folder):
                    if file.lower().endswith(tuple(extensions)):
                        if target_keyword in file.lower():
                            full_path = os.path.join(folder, file)
                            return pv.read_texture(full_path)
            except: pass
            
        return None

    def _try_load_from_mtl(self, obj_path):
        folder = os.path.dirname(obj_path)
        mtl_filename = None
        try:
            with open(obj_path, 'r') as f:
                for line in f:
                    if line.startswith('mtllib'):
                        mtl_filename = line.split()[1]
                        break
        except: pass
        
        if not mtl_filename:
            mtl_filename = os.path.splitext(os.path.basename(obj_path))[0] + ".mtl"
            
        mtl_path = os.path.join(folder, mtl_filename)
        if not os.path.exists(mtl_path): return None
            
        texture_name = None
        try:
            with open(mtl_path, 'r') as f:
                for line in f:
                    if 'map_Kd' in line:
                        texture_name = line.split()[-1]
                        break
        except: pass
        
        if texture_name:
            tex_path = os.path.join(folder, texture_name)
            if os.path.exists(tex_path):
                return pv.read_texture(tex_path)
        return None
    
    def batch_delete_meshes(self, names_list):
        if not names_list: return
        cmd = MultiDeleteCommand(self, names_list)
        self.command_manager.execute(cmd)

    def remove_floating_islands(self):
        """
        Separates the mesh into connected components and keeps only the largest one.
        """
        if not self.active_actor:
            QMessageBox.warning(self, "No Selection", "Please select a mesh to clean.")
            return

        # 1. Get the current mesh
        original_mesh = self.active_actor.mapper.dataset
        
        # 2. Check if it's empty
        if original_mesh.n_points == 0:
            return

        self.is_processing = True
        try:
            # 3. Use PyVista's built-in extract_largest
            # This handles the connectivity algorithm automatically
            largest_part = original_mesh.extract_largest()

            # 4. Check if any change actually happened
            if largest_part.n_cells == original_mesh.n_cells:
                self.statusBar().showMessage("Mesh is already a single continuous part.", 3000)
                self.is_processing = False
                return

            # 5. Execute Command (allows Undo)
            # We copy original_mesh so the undo stack has a preserved state
            cmd = ReplaceGeometryCommand(
                self, 
                self.active_actor, 
                original_mesh.copy(), 
                largest_part
            )
            self.command_manager.execute(cmd)
            
            diff = original_mesh.n_cells - largest_part.n_cells
            self.statusBar().showMessage(f"Deleted {diff} floating cells.", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to remove floating parts: {str(e)}")
        
        self.is_processing = False

    def delete_selected_mesh_wrapper(self):
        self.is_processing = True

        if self.brush_active:
            if self.brush_indices:
                if self.active_actor:
                    if getattr(self, 'cursor_patch_actor', None):
                          if self.cursor_patch_actor.GetMapper():
                              self.cursor_patch_actor.GetMapper().RemoveAllInputs()
                          self.plotter.remove_actor(self.cursor_patch_actor)
                          self.cursor_patch_actor = None
                    
                    self.plotter.render() 
                    
                    mesh_copy = self.active_actor.mapper.dataset.copy()
                    
                    cmd = DeleteCellsCommand(self, self.active_actor, self.brush_indices.copy(), mesh_copy)
                    self.command_manager.execute(cmd)
                    
                    self.brush_indices.clear()
                    
                    new_mesh = self.active_actor.mapper.dataset
                    if "_brush_mask" not in new_mesh.cell_data:
                        new_mesh.cell_data["_brush_mask"] = np.zeros(new_mesh.n_cells, dtype=int)
                    
                    self.brush_locator = vtk.vtkPointLocator()
                    self.brush_locator.SetDataSet(new_mesh)
                    self.brush_locator.BuildLocator()

                    self.cursor_patch_actor = vtk.vtkActor()
                    patch_mapper = vtk.vtkDataSetMapper()
                    patch_mapper.SetResolveCoincidentTopologyToPolygonOffset()
                    patch_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0, -66)
                    self.cursor_patch_actor.SetMapper(patch_mapper)
                    self.cursor_patch_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
                    self.cursor_patch_actor.GetProperty().SetLighting(False)
                    self.cursor_patch_actor.SetPickable(False)
                    self.cursor_patch_actor.SetVisibility(False)
                    self.plotter.add_actor(self.cursor_patch_actor)
            
            else:
                print("Brush Mode Active: Nothing painted to delete.")

            self.is_processing = False
            return 

        if self.hierarchy_panel.tree.hasFocus():
            item = self.hierarchy_panel.tree.currentItem()
            if item:
                self.hierarchy_panel.delete_item_recursive(item)
                self.is_processing = False
                return

        if not self.active_actor: 
            self.is_processing = False
            return
        
        is_node = (self.active_actor.name and "BezierNode" in self.active_actor.name)
        if is_node:
            if hasattr(self.dental_wizard, 'maxilla_wizard') and self.dental_wizard.maxilla_wizard.bezier_tool:
                self.dental_wizard.maxilla_wizard.bezier_tool.delete_node(self.active_actor)
                self.is_processing = False
                return 
        
        active_name = next((k for k, v in self.actors.items() if v == self.active_actor), None)
        if active_name:
            cmd = DeleteMeshCommand(self, active_name, self.active_actor)
            self.command_manager.execute(cmd)
        else:
            self.plotter.remove_actor(self.active_actor)
            self.plotter.render()

        self.is_processing = False