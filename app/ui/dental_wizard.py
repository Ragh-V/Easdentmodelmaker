import sys
import time
import numpy as np
import pyvista as pv
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QStackedWidget, QHBoxLayout, QMessageBox,
    QDoubleSpinBox, QFormLayout, QGroupBox,
    QListWidget, QApplication, QSlider, QButtonGroup, QRadioButton, QCheckBox, QFrame
)
from PySide6.QtCore import Qt, QTimer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Assuming your main file is named 'main.py' or similar structure
from core.commands import AddMeshCommand, MultiCommand, DeleteMeshCommand

# Import logic from your existing utils file
from utils.generator import ModelGenerator
from tools.surveyor import UndercutSurveyor
from tools.bezier import BezierMarkerTool
from tools.sculptor import SculptTool

# ==========================================
#       STEP 1: ALIGNMENT WIDGET
# ==========================================
class AlignmentStep(QWidget):
    def __init__(self, app_interface):
        super().__init__()
        self.app = app_interface
        self.align_points = []
        self.align_markers = []
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        self.layout.addWidget(QLabel("<b>HORIZONTAL PLANE ALIGNMENT</b>"))
        self.layout.addWidget(QLabel("Pick 3 points on the occlusal plane (e.g. molars + incisor)."))
        
        self.lbl_status = QLabel("Status: Ready.")
        self.lbl_status.setStyleSheet("color: #e67e22; font-weight: bold;")
        self.layout.addWidget(self.lbl_status)

        self.btn_pick = QPushButton("Start 3-Point Pick")
        self.btn_pick.setCheckable(True)
        self.btn_pick.clicked.connect(self.toggle_picking)
        self.layout.addWidget(self.btn_pick)

        self.btn_align = QPushButton("Align Horizontal")
        self.btn_align.clicked.connect(self.run_alignment)
        self.btn_align.setEnabled(False)
        self.layout.addWidget(self.btn_align)

        self.btn_reset = QPushButton("Reset Points")
        self.btn_reset.clicked.connect(self.reset_points)
        self.layout.addWidget(self.btn_reset)
        self.layout.addStretch()

    def toggle_picking(self, checked):
        if checked:
            if not self.app.active_actor:
                self.btn_pick.setChecked(False)
                QMessageBox.warning(self, "No Mesh", "Please select a mesh in the scene first.")
                return
            self.lbl_status.setText("Pick Point 1/3")
            self.app.reset_picker()
            self.app.plotter.enable_point_picking(
                callback=self._on_pick, show_message=False, show_point=False, use_picker=True
            )
        else:
            self.app.reset_picker()
            self.app.setup_picking() 
            self.lbl_status.setText("Picking paused.")

    def _on_pick(self, mesh, idx):
        if mesh is None: return
        point = mesh.points[idx]
        self.align_points.append(point)
        
        sphere = pv.Sphere(radius=1.5, center=point)
        actor = self.app.plotter.add_mesh(sphere, color="cyan", pickable=False)
        self.align_markers.append(actor)
        
        count = len(self.align_points)
        if count < 3:
            self.lbl_status.setText(f"Pick Point {count+1}/3")
        else:
            self.lbl_status.setText("Ready to Align.")
            self.app.reset_picker()
            self.btn_pick.setChecked(False)
            self.btn_align.setEnabled(True)
            self.app.setup_picking()

    def run_alignment(self):
        if len(self.align_points) != 3: return
        p1, p2, p3 = np.array(self.align_points)
        
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        if normal[2] < 0: normal = -normal 

        target = np.array([0.0, 0.0, 1.0])
        rot_axis = np.cross(normal, target)
        angle_rad = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))
        
        matrix = pv.transformations.axis_angle_rotation(rot_axis, np.degrees(angle_rad))
        
        actor = self.app.active_actor
        current_mat = self.app._get_matrix_as_array(actor)
        
        centroid = (p1 + p2 + p3) / 3.0
        trans_to_origin = np.eye(4); trans_to_origin[:3, 3] = -centroid
        
        final_mat = matrix @ trans_to_origin
        new_combined = final_mat @ current_mat
        
        # Depending on your codebase, TransformCommand might be in core.commands
        from core.commands import TransformCommand 
        cmd = TransformCommand(self.app, actor, current_mat, new_combined)
        self.app.command_manager.execute(cmd)
        
        self.reset_points()
        self.app.plotter.reset_camera()
        self.lbl_status.setText("Aligned to Z-Axis.")

    def reset_points(self):
        self.align_points = []
        for m in self.align_markers: self.app.plotter.remove_actor(m)
        self.align_markers = []
        self.btn_align.setEnabled(False)
        self.app.plotter.render()
        self.lbl_status.setText("Points reset.")

# ==========================================
#       STEP 2: MAXILLARY WORKFLOW
# ==========================================
class MaxillarySteps(QWidget):
    def __init__(self, app_interface):
        super().__init__()
        self.app = app_interface
        
        # --- Tools ---
        self.bezier_tool = BezierMarkerTool(self.app.plotter, self.app)
        self.survey_tool = None
        self.sculpt_tool = None
        self.border_tool = None 
        
        # --- Data States ---
        self.pre_cleanup_mesh = None
        self.last_base_name = None 
        self.incisor_actor = None 
        self.last_border_path = None # Stores the loop from generator

        # --- Layout ---
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5,5,5,5)

        # --- Stack Initialization (MUST BE BEFORE _init_pages) ---
        self.stack = QStackedWidget()
        self.stack.currentChanged.connect(self._on_stack_changed)
        
        # --- Init Pages ---
        self._init_pages()
        
        # Add stack to layout
        self.layout.addWidget(self.stack)

        # --- Navigation Buttons ---
        h = QHBoxLayout()
        self.btn_back = QPushButton("< Back")
        self.btn_next = QPushButton("Next >")
        self.btn_back.clicked.connect(lambda: self.nav(-1))
        self.btn_next.clicked.connect(lambda: self.nav(1))
        h.addWidget(self.btn_back)
        h.addWidget(self.btn_next)
        self.layout.addLayout(h)

    def nav(self, delta):
        self._stop_active_tools()
        curr = self.stack.currentIndex()
        self.stack.setCurrentIndex(max(0, min(self.stack.count()-1, curr + delta)))

    # ==========================================
    #       VISIBILITY & CLEANUP
    # ==========================================
    def _on_stack_changed(self, index):
        # Papilla Visibility
        if hasattr(self, 'page_papilla'):
            is_papilla_page = (self.stack.widget(index) == self.page_papilla)
            self.set_papilla_visibility(is_papilla_page)

    def set_papilla_visibility(self, visible):
        if self.incisor_actor:
            self.incisor_actor.SetVisibility(visible)
            self.app.plotter.render()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.set_papilla_visibility(False)
        self._stop_active_tools()

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, 'page_papilla') and self.stack.currentWidget() == self.page_papilla:
            self.set_papilla_visibility(True)

    def _stop_active_tools(self):
        # Stop Bezier
        if self.bezier_tool: 
            self.bezier_tool.stop()
            self.app.setup_picking()
            
        # Stop Surveyor
        if self.survey_tool:
            self.survey_tool.stop()
            self.survey_tool = None
            
        # Stop Sculptor
        if self.sculpt_tool:
            self.sculpt_tool.stop()
            self.sculpt_tool = None
        
        # Stop Border Tool
        if self.border_tool:
            self.border_tool.stop()
            self.border_tool = None

        # Reset Buttons
        if hasattr(self, 'btn_manual_border'): self.btn_manual_border.setChecked(False)
        if hasattr(self, 'btn_papilla'): self.btn_papilla.setChecked(False)
        if hasattr(self, 'btn_sculpt_toggle'): self.btn_sculpt_toggle.setChecked(False)
        if hasattr(self, 'btn_survey'): self.btn_survey.setChecked(False)
        if hasattr(self, 'btn_border_tool'): self.btn_border_tool.setChecked(False)

    # ==========================================
    #       PAGE INITIALIZATION
    # ==========================================
    def _init_pages(self):
        # PAGE 1: IMPORT
        p1 = QWidget(); l1 = QVBoxLayout(p1)
        l1.addWidget(QLabel("<b>1. Import Maxilla</b>"))
        btn_load = QPushButton("Load STL/PLY")
        btn_load.clicked.connect(self.app.open_file_dialog)
        l1.addWidget(btn_load); l1.addStretch(); self.stack.addWidget(p1)

        # PAGE 2: CLEANUP
        p2 = QWidget(); l2 = QVBoxLayout(p2)
        l2.addWidget(QLabel("<b>2. Deep Artifact Cleanup</b>"))
        form = QFormLayout()
        self.spin_thresh = QDoubleSpinBox(); self.spin_thresh.setValue(0.5)
        self.spin_rescue = QDoubleSpinBox(); self.spin_rescue.setValue(2.1)
        form.addRow("Depth Thresh:", self.spin_thresh)
        form.addRow("Rescue Dist:", self.spin_rescue)
        l2.addLayout(form)
        btn_clean = QPushButton("Run Deep Cleanup")
        btn_clean.clicked.connect(self.run_cleanup)
        l2.addWidget(btn_clean)
        self.btn_undo = QPushButton("Undo Cleanup")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_cleanup)
        l2.addWidget(self.btn_undo); l2.addStretch(); self.stack.addWidget(p2)

        # PAGE 3: BASE
        p3 = QWidget(); l3 = QVBoxLayout(p3)
        l3.addWidget(QLabel("<b>3. Define Border & Base</b>"))
        self.btn_manual_border = QPushButton("Manual Marking Tool")
        self.btn_manual_border.setCheckable(True)
        self.btn_manual_border.clicked.connect(self.toggle_border_tool_manual)
        l3.addWidget(self.btn_manual_border)
        btn_extract = QPushButton("Extract Selection Patch")
        btn_extract.clicked.connect(self.extract_selection)
        l3.addWidget(btn_extract); l3.addSpacing(10)
        
        # Base Params
        params_group = QGroupBox("Base Parameters"); params_layout = QFormLayout()
        self.slider_height = QSlider(Qt.Horizontal); self.slider_height.setRange(5, 60); self.slider_height.setValue(20)
        self.spin_height = QDoubleSpinBox(); self.spin_height.setRange(5.0, 60.0); self.spin_height.setValue(20.0)
        self.slider_height.valueChanged.connect(lambda v: self.spin_height.setValue(v))
        self.spin_height.valueChanged.connect(lambda v: self.slider_height.setValue(int(v)))
        h_layout = QHBoxLayout(); h_layout.addWidget(self.slider_height); h_layout.addWidget(self.spin_height)
        params_layout.addRow("Height:", h_layout)
        
        self.slider_skirt = QSlider(Qt.Horizontal); self.slider_skirt.setRange(0, 50); self.slider_skirt.setValue(10)
        self.spin_skirt = QDoubleSpinBox(); self.spin_skirt.setRange(0.0, 5.0); self.spin_skirt.setValue(1.0)
        self.slider_skirt.valueChanged.connect(lambda v: self.spin_skirt.setValue(v/10.0))
        self.spin_skirt.valueChanged.connect(lambda v: self.slider_skirt.setValue(int(v*10)))
        s_layout = QHBoxLayout(); s_layout.addWidget(self.slider_skirt); s_layout.addWidget(self.spin_skirt)
        params_layout.addRow("Skirt:", s_layout)
        params_group.setLayout(params_layout); l3.addWidget(params_group)
        
        btn_base = QPushButton("Generate Solid Base")
        btn_base.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        btn_base.clicked.connect(self.generate_base)
        l3.addWidget(btn_base)
        self.btn_undo_base = QPushButton("Undo Base Generation")
        self.btn_undo_base.setEnabled(False)
        self.btn_undo_base.clicked.connect(self.undo_base)
        l3.addWidget(self.btn_undo_base); l3.addStretch(); self.stack.addWidget(p3)

        # PAGE 3b: BORDER MOD (NEW)
        self.page_border = self._create_border_mod_page()
        self.stack.addWidget(self.page_border)

        # PAGE 4: PAPILLA
        self.page_papilla = self._create_papilla_page()
        self.stack.addWidget(self.page_papilla)

        # PAGE 5: MODIFY
        self.stack.addWidget(self._create_modify_page())

        # PAGE 6: SURVEY
        p6 = QWidget(); l6 = QVBoxLayout(p6)
        l6.addWidget(QLabel("<b>6. Survey Undercuts</b>"))
        self.btn_survey = QPushButton("Enable Surveyor")
        self.btn_survey.setCheckable(True)
        self.btn_survey.clicked.connect(self.toggle_survey)
        l6.addWidget(self.btn_survey)
        self.spin_tilt = QDoubleSpinBox(); self.spin_tilt.setRange(0, 180); self.spin_tilt.setValue(180)
        self.spin_tilt.valueChanged.connect(self.update_survey_vector)
        l6.addWidget(QLabel("Tilt Angle (Theta):"))
        l6.addWidget(self.spin_tilt)
        btn_reset_path = QPushButton("See Original Path of Insertion")
        btn_reset_path.clicked.connect(self.reset_survey_path)
        l6.addWidget(btn_reset_path)
        l6.addStretch(); self.stack.addWidget(p6)

    # ==========================================
    #       NEW BORDER MOD PAGE LOGIC
    # ==========================================
    def _create_border_mod_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<b>3b. Modify Denture Border</b>"))
        layout.addWidget(QLabel("Drag points to adjust border extension."))
        
        # Tool Toggle
        self.btn_border_tool = QPushButton("Activate Border Handles")
        self.btn_border_tool.setCheckable(True)
        self.btn_border_tool.clicked.connect(self.toggle_border_tool)
        layout.addWidget(self.btn_border_tool)
        
        # Parameters
        grp = QGroupBox("Influence Settings")
        form = QFormLayout(grp)
        self.slider_inf_radius = QSlider(Qt.Horizontal)
        self.slider_inf_radius.setRange(5, 50); self.slider_inf_radius.setValue(15)
        self.slider_inf_radius.valueChanged.connect(self.update_border_params)
        form.addRow("Influence Radius:", self.slider_inf_radius)
        layout.addWidget(grp)
        
        # Reset
        btn_reset = QPushButton("Reset Mesh")
        btn_reset.clicked.connect(self.reset_border_mesh)
        layout.addWidget(btn_reset)
        layout.addStretch()
        return page

    def toggle_border_tool(self, checked):
        # Local import to prevent circular dependency
        from tools.border_tool import BorderDeformTool 
        if checked:
            if not self.app.active_actor:
                self.btn_border_tool.setChecked(False)
                QMessageBox.warning(self, "Error", "No mesh loaded.")
                return
            
            # Check for stored path from generation
            if not hasattr(self, 'last_border_path') or self.last_border_path is None:
                QMessageBox.warning(self, "Error", "No generated border data found.\nPlease generate a base first.")
                self.btn_border_tool.setChecked(False)
                return

            if not self.border_tool:
                self.border_tool = BorderDeformTool(self.app.plotter, self.app.active_actor)
            
            self.border_tool.set_radius(self.slider_inf_radius.value())
            # PASS THE EXPLICIT PATH HERE
            self.border_tool.start(explicit_path=self.last_border_path)
        else:
            if self.border_tool:
                self.border_tool.stop()
                self.border_tool = None

    def update_border_params(self):
        if self.border_tool:
            self.border_tool.set_radius(self.slider_inf_radius.value())

    def reset_border_mesh(self):
        if self.border_tool:
            self.border_tool.stop()
            self.border_tool = None
            self.btn_border_tool.setChecked(False)
        QMessageBox.information(self, "Info", "Use Undo (Step 2/3) to revert generation.")

    # ==========================================
    #       EXISTING PAGE CREATORS
    # ==========================================
    def _create_papilla_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<b>4. Mark Incisor Papilla</b>"))
        layout.addWidget(QLabel("Outline the incisive papilla region."))
        self.btn_papilla = QPushButton("Draw Region (Bezier)")
        self.btn_papilla.setCheckable(True)
        self.btn_papilla.clicked.connect(self.toggle_papilla_tool)
        layout.addWidget(self.btn_papilla)
        btn_clear_papilla = QPushButton("Clear/Reset Mark")
        btn_clear_papilla.clicked.connect(self.clear_papilla_mark)
        layout.addWidget(btn_clear_papilla)
        layout.addWidget(QLabel("<i>Note: Marking will hide when leaving this tab.</i>"))
        layout.addStretch()
        return page

    def _create_modify_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<b>5. Smooth / Modify Mesh</b>"))
        
        grp_auto = QGroupBox("Autosmooth")
        v_auto = QVBoxLayout(grp_auto)
        self.spin_crater_depth = QDoubleSpinBox()
        self.spin_crater_depth.setRange(0.01, 1.0); self.spin_crater_depth.setValue(0.05)
        v_auto.addWidget(QLabel("Depth Threshold:"))
        v_auto.addWidget(self.spin_crater_depth)
        btn_autosmooth = QPushButton("Run Autosmooth")
        btn_autosmooth.clicked.connect(self.run_autosmooth)
        v_auto.addWidget(btn_autosmooth)
        layout.addWidget(grp_auto)

        grp_sculpt = QGroupBox("Sculpting")
        v_sculpt = QVBoxLayout(grp_sculpt)
        self.btn_sculpt_toggle = QPushButton("Enable Sculpting")
        self.btn_sculpt_toggle.setCheckable(True)
        self.btn_sculpt_toggle.clicked.connect(self.toggle_sculpting)
        v_sculpt.addWidget(self.btn_sculpt_toggle)
        
        self.chk_dyntopo = QCheckBox("Dynamic Topology")
        self.chk_dyntopo.toggled.connect(self.update_sculpt_params)
        v_sculpt.addWidget(self.chk_dyntopo)
        
        self.slider_detail = QSlider(Qt.Horizontal); self.slider_detail.setRange(1, 100); self.slider_detail.setValue(20)
        self.slider_detail.valueChanged.connect(self.update_sculpt_params)
        v_sculpt.addWidget(QLabel("Detail Level:")); v_sculpt.addWidget(self.slider_detail)
        
        self.radio_group = QButtonGroup()
        r_smooth = QRadioButton("Smooth Brush"); r_smooth.setChecked(True)
        r_remove = QRadioButton("Remove Brush")
        r_add = QRadioButton("Add Brush")
        self.radio_group.addButton(r_smooth); self.radio_group.addButton(r_remove); self.radio_group.addButton(r_add)
        self.radio_group.buttonClicked.connect(self.update_sculpt_params)
        v_sculpt.addWidget(r_smooth); v_sculpt.addWidget(r_remove); v_sculpt.addWidget(r_add)
        
        self.slider_radius = QSlider(Qt.Horizontal); self.slider_radius.setRange(1, 100); self.slider_radius.setValue(30)
        self.slider_radius.valueChanged.connect(self.update_sculpt_params)
        v_sculpt.addWidget(QLabel("Brush Radius:")); v_sculpt.addWidget(self.slider_radius)
        
        self.slider_power = QSlider(Qt.Horizontal); self.slider_power.setRange(1, 100); self.slider_power.setValue(50)
        self.slider_power.valueChanged.connect(self.update_sculpt_params)
        v_sculpt.addWidget(QLabel("Brush Power:")); v_sculpt.addWidget(self.slider_power)
        
        layout.addWidget(grp_sculpt); layout.addStretch()
        return page

    # ==========================================
    #       TOOL IMPLEMENTATIONS
    # ==========================================
    def toggle_papilla_tool(self, checked):
        if checked:
            if not self.app.active_actor:
                QMessageBox.warning(self, "Error", "No mesh loaded.")
                self.btn_papilla.setChecked(False)
                return
            if self.incisor_actor:
                self.app.plotter.remove_actor(self.incisor_actor)
                self.incisor_actor = None
            self.bezier_tool.start()
        else:
            patch = self.bezier_tool.get_selected_region()
            if patch:
                self.incisor_actor = self.app.plotter.add_mesh(patch, color="#ff4081", pickable=False, name="Incisor_Papilla_Mark")
            self.bezier_tool.stop()
            self.bezier_tool.clear_all_markup()

    def clear_papilla_mark(self):
        if self.incisor_actor:
            self.app.plotter.remove_actor(self.incisor_actor)
            self.incisor_actor = None
        self.bezier_tool.stop()
        self.bezier_tool.clear_all_markup()
        self.btn_papilla.setChecked(False)
        self.app.plotter.render()

    def toggle_border_tool_manual(self, checked):
        if checked:
            if not self.app.active_actor:
                QMessageBox.warning(self, "Error", "No mesh loaded.")
                self.btn_manual_border.setChecked(False)
                return
            self.bezier_tool.start()
        else:
            self.bezier_tool.stop()

    def extract_selection(self):
        patch = self.bezier_tool.get_selected_region()
        if patch:
            self.app._internal_add_mesh(patch, f"Extracted_Patch_{int(time.time())}")
            self.bezier_tool.clear_all_markup()

    def generate_base(self):
        patch = self.bezier_tool.get_selected_region()
        if not patch:
            QMessageBox.warning(self, "Error", "No valid selection found.")
            return

        # Handle tuple return (mesh, path)
        result = ModelGenerator.create_base_from_selection(
            patch, 
            base_height=self.spin_height.value(), 
            skirt_size=self.spin_skirt.value()
        )
        
        if result:
            base_mesh, border_path = result # Unpack the tuple
            
            # Store the path for the tool
            self.last_border_path = border_path
            
            base_name = f"Solid_Base_{int(time.time())}"
            cmd_add = AddMeshCommand(self.app, base_mesh, base_name)
            cmd_clear = self.bezier_tool.clear_markup_cmd()
            
            composite = MultiCommand([cmd_add, cmd_clear])
            self.app.command_manager.execute(composite)
            
            self.last_base_name = base_name 
            self.btn_undo_base.setEnabled(True)
            QMessageBox.information(self, "Success", "Base Generated.")
        else:
            QMessageBox.warning(self, "Error", "Failed to generate base.")

    def undo_base(self):
        if hasattr(self, 'last_base_name') and self.last_base_name:
            if self.last_base_name in self.app.actors:
                actor = self.app.actors[self.last_base_name]
                cmd = DeleteMeshCommand(self.app, self.last_base_name, actor)
                self.app.command_manager.execute(cmd)
            self.last_base_name = None
            self.btn_undo_base.setEnabled(False)

    def run_autosmooth(self):
        if not self.app.active_actor: return
        self.setCursor(Qt.WaitCursor)
        mesh = self.app.active_actor.mapper.dataset
        # Assuming auto_patch_craters is imported or available in context, or add import if needed
        from utils.generator import auto_patch_craters # Safety import if not global
        patched = auto_patch_craters(mesh, depth_threshold=self.spin_crater_depth.value())
        if patched:
            self.app.active_actor.mapper.SetInputData(patched)
            self.app.active_actor.mapper.Update()
            self.app.plotter.render()
            QMessageBox.information(self, "Autosmooth", "Patched craters.")
        self.setCursor(Qt.ArrowCursor)

    def toggle_sculpting(self, checked):
        if checked:
            if not self.app.active_actor: 
                self.btn_sculpt_toggle.setChecked(False)
                return
            if not self.sculpt_tool: 
                self.sculpt_tool = SculptTool(self.app, self.app.plotter)
            self.sculpt_tool.start()
            self.update_sculpt_params()
            if hasattr(self.app, 'apply_sculpt_visuals'):
                self.app.apply_sculpt_visuals()
        else:
            if self.sculpt_tool: 
                self.sculpt_tool.stop()
            if hasattr(self.app, 'reset_clinical_visuals'):
                self.app.reset_clinical_visuals()

    def update_sculpt_params(self, *args):
        if not self.sculpt_tool: return
        text = self.radio_group.checkedButton().text()
        if "Remove" in text: mode = "REMOVE"
        elif "Add" in text: mode = "ADD"
        else: mode = "SMOOTH"
        
        radius = self.slider_radius.value() / 5.0
        strength = self.slider_power.value() / 100.0
        dyntopo = self.chk_dyntopo.isChecked()
        detail_size = 3.0 - (self.slider_detail.value() / 100.0 * 2.8) 
        self.sculpt_tool.set_params(mode, radius, strength, dyntopo, detail_size)

    def run_cleanup(self):
        if not self.app.active_actor: return
        mesh = self.app.active_actor.mapper.dataset
        self.pre_cleanup_mesh = mesh.copy()
        cleaned = ModelGenerator.clean_undesirable_artifacts(mesh, threshold=self.spin_thresh.value(), rescue_threshold=self.spin_rescue.value())
        if cleaned:
            self.app.active_actor.mapper.SetInputData(cleaned)
            self.app.active_actor.mapper.Update()
            self.app.plotter.render()
            self.btn_undo.setEnabled(True)
            QMessageBox.information(self, "Result", "Cleanup complete.")

    def undo_cleanup(self):
        if self.pre_cleanup_mesh:
            self.app.active_actor.mapper.SetInputData(self.pre_cleanup_mesh)
            self.app.active_actor.mapper.Update()
            self.app.plotter.render()
            self.pre_cleanup_mesh = None
            self.btn_undo.setEnabled(False)

    def toggle_survey(self, checked):
        if checked:
            if not self.app.active_actor: return
            self.survey_tool = UndercutSurveyor(self.app.plotter, self.app.active_actor)
            self.survey_tool.start()
            self.update_survey_vector()
        else:
            if self.survey_tool: self.survey_tool.stop(); self.survey_tool = None

    def update_survey_vector(self):
        if self.survey_tool: self.survey_tool.update_direction(self.spin_tilt.value(), 0)

    def reset_survey_path(self):
        if not self.survey_tool:
            QMessageBox.information(self, "Info", "Please enable the surveyor first.")
            return
        self.survey_tool.reset_to_original_vector()
        self.spin_tilt.blockSignals(True)
        self.spin_tilt.setValue(180)
        self.spin_tilt.blockSignals(False)


# ==========================================
#       STEP 3: MANDIBULAR WORKFLOW
# ==========================================
class MandibularSteps(QWidget):
    def __init__(self, app_interface):
        super().__init__()
        self.app = app_interface
        
        # Tools
        self.bezier_tool = BezierMarkerTool(self.app.plotter, self.app)
        self.survey_tool = None
        self.sculpt_tool = None
        
        # Data States
        self.pre_cleanup_mesh = None
        self.last_base_name = None 
        
        # Mandibular Specific Actors
        self.pad_left_actor = None
        self.pad_right_actor = None
        self.active_pad_side = None # 'left' or 'right' during marking

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5,5,5,5)

        self.stack = QStackedWidget()
        self.stack.currentChanged.connect(self._on_stack_changed)
        
        self._init_pages()
        self.layout.addWidget(self.stack)

        # Navigation
        h = QHBoxLayout()
        self.btn_back = QPushButton("< Back")
        self.btn_next = QPushButton("Next >")
        self.btn_back.clicked.connect(lambda: self.nav(-1))
        self.btn_next.clicked.connect(lambda: self.nav(1))
        h.addWidget(self.btn_back)
        h.addWidget(self.btn_next)
        self.layout.addLayout(h)

    def nav(self, delta):
        self._stop_active_tools()
        curr = self.stack.currentIndex()
        self.stack.setCurrentIndex(max(0, min(self.stack.count()-1, curr + delta)))

    # ==========================================
    #       VISIBILITY LOGIC
    # ==========================================
    def _on_stack_changed(self, index):
        if hasattr(self, 'page_pads'):
            is_pad_page = (self.stack.widget(index) == self.page_pads)
            self.set_pads_visibility(is_pad_page)

    def set_pads_visibility(self, visible):
        if self.pad_left_actor: self.pad_left_actor.SetVisibility(visible)
        if self.pad_right_actor: self.pad_right_actor.SetVisibility(visible)
        self.app.plotter.render()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.set_pads_visibility(False)
        self._stop_active_tools()

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, 'page_pads') and self.stack.currentWidget() == self.page_pads:
            self.set_pads_visibility(True)

    def _stop_active_tools(self):
        if self.bezier_tool: 
            self.bezier_tool.stop()
            self.app.setup_picking()
            
        if self.survey_tool:
            self.survey_tool.stop()
            self.survey_tool = None
            
        if self.sculpt_tool:
            self.sculpt_tool.stop()
            self.sculpt_tool = None
            if hasattr(self.app, 'reset_clinical_visuals'):
                self.app.reset_clinical_visuals()
            
        if hasattr(self, 'btn_sculpt_toggle'): 
            self.btn_sculpt_toggle.setChecked(False)

        # Reset Buttons
        if hasattr(self, 'btn_manual_border'): self.btn_manual_border.setChecked(False)
        if hasattr(self, 'btn_pad_left'): self.btn_pad_left.setChecked(False)
        if hasattr(self, 'btn_pad_right'): self.btn_pad_right.setChecked(False)
        if hasattr(self, 'btn_sculpt_toggle'): self.btn_sculpt_toggle.setChecked(False)
        if hasattr(self, 'btn_survey'): self.btn_survey.setChecked(False)

    def _init_pages(self):
        # PAGE 1: IMPORT
        p1 = QWidget(); l1 = QVBoxLayout(p1)
        l1.addWidget(QLabel("<b>1. Import Mandible</b>"))
        btn_load = QPushButton("Load STL/PLY")
        btn_load.clicked.connect(self.app.open_file_dialog)
        l1.addWidget(btn_load); l1.addStretch(); self.stack.addWidget(p1)

        # PAGE 2: CLEANUP
        p2 = QWidget(); l2 = QVBoxLayout(p2)
        l2.addWidget(QLabel("<b>2. Deep Artifact Cleanup</b>"))
        form = QFormLayout()
        self.spin_thresh = QDoubleSpinBox(); self.spin_thresh.setValue(0.5)
        self.spin_rescue = QDoubleSpinBox(); self.spin_rescue.setValue(2.1)
        form.addRow("Depth Thresh:", self.spin_thresh)
        form.addRow("Rescue Dist:", self.spin_rescue)
        l2.addLayout(form)
        btn_clean = QPushButton("Run Deep Cleanup")
        btn_clean.clicked.connect(self.run_cleanup)
        l2.addWidget(btn_clean)
        self.btn_undo = QPushButton("Undo Cleanup")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_cleanup)
        l2.addWidget(self.btn_undo); l2.addStretch(); self.stack.addWidget(p2)

        # PAGE 3: BASE (INVERTED LOGIC)
        p3 = QWidget(); l3 = QVBoxLayout(p3)
        l3.addWidget(QLabel("<b>3. Define Border & Base</b>"))
        l3.addWidget(QLabel("<i>Extrudes downwards for Mandible</i>"))
        
        self.btn_manual_border = QPushButton("Manual Marking Tool")
        self.btn_manual_border.setCheckable(True)
        self.btn_manual_border.clicked.connect(self.toggle_border_tool)
        l3.addWidget(self.btn_manual_border)
        
        # Base Params
        params_group = QGroupBox("Base Parameters"); params_layout = QFormLayout()
        self.spin_height = QDoubleSpinBox(); self.spin_height.setRange(5.0, 60.0); self.spin_height.setValue(20.0)
        self.spin_skirt = QDoubleSpinBox(); self.spin_skirt.setRange(0.0, 5.0); self.spin_skirt.setValue(1.0)
        params_layout.addRow("Height:", self.spin_height)
        params_layout.addRow("Skirt:", self.spin_skirt)
        params_group.setLayout(params_layout); l3.addWidget(params_group)
        
        btn_base = QPushButton("Generate Solid Base")
        btn_base.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold;")
        btn_base.clicked.connect(self.generate_base)
        l3.addWidget(btn_base)
        
        self.btn_undo_base = QPushButton("Undo Base Generation")
        self.btn_undo_base.setEnabled(False)
        self.btn_undo_base.clicked.connect(self.undo_base)
        l3.addWidget(self.btn_undo_base); l3.addStretch(); self.stack.addWidget(p3)

        # PAGE 4: RETROMOLAR PADS (New)
        self.page_pads = self._create_retromolar_page()
        self.stack.addWidget(self.page_pads)

        # PAGE 5: MODIFY
        self.stack.addWidget(self._create_modify_page())

        # PAGE 6: SURVEY (INVERTED DEFAULT)
        p6 = QWidget(); l6 = QVBoxLayout(p6)
        l6.addWidget(QLabel("<b>6. Survey Undercuts</b>"))
        
        self.btn_survey = QPushButton("Enable Surveyor")
        self.btn_survey.setCheckable(True)
        self.btn_survey.clicked.connect(self.toggle_survey)
        l6.addWidget(self.btn_survey)
        
        # Tilt Control (Default 0 for Mandible/Top-down view)
        self.spin_tilt = QDoubleSpinBox(); self.spin_tilt.setRange(0, 180); self.spin_tilt.setValue(0)
        self.spin_tilt.valueChanged.connect(self.update_survey_vector)
        l6.addWidget(QLabel("Tilt Angle (Theta):"))
        l6.addWidget(self.spin_tilt)
        
        btn_reset_path = QPushButton("Reset Path (Vertical)")
        btn_reset_path.clicked.connect(self.reset_survey_path)
        l6.addWidget(btn_reset_path)
        
        l6.addStretch(); self.stack.addWidget(p6)

    def _create_retromolar_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<b>4. Mark Retromolar Pads</b>"))
        layout.addWidget(QLabel("Outline the left and right pads."))
        
        # Left Pad
        self.btn_pad_left = QPushButton("Mark Left Pad")
        self.btn_pad_left.setCheckable(True)
        self.btn_pad_left.clicked.connect(lambda c: self.toggle_pad_tool(c, 'left'))
        layout.addWidget(self.btn_pad_left)

        # Right Pad
        self.btn_pad_right = QPushButton("Mark Right Pad")
        self.btn_pad_right.setCheckable(True)
        self.btn_pad_right.clicked.connect(lambda c: self.toggle_pad_tool(c, 'right'))
        layout.addWidget(self.btn_pad_right)
        
        btn_clear_pads = QPushButton("Clear All Marks")
        btn_clear_pads.clicked.connect(self.clear_pad_marks)
        layout.addWidget(btn_clear_pads)
        
        layout.addStretch()
        return page

    def _create_modify_page(self):
        # (This is identical to Maxilla - could be refactored to a shared method)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<b>5. Smooth / Modify Mesh</b>"))
        
        grp_auto = QGroupBox("Autosmooth")
        v_auto = QVBoxLayout(grp_auto)
        self.spin_crater_depth = QDoubleSpinBox()
        self.spin_crater_depth.setRange(0.01, 1.0); self.spin_crater_depth.setValue(0.05)
        v_auto.addWidget(QLabel("Depth Threshold:")); v_auto.addWidget(self.spin_crater_depth)
        btn_autosmooth = QPushButton("Run Autosmooth")
        btn_autosmooth.clicked.connect(self.run_autosmooth)
        v_auto.addWidget(btn_autosmooth)
        layout.addWidget(grp_auto)

        grp_sculpt = QGroupBox("Sculpting")
        v_sculpt = QVBoxLayout(grp_sculpt)
        self.btn_sculpt_toggle = QPushButton("Enable Sculpting")
        self.btn_sculpt_toggle.setCheckable(True)
        self.btn_sculpt_toggle.clicked.connect(self.toggle_sculpting)
        v_sculpt.addWidget(self.btn_sculpt_toggle)
        
        self.chk_dyntopo = QCheckBox("Dynamic Topology")
        self.chk_dyntopo.toggled.connect(self.update_sculpt_params)
        v_sculpt.addWidget(self.chk_dyntopo)
        
        self.slider_detail = QSlider(Qt.Horizontal); self.slider_detail.setRange(1, 100); self.slider_detail.setValue(20)
        self.slider_detail.valueChanged.connect(self.update_sculpt_params)
        v_sculpt.addWidget(QLabel("Detail Level:")); v_sculpt.addWidget(self.slider_detail)
        
        self.radio_group = QButtonGroup()
        r_smooth = QRadioButton("Smooth Brush"); r_smooth.setChecked(True)
        r_remove = QRadioButton("Remove Brush")
        r_add = QRadioButton("Add Brush")
        self.radio_group.addButton(r_smooth); self.radio_group.addButton(r_remove); self.radio_group.addButton(r_add)
        self.radio_group.buttonClicked.connect(self.update_sculpt_params)
        v_sculpt.addWidget(r_smooth); v_sculpt.addWidget(r_remove); v_sculpt.addWidget(r_add)
        
        self.slider_radius = QSlider(Qt.Horizontal); self.slider_radius.setRange(1, 100); self.slider_radius.setValue(30)
        self.slider_radius.valueChanged.connect(self.update_sculpt_params)
        v_sculpt.addWidget(QLabel("Brush Radius:")); v_sculpt.addWidget(self.slider_radius)
        
        self.slider_power = QSlider(Qt.Horizontal); self.slider_power.setRange(1, 100); self.slider_power.setValue(50)
        self.slider_power.valueChanged.connect(self.update_sculpt_params)
        v_sculpt.addWidget(QLabel("Brush Power:")); v_sculpt.addWidget(self.slider_power)
        
        layout.addWidget(grp_sculpt); layout.addStretch()
        return page

    # ==========================================
    #       LOGIC IMPLEMENTATION
    # ==========================================

    # --- RETROMOLAR TOOLS ---
    def toggle_pad_tool(self, checked, side):
        if checked:
            if not self.app.active_actor:
                QMessageBox.warning(self, "Error", "No mesh loaded.")
                if side == 'left': self.btn_pad_left.setChecked(False)
                else: self.btn_pad_right.setChecked(False)
                return
            
            # Ensure other button is unchecked
            if side == 'left': self.btn_pad_right.setChecked(False)
            else: self.btn_pad_left.setChecked(False)
            
            # Clear existing actor for this side if exists
            if side == 'left' and self.pad_left_actor:
                self.app.plotter.remove_actor(self.pad_left_actor)
                self.pad_left_actor = None
            elif side == 'right' and self.pad_right_actor:
                self.app.plotter.remove_actor(self.pad_right_actor)
                self.pad_right_actor = None
                
            self.active_pad_side = side
            self.bezier_tool.start()
        else:
            # Button unchecked -> finalize
            patch = self.bezier_tool.get_selected_region()
            if patch:
                color = "#e74c3c" if self.active_pad_side == 'left' else "#3498db"
                name = f"Pad_{self.active_pad_side}_{int(time.time())}"
                actor = self.app.plotter.add_mesh(patch, color=color, pickable=False, name=name)
                
                if self.active_pad_side == 'left': self.pad_left_actor = actor
                else: self.pad_right_actor = actor
            
            self.bezier_tool.stop()
            self.bezier_tool.clear_all_markup()
            self.active_pad_side = None

    def clear_pad_marks(self):
        if self.pad_left_actor: self.app.plotter.remove_actor(self.pad_left_actor)
        if self.pad_right_actor: self.app.plotter.remove_actor(self.pad_right_actor)
        self.pad_left_actor = None
        self.pad_right_actor = None
        self.bezier_tool.stop()
        self.bezier_tool.clear_all_markup()
        self.btn_pad_left.setChecked(False)
        self.btn_pad_right.setChecked(False)
        self.app.plotter.render()

    # --- BASE GENERATION (INVERTED) ---
    def generate_base(self):
        patch = self.bezier_tool.get_selected_region()
        if not patch:
            QMessageBox.warning(self, "Error", "No valid selection found.")
            return

        # NOTE: We invert the height to extrude downwards
        inverted_height = -1 * self.spin_height.value()
        
        # Update: create_base_from_selection now returns a tuple (mesh, path)
        result = ModelGenerator.create_base_from_selection(
            patch, 
            base_height=inverted_height, # Negative for Mandible/Downwards
            skirt_size=self.spin_skirt.value()
        )
        
        if result:
            base_mesh, _ = result # Unpack mesh, ignore path for now
            
            base_name = f"Mandible_Base_{int(time.time())}"
            cmd_add = AddMeshCommand(self.app, base_mesh, base_name)
            cmd_clear = self.bezier_tool.clear_markup_cmd()
            composite = MultiCommand([cmd_add, cmd_clear])
            self.app.command_manager.execute(composite)
            
            self.last_base_name = base_name 
            self.btn_undo_base.setEnabled(True)
            QMessageBox.information(self, "Success", "Mandibular Base Generated (Downwards).")
        else:
            QMessageBox.warning(self, "Error", "Failed to generate base.")
            
    def undo_base(self):
        if hasattr(self, 'last_base_name') and self.last_base_name:
            if self.last_base_name in self.app.actors:
                actor = self.app.actors[self.last_base_name]
                cmd = DeleteMeshCommand(self.app, self.last_base_name, actor)
                self.app.command_manager.execute(cmd)
            self.last_base_name = None
            self.btn_undo_base.setEnabled(False)

    # --- STANDARD TOOLS (Border, Sculpt, Cleanup) ---
    def toggle_border_tool(self, checked):
        if checked:
            if not self.app.active_actor:
                self.btn_manual_border.setChecked(False); return
            self.bezier_tool.start()
        else:
            self.bezier_tool.stop()

    def run_autosmooth(self):
        if not self.app.active_actor: return
        self.setCursor(Qt.WaitCursor)
        mesh = self.app.active_actor.mapper.dataset
        from utils.generator import auto_patch_craters # Safety import
        patched = auto_patch_craters(mesh, depth_threshold=self.spin_crater_depth.value())
        if patched:
            self.app.active_actor.mapper.SetInputData(patched)
            self.app.active_actor.mapper.Update()
            self.app.plotter.render()
        self.setCursor(Qt.ArrowCursor)

    def toggle_sculpting(self, checked):
        if checked:
            if not self.app.active_actor: 
                self.btn_sculpt_toggle.setChecked(False)
                return
            
            # Start Tool
            if not self.sculpt_tool: 
                self.sculpt_tool = SculptTool(self.app, self.app.plotter)
            self.sculpt_tool.start()
            self.update_sculpt_params()

            # --- APPLY VISUALS HERE ---
            if hasattr(self.app, 'apply_sculpt_visuals'):
                self.app.apply_sculpt_visuals()
                
        else:
            # Stop Tool
            if self.sculpt_tool: 
                self.sculpt_tool.stop()

            # --- RESET VISUALS HERE ---
            if hasattr(self.app, 'reset_clinical_visuals'):
                self.app.reset_clinical_visuals()

    def update_sculpt_params(self, *args):
        if not self.sculpt_tool: return
        text = self.radio_group.checkedButton().text()
        mode = "REMOVE" if "Remove" in text else ("ADD" if "Add" in text else "SMOOTH")
        radius = self.slider_radius.value() / 5.0
        strength = self.slider_power.value() / 100.0
        dyntopo = self.chk_dyntopo.isChecked()
        detail_size = 3.0 - (self.slider_detail.value() / 100.0 * 2.8) 
        self.sculpt_tool.set_params(mode, radius, strength, dyntopo, detail_size)

    def run_cleanup(self):
        if not self.app.active_actor: return
        
        mesh = self.app.active_actor.mapper.dataset
        self.pre_cleanup_mesh = mesh.copy()
        
        # === FLIPPED VECTOR FOR MANDIBLE ===
        # We pass direction=[0, 0, -1] so the algorithm knows "up" is inverted
        cleaned = ModelGenerator.clean_undesirable_artifacts(
            mesh, 
            threshold=self.spin_thresh.value(), 
            rescue_threshold=self.spin_rescue.value(),
            direction=[0, 0, -1] 
        )
        
        if cleaned:
            self.app.active_actor.mapper.SetInputData(cleaned)
            self.app.active_actor.mapper.Update()
            self.app.plotter.render()
            self.btn_undo.setEnabled(True)
            QMessageBox.information(self, "Result", "Mandibular Deep Cleanup complete.")

    def undo_cleanup(self):
        if self.pre_cleanup_mesh:
            self.app.active_actor.mapper.SetInputData(self.pre_cleanup_mesh)
            self.app.active_actor.mapper.Update()
            self.app.plotter.render()
            self.pre_cleanup_mesh = None
            self.btn_undo.setEnabled(False)

    # --- SURVEYOR (MANDIBLE) ---
    def toggle_survey(self, checked):
        if checked:
            if not self.app.active_actor: return
            self.survey_tool = UndercutSurveyor(self.app.plotter, self.app.active_actor)
            self.survey_tool.start()
            self.update_survey_vector()
        else:
            if self.survey_tool: self.survey_tool.stop(); self.survey_tool = None

    def update_survey_vector(self):
        # 0 degrees is typically Vertical (Top-Down) in these tools
        if self.survey_tool: self.survey_tool.update_direction(self.spin_tilt.value(), 0)

    def reset_survey_path(self):
        if not self.survey_tool: return
        self.survey_tool.reset_to_original_vector()
        self.spin_tilt.blockSignals(True)
        self.spin_tilt.setValue(0) # Reset to 0 for Mandible
        self.spin_tilt.blockSignals(False)


# ==========================================
#       CLASS: DentalWizardSidebar (Host)
# ==========================================
class DentalWizardSidebar(QWidget):
    def __init__(self, app_interface):
        super().__init__()
        self.app = app_interface 
        self.setFixedWidth(280)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        self.layout.addWidget(QLabel("<b>DENTAL WORKFLOW</b>"))
        self.main_list = QListWidget()
        self.main_list.setFixedHeight(160)
        
        # Updated List Items
        self.main_list.addItems([
            "1. Setup Maxilla", 
            "2. Align Horizontal", 
            "3. Setup Mandible", 
            "4. Mark Impression"
        ])
        
        self.main_list.currentRowChanged.connect(self._on_main_change)
        self.layout.addWidget(self.main_list)

        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)

        # Pages
        self.maxilla_wizard = MaxillarySteps(self.app)
        self.stack.addWidget(self.maxilla_wizard)     # Index 0
        
        self.align_wizard = AlignmentStep(self.app)
        self.stack.addWidget(self.align_wizard)       # Index 1
        
        self.mandible_wizard = MandibularSteps(self.app)
        self.stack.addWidget(self.mandible_wizard)    # Index 2
        
        self.stack.addWidget(self._init_placeholder_page("Impression")) # Index 3
        
        self.main_list.setCurrentRow(0)

    def _on_main_change(self, index):
        # 1. CLEANUP PREVIOUS PAGES
        
        # If leaving Maxilla
        if index != 0: 
            self.maxilla_wizard._stop_active_tools()
            self.maxilla_wizard.set_papilla_visibility(False)
            
        # If leaving Alignment
        if index != 1: 
            self.align_wizard.reset_points()
            
        # If leaving Mandible
        if index != 2:
            self.mandible_wizard._stop_active_tools()
            self.mandible_wizard.set_pads_visibility(False)

        # 2. ACTIVATE NEW PAGE
        self.stack.setCurrentIndex(index)
        
        # Auto-trigger visibility for actors if returning to specific sub-pages
        if index == 0:
            # Trigger showEvent logic manually if needed
            self.maxilla_wizard.showEvent(None)
        elif index == 2:
            self.mandible_wizard.showEvent(None)

    def _init_placeholder_page(self, title):
        p = QWidget()
        l = QVBoxLayout(p)
        l.addWidget(QLabel(f"<b>{title}</b>"))
        l.addStretch()
        return p