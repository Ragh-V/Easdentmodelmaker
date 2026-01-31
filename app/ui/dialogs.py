from PySide6.QtWidgets import (QDialog, QVBoxLayout, QGroupBox, QLabel, 
                               QSlider, QPushButton, QHBoxLayout, QMessageBox)
from PySide6.QtCore import Qt
import numpy as np

class HoleFillDialog(QDialog):
    def __init__(self, parent_app, actor):
        super().__init__(parent_app)
        self.app = parent_app
        self.actor = actor
        self.mesh = actor.mapper.dataset
        self.setWindowTitle("Fill Holes")
        self.resize(300, 150)
        self.setModal(False) 
        
        self.debug_actors = []
        
        self.boundary_edges = self.mesh.extract_feature_edges(
            boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
        )
        
        if self.boundary_edges.n_points == 0:
            self.boundary_strips = None
            QMessageBox.information(self, "Info", "Mesh is watertight. No holes detected.")
        else:
            self.boundary_strips = self.boundary_edges.connectivity(largest=False)
        
        self.setup_ui()
        
        if self.boundary_strips is not None:
            self.update_visualization()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        grp = QGroupBox("Hole Size Threshold (mm)")
        vbox = QVBoxLayout()
        
        self.lbl_size = QLabel("Max Radius: 10.0 mm")
        vbox.addWidget(self.lbl_size)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000) 
        self.slider.setValue(100) 
        self.slider.valueChanged.connect(self.on_slider_change)
        vbox.addWidget(self.slider)
        grp.setLayout(vbox)
        layout.addWidget(grp)
        
        btn_layout = QHBoxLayout()
        self.btn_fill = QPushButton("Fill Holes")
        self.btn_fill.clicked.connect(self.fill_holes)
        self.btn_cancel = QPushButton("Close")
        self.btn_cancel.clicked.connect(self.close)
        
        btn_layout.addWidget(self.btn_fill)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def on_slider_change(self, val):
        radius = val / 10.0
        self.lbl_size.setText(f"Max Radius: {radius:.1f} mm")
        self.update_visualization()
        
    def update_visualization(self):
        if self.boundary_strips is None: return

        for act in self.debug_actors:
            self.app.plotter.remove_actor(act)
        self.debug_actors.clear()
        
        limit = self.slider.value() / 10.0
        
        try:
            scalar_name = self.boundary_strips.active_scalars_name
            if not scalar_name: return 

            region_ids = np.unique(self.boundary_strips[scalar_name])
            
            for region_id in region_ids:
                hole_edge = self.boundary_strips.threshold([region_id, region_id], scalars=scalar_name, preference='point')
                if hole_edge.n_points == 0: continue
                
                bounds = hole_edge.bounds
                diag = np.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)
                radius = diag / 2.0
                
                if radius <= limit:
                    act = self.app.plotter.add_mesh(hole_edge, color='red', line_width=4, render_lines_as_tubes=True, pickable=False)
                    self.debug_actors.append(act)
                else:
                    act = self.app.plotter.add_mesh(hole_edge, color='green', line_width=2, pickable=False)
                    self.debug_actors.append(act)
                    
            self.app.plotter.render()
        except Exception as e:
            print(f"Viz Error: {e}")


    def fill_holes(self):
        if self.boundary_strips is None: return
        limit = self.slider.value() / 10.0
        
        try:
            filled_mesh = self.mesh.fill_holes(hole_size=limit**2 * 3.14) 
            
            filled_mesh.compute_normals(
                cell_normals=True, 
                point_normals=True, 
                inplace=True, 
                auto_orient_normals=True,
                consistent_normals=True,  
                split_vertices=False        
            )

            self.actor.mapper.dataset.copy_from(filled_mesh)
            self.actor.mapper.dataset.Modified()
            self.mesh = self.actor.mapper.dataset 
            
            self.boundary_edges = self.mesh.extract_feature_edges(
                boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
            )
            
            if self.boundary_edges.n_points > 0:
                self.boundary_strips = self.boundary_edges.connectivity(largest=False)
                self.update_visualization()
            else:
                self.boundary_strips = None
                self.update_visualization()
                QMessageBox.information(self, "Success", "All holes filled!")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def closeEvent(self, event):
        for act in self.debug_actors:
            self.app.plotter.remove_actor(act)
        super().closeEvent(event)