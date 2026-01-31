from PySide6.QtWidgets import (QWidget, QHBoxLayout, QCheckBox, QLabel, QSlider, 
                               QTreeWidget, QTreeWidgetItem, QAbstractItemView, 
                               QVBoxLayout, QMenu, QInputDialog, QMessageBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent

class HierarchyItemWidget(QWidget):
    def __init__(self, name, actor, plotter_ref, tree_ref, tree_item_ref):
        super().__init__()
        self.actor = actor
        self.plotter_ref = plotter_ref
        self.name = name
        self.tree_ref = tree_ref
        self.tree_item_ref = tree_item_ref
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        self.chk_visible = QCheckBox()
        self.chk_visible.setChecked(self.actor.GetVisibility())
        self.chk_visible.toggled.connect(self.toggle_visibility)
        layout.addWidget(self.chk_visible)
        
        self.lbl_name = QLabel(name)
        layout.addWidget(self.lbl_name)
        
        layout.addStretch()
        
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setFixedWidth(60)
        
        current_op = self.actor.GetProperty().GetOpacity()
        self.slider_opacity.setValue(int(current_op * 100))
        self.slider_opacity.valueChanged.connect(self.change_opacity)
        self.slider_opacity.setVisible(False) 
        layout.addWidget(self.slider_opacity)

    def set_name(self, new_name):
        self.name = new_name
        self.lbl_name.setText(new_name)

    def toggle_visibility(self, checked):
        if self.actor:
            self.actor.SetVisibility(checked)
            self.plotter_ref.render()

    def change_opacity(self, value):
        if self.actor:
            self.actor.GetProperty().SetOpacity(value / 100.0)
            self.plotter_ref.render()

    def enterEvent(self, event):
        self.slider_opacity.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.slider_opacity.setVisible(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.tree_ref.setCurrentItem(self.tree_item_ref)
            self.tree_ref.itemClicked.emit(self.tree_item_ref, 0)
        super().mousePressEvent(event)


class FolderItemWidget(QWidget):
    def __init__(self, name, tree_ref, tree_item_ref):
        super().__init__()
        self.name = name
        self.tree_ref = tree_ref
        self.tree_item_ref = tree_item_ref
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        self.chk_visible = QCheckBox()
        self.chk_visible.setChecked(True)
        self.chk_visible.toggled.connect(self.toggle_recursive_visibility)
        layout.addWidget(self.chk_visible)
        
        icon_label = QLabel("📁") 
        layout.addWidget(icon_label)
        
        self.lbl_name = QLabel(name)
        layout.addWidget(self.lbl_name)
        layout.addStretch()

    def set_name(self, new_name):
        self.name = new_name
        self.lbl_name.setText(new_name)

    def toggle_recursive_visibility(self, checked):
        count = self.tree_item_ref.childCount()
        for i in range(count):
            child_item = self.tree_item_ref.child(i)
            widget = self.tree_ref.itemWidget(child_item, 0)
            if isinstance(widget, HierarchyItemWidget):
                widget.chk_visible.setChecked(checked)
            elif isinstance(widget, FolderItemWidget):
                widget.chk_visible.setChecked(checked)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.tree_ref.setCurrentItem(self.tree_item_ref)
        super().mousePressEvent(event)


class HierarchyTreeWidget(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.header().hide()
        
    def dropEvent(self, event):
        source_item = self.currentItem()
        if not source_item: 
            super().dropEvent(event)
            return

        old_widget = self.itemWidget(source_item, 0)
        widget_data = None
        
        if isinstance(old_widget, HierarchyItemWidget):
            widget_data = {
                'type': 'mesh',
                'name': old_widget.name,
                'actor': old_widget.actor,
                'plotter': old_widget.plotter_ref
            }
        elif isinstance(old_widget, FolderItemWidget):
             widget_data = {
                'type': 'folder',
                'name': old_widget.name
            }

        super().dropEvent(event)
        
        if widget_data:
            if widget_data['type'] == 'mesh':
                new_widget = HierarchyItemWidget(
                    widget_data['name'], widget_data['actor'], 
                    widget_data['plotter'], self, source_item
                )
            else:
                new_widget = FolderItemWidget(
                    widget_data['name'], self, source_item
                )
            self.setItemWidget(source_item, 0, new_widget)
            source_item.setExpanded(True)


class HierarchyPanel(QWidget):
    item_removed = Signal(str) 
    item_selected = Signal(str)
    item_renamed = Signal(str, str)
    delete_requested = Signal(list)

    def __init__(self, parent=None, plotter=None):
        super().__init__(parent)
        self.plotter = plotter
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.tree = HierarchyTreeWidget()
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        self.tree.itemClicked.connect(self.on_item_clicked)
        
        self.layout.addWidget(self.tree)
        self.items_map = {} 

    def on_item_clicked(self, item, column):
        widget = self.tree.itemWidget(item, 0)
        if isinstance(widget, HierarchyItemWidget):
            self.item_selected.emit(widget.name)

    def show_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        menu = QMenu()
        
        new_folder_action = menu.addAction("Create Folder")
        menu.addSeparator()
        
        rename_action = None
        delete_action = None
        
        if item:
            rename_action = menu.addAction("Rename")
            delete_action = menu.addAction("Delete")
        
        action = menu.exec(self.tree.mapToGlobal(pos))
        
        if action == new_folder_action:
            self.create_folder(item) 
        elif action == rename_action and item:
            self.prompt_rename(item)
        elif action == delete_action and item:
            self.delete_item_recursive(item)

    def create_folder(self, parent_item=None):
        name, ok = QInputDialog.getText(self, "New Folder", "Folder Name:")
        if not ok or not name: return
        
        folder_item = QTreeWidgetItem()
        folder_item.setFlags(folder_item.flags() | Qt.ItemIsDropEnabled)
        
        if parent_item and isinstance(self.tree.itemWidget(parent_item, 0), FolderItemWidget):
            parent_item.addChild(folder_item)
            parent_item.setExpanded(True)
        else:
            self.tree.addTopLevelItem(folder_item)
            
        widget = FolderItemWidget(name, self.tree, folder_item)
        self.tree.setItemWidget(folder_item, 0, widget)

    def prompt_rename(self, item):
        widget = self.tree.itemWidget(item, 0)
        if not widget: return
        
        old_name = widget.name
        new_name, ok = QInputDialog.getText(self, "Rename", "New Name:", text=old_name)
        
        if ok and new_name and new_name != old_name:
            if isinstance(widget, HierarchyItemWidget):
                if new_name in self.items_map:
                    QMessageBox.warning(self, "Error", "Name already exists.")
                    return
                self.item_renamed.emit(old_name, new_name)
            else:
                widget.set_name(new_name)

    def delete_item_recursive(self, item):
        actors_to_delete = []
        
        def collect_actors(tree_item):
            widget = self.tree.itemWidget(tree_item, 0)
            if isinstance(widget, HierarchyItemWidget):
                actors_to_delete.append(widget.name)
            for i in range(tree_item.childCount()):
                collect_actors(tree_item.child(i))

        collect_actors(item)
        
        if not actors_to_delete:
            root = self.tree.invisibleRootItem()
            (item.parent() or root).removeChild(item)
        else:
            self.delete_requested.emit(actors_to_delete)
            widget = self.tree.itemWidget(item, 0)
            if isinstance(widget, FolderItemWidget):
                 root = self.tree.invisibleRootItem()
                 (item.parent() or root).removeChild(item)

    def update_item_name(self, old_name, new_name):
        if old_name in self.items_map:
            item = self.items_map.pop(old_name)
            widget = self.tree.itemWidget(item, 0)
            if widget:
                widget.set_name(new_name)
            self.items_map[new_name] = item

    def add_mesh_item(self, name, actor):
        if name in self.items_map: return 
        
        list_item = QTreeWidgetItem()
        list_item.setFlags(list_item.flags() & ~Qt.ItemIsDropEnabled)
        
        current = self.tree.currentItem()
        if current and isinstance(self.tree.itemWidget(current, 0), FolderItemWidget):
            current.addChild(list_item)
            current.setExpanded(True)
        else:
            self.tree.addTopLevelItem(list_item)
        
        custom_widget = HierarchyItemWidget(name, actor, self.plotter, self.tree, list_item)
        self.tree.setItemWidget(list_item, 0, custom_widget)
        self.items_map[name] = list_item

    def remove_mesh_item(self, name):
        if name in self.items_map:
            item = self.items_map[name]
            parent = item.parent() or self.tree.invisibleRootItem()
            parent.removeChild(item)
            del self.items_map[name]

    def select_item(self, name):
        if name in self.items_map:
            item = self.items_map[name]
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(item)
            self.tree.blockSignals(False)
        else:
            self.tree.clearSelection()