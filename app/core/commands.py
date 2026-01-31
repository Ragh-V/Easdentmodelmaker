from abc import ABC, abstractmethod
import numpy as np
from weakref import WeakKeyDictionary

class ICommand(ABC):
    @abstractmethod
    def execute(self): pass
    @abstractmethod
    def undo(self): pass




class CommandManager:
    def __init__(self, app_interface):
        self.undo_stack = []
        self.redo_stack = []
        self.app = app_interface

    def execute(self, command: ICommand):
        command.execute()
        self.undo_stack.append(command)
        self.redo_stack.clear()
        self.app.update_undo_redo_ui()

    def push_existing(self, command: ICommand):
        self.undo_stack.append(command)
        self.redo_stack.clear()
        self.app.update_undo_redo_ui()

    def undo(self):
        if not self.undo_stack: return
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        self.app.update_undo_redo_ui()

    def redo(self):
        if not self.redo_stack: return
        command = self.redo_stack.pop()
        command.execute()
        self.undo_stack.append(command)
        self.app.update_undo_redo_ui()

# --- CONCRETE COMMANDS ---

class MultiCommand(ICommand):
    def __init__(self, commands_list):
        self.commands = commands_list

    def execute(self):
        for cmd in self.commands:
            cmd.execute()

    def undo(self):
        for cmd in reversed(self.commands):
            cmd.undo()


class ReplaceGeometryCommand(ICommand):
    """
    Replaces the dataset of an actor with a new mesh.
    Useful for filtering operations like 'extract_largest'.
    """
    def __init__(self, app, actor, old_mesh, new_mesh):
        # Determine if we need to call super().__init__(app) based on your Command definition
        try:
            super().__init__(app)
        except:
            self.app = app
            
        self.actor = actor
        self.old_mesh = old_mesh
        self.new_mesh = new_mesh

    def execute(self):
        # DeepCopy ensures the actor updates in place without breaking pointers
        self.actor.mapper.dataset.DeepCopy(self.new_mesh)
        self.app.plotter.render()

    def undo(self):
        self.actor.mapper.dataset.DeepCopy(self.old_mesh)
        self.app.plotter.render()


class AddMeshCommand(ICommand):
    def __init__(self, app, mesh, name, texture=None):
        self.app = app
        self.mesh = mesh
        self.name = name
        self.texture = texture
        self.actor = None

    def execute(self):
        color_arg = 'white'
        rgb_arg = False
        scalars_arg = None
        
        if self.texture is not None:
            color_arg = 'white'
            rgb_arg = False
        else:
            active_name = self.mesh.active_scalars_name
            found_rgb = False
            if active_name and ('RGB' in active_name.upper() or 'COLOR' in active_name.upper()):
                scalars_arg = active_name
                found_rgb = True
            
            if not found_rgb and hasattr(self.mesh, 'array_names'):
                for arr_name in self.mesh.array_names:
                    if 'RGB' in arr_name.upper():
                        self.mesh.set_active_scalars(arr_name)
                        scalars_arg = arr_name
                        found_rgb = True
                        break
            
            if found_rgb:
                color_arg = None  
                rgb_arg = True
        
        try: self.app.plotter.disable_picking()
        except: pass
            
        res = self.app.plotter.add_mesh(
            self.mesh, 
            texture=self.texture,
            color=color_arg, 
            scalars=scalars_arg, 
            rgb=rgb_arg,
            specular=0.2, diffuse=0.7, ambient=0.3, smooth_shading=True,
            show_edges=False, name=self.name, pickable=True, reset_camera=False
        )
        self.actor = res[0] if isinstance(res, tuple) else res
        
        if self.texture:
            self.app.original_textures[self.actor] = self.texture

        self.app.enable_object_selection_mode()
        self.app.actors[self.name] = self.actor
        self.app.set_active_actor(self.actor)
        
        if hasattr(self.app, 'hierarchy_panel'):
            self.app.hierarchy_panel.add_mesh_item(self.name, self.actor)

    def undo(self):
        if self.actor:
            if self.app.active_actor == self.actor:
                self.app.set_active_actor(None)
            try: self.app.plotter.disable_picking()
            except: pass
            
            self.app.plotter.remove_actor(self.actor)
            self.app.enable_object_selection_mode()
            
            if hasattr(self.app, 'hierarchy_panel'):
                self.app.hierarchy_panel.remove_mesh_item(self.name)

            if self.name in self.app.actors:
                del self.app.actors[self.name]
            self.app.plotter.render()

class DeleteMeshCommand(ICommand):
    def __init__(self, app, actor_name, actor_obj):
        self.app = app
        self.name = actor_name
        self.actor_obj = actor_obj

    def execute(self):
        if hasattr(self.app, 'dental_wizard') and hasattr(self.app.dental_wizard, 'maxilla_wizard'):
            wiz = self.app.dental_wizard.maxilla_wizard
            if hasattr(wiz, 'bezier_tool') and wiz.bezier_tool:
                if wiz.bezier_tool.target_actor == self.actor_obj:
                    wiz.bezier_tool._internal_hard_reset(keep_actors=False)
                    wiz.bezier_tool.stop()

        if self.app.active_actor == self.actor_obj:
            self.app.set_active_actor(None)
        
        try: self.app.plotter.disable_picking()
        except: pass
            
        self.app.plotter.remove_actor(self.actor_obj)
        self.app.enable_object_selection_mode()
        
        if self.name in self.app.actors:
            del self.app.actors[self.name]
            
        if hasattr(self.app, 'hierarchy_panel'):
            self.app.hierarchy_panel.remove_mesh_item(self.name)
            
        self.app.plotter.render()

    def undo(self):
        try: self.app.plotter.disable_picking()
        except: pass

        self.app.plotter.add_actor(self.actor_obj)
        self.app.enable_object_selection_mode()
        
        self.app.actors[self.name] = self.actor_obj
        self.app.set_active_actor(self.actor_obj)
        
        if hasattr(self.app, 'hierarchy_panel'):
            self.app.hierarchy_panel.add_mesh_item(self.name, self.actor_obj)
            
        self.app.plotter.render()

class TransformCommand(ICommand):
    def __init__(self, app, actor, old_matrix, new_matrix):
        self.app = app
        self.actor = actor
        self.old_matrix = old_matrix
        self.new_matrix = new_matrix

    def execute(self):
        self._apply_matrix(self.new_matrix)

    def undo(self):
        self._apply_matrix(self.old_matrix)

    def _apply_matrix(self, matrix):
        self.actor.user_matrix = matrix
        self.actor.position = (0, 0, 0)
        self.actor.orientation = (0, 0, 0)
        self.actor.scale = (1, 1, 1)
        self.app.update_gizmo_target()
        self.app.sync_highlight_motion()
        self.app.plotter.render()

class MultiDeleteCommand(ICommand):
    def __init__(self, app, names_list):
        self.app = app
        self.names = names_list
        self.sub_commands = []

    def execute(self):
        self.sub_commands = []
        for name in self.names:
            if name in self.app.actors:
                actor = self.app.actors[name]
                cmd = DeleteMeshCommand(self.app, name, actor)
                cmd.execute()
                self.sub_commands.append(cmd)

    def undo(self):
        for cmd in reversed(self.sub_commands):
            cmd.undo()

class DeleteCellsCommand(ICommand):
    def __init__(self, app, actor, deleted_cell_ids, original_mesh_copy):
        self.app = app
        self.actor = actor
        self.ids = deleted_cell_ids
        self.original_mesh = original_mesh_copy 
        self.new_mesh = None

    def execute(self):
        mesh = self.actor.mapper.dataset
        self.new_mesh = mesh.remove_cells(list(self.ids))
        
        if self.new_mesh.points.dtype == np.float64:
            self.new_mesh.points = self.new_mesh.points.astype(np.float32)
        
        self.actor.mapper.SetInputData(self.new_mesh)
        self.actor.mapper.Update() 
        
        if hasattr(self.new_mesh, "compute_normals"):
            self.new_mesh.compute_normals(inplace=True)
            
        self.app.plotter.render()

    def undo(self):
        self.actor.mapper.SetInputData(self.original_mesh)
        self.actor.mapper.Update()
        self.app.plotter.render()      

class MaterialChangeCommand(ICommand):
    def __init__(self, app, actor, new_props):
        self.app = app
        self.actor = actor
        self.new_props = new_props
        self.old_props = self._capture_props(actor)

    def _capture_props(self, actor):
        prop = actor.GetProperty()
        return {
            'color': prop.GetColor(),
            'diffuse': prop.GetDiffuse(),
            'specular': prop.GetSpecular(),
            'specular_power': prop.GetSpecularPower(),
            'ambient': prop.GetAmbient(),
            'interpolation': prop.GetInterpolation(),
            'metallic': prop.GetMetallic() if hasattr(prop, 'GetMetallic') else 0.0,
            'roughness': prop.GetRoughness() if hasattr(prop, 'GetRoughness') else 1.0,
            'texture': actor.GetTexture()
        }

    def execute(self):
        self._apply(self.new_props)

    def undo(self):
        self._apply(self.old_props)

    def _apply(self, props):
        prop = self.actor.GetProperty()
        if 'color' in props: prop.SetColor(props['color'])
        if 'diffuse' in props: prop.SetDiffuse(props['diffuse'])
        if 'specular' in props: prop.SetSpecular(props['specular'])
        if 'specular_power' in props: prop.SetSpecularPower(props['specular_power'])
        if 'ambient' in props: prop.SetAmbient(props['ambient'])
        if 'interpolation' in props: prop.SetInterpolation(props['interpolation'])
        
        if hasattr(prop, 'SetMetallic') and 'metallic' in props:
            prop.SetMetallic(props['metallic'])
        if hasattr(prop, 'SetRoughness') and 'roughness' in props:
            prop.SetRoughness(props['roughness'])
            
        if 'texture' in props:
            self.actor.SetTexture(props['texture'])
            
        self.app.plotter.render()