import numpy as np
import pyvista as pv
import vtk
import math
from scipy.spatial import cKDTree
from collections import deque
from typing import List, Set, Tuple, Optional
from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QCursor

# --- COMMANDS ---

class BezierCmdAdd:
    def __init__(self, tool, position, insert_idx=-1):
        self.tool = tool
        self.pos = position
        self.insert_idx = insert_idx
        self.added_actor = None
        self.added_pid = None
        self.final_idx = -1

    def execute(self):
        # The tool handles the logic of "appending" vs "inserting" if insert_idx is -1
        self.added_actor, self.added_pid = self.tool._internal_add_node(self.pos, self.insert_idx)
        self.final_idx = self.tool.nodes.index(self.added_actor)

    def undo(self):
        if self.added_actor:
            self.tool._internal_remove_node_by_actor(self.added_actor)

class BezierCmdDelete:
    def __init__(self, tool, actor):
        self.tool = tool
        self.actor = actor
        # Snapshot state before deletion
        self.idx = tool.nodes.index(actor) if actor in tool.nodes else -1
        self.pid = tool.node_ids[self.idx] if self.idx != -1 else -1
        self.pos = np.array(actor.GetPosition())
        self.was_closed = tool.is_closed

    def execute(self):
        self.tool._internal_remove_node_by_actor(self.actor)

    def undo(self):
        if self.idx != -1:
            self.tool._internal_restore_node(self.actor, self.pid, self.idx, self.was_closed)

class BezierCmdMove:
    """Handles Undo/Redo for dragging nodes."""
    def __init__(self, tool, actor, old_pos, new_pos, old_pid, new_pid):
        self.tool = tool
        self.actor = actor
        self.old_pos = old_pos
        self.new_pos = new_pos
        self.old_pid = old_pid
        self.new_pid = new_pid

    def execute(self):
        self.tool._internal_update_node_pos(self.actor, self.new_pos, self.new_pid)

    def undo(self):
        self.tool._internal_update_node_pos(self.actor, self.old_pos, self.old_pid)

class BezierCmdCloseLoop:
    def __init__(self, tool):
        self.tool = tool
        self.prev_state = tool.is_closed

    def execute(self):
        self.tool.is_closed = True
        self.tool.update_curve()
        self.tool.plotter.render()

    def undo(self):
        self.tool.is_closed = self.prev_state
        self.tool.update_curve()
        self.tool.plotter.render()

class BezierCmdClear:
    def __init__(self, tool):
        self.tool = tool
        self.saved_nodes = list(tool.nodes)
        self.saved_ids = list(tool.node_ids)
        self.saved_closed = tool.is_closed
        
    def execute(self):
        self.tool._internal_hard_reset(keep_actors=False) 

    def undo(self):
        self.tool._internal_restore_full_state(self.saved_nodes, self.saved_ids, self.saved_closed)

# --- INTERACTOR ---

class BezierInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent_tool):
        super().__init__()
        self.tool = parent_tool
        
        self.AddObserver("LeftButtonPressEvent", self.OnLeftDown)
        self.AddObserver("LeftButtonReleaseEvent", self.OnLeftUp)
        self.AddObserver("MouseMoveEvent", self.OnMouseMove)
        self.AddObserver("RightButtonPressEvent", self.OnRightDown)
        self.AddObserver("RightButtonReleaseEvent", self.OnRightUp) # Added for consistency
        self.AddObserver("KeyPressEvent", self.OnKeyPress)

    def OnLeftDown(self, obj, event):
        # 1. PRIORITY: Pick Existing Node (Drag)
        picked_node = self.tool.try_pick_node()
        if picked_node:
            self.tool.active_node = picked_node
            self.tool.active_idx = self.tool.nodes.index(picked_node)
            self.tool.is_dragging = True
            
            # Record start state for UndoCmd
            self.tool._drag_start_pos = np.array(self.tool.active_node.GetPosition())
            self.tool._drag_start_pid = self.tool.node_ids[self.tool.active_idx]
            
            self.tool._update_node_colors()
            self.tool.plotter.render()
            return 

        # 2. PRIORITY: Pick Mesh (Append or Insert)
        hit_pos = self.tool.try_pick_mesh()
        if hit_pos is not None:
            # Check if this click is close to an existing curve segment
            insert_idx = self.tool._resolve_insertion_index(hit_pos)
            self.tool.add_node_at(hit_pos, insert_idx)
            return 

        # 3. Background: Rotate
        self.StartRotate()

    def OnLeftUp(self, obj, event):
        if self.tool.is_dragging and self.tool.active_node:
            final_pos = np.array(self.tool.active_node.GetPosition())
            # Re-snap to closest mesh point to ensure validity
            end_pid = self.tool.mesh.find_closest_point(final_pos)
            
            # If position changed, push a Move Command
            if self.tool._drag_start_pid != end_pid:
                cmd = BezierCmdMove(
                    self.tool, self.tool.active_node, 
                    self.tool._drag_start_pos, final_pos,
                    self.tool._drag_start_pid, end_pid
                )
                self.tool.app.command_manager.execute(cmd)
            else:
                # Just visually snap if no effective move
                self.tool.active_node.SetPosition(self.tool.mesh.points[end_pid])
                self.tool.plotter.render()

            self.tool.active_node = None
            self.tool.active_idx = -1
            self.tool.is_dragging = False
            
            self.tool._update_node_colors()
            self.tool.update_curve()
            self.tool.plotter.render()
        
        self.EndRotate()

    def OnMouseMove(self, obj, event):
        if self.tool.is_dragging and self.tool.active_node:
            self.tool.drag_active_node_visual()
        else:
            super().OnMouseMove()

    def OnRightDown(self, obj, event):
        if self.tool.try_pick_node():
            self.tool.handle_right_click()
            return
        self.StartPan()

    def OnRightUp(self, obj, event):
        self.EndPan()
        
    def OnKeyPress(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if self.tool.handle_key(key): return 
        super().OnKeyPress()

# --- TOOL ---

class BezierMarkerTool:
    def __init__(self, plotter, app):
        self.plotter = plotter
        self.app = app
        
        self.nodes = []       
        self.node_ids = []    
        self.path_cache = {}  
        self.is_closed = False
        
        self.line_actor = None
        self.target_actor = None
        self.node_radius = 1.0 
        self.curve_color = "#e67e22" 
        
        self.mesh = None
        self.locator = None 
        self.active_node = None      
        self.active_idx = -1
        self.is_dragging = False
        self._drag_start_pos = None
        self._drag_start_pid = -1
        
        self.node_picker = vtk.vtkPropPicker()
        self._style = None
        self._prev_style = None

    def start(self):
        if not self.app.active_actor: return
        
        try: self.plotter.disable_picking()
        except: pass

        self.target_actor = self.app.active_actor
        self.mesh = self.target_actor.mapper.dataset
        if isinstance(self.mesh, pv.UnstructuredGrid):
            self.mesh = self.mesh.extract_surface()
            self.target_actor.mapper.SetInputData(self.mesh)
        
        self.mesh.BuildLinks()
        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(self.mesh)
        self.locator.BuildLocator()
        
        if "Normals" not in self.mesh.point_data:
            self.mesh.compute_normals(inplace=True, cell_normals=True)
        
        self.node_radius = self.mesh.length * 0.00375
        
        self.target_actor.SetPickable(True)
        for actor in self.plotter.actors.values():
            if actor != self.target_actor: actor.SetPickable(False)

        self._prev_style = self.plotter.iren.interactor.GetInteractorStyle()
        self._style = BezierInteractorStyle(self)
        self._style.SetDefaultRenderer(self.plotter.renderer)
        self.plotter.iren.interactor.SetInteractorStyle(self._style)
        
    def stop(self):
        if self._prev_style:
            self.plotter.iren.interactor.SetInteractorStyle(self._prev_style)
        else:
            self.plotter.enable_trackball_style()
        self.app.enable_object_selection_mode()
        self.active_node = None
        self.locator = None
        for actor in self.plotter.actors.values(): actor.SetPickable(True)
        self.plotter.render()

    # --- PUBLIC API ---

    def add_node_at(self, world_pos, insert_idx=-1):
        cmd = BezierCmdAdd(self, world_pos, insert_idx)
        self.app.command_manager.execute(cmd)

    def delete_node_cmd(self, actor):
        cmd = BezierCmdDelete(self, actor)
        self.app.command_manager.execute(cmd)

    def close_loop(self):
        if len(self.nodes) > 2 and not self.is_closed:
            cmd = BezierCmdCloseLoop(self)
            self.app.command_manager.execute(cmd)

    def clear_markup_cmd(self):
        return BezierCmdClear(self)

    # --- GEOMETRY CALCULATION ---

    def _dist_sq_to_segment(self, p, a, b):
        """Squared distance from point p to line segment ab."""
        ab = b - a
        ap = p - a
        len_sq = np.dot(ab, ab)
        if len_sq == 0: return np.dot(ap, ap) 
        
        t = max(0, min(1, np.dot(ap, ab) / len_sq))
        projection = a + t * ab
        dist_vec = p - projection
        return np.dot(dist_vec, dist_vec)

    def _resolve_insertion_index(self, hit_pos):
        """
        Determines closest segment for node insertion.
        Returns index i (meaning insert between i and i+1) or -1.
        """
        if len(self.nodes) < 2: return -1
        
        threshold_sq = (self.node_radius * 3.5) ** 2
        best_idx = -1
        min_dist_sq = float('inf')
        
        count = len(self.nodes)
        loop_range = count if self.is_closed else count - 1

        for i in range(loop_range):
            pts = self._get_segment_points(i, (i+1)%count)
            if len(pts) < 2: continue
            
            # Check distance against the detailed polyline segments
            for k in range(len(pts) - 1):
                d_sq = self._dist_sq_to_segment(hit_pos, pts[k], pts[k+1])
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
                    best_idx = i
        
        if min_dist_sq < threshold_sq:
            return best_idx
            
        return -1

    # --- INTERNAL LOGIC ---

    def _internal_add_node(self, pos, insert_after_idx=-1):
        pid = self.mesh.find_closest_point(pos)
        
        if self.node_ids and insert_after_idx == -1 and self.node_ids[-1] == pid:
             return self.nodes[-1], self.node_ids[-1]
             
        sphere = pv.Sphere(radius=self.node_radius)
        actor = self.plotter.add_mesh(
            sphere, color="yellow", render_points_as_spheres=False,
            reset_camera=False, pickable=True, name=f"BezierNode_{len(self.nodes)}"
        )
        actor.SetPosition(self.mesh.points[pid])
        
        if insert_after_idx != -1:
            idx = insert_after_idx + 1
            self.nodes.insert(idx, actor)
            self.node_ids.insert(idx, pid)
            self._invalidate_neighbors(insert_after_idx)
            self._invalidate_neighbors(idx)
        else:
            self.nodes.append(actor)
            self.node_ids.append(pid)
        
        self._update_node_colors()
        self.update_curve()
        self.plotter.render()
        return actor, pid

    def _internal_remove_node_by_actor(self, actor):
        try:
            idx = self.nodes.index(actor)
            self.plotter.remove_actor(actor)
            self._invalidate_neighbors(idx - 1)
            self.nodes.pop(idx)
            self.node_ids.pop(idx)
            if len(self.nodes) < 3: self.is_closed = False
            self._update_node_colors()
            self.update_curve()
            self.plotter.render()
        except ValueError: pass

    def _internal_restore_node(self, actor, pid, idx, was_closed):
        self.plotter.add_actor(actor)
        self.nodes.insert(idx, actor)
        self.node_ids.insert(idx, pid)
        self.is_closed = was_closed
        self._invalidate_neighbors(idx - 1)
        self._invalidate_neighbors(idx)
        self._update_node_colors()
        self.update_curve()
        self.plotter.render()

    def _internal_update_node_pos(self, actor, pos, pid):
        actor.SetPosition(pos)
        try:
            idx = self.nodes.index(actor)
            self.node_ids[idx] = pid
            self._invalidate_neighbors(idx - 1)
            self._invalidate_neighbors(idx)
            self.update_curve()
            self.plotter.render()
        except: pass

    def _internal_hard_reset(self, keep_actors=False):
        if self.line_actor: 
            self.plotter.remove_actor(self.line_actor)
            self.line_actor = None
        for name in ["Debug_Raw_Selection", "Debug_Barrier_Edge", "Debug_Hole_Fill", "Debug_Original_Overlay"]:
            self.plotter.remove_actor(name)
            
        if not keep_actors:
            for n in self.nodes: self.plotter.remove_actor(n)
            
        self.nodes = []
        self.node_ids = []
        self.path_cache = {}
        self.is_closed = False
        self.plotter.render()

    def _internal_restore_full_state(self, nodes_list, ids_list, closed_state):
        self._internal_hard_reset(keep_actors=True)
        for i, actor in enumerate(nodes_list):
            if actor not in self.plotter.actors.values():
                self.plotter.add_actor(actor)
            self.nodes.append(actor)
            self.node_ids.append(ids_list[i])
        self.is_closed = closed_state
        self._update_node_colors()
        self.update_curve()
        self.plotter.render()

    # --- INTERACTION HELPERS ---

    def try_pick_node(self):
        pos = self.plotter.iren.get_event_position()
        if self.node_picker.PickProp(pos[0], pos[1], self.plotter.renderer):
            actor = self.node_picker.GetActor()
            if actor in self.nodes: return actor
        return None

    def try_pick_mesh(self):
        if not self.locator: return None
        x, y = self.plotter.iren.get_event_position()
        renderer = self.plotter.renderer
        renderer.SetDisplayPoint(x, y, 0); renderer.DisplayToWorld()
        near = np.array(renderer.GetWorldPoint()[:3])
        renderer.SetDisplayPoint(x, y, 1); renderer.DisplayToWorld()
        far = np.array(renderer.GetWorldPoint()[:3])
        t = vtk.mutable(0.0); world = [0.0]*3; pcoords = [0.0]*3; subId = vtk.mutable(0)
        hit = self.locator.IntersectWithLine(near, far, 0.001, t, world, pcoords, subId)
        if hit: return np.array(world)
        return None

    def drag_active_node_visual(self):
        hit_pos = self.try_pick_mesh()
        if hit_pos is not None:
            pid = self.mesh.find_closest_point(hit_pos)
            snapped_pos = self.mesh.points[pid]
            self.active_node.SetPosition(snapped_pos)
            self.plotter.render()

    def handle_right_click(self):
        node = self.try_pick_node()
        if node:
            menu = QMenu()
            menu.setStyleSheet("QMenu { background-color: #333; color: white; }")
            menu.addAction("Delete Point").triggered.connect(lambda: self.delete_node_cmd(node))
            if self.nodes.index(node) == len(self.nodes) - 1:
                 menu.addAction("Close Loop").triggered.connect(self.close_loop)
            menu.exec_(QCursor.pos())

    def handle_key(self, key):
        if key in ["Return", "Enter"]: self.close_loop(); return True
        elif key in ["Delete", "BackSpace"] and self.nodes:
            target = self.active_node if self.active_node else self.nodes[-1]
            self.delete_node_cmd(target)
            return True
        return False

    def _update_node_colors(self):
        for i, node in enumerate(self.nodes):
            prop = node.GetProperty()
            if node == self.active_node: prop.SetColor(1.0, 0.0, 0.0) 
            elif i == len(self.nodes) - 1 and not self.is_closed: prop.SetColor(0.2, 0.6, 1.0)
            else: prop.SetColor(1.0, 1.0, 0.0)

    # --- GEOMETRY HELPERS (Advanced Pathing) ---

    def _invalidate_neighbors(self, idx):
        self.path_cache.clear()

    def _get_segment_points(self, idx_a, idx_b):
        """
        Approximates a geodesic path between two nodes by linear interpolation 
        snapped to the nearest mesh vertices using cKDTree.
        """
        pid_a, pid_b = self.node_ids[idx_a], self.node_ids[idx_b]
        key = tuple(sorted((pid_a, pid_b)))
        if key in self.path_cache: return self.path_cache[key]
        
        p_a, p_b = self.mesh.points[pid_a], self.mesh.points[pid_b]
        dist = np.linalg.norm(p_b - p_a)
        
        # Density: Approx one point every 0.5 units, min 10 points
        num = int(max(10, dist / 0.5))
        t = np.linspace(0, 1, num)
        
        # Linear interp
        interp = p_a + np.outer(t, (p_b - p_a))
        
        # Snap to mesh
        _, ids = cKDTree(self.mesh.points).query(interp)
        snapped = self.mesh.points[ids]
        
        self.path_cache[key] = snapped
        return snapped

    def _get_full_path_points(self):
        if len(self.nodes) < 2: return np.array([])
        full = []
        count = len(self.nodes)
        rng = count if self.is_closed else count - 1
        for i in range(rng):
            pts = self._get_segment_points(i, (i+1)%count)
            # Avoid duplicating connecting points
            full.append(pts[1:] if full else pts)
        return np.concatenate(full) if full else np.array([])

    def update_curve(self):
        """Visualizes the path as a smooth spline tube."""
        if self.line_actor: 
            self.plotter.remove_actor(self.line_actor)
            self.line_actor = None
            
        pts = self._get_full_path_points()
        if len(pts) < 2: return
        
        try:
            if len(pts) > 3:
                # Smooth spline visualization
                spline = pv.Spline(pts, n_points=len(pts))
                tube = spline.tube(radius=self.node_radius * 0.5)
                self.line_actor = self.plotter.add_mesh(
                    tube, color=self.curve_color, pickable=False, 
                    reset_camera=False, name="BezierCurve"
                )
            else:
                # Fallback to lines for very short segments
                self.line_actor = self.plotter.add_lines(
                    pts, color=self.curve_color, width=3, 
                    name="BezierCurve", pickable=False
                )
        except: pass

    # --- REGION SELECTION LOGIC ---

    def get_selected_region(self):
        """
        Uses the closed loop to 'cut' the mesh and return the inner region.
        Implements ray-tracing to find seed point and flood fill.
        """
        # Clean debug actors
        self.plotter.remove_actor("Debug_Raw_Selection")
        self.plotter.remove_actor("Debug_Barrier_Edge")
        
        if not self.is_closed:
            print("Loop must be closed first.")
            return None
            
        if self.target_actor and self.target_actor.mapper:
            current_mesh = self.target_actor.mapper.dataset
            if isinstance(current_mesh, pv.UnstructuredGrid): 
                current_mesh = current_mesh.extract_surface()
            self.mesh = current_mesh
        if not self.mesh: return None

        # 1. Calculate approximate center and normal of the loop
        raw_path_points = self._get_full_path_points()
        if len(raw_path_points) < 3: return None
        
        center = np.mean(raw_path_points, axis=0)
        v1 = raw_path_points[0] - center
        v2 = raw_path_points[len(raw_path_points)//3] - center
        normal = np.cross(v1, v2)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-6: normal = np.array([0,0,1])
        else: normal = normal / norm_mag
        
        # Heuristic: Ensure normal points somewhat 'up' or towards camera? 
        # For dental, usually Z-up, but checking dot with Z=-1 might be safer if flipped.
        if normal[2] < 0: normal = -normal 

        # 2. Ray trace to find a seed point INSIDE the loop
        ray_start = center + (normal * 200)
        ray_end = center - (normal * 200)
        try: pts, _ = self.mesh.ray_trace(ray_start, ray_end)
        except: return None
        
        seed_coords = pts[0] if len(pts) > 0 else center

        # 3. Create a barrier on the mesh vertices corresponding to the path
        try:
            tree = cKDTree(self.mesh.points)
            dists, candidates = tree.query(raw_path_points, k=5)
        except: return None
        
        if "Normals" not in self.mesh.point_data: 
            self.mesh.compute_normals(inplace=True)
        mesh_normals = self.mesh.point_data["Normals"]
        
        clean_path_ids = []
        for i, pt_candidates in enumerate(candidates):
            best_pid = pt_candidates[0]
            # Heuristic: Pick point with normal somewhat facing our loop normal
            for pid in pt_candidates:
                if pid < len(mesh_normals): 
                    if np.dot(mesh_normals[pid], normal) > -0.2: 
                        best_pid = pid; break
            clean_path_ids.append(best_pid)

        # 4. Make barrier watertight
        barrier_ids = self._make_barrier_watertight(clean_path_ids)
        
        # 5. Scalar thresholding to cut the mesh
        n_points = self.mesh.n_points
        scalars = np.zeros(n_points, dtype=int)
        valid_barrier_ids = [pid for pid in barrier_ids if pid < n_points]
        scalars[valid_barrier_ids] = 1
        self.mesh.point_data["is_barrier"] = scalars
        
        # Cut out the barrier (creating holes)
        cut_mesh = self.mesh.threshold(0.5, scalars="is_barrier", invert=True)
        
        # 6. Connectivity Analysis (Flood Fill)
        connected = cut_mesh.connectivity(extraction_mode='all')
        
        # Find which region the seed point belongs to
        cut_tree = cKDTree(connected.points)
        _, closest_idx = cut_tree.query(seed_coords)
        region_ids = connected.point_data['RegionId']
        target_region = region_ids[closest_idx]
        
        # Extract that region
        final_selection = connected.threshold(
            [target_region-0.1, target_region+0.1], 
            scalars='RegionId'
        ).extract_surface().clean()
        
        if final_selection.n_points < 10: return None
        
        # Debug Visualization
        debug_mesh = final_selection.copy()
        if "Normals" in debug_mesh.point_data: 
            debug_mesh.points += debug_mesh.point_data["Normals"] * 0.05
        self.plotter.add_mesh(
            debug_mesh, name="Debug_Raw_Selection", 
            color="red", opacity=0.6, lighting=False, pickable=False
        )
        
        return final_selection
    

    # Add this method to BezierMarkerTool
    def clear_all_markup(self):
        """Clears all nodes and lines immediately."""
        self._internal_hard_reset(keep_actors=False)

    def _make_barrier_watertight(self, initial_ids: List[int]) -> Set[int]:
        """BFS to connect separated barrier points to ensure the cut is continuous."""
        if not initial_ids: return set()
        
        ordered_ids = [initial_ids[0]]
        for pid in initial_ids[1:]:
            if pid != ordered_ids[-1]: ordered_ids.append(pid)
            
        final = set(ordered_ids)
        point_cell_ids = self.mesh.point_cell_ids; GetCell = self.mesh.GetCell
        
        for i in range(len(ordered_ids) - 1):
            s, e = ordered_ids[i], ordered_ids[i+1]
            
            # If s and e share a cell, they are connected
            if set(point_cell_ids(s)) & set(point_cell_ids(e)): continue
            
            # BFS to find path between s and e
            q = deque([(s, [])]); vis = {s}; found = None
            while q:
                cur, path = q.popleft()
                if len(path) > 20: continue # Limit search depth
                
                for cid in point_cell_ids(cur):
                    cell = GetCell(cid)
                    for k in range(cell.GetNumberOfPoints()):
                        neigh = cell.GetPointId(k)
                        if neigh == e: found = path; break
                        if neigh not in vis: 
                            vis.add(neigh)
                            q.append((neigh, path+[neigh]))
                    if found: break
                if found: break
            
            if found: final.update(found)
            
        return final