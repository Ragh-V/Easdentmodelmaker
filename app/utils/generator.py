import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from typing import Optional
from typing import Optional, Tuple
class ModelGenerator:
    """Robust Dental Base Generator."""
    
    @staticmethod
    def clean_undesirable_artifacts(mesh: pv.PolyData, threshold=0.1, rescue_threshold=2.1, grid_resolution=100):
        if not mesh: return None
        if isinstance(mesh, pv.UnstructuredGrid): mesh = mesh.extract_surface()
        
        work = mesh.copy()
        work.compute_normals(inplace=True, auto_orient_normals=False)
        pts = work.points
        z = pts[:, 2]
        
        if len(pts) == 0: return work
        
        min_x, max_x = np.min(pts[:,0]), np.max(pts[:,0])
        min_y, max_y = np.min(pts[:,1]), np.max(pts[:,1])
        if (max_x - min_x) == 0: return work
        
        x_bins = np.clip(np.floor((pts[:,0]-min_x)/(max_x-min_x)*grid_resolution).astype(int), 0, grid_resolution-1)
        y_bins = np.clip(np.floor((pts[:,1]-min_y)/(max_y-min_y)*grid_resolution).astype(int), 0, grid_resolution-1)
        flat = x_bins * grid_resolution + y_bins
        
        max_z = np.full(grid_resolution**2, -np.inf)
        np.maximum.at(max_z, flat, z)
        depths = max_z[flat] - z
        
        is_wall = np.abs(work.point_data["Normals"][:, 2]) < 0.4
        mask = (depths <= threshold) | is_wall
        
        bad = np.where(~mask)[0]; good = np.where(mask)[0]
        if len(bad) > 0 and len(good) > 0:
            tree = cKDTree(pts[good])
            dists, _ = tree.query(pts[bad], distance_upper_bound=rescue_threshold)
            mask[bad[dists <= rescue_threshold]] = True
            
        return work.extract_points(mask, adjacent_cells=False).extract_surface().clean()
    
    @staticmethod
    def create_base_from_selection(region_poly: pv.PolyData, base_height=20.0, skirt_size=1.0) -> Optional[Tuple[pv.PolyData, np.ndarray]]:
        """
        Generates a base and returns the specific border loop used to create it.
        Returns: (Generated_Mesh, Smoothed_Border_Points) or None
        """
        # --- 1. PREP & CLEAN ---
        if not region_poly or region_poly.n_points < 3: return None
        if isinstance(region_poly, pv.UnstructuredGrid): region_poly = region_poly.extract_surface()
        
        base = region_poly.copy().clean()
        if "Normals" not in base.point_data:
            base.compute_normals(inplace=True, auto_orient_normals=True)
            
        # --- 2. EXTRACT RAW BOUNDARY ---
        edges = base.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        strips = edges.strip(join=True)
        if strips.n_lines == 0: return None
        
        # Get largest loop
        lines = strips.lines
        cursor = 0; max_pts = -1; pts_ids = []
        while cursor < len(lines):
            n = lines[cursor]; cursor += 1
            if n > max_pts: max_pts = n; pts_ids = lines[cursor : cursor + n]
            cursor += n
            
        if len(pts_ids) < 3: return None
        boundary_coords = strips.points[pts_ids]
        
        # --- 3. SMOOTH BOUNDARY (The "Golden" Path) ---
        # We smooth the raw jagged edge to get the nice denture border.
        # We MUST return this exact geometry so the tool knows where the handles go.
        smoothed_boundary = boundary_coords.copy()
        for _ in range(10): 
            smoothed_boundary = (np.roll(smoothed_boundary, 1, axis=0) + 
                                 np.roll(smoothed_boundary, -1, axis=0)) * 0.5
            
        # --- 4. GEOMETRY GENERATION ---
        # Generate Skirt Vectors
        center = np.mean(smoothed_boundary, axis=0)
        tangents = np.roll(smoothed_boundary, -1, axis=0) - np.roll(smoothed_boundary, 1, axis=0)
        tangents[:, 2] = 0
        
        skirt_normals = np.zeros_like(tangents)
        skirt_normals[:, 0] = -tangents[:, 1]
        skirt_normals[:, 1] = tangents[:, 0]
        
        mags = np.linalg.norm(skirt_normals, axis=1, keepdims=True)
        mags[mags == 0] = 1 
        skirt_normals /= mags
        
        if np.mean(np.sum(skirt_normals * (smoothed_boundary - center), axis=1)) < 0:
            skirt_normals *= -1
            
        skirt_coords = smoothed_boundary + (skirt_normals * skirt_size)
        
        # Wall Bottom
        wall_bottom_coords = skirt_coords.copy()
        target_z = np.mean(smoothed_boundary[:, 2]) + base_height
        wall_bottom_coords[:, 2] = target_z
        
        # Stitch Wall
        wall_points = np.vstack([smoothed_boundary, skirt_coords, wall_bottom_coords])
        L = len(smoothed_boundary)
        faces = []
        for i in range(L):
            curr = i; nxt = (i + 1) % L
            faces.extend([4, curr, nxt, nxt + L, curr + L])          # Skirt
            faces.extend([4, curr + L, nxt + L, nxt + 2*L, curr + 2*L]) # Wall
            
        wall_mesh = pv.PolyData(wall_points, faces)
        
        # Cap
        cap_poly = pv.PolyData(wall_bottom_coords)
        cap_mesh = cap_poly.delaunay_2d()
        
        # --- 5. SNAP BASE TO SMOOTHED BOUNDARY ---
        # To ensure the base matches the wall perfectly, we move the original
        # jagged boundary points to the smoothed positions.
        base_tree = cKDTree(base.points)
        # Query using the RAW coords to find indices, but move to SMOOTHED coords
        dists, base_indices = base_tree.query(boundary_coords)
        base.points[base_indices] = smoothed_boundary

        # --- 6. FINALIZE ---
        final_mesh = base + wall_mesh + cap_mesh
        final_mesh = final_mesh.clean().compute_normals(auto_orient_normals=True)
        
        # RETURN TUPLE: (Mesh, The_Exact_Border_Loop)
        return final_mesh, smoothed_boundary
    
    # (Include clean_undesirable_artifacts here if needed, unchanged)