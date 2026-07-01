# geometry.py
"""
Handles loading, scaling, and refining of mesh geometry from folder-based definitions.
Includes a caching system to store refined meshes for faster startup.
"""
import trimesh
import numpy as np
import os
import glob
from tqdm import tqdm


def load_scene(geometry_folders, cache_dir=None):
    """
    Loads geometry from specified folders, applies group settings for scaling
    and refinement, and uses a cache to speed up loading of already-processed meshes.
    Returns the meshes grouped by folder name.
    
    Args:
        geometry_folders (dict): Configuration dictionary for geometry folders.
        cache_dir (str, optional): Path to the directory for storing/loading cached
                                   refined meshes. Defaults to None (caching disabled).
    
    Returns:
        dict: A dictionary of {folder_name: [list_of_trimesh_objects]}.
    """

    # The main data structure is a dictionary, e.g.:
    # {'NEU': [mesh1, mesh2], 'RID': [mesh3, mesh4]}
    grouped_meshes = {}

    print("\nLoading and processing geometry from folders...")
    
    # Create the cache directory if it doesn't exist and caching is enabled
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using geometry cache directory: '{cache_dir}'")

    # Iterate through the high-level folder definitions from the config
    for folder_path, settings in geometry_folders.items():
        if not os.path.isdir(folder_path):
            print(f"WARNING: Geometry folder not found: '{folder_path}'. Skipping.")
            continue
            
        scale = settings.get("scale", 1.0)
        target_length = settings.get("target_length", None)
        
        # Find all .stl files inside this specific folder
        search_path = os.path.join(folder_path, '*.stl')
        stl_files_in_folder = glob.glob(search_path)
        
        if not stl_files_in_folder:
            print(f"INFO: No .stl files found in folder '{folder_path}'.")
            continue

        print(f"Processing {len(stl_files_in_folder)} files from '{folder_path}'...")
        
        meshes_in_folder = []
        for f in tqdm(stl_files_in_folder, desc=f"Folder '{folder_path}'"):
            try:
                mesh = None
                basename = os.path.basename(f)
                
                # --- Caching Logic ---
                if cache_dir:
                    # Create a unique filename based on the original name, scale, and target length.
                    # This ensures that if you change parameters, a new cache file is generated.
                    # Note: We only cache refined meshes, as unrefined meshes load fast anyway.
                    cache_filename = ""
                    if target_length:
                        cache_filename = f"{os.path.splitext(basename)[0]}_L{target_length}_S{scale}.stl"
                    
                    if cache_filename:
                        cache_path = os.path.join(cache_dir, cache_filename)
                        if os.path.exists(cache_path) and \
                                os.path.getmtime(cache_path) >= os.path.getmtime(f):
                            # Load the already-refined mesh directly from the cache.
                            # The mtime check ensures we re-process if the source file
                            # has been updated since the cache was written.
                            mesh = trimesh.load_mesh(cache_path)
                
                # If mesh was not loaded from cache, do the full processing
                if mesh is None:
                    loaded = trimesh.load(f)

                    # Handle STL files that contain multiple solid bodies (trimesh returns a Scene)
                    if isinstance(loaded, trimesh.Scene):
                        sub_meshes = [m for m in loaded.geometry.values()
                                      if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0]
                        if not sub_meshes:
                            print(f"WARNING: Scene in '{f}' contains no valid geometry. Skipping.")
                            continue
                        if len(sub_meshes) > 1:
                            print(f"  '{basename}': merged {len(sub_meshes)} solids into one mesh.")
                        mesh = trimesh.util.concatenate(sub_meshes)
                    else:
                        mesh = loaded

                    # Apply scaling before refinement
                    if scale != 1.0:
                        mesh.apply_scale(scale)
                    
                    # Apply global refinement if specified
                    if target_length and target_length > 0:
                        mesh = mesh.subdivide_to_size(max_edge=target_length)
                    
                    # If caching is enabled and we refined the mesh, save it to the cache.
                    # Write to a temp file then rename atomically so concurrent simulation
                    # runs cannot read a partially-written cache file.
                    if cache_dir and target_length:
                        tmp_path = cache_path + ".tmp"
                        mesh.export(tmp_path, file_type='stl')
                        os.replace(tmp_path, cache_path)
                
                mesh.metadata['name'] = basename
                meshes_in_folder.append(mesh)
                
            except Exception as e:
                print(f"\nError processing mesh '{f}': {e}. Skipping.")
        
        if meshes_in_folder:
            grouped_meshes[folder_path] = meshes_in_folder

    if not grouped_meshes:
        print("\nFATAL ERROR: No valid geometry was loaded from any folder. Exiting.")
        exit()

    num_total_objects = sum(len(v) for v in grouped_meshes.values())
    print(f"\nScene loaded: {num_total_objects} objects found in {len(grouped_meshes)} geometry groups.")
    
    return grouped_meshes