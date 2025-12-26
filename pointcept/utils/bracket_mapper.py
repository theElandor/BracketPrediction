import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
from pathlib import Path

# def parse_args():
#     p = argparse.ArgumentParser(description="Generates map for stl bracket scan") 
#     p.add_argument("scan")
#     p.add_argument("--debug", action="store_true")
#     return p.parse_args()

class BracketMapper:
    '''
    This class is used to extract the "orientation" map of the bracket
    from the mesh of the tooth. It's main purpose is to help the model
    in understanding the orientation of simmetrical teeth like 47,37,27,17, ecc...
    '''
    def __init__(self, verbose=False):

        self.verbose = verbose
        self.x_axis = np.array([1,0,0])
        self.y_axis = np.array([0,1,0])
        self.z_axis = np.array([0,0,1])

        self.canine_left = (self.y_axis - self.x_axis) / np.linalg.norm(self.y_axis - self.x_axis)
        self.canine_right = (self.y_axis + self.x_axis) / np.linalg.norm(self.y_axis + self.x_axis)

        self.X_MAPPING = {

            # redundant but kept for clarity.
            # each FDI index is mapped to the direction that the
            # bracket should be facing in the reference coordinate system.
            # Y axis going outwards, Z axis upwards, and X axis towards patients' right

            # lower LEFT and upper LEFT.
            38: -self.x_axis, 28: -self.x_axis,
            37: -self.x_axis, 27: -self.x_axis,
            36: -self.x_axis, 26: -self.x_axis,
            35: -self.x_axis, 25: -self.x_axis,
            34: -self.x_axis, 24: -self.x_axis,
            33: self.canine_left, 23: self.canine_left,
            32: self.y_axis,  22: self.y_axis,
            31: self.y_axis,  21: self.y_axis,

            # lower RIGHT and upper RIGHT
            48: self.x_axis, 18: self.x_axis,
            47: self.x_axis, 17: self.x_axis,
            46: self.x_axis, 16: self.x_axis,
            45: self.x_axis, 15: self.x_axis,
            44: self.x_axis, 14: self.x_axis,
            43: self.canine_right, 13: self.canine_right,
            42: self.y_axis, 12: self.y_axis,
            41: self.y_axis, 11: self.y_axis,
        }

    def _get_fdi(self, scan:str) -> int:
        '''
        Given the path of the scan in the STEM_<lower/upper>_FDI_<index>.stl
        format returns the index of the tooth as a integer.
        ''' 
        name = Path(scan).name
        match = re.search(r'FDI_(\d+)', name)
        return int(match.group(1))

    def plot_mesh(self, scan:str, sim_values:np.array) -> None:
        '''
        Given the scan and the similarity values obtained with the
        get_sim method, makes a plot that visualizes the mask in 3D.
        '''
        mesh = trimesh.load(scan)
        fig = plt.figure(figsize=(18, 6))
        views = [
            (30, 45, "Front-Right View"),
            (30, 135, "Front-Left View"),
            (30, -90, "Side View")
        ]
        cmap = cm.get_cmap('jet')
        colors = cmap(sim_values)
        
        for idx, (elev, azim, title) in enumerate(views, 1):
            ax = fig.add_subplot(1, 3, idx, projection='3d')
            
            # Simple scatter plot of vertices
            ax.scatter(mesh.vertices[:, 0], 
                    mesh.vertices[:, 1], 
                    mesh.vertices[:, 2], 
                    c=sim_values, 
                    cmap='jet', 
                    s=1, 
                    vmin=0, 
                    vmax=1)
            
            # Get the data limits
            x_min, x_max = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
            y_min, y_max = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()
            z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
            
            # Draw axis lines from origin
            origin = [0, 0, 0]
            axis_length = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.7
            
            ax.plot([origin[0], origin[0] + axis_length], 
                    [origin[1], origin[1]], 
                    [origin[2], origin[2]], 
                    'r-', linewidth=2, label='X')
            
            ax.plot([origin[0], origin[0]], 
                    [origin[1], origin[1] + axis_length], 
                    [origin[2], origin[2]], 
                    'g-', linewidth=2, label='Y')
            
            ax.plot([origin[0], origin[0]], 
                    [origin[1], origin[1]], 
                    [origin[2], origin[2] + axis_length], 
                    'b-', linewidth=2, label='Z')
            
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)
            ax.legend()
            
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        
        plt.tight_layout()
        name = f"{Path(scan).stem}_visual.png"
        full_path = Path(scan).parent / name
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        print("Saved visualization to mesh_heatmap.png")
        plt.close()
    
    def get_sim_values(self, scan:str, mesh:trimesh.Geometry|None=None, save:bool=False) -> np.ndarray:
        '''
        Given the scan returns the array containing the orientation map.
        If specified can be saved to a npy file. 
        '''
        name = Path(scan).stem
        FDI_idx = self._get_fdi(name)
        if not mesh: # load the mesh if not given
            mesh = trimesh.load(scan)
        vertex_count = mesh.vertices.shape[0]
        faces = mesh.faces
        face_normals = mesh.face_normals
        vertex_normals = trimesh.geometry.mean_vertex_normals(vertex_count, faces, face_normals)
        # For each vertex compute the cosine similarity with -x axis and store that in a array
        vertex_normals_n = vertex_normals / np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        cos_sim = np.dot(vertex_normals_n, self.X_MAPPING[FDI_idx])
        sim_values = (cos_sim+1)/2
        if save:
            output_path = Path(scan).parent / f"{name}_orient.npy"
            np.save(output_path, sim_values)
            if self.verbose:
                print(f"Saved {output_path}.")
        return sim_values

# Example usage
# args = parse_args()
# if args.debug == True:
#     print("Hello")
#     debugpy.listen(("0.0.0.0", 5681))
#     print(">>> Debugger is listening on port 5681. Waiting for client to attach...")
#     debugpy.wait_for_client()
#     print(">>> Debugger attached. Resuming execution.")
# 
# bm = BracketMapper()
# mesh = trimesh.load(args.scan)
# sim_values = bm.get_sim_values(args.scan, mesh, save=True)
# bm.plot_mesh(mesh, sim_values)
# print("Finished!")