# import open3d as o3d
import pickle
from pathlib import Path
import numpy as np


# def read_pcd(filename: Path):
#     return np.asarray(o3d.io.read_point_cloud(str(filename)).points)


def load_poses_and_object_info(root: Path):
    with open(root / 'st_map_inst.pkl', 'rb') as fd:
        st_map_inst = pickle.load(fd)

    with open(root / 'iter_st_map_world_T_sensor.pkl', 'rb') as fd:
        iter_st_map_w_T_s = pickle.load(fd)

    with open(root / 'inst_map_size.pkl', 'rb') as fd:
        inst_map_size = pickle.load(fd)

    with open(root / 'inst_map_category.pkl', 'rb') as fd:
        inst_map_cate = pickle.load(fd)

    inst_map_iter_st_map_o_T_s = {}
    for inst in inst_map_size:
        with open(root / f'{inst}_iter_st_map_object_T_sensor.pkl', 'rb') as fd:
            inst_map_iter_st_map_o_T_s[inst] = pickle.load(fd)
    return inst_map_iter_st_map_o_T_s, iter_st_map_w_T_s, st_map_inst, inst_map_size, inst_map_cate


# def load_points(root: Path):
#     with open(root / 'st_map_ts.pkl', 'rb') as fd:
#         st_map_ts = pickle.load(fd)
#     sweep_dir = root / 'sweeps'
#     st_map_xyz = {st: read_pcd(sweep_dir / f'{st_map_ts[st]}.pcd') for st in st_map_ts}

#     with open(root / 'st_map_into.pkl', 'rb') as fd:
#         st_map_into = pickle.load(fd)

#     with open(root / 'meta.pkl', 'rb') as fd:
#         sweep_length = pickle.load(fd)

#     return st_map_xyz, st_map_into, st_map_ts, sweep_length


def load_eval(root: Path):
    with open(root / 'inst_map_input_st_list.pkl', 'rb') as fd:
        inst_map_input_st_list = pickle.load(fd)
    with open(root / 'inst_map_st_map_interp_o_T_s.pkl', 'rb') as fd:
        inst_map_st_map_interp_o_T_s = pickle.load(fd)
    with open(root / 'inst_map_st_map_gt_o_T_s.pkl', 'rb') as fd:
        inst_map_st_map_gt_o_T_s = pickle.load(fd)        

    return inst_map_input_st_list, inst_map_st_map_interp_o_T_s, inst_map_st_map_gt_o_T_s


def load_cam_info(root: Path):
    with open(root / 'cam_map_ts_map_imgf.pkl', 'rb') as fd:
        cam_map_ts_map_imgf = pickle.load(fd)

    with open(root / 'cam_map_ts_map_w_T_c.pkl', 'rb') as fd:
        cam_map_ts_map_w_T_c = pickle.load(fd)

    with open(root / 'cam_map_intr.pkl', 'rb') as fd:
        cam_map_intr = pickle.load(fd)

    depth_dir = root / 'depth'
    cam_map_ts_map_depthf = {cam: {ts: depth_dir / f'{cam}_{ts}_depth.tiff' for ts in ts_map_imgf}
                             for cam, ts_map_imgf in cam_map_ts_map_imgf.items()}

    return cam_map_ts_map_imgf, cam_map_ts_map_depthf, cam_map_ts_map_w_T_c, cam_map_intr
