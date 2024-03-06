from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import configargparse
import torch
from smart_open import open
from tqdm import tqdm

from metadata_utils import get_frame_range, get_bounds_from_depth, get_neighbor, \
    write_metadata, get_val_frames, scale_bounds, OPENCV_TO_OPENGL, normalize_timestamp
from suds.data.image_metadata import ImageMetadata
from suds.stream_utils import image_from_stream, get_filesystem
from smor_io import load_poses_and_object_info, load_cam_info
GROUND_PLANE_Z = torch.DoubleTensor([[1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, 0, 1]])


def get_smor_items(smor_path: str) -> \
        Tuple[List[ImageMetadata], List[str], torch.Tensor, float, torch.Tensor]:

    cam_map_ts_map_imgf, cam_map_ts_map_depthf, cam_map_ts_map_w_T_c, cam_map_intr = load_cam_info(Path(smor_path))

    cameras = list(cam_map_ts_map_imgf.keys())
    num_frames = max([len(cam_map_ts_map_imgf[cam]) for cam in cameras])

    val_frames = set() #get_val_frames(num_frames, test_every, train_every)
    metadata_items: List[ImageMetadata] = []
    item_frame_ranges: List[Tuple[int]] = []
    static_masks = []
    min_bounds = None
    max_bounds = None

    use_masks = True
    min_frame = 0
    max_frame = sum([len(cam_map_ts_map_imgf[cam]) for cam in cameras])

    cam_ts = [(cam, ts) for cam in cameras for ts in cam_map_ts_map_imgf[cam]]
    cam_map_ts_map_index = {cam: {} for cam in cameras}
    for i, (cam, ts) in enumerate(cam_ts):
        cam_map_ts_map_index[cam][ts] = i

    min_ts = min([min(list(cam_map_ts_map_imgf[cam].keys())) for cam in cameras])
    max_ts = max([max(list(cam_map_ts_map_imgf[cam].keys())) for cam in cameras])
    
    for cam, ts in cam_ts:

        path = cam_map_ts_map_imgf[cam][ts]
        image = Image.open(path)
        K = cam_map_intr[cam]
        image_index = len(metadata_items)
        assert image_index == cam_map_ts_map_index[cam][ts]

        time = ((ts - min_ts) / (max_ts - min_ts)) * 2 - 1

        tss = sorted(list(cam_map_ts_map_index[cam].keys()))
        cam_frame_idx = tss.index(ts)

        if cam_frame_idx < len(tss) - 1:
            forward_neighbor_ts = tss[cam_frame_idx + 1]
            forward_neighbor = cam_map_ts_map_index[cam][forward_neighbor_ts]
            forward_flow_path = f'{smor_path}/dino_correspondences/forward/{cam}/{ts}.parquet'
        else:
            forward_neighbor = None
            forward_flow_path = None

        if cam_frame_idx > 0:
            backward_neighbor_ts = tss[cam_frame_idx - 1]
            backward_neighbor = cam_map_ts_map_index[cam][backward_neighbor_ts]
            backward_flow_path = f'{smor_path}/dino_correspondences/backward/{cam}/{ts}.parquet'
        else:
            backward_neighbor = None
            backward_flow_path = None

        item = ImageMetadata(
            str(cam_map_ts_map_imgf[cam][ts]),
            torch.from_numpy(cam_map_ts_map_w_T_c[cam][ts])[:3],
            image.size[0],
            image.size[1],
            torch.FloatTensor([K[0,0], K[1, 1], K[0, 2], K[1, 2]]), # intr: [K[0,0], K[1, 1], K[0, 2], L[1, 2]
            image_index,
            time, # time [-1, 1]
            0, # no idea
            str(cam_map_ts_map_depthf[cam][ts]), # depth path
            None, # mask path
            None, # sky mask
            None, # dino path
            backward_flow_path, # pathname that backward flow will be written to or None if out of range
            forward_flow_path, # pathname that forward flow will be written to or None if out of range
            backward_neighbor, # index of backward neighbor for flow computation or None if out of range
            forward_neighbor, # index of foward neighbor for flow computation or None if out of range
            False, # is val
            1, # pose scale
            None # local cache
        )

        metadata_items.append(item)
        min_bounds, max_bounds = get_bounds_from_depth(item, min_bounds, max_bounds)

    origin, pose_scale_factor, scene_bounds = scale_bounds(metadata_items, min_bounds, max_bounds)

    return metadata_items, static_masks, origin, pose_scale_factor, scene_bounds


def _get_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--smor_path', type=str, required=True)

    return parser.parse_args()


def main(hparams: Namespace) -> None:
    metadata_items, static_masks, origin, pose_scale_factor, scene_bounds = get_smor_items(hparams.smor_path)
    write_metadata(hparams.output_path, metadata_items, static_masks, origin, pose_scale_factor, scene_bounds)


if __name__ == '__main__':
    main(_get_opts())
