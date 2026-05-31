import argparse
import json
import os
import pathlib
import pickle

import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import create_robot_motion_visualizer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.smpl import (
    get_gvhmr_data_offline_fast,
    get_smplx_data_offline_fast,
    load_gvhmr_pred_file,
    load_smplx_file,
)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _default_output_path(robot, motion_path):
    motion_name = pathlib.Path(motion_path).stem
    return f"videos/{robot}_{motion_name}.mp4"


def _save_robot_motion(save_path, fps, qpos_list):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    motion_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"Saved motion to {save_path}")


def _load_motion_frames(args):
    if args.source == "smplx":
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            args.motion, args.smplx_model_path
        )
        return get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=args.fps,
        ) + (actual_human_height, "smplx")

    if args.source == "gvhmr":
        smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
            args.motion, args.smplx_model_path
        )
        return get_gvhmr_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=args.fps,
        ) + (actual_human_height, "smplx")

    if args.source in {"bvh_lafan1", "bvh_nokov"}:
        bvh_format = args.source.removeprefix("bvh_")
        frames, actual_human_height = load_bvh_file(args.motion, format=bvh_format)
        return frames, args.fps, actual_human_height, args.source

    raise ValueError(f"Unsupported source: {args.source}")


def _apply_config(args, config):
    field_map = {
        "motion": ["motion", "motion_file", "smplx_file", "gvhmr_pred_file", "bvh_file"],
        "source": ["source", "src_human", "retarget_source"],
        "robot": ["robot", "tgt_robot", "retarget_target"],
        "output": ["output", "video_path"],
        "record_video": ["record_video"],
        "save_path": ["save_path"],
        "smplx_model_path": ["smplx_model_path"],
        "fps": ["fps", "motion_fps"],
        "ground_clearance": ["ground_clearance"],
        "offset_to_ground": ["offset_to_ground"],
        "hide_targets": ["hide_targets"],
        "rate_limit": ["rate_limit"],
        "video_width": ["video_width"],
        "video_height": ["video_height"],
        "camera_azimuth": ["camera_azimuth"],
        "camera_elevation": ["camera_elevation"],
        "camera_distance": ["camera_distance"],
    }
    for arg_name, keys in field_map.items():
        current = getattr(args, arg_name)
        for key in keys:
            if key in config and current == parser.get_default(arg_name):
                setattr(args, arg_name, config[key])
                break


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
parser.add_argument(
    "--source",
    choices=["smplx", "gvhmr", "bvh_lafan1", "bvh_nokov"],
    default="smplx",
)
parser.add_argument("--motion", type=str, default=None, help="Input motion file.")
parser.add_argument("--robot", type=str, default="unitree_g1")
parser.add_argument(
    "--viewer",
    choices=["auto", "gl", "offscreen", "none"],
    default="auto",
    help="auto exports video offscreen unless --no-record_video is set; gl opens a MuJoCo window.",
)
parser.add_argument(
    "--record_video",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Export an MP4 video.",
)
parser.add_argument("--output", type=str, default=None, help="Output MP4 path.")
parser.add_argument("--save_path", type=str, default=None, help="Optional output robot motion pkl.")
parser.add_argument("--smplx_model_path", type=str, default="/media/sky/Data/SMPL/smplx")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--rate_limit", action="store_true", default=False)
parser.add_argument("--offset_to_ground", action="store_true", default=False)
parser.add_argument("--ground_clearance", type=float, default=0.0)
parser.add_argument("--hide_targets", action="store_true", default=False)
parser.add_argument("--video_width", type=int, default=960)
parser.add_argument("--video_height", type=int, default=544)
parser.add_argument("--camera_azimuth", type=float, default=135.0)
parser.add_argument("--camera_elevation", type=float, default=-12.0)
parser.add_argument("--camera_distance", type=float, default=None)
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=None)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.config is not None:
        _apply_config(args, _load_json(args.config))

    if args.motion is None:
        raise ValueError("Please provide --motion or set motion in --config.")

    frames, motion_fps, actual_human_height, src_human = _load_motion_frames(args)
    start = min(args.start_frame, max(len(frames) - 1, 0))
    stop = len(frames) if args.max_frames is None else min(start + args.max_frames, len(frames))
    frames = frames[start:stop]

    output = args.output or _default_output_path(args.robot, args.motion)
    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human=src_human,
        tgt_robot=args.robot,
        ground_clearance=args.ground_clearance,
    )
    viewer = create_robot_motion_visualizer(
        robot_type=args.robot,
        viewer=args.viewer,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=output,
        video_width=args.video_width,
        video_height=args.video_height,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        camera_distance=args.camera_distance,
    )

    qpos_list = []
    for frame in tqdm(frames, desc="Retargeting"):
        qpos = retargeter.retarget(frame, offset_to_ground=args.offset_to_ground)
        viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=None if args.hide_targets else retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)

    viewer.close()
    if args.save_path is not None:
        _save_robot_motion(args.save_path, motion_fps, qpos_list)
