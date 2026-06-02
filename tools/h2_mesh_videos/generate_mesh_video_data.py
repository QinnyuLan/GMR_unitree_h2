import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np
import torch
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.utils.smpl import (
    estimate_smplx_ground_offset,
    get_smplx_data_offline_fast,
    load_smplx_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/h2_mesh_videos/data"
DEFAULT_SMPLX_MODEL_PATH = "/mnt/data/SMPL-series/smplx"


def _resolve_smplx_model_path(path):
    model_path = Path(path).expanduser()
    if model_path.exists():
        return model_path

    fallback = REPO_ROOT / "assets" / "body_models"
    if fallback.exists():
        return fallback

    return model_path


def _sample_indices(num_frames, source_fps, output_fps, max_seconds=None):
    duration = num_frames / float(source_fps)
    if max_seconds is not None:
        duration = min(duration, max_seconds)
    sample_count = max(2, int(np.ceil(duration * output_fps)))
    times = np.arange(sample_count, dtype=np.float32) / float(output_fps)
    indices = np.clip(np.round(times * source_fps).astype(np.int64), 0, num_frames - 1)
    return times, indices, duration


def _robot_body_transforms(robot_type, qpos_list, robot_offset):
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot_type]))
    data = mj.MjData(model)
    body_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    ]

    body_q = np.empty((len(qpos_list), len(body_names), 7), dtype=np.float32)
    for frame_idx, qpos in enumerate(qpos_list):
        data.qpos[:3] = qpos[:3] + robot_offset
        data.qpos[3:7] = qpos[3:7]
        data.qpos[7:] = qpos[7:]
        mj.mj_forward(model, data)
        for body_idx, body_name in enumerate(body_names):
            body_id = model.body(body_name).id
            xpos = data.xpos[body_id]
            xquat_wxyz = data.xquat[body_id]
            body_q[frame_idx, body_idx, 0:3] = xpos
            body_q[frame_idx, body_idx, 3:7] = xquat_wxyz[[1, 2, 3, 0]]

    return body_q, body_names


def _sample_smplx_vertices(smplx_output, frame_indices, human_offset):
    vertices = smplx_output.vertices.detach().cpu().numpy()[frame_indices].astype(np.float32)
    vertices += human_offset.astype(np.float32)
    return vertices


def _retarget_qpos(frames, actual_human_height, robot_type, ground_clearance, offset_to_ground):
    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot_type,
        ground_clearance=ground_clearance,
        verbose=False,
    )
    if offset_to_ground:
        retargeter.set_ground_offset(
            estimate_smplx_ground_offset(frames, retargeter, ground_clearance)
        )

    qpos_list = []
    for frame in tqdm(frames, desc="Retargeting"):
        qpos_list.append(retargeter.retarget(frame, offset_to_ground=False))
    return qpos_list


def _write_motion_data(
    motion_path,
    out_path,
    robot_type,
    fps,
    smplx_model_path,
    max_seconds,
    robot_offset,
    human_offset,
    ground_clearance,
    offset_to_ground,
):
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        motion_path, smplx_model_path
    )
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=fps
    )
    times, frame_indices, duration = _sample_indices(
        len(frames), aligned_fps, fps, max_seconds=max_seconds
    )
    sampled_frames = [frames[int(idx)] for idx in frame_indices]

    qpos_list = _retarget_qpos(
        sampled_frames,
        actual_human_height,
        robot_type,
        ground_clearance,
        offset_to_ground,
    )
    h2_body_q, body_names = _robot_body_transforms(robot_type, qpos_list, robot_offset)
    source_indices = np.clip(
        np.round(times * float(smplx_data["mocap_frame_rate"].item())).astype(np.int64),
        0,
        smplx_output.vertices.shape[0] - 1,
    )
    smplx_vertices = _sample_smplx_vertices(smplx_output, source_indices, human_offset)

    metadata = {
        "motion": str(motion_path),
        "robot_type": robot_type,
        "fps": fps,
        "duration": float(duration),
        "body_names": body_names,
        "smplx_faces": int(body_model.faces.shape[0]),
        "robot_offset": robot_offset.tolist(),
        "human_offset": human_offset.tolist(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        times=times,
        h2_body_q=h2_body_q,
        body_names=np.array(body_names),
        smplx_vertices=smplx_vertices,
        smplx_faces=body_model.faces.astype(np.int32),
        metadata=np.array(json.dumps(metadata)),
    )
    print(f"[OK] wrote {out_path} ({len(times)} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(
        description="Cache GMR-retargeted H2 and source SMPL-X mesh poses for Blender rendering."
    )
    parser.add_argument("--robot-type", default="unitree_h2", choices=["unitree_h2"])
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smplx-model-path", default=DEFAULT_SMPLX_MODEL_PATH)
    parser.add_argument("--max-seconds", type=float, default=6.0)
    parser.add_argument("--ground-clearance", type=float, default=0.0)
    parser.add_argument("--offset-to-ground", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot-offset", type=float, nargs=3, default=[0.0, -0.55, 0.0])
    parser.add_argument("--human-offset", type=float, nargs=3, default=[0.0, 0.55, 0.0])
    parser.add_argument("motions", nargs="+", help="SMPL-X .npz motion files to cache.")
    args = parser.parse_args()

    smplx_model_path = _resolve_smplx_model_path(args.smplx_model_path)
    out_dir = Path(args.out_dir)
    for motion in args.motions:
        motion_path = Path(motion)
        out_path = out_dir / f"{motion_path.stem}.npz"
        _write_motion_data(
            str(motion_path),
            out_path,
            args.robot_type,
            args.fps,
            smplx_model_path,
            args.max_seconds,
            np.array(args.robot_offset, dtype=np.float32),
            np.array(args.human_offset, dtype=np.float32),
            args.ground_clearance,
            args.offset_to_ground,
        )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
