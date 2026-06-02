import numpy as np
from pathlib import Path
import smplx
import tempfile
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from scipy.interpolate import interp1d

import general_motion_retargeting.utils.lafan_vendor.utils as utils


SMPLX_EXTRA_JOINT_ORIENTATION_FALLBACKS = {
    "left_big_toe": "left_foot",
    "left_small_toe": "left_foot",
    "left_heel": "left_foot",
    "right_big_toe": "right_foot",
    "right_small_toe": "right_foot",
    "right_heel": "right_foot",
}

SOMA_TO_SMPLX_BODY_JOINTS = {
    "LeftLeg": "left_hip",
    "RightLeg": "right_hip",
    "Spine1": "spine1",
    "LeftShin": "left_knee",
    "RightShin": "right_knee",
    "Spine2": "spine2",
    "LeftFoot": "left_ankle",
    "RightFoot": "right_ankle",
    "Chest": "spine3",
    "LeftToeBase": "left_foot",
    "RightToeBase": "right_foot",
    "Neck1": "neck",
    "LeftShoulder": "left_collar",
    "RightShoulder": "right_collar",
    "Head": "head",
    "LeftArm": "left_shoulder",
    "RightArm": "right_shoulder",
    "LeftForeArm": "left_elbow",
    "RightForeArm": "right_elbow",
    "LeftHand": "left_wrist",
    "RightHand": "right_wrist",
}

SOMA_Y_UP_TO_Z_UP = R.from_euler("x", 90, degrees=True)


def _convert_soma_rotvecs_to_z_up(rotvecs):
    rotations = R.from_rotvec(rotvecs.reshape(-1, 3))
    converted = SOMA_Y_UP_TO_Z_UP * rotations * SOMA_Y_UP_TO_Z_UP.inv()
    return converted.as_rotvec().reshape(rotvecs.shape)


def _convert_soma_positions_to_z_up(positions):
    return SOMA_Y_UP_TO_Z_UP.apply(positions.reshape(-1, 3)).reshape(positions.shape)


def _add_smplx_extra_joints(result, joints):
    """Expose useful SMPL-X extra joints that are not part of the kinematic tree."""
    for joint_name, fallback_joint_name in SMPLX_EXTRA_JOINT_ORIENTATION_FALLBACKS.items():
        try:
            joint_index = JOINT_NAMES.index(joint_name)
        except ValueError:
            continue
        if joint_index >= len(joints) or joint_name in result:
            continue

        fallback_entry = result.get(fallback_joint_name, result.get("pelvis"))
        if fallback_entry is None:
            continue
        result[joint_name] = (joints[joint_index], fallback_entry[1])

    for side in ("left", "right"):
        big_toe_name = f"{side}_big_toe"
        small_toe_name = f"{side}_small_toe"
        heel_name = f"{side}_heel"
        foot_name = f"{side}_foot"
        if big_toe_name not in result or small_toe_name not in result:
            continue
        big_toe_pos = result[big_toe_name][0]
        small_toe_pos = result[small_toe_name][0]
        foot_entry = result.get(foot_name, result[big_toe_name])
        result[f"{side}_toe_center"] = (
            0.5 * (big_toe_pos + small_toe_pos),
            foot_entry[1],
        )
        if heel_name not in result:
            continue
        heel_pos = result[heel_name][0]
        result[f"{side}_sole_center"] = (
            0.5 * (result[f"{side}_toe_center"][0] + heel_pos),
            foot_entry[1],
        )
        half_width = 0.5 * (big_toe_pos - small_toe_pos)
        result[f"{side}_heel_inner"] = (heel_pos + half_width, foot_entry[1])
        result[f"{side}_heel_outer"] = (heel_pos - half_width, foot_entry[1])

    return result


def estimate_smplx_ground_offset(
    human_data_frames,
    retargeter,
    ground_clearance=0.0,
    percentile=5.0,
):
    """Estimate one fixed ground offset from the active foot targets."""
    active_body_names = set()
    for task_table in (retargeter.ik_match_table1, retargeter.ik_match_table2):
        for body_name, pos_weight, rot_weight, *_ in task_table.values():
            if (pos_weight != 0 or rot_weight != 0) and any(
                keyword in body_name for keyword in ("foot", "toe", "heel")
            ):
                active_body_names.add(body_name)
    ground_body_names = sorted(active_body_names)
    foot_heights = []
    for frame in human_data_frames:
        human_data = {
            body_name: [
                np.asarray(pos).copy(),
                np.asarray(quat).copy(),
            ]
            for body_name, (pos, quat) in frame.items()
        }
        human_data = retargeter.to_numpy(human_data)
        human_data = retargeter.scale_human_data(
            human_data,
            retargeter.human_root_name,
            retargeter.human_scale_table,
        )
        human_data = retargeter.offset_human_data(
            human_data,
            retargeter.pos_offsets1,
            retargeter.rot_offsets1,
        )
        frame_heights = [
            human_data[body_name][0][2]
            for body_name in ground_body_names
            if body_name in human_data
        ]
        if not frame_heights:
            frame_heights = [
                human_data[body_name][0][2]
                for body_name in human_data
                if "Foot" in body_name or "foot" in body_name
            ]
        if frame_heights:
            foot_heights.append(min(frame_heights))

    if not foot_heights:
        return 0.0
    return float(np.percentile(foot_heights, percentile) - ground_clearance)


def _as_gender(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).lower()


def _resolve_smplx_model_file(smplx_body_model_path, gender):
    model_path = Path(smplx_body_model_path).expanduser()
    gender = _as_gender(gender)
    ext_order = ("npz", "pkl")

    if model_path.is_file():
        return model_path, model_path.suffix.lstrip(".")

    candidates = []
    for ext in ext_order:
        candidates.extend(
            [
                model_path / f"SMPLX_{gender.upper()}.{ext}",
                model_path / "smplx" / f"SMPLX_{gender.upper()}.{ext}",
                model_path / gender / f"model.{ext}",
                model_path / "smplx" / gender / f"model.{ext}",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate, candidate.suffix.lstrip(".")

    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find SMPL-X model for gender '{gender}' under {model_path}. "
        f"Searched:\n{searched}"
    )


def _ensure_smplx_model_compatibility(model_file):
    if model_file.suffix != ".npz":
        return model_file

    model_data = dict(np.load(model_file, allow_pickle=True))
    extra_fields = {
        "hands_componentsl": np.zeros((45, 45), dtype=np.float32),
        "hands_componentsr": np.zeros((45, 45), dtype=np.float32),
        "hands_meanl": np.zeros(45, dtype=np.float32),
        "hands_meanr": np.zeros(45, dtype=np.float32),
        "lmk_faces_idx": np.zeros(0, dtype=np.int64),
        "lmk_bary_coords": np.zeros((0, 3), dtype=np.float32),
    }
    missing_fields = [key for key in extra_fields if key not in model_data]
    if not missing_fields:
        return model_file

    for key in missing_fields:
        model_data[key] = extra_fields[key]

    cache_dir = Path(tempfile.gettempdir()) / "gmr_smplx_model_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{model_file.parent.name}_{model_file.stem}_patched.npz"
    if not cache_file.exists():
        np.savez(cache_file, **model_data)
    return cache_file


def _create_smplx_body_model(smplx_body_model_path, gender):
    model_file, ext = _resolve_smplx_model_file(smplx_body_model_path, gender)
    model_file = _ensure_smplx_model_compatibility(model_file)
    return smplx.SMPLX(
        str(model_file),
        gender=_as_gender(gender),
        use_pca=False,
        ext=ext,
    )


def _fit_betas_to_model(betas, body_model):
    betas = np.asarray(betas, dtype=np.float32)
    betas = np.reshape(betas, (-1,))
    num_betas = int(getattr(body_model, "num_betas", body_model.shapedirs.shape[-1]))
    if betas.shape[0] == num_betas:
        return betas
    fitted_betas = np.zeros(num_betas, dtype=np.float32)
    fitted_betas[: min(num_betas, betas.shape[0])] = betas[:num_betas]
    return fitted_betas


def load_smpl_file(smpl_file):
    smpl_data = np.load(smpl_file, allow_pickle=True)
    return smpl_data


def _normalize_smplx_motion_data(smplx_data):
    data_fields = set(smplx_data.files)
    if {"pose_body", "root_orient", "mocap_frame_rate"}.issubset(data_fields):
        return smplx_data

    if "poses" not in data_fields:
        return smplx_data

    poses = smplx_data["poses"]
    normalized_data = {key: smplx_data[key] for key in smplx_data.files}
    if {"joint_names", "identity_model_type", "frame_time"}.issubset(data_fields) and poses.ndim == 3:
        if str(smplx_data["identity_model_type"].item()).lower() != "smplx":
            return smplx_data
        joint_names = [str(name) for name in smplx_data["joint_names"]]
        joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        poses = _convert_soma_rotvecs_to_z_up(poses)
        body_pose = np.zeros((poses.shape[0], 21, 3), dtype=poses.dtype)
        for soma_name, smplx_name in SOMA_TO_SMPLX_BODY_JOINTS.items():
            if soma_name not in joint_name_to_idx:
                continue
            try:
                smplx_idx = JOINT_NAMES.index(smplx_name)
            except ValueError:
                continue
            if not 1 <= smplx_idx <= 21:
                continue
            body_pose[:, smplx_idx - 1] = poses[:, joint_name_to_idx[soma_name]]

        normalized_data["gender"] = normalized_data.get("gender", np.asarray("neutral"))
        normalized_data["betas"] = normalized_data.get(
            "betas",
            np.asarray(smplx_data.get("identity_coeffs", np.zeros((1, 10), dtype=np.float32))),
        )
        normalized_data["root_orient"] = poses[:, joint_name_to_idx.get("Hips", 0)]
        normalized_data["pose_body"] = body_pose.reshape(poses.shape[0], -1)
        trans = normalized_data.get("trans", smplx_data.get("transl"))
        normalized_data["trans"] = _convert_soma_positions_to_z_up(trans)
        normalized_data["mocap_frame_rate"] = np.asarray(round(1.0 / float(smplx_data["frame_time"])))
        return normalized_data

    normalized_data["root_orient"] = poses[:, :3]
    normalized_data["pose_body"] = poses[:, 3:66]
    if "mocap_frame_rate" not in normalized_data and "mocap_framerate" in normalized_data:
        normalized_data["mocap_frame_rate"] = np.asarray(normalized_data["mocap_framerate"])
    return normalized_data


def load_smplx_file(smplx_file, smplx_body_model_path):
    smplx_data = _normalize_smplx_motion_data(np.load(smplx_file, allow_pickle=True))
    body_model = _create_smplx_body_model(smplx_body_model_path, smplx_data["gender"])
    betas = _fit_betas_to_model(smplx_data["betas"], body_model)
    # print(smplx_data["pose_body"].shape)
    # print(smplx_data["betas"].shape)
    # print(smplx_data["root_orient"].shape)
    # print(smplx_data["trans"].shape)
    
    num_frames = smplx_data["pose_body"].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(betas).float().view(1, -1), # (16,)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    
    if len(betas.shape)==1:
        human_height = 1.66 + 0.1 * betas[0]
    else:
        human_height = 1.66 + 0.1 * betas[0, 0]
    
    return smplx_data, body_model, smplx_output, human_height


def load_gvhmr_pred_file(gvhmr_pred_file, smplx_body_model_path):
    gvhmr_pred = torch.load(gvhmr_pred_file)
    smpl_params_global = gvhmr_pred['smpl_params_global']
    # print(smpl_params_global['body_pose'].shape)
    # print(smpl_params_global['betas'].shape)
    # print(smpl_params_global['global_orient'].shape)
    # print(smpl_params_global['transl'].shape)
    
    betas = np.pad(smpl_params_global['betas'][0], (0,6))
    
    # correct rotations
    # rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    
    # smpl_params_global['body_pose'] = smpl_params_global['body_pose'] @ rotation_matrix
    # smpl_params_global['global_orient'] = smpl_params_global['global_orient'] @ rotation_quat
    
    smplx_data = {
        'pose_body': smpl_params_global['body_pose'].numpy(),
        'betas': betas,
        'root_orient': smpl_params_global['global_orient'].numpy(),
        'trans': smpl_params_global['transl'].numpy(),
        "mocap_frame_rate": torch.tensor(30),
    }

    body_model = _create_smplx_body_model(smplx_body_model_path, "neutral")
    
    num_frames = smpl_params_global['body_pose'].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1), # (16,)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    
    if len(smplx_data['betas'].shape)==1:
        human_height = 1.66 + 0.1 * smplx_data['betas'][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data['betas'][0, 0]
    
    return smplx_data, body_model, smplx_output, human_height


def get_smplx_data(smplx_data, body_model, smplx_output, curr_frame):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    global_orient = smplx_output.global_orient[curr_frame].squeeze()
    full_body_pose = smplx_output.full_pose[curr_frame].reshape(-1, 3)
    joints = smplx_output.joints[curr_frame].detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    result = {}
    joint_orientations = []
    for i, joint_name in enumerate(joint_names):
        if i == 0:
            rot = R.from_rotvec(global_orient)
        else:
            rot = joint_orientations[parents[i]] * R.from_rotvec(
                full_body_pose[i].squeeze()
            )
        joint_orientations.append(rot)
        result[joint_name] = (joints[i], rot.as_quat(scalar_first=True))

    _add_smplx_extra_joints(result, joints)
  
    return result


def slerp(rot1, rot2, t):
    """Spherical linear interpolation between two rotations."""
    # Convert to quaternions
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()
    
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.sum(q1 * q2)
    
    # If the dot product is negative, slerp won't take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If the inputs are too close, linearly interpolate
    if dot > 0.9995:
        return R.from_quat(q1 + t * (q2 - q1))
    
    # Perform SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q1 + s1 * q2
    
    return R.from_quat(q)

def get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    
    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        new_num_frames = num_frames // frame_skip
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))

        _add_smplx_extra_joints(result, single_joints)

        smplx_data_frames.append(result)

    return smplx_data_frames, aligned_fps



def get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    
    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        new_num_frames = num_frames // frame_skip
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))

        _add_smplx_extra_joints(result, single_joints)

        smplx_data_frames.append(result)
        
    # add correct rotations
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    for result in smplx_data_frames:
        for joint_name in result.keys():
            orientation = utils.quat_mul(rotation_quat, result[joint_name][1])
            position = result[joint_name][0] @ rotation_matrix.T
            result[joint_name] = (position, orientation)
            

    return smplx_data_frames, aligned_fps
