import os
import time

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                camera_azimuth=None,
                camera_elevation=-10,
                camera_distance=None,
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480,
                keyboard_callback=None,
                ):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation
        self.camera_distance = camera_distance or self.viewer_cam_distance
        self.record_video = record_video


        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False, 
            key_callback=keyboard_callback
            )      

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
        
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos
        
        mj.mj_forward(self.model, self.data)
        
        if self.camera_follow and follow_camera:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.viewer.cam.distance = self.camera_distance
            self.viewer.cam.elevation = self.camera_elevation
            if self.camera_azimuth is not None:
                self.viewer.cam.azimuth = self.camera_azimuth
        
        if human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0
            # Draw the task targets for reference
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None
                    )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
    
    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")


class RobotMotionRenderer:
    """Offscreen MuJoCo MP4 renderer for robot qpos trajectories.

    This is the default backend for script-level video export. It avoids opening
    a passive viewer and uses a camera that follows the robot base, which makes
    exported walking/running clips easier to compare across tuning variants.
    """

    def __init__(
        self,
        robot_type,
        motion_fps=30,
        video_path=None,
        video_width=960,
        video_height=544,
        transparent_robot=0,
        camera_follow=True,
        camera_azimuth=135.0,
        camera_elevation=-12.0,
        camera_distance=None,
        keyboard_callback=None,
    ):
        del keyboard_callback
        os.environ.setdefault("MUJOCO_GL", "egl")

        if video_path is None:
            raise ValueError("Please provide video_path for offscreen rendering")

        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.model.vis.global_.offwidth = video_width
        self.model.vis.global_.offheight = video_height
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.robot_base_id = self.model.body(self.robot_base).id
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        self.motion_fps = motion_fps
        self.camera_follow = camera_follow
        self.camera_distance = camera_distance or self.viewer_cam_distance
        self.video_path = video_path

        video_dir = os.path.dirname(self.video_path)
        if video_dir and not os.path.exists(video_dir):
            os.makedirs(video_dir)

        self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
        self.camera = mj.MjvCamera()
        mj.mjv_defaultCamera(self.camera)
        self.camera.azimuth = camera_azimuth
        self.camera.elevation = camera_elevation
        self.camera.distance = self.camera_distance

        self.scene_option = mj.MjvOption()
        mj.mjv_defaultOption(self.scene_option)
        self.scene_option.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot

        self.mp4_writer = imageio.get_writer(self.video_path, fps=int(round(self.motion_fps)))
        print(f"Rendering video to {self.video_path}")

    def step(
        self,
        root_pos,
        root_rot,
        dof_pos,
        human_motion_data=None,
        show_human_body_name=False,
        human_point_scale=0.1,
        human_pos_offset=np.array([0.0, 0.0, 0]),
        rate_limit=True,
        follow_camera=True,
    ):
        del human_motion_data, show_human_body_name, human_point_scale, human_pos_offset, rate_limit

        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot
        self.data.qpos[7:] = dof_pos
        mj.mj_forward(self.model, self.data)

        if self.camera_follow and follow_camera:
            self.camera.lookat[:] = self.data.xpos[self.robot_base_id]
            self.camera.distance = self.camera_distance

        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        self.mp4_writer.append_data(self.renderer.render())

    def close(self):
        self.mp4_writer.close()
        self.renderer.close()
        print(f"Video saved to {self.video_path}")


class NullRobotMotionViewer:
    """No-op visualization backend used for save-only conversion."""

    def __init__(self, motion_fps=30, **kwargs):
        del kwargs
        self.rate_limiter = RateLimiter(frequency=motion_fps, warn=False)

    def step(
        self,
        root_pos,
        root_rot,
        dof_pos,
        human_motion_data=None,
        show_human_body_name=False,
        human_point_scale=0.1,
        human_pos_offset=np.array([0.0, 0.0, 0]),
        rate_limit=True,
        follow_camera=True,
    ):
        del root_pos, root_rot, dof_pos, human_motion_data, show_human_body_name
        del human_point_scale, human_pos_offset, follow_camera
        if rate_limit:
            self.rate_limiter.sleep()

    def close(self):
        pass


def create_robot_motion_visualizer(
    robot_type,
    viewer="auto",
    motion_fps=30,
    transparent_robot=0,
    record_video=False,
    video_path=None,
    video_width=960,
    video_height=544,
    camera_follow=True,
    camera_azimuth=135.0,
    camera_elevation=-12.0,
    camera_distance=None,
    keyboard_callback=None,
):
    if viewer not in {"auto", "gl", "offscreen", "none"}:
        raise ValueError(f"Unknown viewer backend: {viewer}")

    backend = "offscreen" if viewer == "auto" and record_video else viewer
    if backend == "auto":
        backend = "gl"

    if backend == "gl":
        return RobotMotionViewer(
            robot_type=robot_type,
            motion_fps=motion_fps,
            transparent_robot=transparent_robot,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            camera_distance=camera_distance,
            record_video=record_video,
            video_path=video_path,
            video_width=video_width,
            video_height=video_height,
            camera_follow=camera_follow,
            keyboard_callback=keyboard_callback,
        )

    if backend == "offscreen":
        return RobotMotionRenderer(
            robot_type=robot_type,
            motion_fps=motion_fps,
            transparent_robot=transparent_robot,
            video_path=video_path,
            video_width=video_width,
            video_height=video_height,
            camera_follow=camera_follow,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            camera_distance=camera_distance,
            keyboard_callback=keyboard_callback,
        )

    return NullRobotMotionViewer(motion_fps=motion_fps)
