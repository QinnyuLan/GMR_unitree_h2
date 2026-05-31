from general_motion_retargeting import create_robot_motion_visualizer, load_robot_motion
import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
    parser.add_argument(
        "--viewer",
        choices=["auto", "gl", "offscreen", "none"],
        default="auto",
        help="Visualization backend. auto uses offscreen for --record_video, otherwise gl.",
    )
    parser.add_argument("--video_width", type=int, default=960)
    parser.add_argument("--video_height", type=int, default=544)
    parser.add_argument("--camera_azimuth", type=float, default=135.0)
    parser.add_argument("--camera_elevation", type=float, default=-12.0)
    parser.add_argument("--camera_distance", type=float, default=None)
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loop playback. Disable this for finite offscreen video exports.",
    )
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    
    env = create_robot_motion_visualizer(
        robot_type=robot_type,
        motion_fps=motion_fps,
        camera_follow=True,
        viewer=args.viewer,
        record_video=args.record_video,
        video_path=args.video_path,
        video_width=args.video_width,
        video_height=args.video_height,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        camera_distance=args.camera_distance,
    )
    
    frame_idx = 0
    while True:
        env.step(motion_root_pos[frame_idx], 
                motion_root_rot[frame_idx], 
                motion_dof_pos[frame_idx], 
                rate_limit=True)
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            if not args.loop:
                break
            frame_idx = 0
    env.close()
