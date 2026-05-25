# Unitree H2 MuJoCo Joint Order

`h2_mujoco_joint_order.json` records the order used by this repository when saving
retargeted H2 motion:

- `root_pos` is `qpos[0:3]`.
- `root_rot` is `qpos[3:7]` in MuJoCo `wxyz` order before export conversion.
- `dof_pos` is `qpos[7:]` and follows the table below.
- `actuator_index` is the MuJoCo actuator/control index in `h2.xml`.
- `unitree_sdk_motor_index` is intentionally left `null` in the JSON until it is
  checked against the real Unitree H2 SDK or control stack.

| dof_pos | qpos | actuator | joint |
| ---: | ---: | ---: | --- |
| 0 | 7 | 0 | `left_hip_pitch_joint` |
| 1 | 8 | 1 | `left_hip_roll_joint` |
| 2 | 9 | 2 | `left_hip_yaw_joint` |
| 3 | 10 | 3 | `left_knee_joint` |
| 4 | 11 | 4 | `left_ankle_roll_joint` |
| 5 | 12 | 5 | `left_ankle_pitch_joint` |
| 6 | 13 | 6 | `right_hip_pitch_joint` |
| 7 | 14 | 7 | `right_hip_roll_joint` |
| 8 | 15 | 8 | `right_hip_yaw_joint` |
| 9 | 16 | 9 | `right_knee_joint` |
| 10 | 17 | 10 | `right_ankle_roll_joint` |
| 11 | 18 | 11 | `right_ankle_pitch_joint` |
| 12 | 19 | 12 | `waist_yaw_joint` |
| 13 | 20 | 13 | `waist_roll_joint` |
| 14 | 21 | 14 | `waist_pitch_joint` |
| 15 | 22 | 15 | `head_pitch_joint` |
| 16 | 23 | 16 | `head_yaw_joint` |
| 17 | 24 | 17 | `left_shoulder_pitch_joint` |
| 18 | 25 | 18 | `left_shoulder_roll_joint` |
| 19 | 26 | 19 | `left_shoulder_yaw_joint` |
| 20 | 27 | 20 | `left_elbow_joint` |
| 21 | 28 | 21 | `left_wrist_roll_joint` |
| 22 | 29 | 22 | `left_wrist_pitch_joint` |
| 23 | 30 | 23 | `left_wrist_yaw_joint` |
| 24 | 31 | 24 | `right_shoulder_pitch_joint` |
| 25 | 32 | 25 | `right_shoulder_roll_joint` |
| 26 | 33 | 26 | `right_shoulder_yaw_joint` |
| 27 | 34 | 27 | `right_elbow_joint` |
| 28 | 35 | 28 | `right_wrist_roll_joint` |
| 29 | 36 | 29 | `right_wrist_pitch_joint` |
| 30 | 37 | 30 | `right_wrist_yaw_joint` |

Before sending retargeted motion to real hardware, verify the SDK motor order,
joint signs, command units, and whether the deployed controller expects head,
waist, and wrist joints.
