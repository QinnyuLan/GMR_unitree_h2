import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_ASSETS = {
    "unitree_h2": REPO_ROOT / "assets" / "unitree_h2",
}


def _parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/h2_mesh_videos"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--preview-frames", type=int, default=0)
    parser.add_argument("data_files", nargs="+")
    return parser.parse_args(argv)


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _material(name, color, roughness=0.55, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    return mat


def _setup_render(width, height, fps):
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        for attr, value in (
            ("taa_render_samples", 16),
            ("use_gtao", True),
            ("gtao_distance", 3.0),
            ("gtao_factor", 1.0),
        ):
            if hasattr(scene.eevee, attr):
                setattr(scene.eevee, attr, value)

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.fps = fps
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.world = bpy.data.worlds.new("GMR dark sky")
    scene.world.color = (0.12, 0.16, 0.22)


def _create_ground():
    bpy.ops.mesh.primitive_plane_add(size=18.0, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "checker ground"

    mat = bpy.data.materials.new("dark checker ground")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    checker = nodes.new(type="ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 9.0
    checker.inputs["Color1"].default_value = (0.15, 0.17, 0.23, 1.0)
    checker.inputs["Color2"].default_value = (0.20, 0.22, 0.30, 1.0)
    if bsdf:
        mat.node_tree.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.85
    ground.data.materials.append(mat)
    return ground


def _look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _setup_camera_and_lights():
    bpy.ops.object.light_add(type="AREA", location=(0, -3.5, 5.5))
    key = bpy.context.object
    key.name = "large softbox"
    key.data.energy = 650.0
    key.data.size = 5.0

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 6))
    sun = bpy.context.object
    sun.name = "soft sun"
    sun.data.energy = 1.6
    sun.rotation_euler = (math.radians(40), 0, math.radians(35))

    bpy.ops.object.camera_add(location=(3.5, -4.7, 2.15))
    camera = bpy.context.object
    camera.name = "tracking camera"
    camera.data.lens = 35
    camera.data.clip_end = 100
    bpy.context.scene.camera = camera
    return camera


def _quat_xyzw_to_matrix(pos, quat_xyzw):
    q = Quaternion((float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])))
    return Matrix.LocRotScale(Vector((float(pos[0]), float(pos[1]), float(pos[2]))), q, None)


def _mjcf_local_matrix(elem):
    pos = [float(v) for v in elem.get("pos", "0 0 0").split()]
    quat_wxyz = [float(v) for v in elem.get("quat", "1 0 0 0").split()]
    q = Quaternion((quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
    return Matrix.LocRotScale(Vector(pos), q, None)


def _parse_h2_visual_geoms(robot_type):
    asset_dir = ROBOT_ASSETS.get(robot_type)
    if asset_dir is None:
        allowed = ", ".join(sorted(ROBOT_ASSETS))
        raise ValueError(f"Unsupported robot_type [{robot_type}]. Allowed values: {allowed}")

    root = ET.parse(asset_dir / "h2.xml").getroot()
    mesh_files = {
        mesh.attrib["name"]: mesh.attrib["file"]
        for mesh in root.find("asset").findall("mesh")
    }
    geoms = []

    def walk(body):
        body_name = body.attrib.get("name")
        for geom in body.findall("geom"):
            mesh_name = geom.attrib.get("mesh")
            if geom.attrib.get("type") == "mesh" and mesh_name and geom.attrib.get("group") == "1":
                rgba = [float(v) for v in geom.attrib.get("rgba", "0.7 0.7 0.7 1").split()]
                geoms.append(
                    {
                        "body": body_name,
                        "mesh": mesh_name,
                        "file": mesh_files[mesh_name],
                        "rgba": rgba,
                        "local": _mjcf_local_matrix(geom),
                    }
                )
        for child in body.findall("body"):
            walk(child)

    for body in root.find("worldbody").findall("body"):
        walk(body)
    return geoms, asset_dir / "meshes"


def _import_stl(path):
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.stl(filepath=str(path))
    after = [obj for obj in bpy.data.objects if obj not in before]
    obj = after[0] if after else bpy.context.object
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def _create_h2_objects(body_names, robot_type):
    body_index = {name: i for i, name in enumerate(body_names)}
    mats = {}
    objects = []
    geoms, mesh_dir = _parse_h2_visual_geoms(robot_type)
    for idx, geom in enumerate(geoms):
        if geom["body"] not in body_index:
            continue
        obj = _import_stl(mesh_dir / geom["file"])
        obj.name = f"h2_{idx:02d}_{geom['body']}_{geom['mesh']}"
        color_key = tuple(round(v, 3) for v in geom["rgba"])
        if color_key not in mats:
            base = geom["rgba"]
            tint = (
                min(base[0] + 0.08, 1.0),
                min(base[1] + 0.10, 1.0),
                min(base[2] + 0.12, 1.0),
                base[3],
            )
            mats[color_key] = _material(f"h2 material {color_key}", tint, roughness=0.38)
        obj.data.materials.append(mats[color_key])
        objects.append((obj, body_index[geom["body"]], geom["local"]))
    return objects


def _create_smplx_mesh(data):
    vertices = data["smplx_vertices"][0]
    faces = data["smplx_faces"]
    mesh = bpy.data.meshes.new("smplx_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    obj = bpy.data.objects.new("smplx_mesh", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_material("SMPL-X yellow mesh", (0.88, 1.0, 0.27, 1.0), roughness=0.48))
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def _update_mesh_vertices(obj, vertices):
    obj.data.vertices.foreach_set("co", vertices.reshape(-1))
    obj.data.update()


def _frame_bounds(h2_body_q, smplx_vertices, frame):
    h2_points = h2_body_q[frame, :, 0:3]
    smplx_points = smplx_vertices[frame]
    all_points = np.concatenate([h2_points, smplx_points], axis=0)
    return all_points.min(axis=0), all_points.max(axis=0)


def _render_data_file(data_path, out_dir, width, height, fps, preview_frames=0):
    _clear_scene()
    _setup_render(width, height, fps)
    _create_ground()
    camera = _setup_camera_and_lights()

    data = np.load(data_path)
    h2_body_q = data["h2_body_q"]
    body_names = [str(name) for name in data["body_names"]]
    smplx_vertices = data["smplx_vertices"]
    meta = json.loads(str(data["metadata"]))
    robot_type = meta.get("robot_type", "unitree_h2")

    h2_objects = _create_h2_objects(body_names, robot_type)
    smplx_object = _create_smplx_mesh(data)
    frame_count = h2_body_q.shape[0]
    if preview_frames > 0:
        frame_count = min(frame_count, preview_frames)

    stem = Path(meta["motion"]).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{stem}_frames_", dir=str(out_dir)))
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = frame_count - 1

    for frame in range(frame_count):
        scene.frame_set(frame)
        for obj, body_idx, local_matrix in h2_objects:
            body = h2_body_q[frame, body_idx]
            obj.matrix_world = _quat_xyzw_to_matrix(body[0:3], body[3:7]) @ local_matrix
        _update_mesh_vertices(smplx_object, smplx_vertices[frame])

        mn, mx = _frame_bounds(h2_body_q, smplx_vertices, frame)
        center = (mn + mx) * 0.5
        target = Vector((float(center[0]), float(center[1]), max(0.95, float(center[2] * 0.58))))
        camera.location = Vector((float(center[0] + 3.5), float(center[1] - 4.7), float(target.z + 1.05)))
        _look_at(camera, target)

        scene.render.filepath = str(tmp_dir / f"frame_{frame:05d}.png")
        bpy.ops.render.render(write_still=True)
        if frame == 0 or frame == frame_count - 1 or (frame + 1) % 10 == 0:
            print(f"[FRAME] {stem}: {frame + 1}/{frame_count}")

    out_path = out_dir / f"{stem}__{robot_type}_smplx_mesh.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(tmp_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-preset",
        "medium",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[OK] wrote {out_path}")
    return out_path


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    for data_file in args.data_files:
        _render_data_file(Path(data_file), out_dir, args.width, args.height, args.fps, args.preview_frames)


if __name__ == "__main__":
    main()
