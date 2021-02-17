"""Generate image renderings for 3DIdent dataset."""

import os
import numpy as np
import argparse
import pathlib
import colorsys
import sys
import site


def main(args):
    args.output_folder = pathlib.Path(args.output_folder).absolute()

    latents_path = os.path.join(args.output_folder, "latents.npy")
    if not os.path.exists(latents_path):
        raise ValueError("Latents could not be found; run latent generation first")

    latents = np.load(latents_path)
    n_samples = len(latents)

    if args.material_names is None:
        args.material_names = ["Rubber"] * ((latents.shape[1] - 1) // 8)
    if args.shape_names is None:
        args.shape_names = ["Teapot"] * ((latents.shape[1] - 1) // 8)

    assert len(args.material_names) == (latents.shape[1] - 1) // 8

    indices = np.array_split(np.arange(n_samples), args.n_batches)[args.batch_index]
    print(f"Rendering samples in range: {min(indices)} - {max(indices)}")

    output_image_folder = os.path.join(args.output_folder, "images")

    initialize_renderer(
        args.shape_names,
        args.material_names,
        not args.no_spotlights,
        render_tile_size=256 if args.use_gpu else 64,
        use_gpu=args.use_gpu,
    )

    for i, idx in enumerate(indices):
        output_filename = os.path.join(
            output_image_folder,
            f"{str(idx).zfill(int(np.ceil(np.log10(n_samples))))}.png",
        )
        if os.path.exists(output_filename):
            print("Skipped file", output_filename)
            continue

        current_latents = latents[idx]
        render_sample(
            current_latents,
            args.material_names,
            not args.no_spotlights,
            output_filename,
            args.save_scene,
        )


def initialize_renderer(
    shape_names,
    material_names,
    include_lights=True,
    width=224,
    height=224,
    render_tile_size=64,
    use_gpu=False,
    render_num_samples=512,
    render_min_bounces=8,
    render_max_bounces=8,
    ground_texture=None,
):
    """Initialize renderer and base scene"""

    base_path = pathlib.Path(__file__).parent.absolute()

    # Load the main blendfile
    base_scene = os.path.join(base_path, "data", "scenes", "base_scene_equal_xyz.blend")
    bpy.ops.wm.open_mainfile(filepath=base_scene)

    # Load materials
    material_dir = os.path.join(base_path, "data", "materials")
    render_utils.load_materials(material_dir)

    # Load segmentation node group
    # node_path = 'data/node_groups/NodeGroupMulti4.blend'
    segm_node_path = os.path.join(base_path, "data/node_groups/NodeGroup.blend")
    with bpy.data.libraries.load(segm_node_path) as (data_from, data_to):
        data_to.objects = data_from.objects
        data_to.materials = data_from.materials
        data_to.node_groups = data_from.node_groups
    segm_node_mat = data_to.materials[0]
    segm_node_group_elems = (
        data_to.node_groups[0].nodes["ColorRamp"].color_ramp.elements
    )

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.resolution_x = width
    render_args.resolution_y = height
    render_args.resolution_percentage = 100
    render_args.tile_x = render_tile_size
    render_args.tile_y = render_tile_size
    if use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = "CUDA"
            bpy.context.user_preferences.system.compute_device = "CUDA_0"
        else:
            # Mark all scene devices as GPU for cycles
            bpy.context.scene.cycles.device = "GPU"

            for scene in bpy.data.scenes:
                scene.cycles.device = "GPU"
                scene.render.resolution_percentage = 100
                scene.cycles.samples = render_num_samples

            # Enable CUDA
            bpy.context.preferences.addons[
                "cycles"
            ].preferences.compute_device_type = "CUDA"

            # Enable and list all devices, or optionally disable CPU
            for devices in bpy.context.preferences.addons[
                "cycles"
            ].preferences.get_devices():
                for d in devices:
                    d.use = True
                    if d.type == "CPU":
                        d.use = False

    # Some CYCLES-specific stuff
    bpy.data.worlds["World"].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = render_max_bounces
    if use_gpu == 1:
        bpy.context.scene.cycles.device = "GPU"

    # activate denoising to make spot lights look nicer
    bpy.context.scene.view_layers["RenderLayer"].cycles.use_denoising = True
    bpy.context.view_layer.cycles.use_denoising = True

    # disable reflections
    bpy.context.scene.cycles.max_bounces = 0

    # Now add objects and spotlights
    add_objects_and_lights(shape_names, material_names, include_lights, base_path)

    max_object_height = max(
        [max(o.dimensions) for o in bpy.data.objects if "Object_" in o.name]
    )

    # Assign texture material to ground
    if ground_texture:
        render_utils.add_texture("Ground", ground_texture)
        # TODO: change z location if texture is used
    else:
        objs = bpy.data.objects
        objs.remove(objs["Ground"], do_unlink=True)

        bpy.ops.mesh.primitive_plane_add(size=1500, location=(0, 0, -max_object_height))
        bpy.context.object.name = "Ground"

        bpy.data.objects["Ground"].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects["Ground"]

        # bpy.data.objects["Ground"].data.materials.clear()
        render_utils.add_material("Rubber", Color=(0.5, 0.5, 0.5, 1.0))

    # Segmentation materials and colors
    n_objects = len(material_names)
    segm_node_mat.node_tree.nodes["Group"].inputs[1].default_value = n_objects
    segm_mat = []
    segm_color = []
    for i in range(n_objects + 1):
        segm_node_mat.node_tree.nodes["Group"].inputs[0].default_value = i
        segm_mat.append(segm_node_mat.copy())
        segm_color.append(list(segm_node_group_elems[i].color))


def add_objects_and_lights(shape_names, material_names, add_lights, base_path):
    shapes_path = os.path.join(base_path, "data", "shapes")

    for i, (shape_name, material_name) in enumerate(zip(shape_names, material_names)):
        print("Adding object", i, shape_name, material_name)
        # add object
        object_name = render_utils.add_object(
            shapes_path, f"Shape{shape_name}", f"Object_{i}", 1.5, (0.0, 0.0, 0.0)
        )

        bpy.data.objects[object_name].data.materials.clear()
        render_utils.add_material(
            material_name, bpy.data.objects[object_name], Color=(0.0, 0.0, 0.0, 1.0)
        )

        if add_lights:
            # add spotlight focusing on the object
            # create light datablock, set attributes
            spotlight_data = bpy.data.lights.new(
                name=f"Spotlight_Object_{i}", type="SPOT"
            )
            spotlight_data.energy = 3000  # 10000, 10000 could be too bright
            spotlight_data.shadow_soft_size = 0.5
            spotlight_data.spot_size = 35 / 180 * np.pi
            spotlight_data.spot_blend = 0.1
            spotlight_data.falloff_type = "CONSTANT"

            spotlight_data.contact_shadow_distance = np.sqrt(3) * 3
            # create new object with our light datablock
            spotlight_object = bpy.data.objects.new(
                name=f"Spotlight_Object_{i}", object_data=spotlight_data
            )
            # link light object
            bpy.context.collection.objects.link(spotlight_object)

            spotlight_object.location = (7, 7, 7)

            ttc = spotlight_object.constraints.new(type="TRACK_TO")
            ttc.target = bpy.data.objects[object_name]
            ttc.track_axis = "TRACK_NEGATIVE_Z"
            # we don't care about the up_axis as long as it is different than TRACK_Z
            ttc.up_axis = "UP_X"

            # update scene, if needed
            dg = bpy.context.evaluated_depsgraph_get()
            dg.update()


def update_objects_and_lights(latents, material_names, update_lights):
    """Parse latents and update the object(s) position, rotation and color
    as well as the spotlight's position and color."""

    objects_latents = np.array_split(latents, (len(latents) - 1) // 8)

    max_object_size = max(
        [max(o.dimensions) for o in bpy.data.objects if "Object_" in o.name]
    )

    for i, (object_latents, material_name) in enumerate(
        zip(objects_latents, material_names)
    ):
        # find correct object name
        object_name = None
        for obj in bpy.data.objects:
            if obj.name.endswith(f"Object_{i}"):
                object_name = obj.name
                break
        assert object_name is not None

        # update object location and rotation
        object = bpy.data.objects[object_name]
        object.location = (
            object_latents[0],
            object_latents[1],
            object_latents[2] + max_object_size / 2,
        )
        object.rotation_euler = tuple(object_latents[3:6])

        # update object color
        rgba_object = colorsys.hsv_to_rgb(
            object_latents[7] / (2.0 * np.pi), 1.0, 1.0
        ) + (1.0,)
        render_utils.change_material(
            bpy.data.objects[object_name].data.materials[-1], Color=rgba_object
        )

        if update_lights:
            # update light color
            rgb_light = colorsys.hsv_to_rgb(object_latents[8] / (2.0 * np.pi), 0.8, 1.0)
            bpy.data.objects[f"Spotlight_Object_{i}"].data.color = rgb_light
            # update light location
            bpy.data.objects[f"Spotlight_Object_{i}"].location = (
                4 * np.sin(object_latents[6]),
                4 * np.cos(object_latents[6]),
                6 + max_object_size,
            )


def render_sample(latents, material_names, include_lights, output_filename, save_scene):
    """Update the scene based on the latents and render the scene and save as an image."""

    # set output path
    bpy.context.scene.render.filepath = output_filename

    # set objects and lights
    update_objects_and_lights(latents, material_names, include_lights)

    rgba_background = colorsys.hsv_to_rgb(latents[9] / (2.0 * np.pi), 0.60, 1.0) + (
        1.0,
    )
    render_utils.change_material(
        bpy.data.objects["Ground"].data.materials[-1], Color=rgba_background
    )

    # set scene background
    bpy.ops.render.render(write_still=True)

    if save_scene:
        # just for debugging
        bpy.ops.wm.save_as_mainfile(
            filepath=f"scene_{os.path.basename(output_filename)}.blend"
        )


if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent.absolute()

    INSIDE_BLENDER = True
    try:
        import bpy, bpy_extras
        from mathutils import Vector
    except ImportError as e:
        INSIDE_BLENDER = False
    if INSIDE_BLENDER:
        try:
            import render_utils
        except ImportError as e:
            try:
                print("Could not import render_utils.py; trying to hot patch it.")
                site.addsitedir(base_path)
                import render_utils
            except ImportError as e:
                print("\nERROR")
                sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", required=True, type=str)
    parser.add_argument("--n-batches", required=True, type=int)
    parser.add_argument("--batch-index", required=True, type=int)
    parser.add_argument("--no-spotlights", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--material-names", nargs="+", type=str)
    parser.add_argument("--shape-names", nargs="+", type=str)
    parser.add_argument("--save-scene", action="store_true")

    if INSIDE_BLENDER:
        # Run normally
        argv = render_utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
    else:
        print("This script is intended to be called from blender like this:")
        print()
        print(
            "blender --background --python generate_3dident_dataset_images.py -- [args]"
        )
        print()
        print("You can also run as a standalone python script to view all")
        print("arguments like this:")
        print()
        print("python render_images.py --help")
