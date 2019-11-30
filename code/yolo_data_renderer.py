#!/usr/bin/python

"""
The script generates synthetic data in Darknet YOLO format by rendering the 3d model of the object in the backgrounds.

XXX: It creates few black images.
XXX: The --blur option doesn't work correctly.

Note: Code taken from "https://github.com/apl-ocean-engineering/rendered-nn-training" and
refactored the code in this file for integration.

"""

# TODO: Currently windowsless feature is not working.
windowless = False

if windowless:
    from panda3d.core import loadPrcFileData
    loadPrcFileData("",
    """
       load-display p3tinydisplay # to force CPU only rendering (to make it available as an option if everything else fail, use aux-display p3tinydisplay)
       window-type offscreen # Spawn an offscreen buffer (use window-type none if you don't need any rendering)
       audio-library-name null # Prevent ALSA errors
       show-frame-rate-meter 0
       sync-video 0
    """)
    from direct.showbase.ShowBase import ShowBase
    base = ShowBase()
else:
    import direct.directbase.DirectStart

from direct.gui.OnscreenImage import OnscreenImage
from direct.filter.CommonFilters import CommonFilters
from panda3d.core import *

import argparse
import glob
import os
import random
import time

def get_args():
    parser = argparse.ArgumentParser(description=
    """
    Get yolo dataset from background images and the 3d model rendered into a non-photo realistic synthetic scene.
    """)
    parser.add_argument('--background-dir',
                        required=True,
                        help="The path to the directory containing the background images.")
    parser.add_argument('--object-model-file',
                        required=True,
                        help="The path to the Panda3d object 3d-model file (.egg).")
    parser.add_argument('--num-scenes',
                        type=int,
                        required=True,
                        help="The number of scenes to generate.")
    parser.add_argument('--out-dir',
                        required=True,
                        help="The path to the output dir to which the yolo dataset need to be written.")
    parser.add_argument('--max-objects',
                        type=int,
                        default=2,
                        help="The maximum number of objects to be rendered on a scene.")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="Enable verbose mode where it outputs a render analysis every 10 frames.")
    parser.add_argument('--blur',
                        action="store_true",
                        help="Toggles the blur shader on the mines.")

    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        raise ValueError("The out-dir provided already exists.")

    return args


###
# XXX: The script uses sleep to wait for renders. Also, it uses 2 dummy images to correct the sequencing.
SLEEP_TIME = 0.01
NUM_DUMMIES = 2
DUMMY_SUFFIX = "dummy"

# Global Variables:
args = get_args()

object_model_file = args.object_model_file
background_dir = args.background_dir
out_dir = args.out_dir

num_scenes = args.num_scenes

minMines = 0
maxMines = args.max_objects

verbose = args.verbose
blur_active = args.blur

# Collect the background image files.
image_file_types = ["*.png", "*.jpg"]
background_files = []
for file_type in image_file_types:
    background_files.extend(glob.glob(os.path.join(background_dir, file_type)))

num_background_files = len(background_files)
background_files = sorted(background_files)

print("Found %s images in the given background-dir: %s" % (num_background_files, background_dir))
if num_background_files < 1:
    raise ValueError("There are no background images.")

# Start the direct show base.
#base.graphicsEngine.renderFrame()

camLens = base.cam.node().getLens()
camLens.setFocalLength(1833)
# Set the scale of the renderspace (this is not image size, just arbitrary Panda units that will be used later
# to set image size) create variables for general parameters that may be useful.
# TODO: How to set lower numbers here to speedup the process.
camLens.setFilmSize(2048, 1536)  # (2048, 1536)
M = camLens.getProjectionMat()
f = camLens.getFocalLength()
r = camLens.getAspectRatio()
w = int(camLens.getFilmSize().getX())
h = int(camLens.getFilmSize().getY())
if blur_active:
    filters2D = CommonFilters(base.win, base.cam2d)
    filters3D = CommonFilters(base.win, base.cam)
    base.cam.node().getDisplayRegion(0).setClearColor((0, 0, 0, 0))

mines = []  # create a static array of mine models:
for i in range(maxMines):
    mines.append(loader.loadModel(object_model_file))
    mines[i].reparentTo(render)
    mines[i].hide()

lights = []
for i in range(maxMines):
    lights.append(Spotlight("slight"))  # create corresponding lighting nodes for each mine

props = WindowProperties()
props.setSize(w, h)  # set the window to be the same size (in pixels) as the renderspace is
base.win.requestProperties(props)  # assign the above properties to the current window


# The core renderer function.
def coordToImagespace(coord):
    # converts from Panda's 3D coordinate system to a relative coordinate space (upper left is 0,0; bottom right is 1,1)
    x = (coord[0] + 1) / 2
    y = (((-1) * coord[2]) + 1) / 2
    return LPoint2f(x, y)


def rerender(task):
    base.cam.node().getDisplayRegion(0).setClearDepthActive(True)
    time.sleep(SLEEP_TIME)
    # base.cam2d.node().getDisplayRegion(0).setClearDepthActive(True)
    if blur_active:
        filters3D.manager.region.setClearDepthActive(True)
        filters2D.setBlurSharpen(1.0)  # 1.0 has no effect, but filter must to be active to draw the background at all
        filters3D.setBlurSharpen(random.random() * 0.75)  # 0 is maximum blur, 1 is no blur
        filters3D.manager.region.setSort(20)
        filters2D.manager.region.setSort(-20)

    # Read arguments from frame name.
    # TODO: Instead read params from the lambda function.
    toks = task.name.split("-")
    count_id = toks[-1]
    is_dummy = False
    if count_id == DUMMY_SUFFIX:
        is_dummy = True
    else:
        count_id = int(count_id)

    selected_background_image = '-'.join(toks[:-1])

    if not os.path.exists(selected_background_image):
        raise ValueError("Background image not found: %s" % selected_background_image)
    else:
        print("Loading background image: %s" % selected_background_image)

    background = OnscreenImage(parent=render2d, image=selected_background_image)  # load background image
    base.cam2d.node().getDisplayRegion(0).setSort(-10)
    # lower numbers render first, higher numbers render in front of lower number,
    # Display regions can also be accessed using base.win.getDisplayRegion(#),
    # which contains all of them as far as I can tell.
    base.cam.node().getDisplayRegion(0).setSort(10)

    spot = []  # create & wipe array of spotlights for new frame
    metadata = []  # wipe metadata for new frame
    for mine in mines:
        mine.hide()  # make sure no mines remain from previous loads

    # The following calculates the 2D bounding box by creating a dummy projection in 2-space and reading the
    # extrema of that node.
    proj_dummy = base.cam.attach_new_node("proj-dummy")  # create a new node to hold the projected model
    line_node = GeomNode("lines")
    line_path = render2d.attach_new_node(line_node)
    proj_mat = camLens.get_projection_mat_inv()  # read the lens' inverse projection matrix

    num_mines = random.randint(minMines, maxMines)  # choose how many mines will appear in this scene
    for i in range(num_mines):
        # display the active mines
        mines[i].show()

        # set random position
        mines[i].setPos(random.uniform(-3.5, 3.5), random.uniform(5, 15), random.uniform(-2.5, 2.5))

        # set random orientation
        mines[i].setHpr(random.uniform(-180, 180), random.uniform(-180, 180), random.uniform(-180, 180))
        mines[i].setColor(1, 1, 1, 0)

        # set light color and intensity
        lights[i].setColor((random.uniform(155, 170) / 255,
                            random.uniform(175, 185) / 255,
                            random.uniform(155, 170) / 255, 1))
        spot.append(render.attachNewNode(lights[i]))

        # Set random position for the light.
        spot[i].setPos(random.uniform(-10, 10), random.uniform(-5, mines[i].getPos()[1]), random.uniform(-10, 10))
        spot[i].lookAt(mines[i])  # point it at the mine, wherever it is
        mines[i].setLight(spot[i])  # assign the light to the mine

        proj_dummy.set_transform(TransformState.makeMat(proj_mat))  # set it as the matrix for the projected dummy
        min, max = mines[i].get_tight_bounds(proj_dummy)  # get the bounding coordinates of the projection in 2-space
        # coordinates of the corners in a format usable by Panda (where y is depth; z is vertical)
        box_LL, box_UR = LPoint3f(min[0], 0, min[1]), LPoint3f(max[0], 0, max[1])

        box_w = (max[0] - min[0])
        box_h = (max[1] - min[1])
        center = LPoint3f(min[0] + box_w / 2, 0, min[1] + (box_h / 2))

        # this next section draws the graphical bounding box as a sanity check. With the background enabled,
        # they are not visible, so this block is pretty much unneeded.
        segs = LineSegs()
        segs.move_to(box_LL)
        segs.draw_to(min[0], 0, max[1])
        segs.draw_to(box_UR)
        segs.draw_to(max[0], 0, min[1])
        segs.draw_to(box_LL)
        segs.create(line_node)

        metadata.append(str(0)
                        + " "
                        + str(coordToImagespace(center).getX())\
                        + " "
                        + str(coordToImagespace(center).getY())
                        + " "
                        + str(box_w / 2)
                        + " "
                        + str(box_h / 2)
                        + "\n")

    # Save the images and labels.
    # XXX: Set the filenames (don't know why images and labels have to be 1 offset, but they do).
    background_img_name = os.path.splitext(os.path.basename(selected_background_image))[0]
    cur_img_file = os.path.join(out_dir, "%s-%s.jpg" % (background_img_name, count_id))
    cur_label_file = os.path.join(out_dir, "%s-%s.txt" % (background_img_name, count_id))

    base.graphicsEngine.renderFrame()
    time.sleep(SLEEP_TIME)
    if not is_dummy:
        base.win.saveScreenshot(cur_img_file)
        time.sleep(SLEEP_TIME)

    if not is_dummy:
        with open(cur_label_file, "w") as labelFile:
            labelFile.writelines(metadata)  # write the label data for separate mines to separate lines
            print(str(num_mines) + " mines in " + cur_img_file + " / " + str(len(metadata)) + " lines in %s" % cur_label_file)

    # Wipes the bounding boxes
    line_node.remove_all_geoms()

    if verbose and type(count_id) == int and count_id / 10.0 == count_id // 10:
        print("\n3D scene analysis:")
        render.analyze()
        print("2D scene analysis:")
        render2d.analyze()

    print("Created scene for %s-%s" % (selected_background_image, count_id))

    return task.done


if __name__ == "__main__":
    start_time = time.time()

    dummy_call_image = background_files[-1]
    for di in range(NUM_DUMMIES):
        cur_dummy_file_prefix = "%s-%s" % (dummy_call_image, DUMMY_SUFFIX)
        base.taskMgr.add(rerender, cur_dummy_file_prefix, priority=di+1)

    for count_id in range(num_scenes):
        selected_background_image = background_files[count_id % num_background_files]
        base.taskMgr.add(rerender, "%s-%s" % (selected_background_image, count_id+1), priority=count_id+3)

    try:
        os.makedirs(out_dir)
        base.run()
    finally:
        base.finalizeExit()
        base.destroy()

    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken to generate synthetic images: %s seconds" % time_taken)
