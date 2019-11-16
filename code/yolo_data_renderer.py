#!/usr/bin/python
import random, time, direct.directbase.DirectStart
from math import *
from sys import argv
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
from direct.filter.CommonFilters import CommonFilters
from panda3d.core import *

# parse command line options (yes, this could be done more elegantly with argparse, but this keeps it more lightweight):
if "-h" in argv or "--help" in argv:
    print("\nUsage:\n\"python " + argv[
        0] + " [starting index] [ending index] [max # of mines per image]\" \nSpecify as many of these arguments as desired, but each must follow the preceding ones in this order to be parsed correctly.\n \n[-h] or [--help] displays this message.\n[-v] outputs a render analysis every 10 frames.\n[-b] or [--blur] toggles the blur shader on the mines.\nThe render task may be aborted at any time with <Ctrl-C>\n\nYou may want to set your starting index a bit below where you actually want the series to begin, as the first few images often must be thrown out.\n")
    go = False
else:
    go = True
if "-v" in argv:
    verbose = True
else:
    verbose = False
if "-b" in argv or "--blur" in argv:
    blurActive = True

else:
    blurActive = False
try:
    start = int(argv[1])
except:
    start = -1
try:
    end = int(argv[2])
except:
    end = 10000
try:
    maxMines = int(argv[3])
    minMines = 0
except:
    maxMines = 3
    minMines = 0


def coordToImagespace(
        coord):  # converts from Panda's 3D coordinate system to a relative coordinate space (upper left is 0,0; bottom right is 1,1)
    x = (coord[0] + 1) / 2
    y = (((-1) * coord[2]) + 1) / 2
    return LPoint2f(x, y)


camLens = base.cam.node().getLens()
camLens.setFocalLength(1833)
camLens.setFilmSize(2048,
                    1536)  # set the scale of the renderspace (this is not image size, just arbitrary Panda units that will be used later to set image size)
# create variables for general parameters that may be useful:
M = camLens.getProjectionMat()
f = camLens.getFocalLength()
r = camLens.getAspectRatio()
w = int(camLens.getFilmSize().getX())
h = int(camLens.getFilmSize().getY())
if blurActive:
    filters2D = CommonFilters(base.win, base.cam2d)
    filters3D = CommonFilters(base.win, base.cam)
    base.cam.node().getDisplayRegion(0).setClearColor((0, 0, 0, 0))
mines = []  # create a static array of mine models:
for i in range(maxMines): mines.append(loader.loadModel("../panda3d_models/Tube.egg")); mines[i].reparentTo(render); mines[i].hide()
lights = []
for i in range(maxMines): lights.append(Spotlight("slight"))  # create corresponding lighting nodes for each mine

props = WindowProperties()
props.setSize(w, h)  # set the window to be the same size (in pixels) as the renderspace is
base.win.requestProperties(props)  # assign the above properties to the current window


def rerender(task):
    base.cam.node().getDisplayRegion(0).setClearDepthActive(True)
    # base.cam2d.node().getDisplayRegion(0).setClearDepthActive(True)
    if blurActive:
        filters3D.manager.region.setClearDepthActive(True)
        filters2D.setBlurSharpen(1.0)  # 1.0 has no effect, but filter must to be active to draw the background at all
        filters3D.setBlurSharpen(random.random() * 0.75)  # 0 is maximum blur, 1 is no blur
        filters3D.manager.region.setSort(20)
        filters2D.manager.region.setSort(-20)

    background = OnscreenImage(parent=render2d, image="./test_img.jpg")  # load background image
    base.cam2d.node().getDisplayRegion(0).setSort(-10)
    base.cam.node().getDisplayRegion(0).setSort(
        10)  # lower numbers render first, higher numbers render in front of lower numbers
    # Display regions can also be accessed using base.win.getDisplayRegion(#), which contains all of them as far as I can tell

    spot = []  # create & wipe array of spotlights for new frame
    metadata = []  # wipe metadata for new frame
    for mine in mines: mine.hide()  # make sure no mines remain from previous loads

    # the following calculates the 2D bounding box by creating a dummy projection in 2-space and reading the extrema of that node
    proj_dummy = base.cam.attach_new_node("proj-dummy")  # create a new node to hold the projected model
    line_node = GeomNode("lines")
    line_path = render2d.attach_new_node(line_node)
    proj_mat = camLens.get_projection_mat_inv()  # read the lens' inverse projection matrix

    count = task.frame + start  # this is how I had to do the counter variable since I couldn't find a way to natively keep track of what iteration the
    # Panda task 'rerender' is on, but since it creates a new image every frame, this works well enough to just count frames
    num_mines = random.randint(minMines, maxMines)  # choose how many mines will appear in this scene

    for i in range(num_mines):
        mines[i].show()  # display the active mines

        mines[i].setPos(random.uniform(-3.5, 3.5), random.uniform(5, 15),
                        random.uniform(-2.5, 2.5))  # set random position
        mines[i].setHpr(random.uniform(-180, 180), random.uniform(-180, 180),
                        random.uniform(-180, 180))  # set random orientation
        mines[i].setColor(1, 1, 1, 0)
        lights[i].setColor((random.uniform(155, 170) / 255, random.uniform(175, 185) / 255,
                            random.uniform(155, 170) / 255, 1))  # set light color and intensity
        spot.append(render.attachNewNode(lights[i]))
        spot[i].setPos(random.uniform(-10, 10), random.uniform(-5, mines[i].getPos()[1]),
                       random.uniform(-10, 10))  # set random position
        spot[i].lookAt(mines[i])  # point it at the mine, wherever it is
        mines[i].setLight(spot[i])  # assign the light to the mine

        proj_dummy.set_transform(TransformState.makeMat(proj_mat))  # set it as the matrix for the projected dummy
        min, max = mines[i].get_tight_bounds(proj_dummy)  # get the bounding coordinates of the projection in 2-space
        box_LL, box_UR = LPoint3f(min[0], 0, min[1]), LPoint3f(max[0], 0, max[
            1])  # coordinates of the corners in a format usable by Panda (where y is depth; z is vertical)

        box_w = (max[0] - min[0])
        box_h = (max[1] - min[1])
        center = LPoint3f(min[0] + box_w / 2, 0, min[1] + (box_h / 2))

        # this next section draws the graphical bounding box as a sanity check. With the background enabled, they are not visible, so this block is pretty much unneeded.
        segs = LineSegs()
        segs.move_to(box_LL)
        segs.draw_to(min[0], 0, max[1])
        segs.draw_to(box_UR)
        segs.draw_to(max[0], 0, min[1])
        segs.draw_to(box_LL)
        segs.create(line_node)

        metadata.append(str(0) + " " + str(coordToImagespace(center).getX()) + " " + str(
            coordToImagespace(center).getY()) + " " + str(box_w / 2) + " " + str(box_h / 2) + "\n")

    cur_img_file = "./test_yolo_renderer/scene_{}.jpg".format(count - 1) # set the filename (don't know why images and labels have to be 1 offset, but they do)
    cur_label_file = "./test_yolo_renderer/scene_{}.txt".format(count) # create the label file
    image = PNMImage()  # create a PNMImage wrapper, an image manipulation class native to Panda
    base.win.getDisplayRegion(0).getScreenshot(image)  # grab a PNM screenshot of the display region
    imageFile = cur_img_file
    image.write(Filename(imageFile))  # write the screenshot to the above file
    labelFile = open(cur_label_file, "w+")
    labelFile.writelines(metadata)  # write the label data for separate mines to separate lines
    print(str(num_mines) + " mines in " + imageFile + " / " + str(len(metadata)) + " lines in {}".format(cur_label_file))
    labelFile.close()
    line_node.remove_all_geoms()  # wipes the bounding boxes

    if verbose and count / 10.0 == count // 10:
        print("\n3D scene analysis:")
        render.analyze()
        print("2D scene analysis:")
        render2d.analyze()

    if count < end:
        return task.cont
    else:
        print("Series complete.")
        return task.done


if go:
    base.taskMgr.add(rerender, "render")
    base.run()
