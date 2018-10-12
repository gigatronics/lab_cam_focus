
#!/bin/bash

# extract first 5 frames
#ffmpeg -i /Users/geewiz/Desktop/yi_cam_test/cam0/YI052601.MP4 -ss 00:00:03.000 -vframes 5 -vf fps=30 /Users/geewiz/Desktop/yi_cam_test/thumb0_%03d.png
#ffmpeg -i /Users/geewiz/Desktop/yi_cam_test/cam1/YI042301.MP4 -ss 00:00:03.000 -vframes 5 -vf fps=30 /Users/geewiz/Desktop/yi_cam_test/thumb1_%03d.png
#ffmpeg -i /Users/geewiz/Desktop/yi_cam_test/cam2/YI028701.MP4 -ss 00:00:03.000 -vframes 5 -vf fps=30 /Users/geewiz/Desktop/yi_cam_test/thumb2_%03d.png

# extract frames
/Users/geewiz/Desktop/180920_focus_test_seq/
/Users/geewiz/Desktop/180928_periph/centre2periph/
ffmpeg -i 60cm_b.avi -ss 4 -t 4 -vf fps=12 ./png/thumb%02d.png


# NOT WORKING - make a video.. 
/Users/geewiz/Desktop/180928_periph/195to0pass2/
ffmpeg -f image2 -i *.png -s 3088x2076 -r 2 -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4


# YI - test if the frames in sync... put only the frames out
d:/yi/messi/YI047501.MP4
d:/yi/neymar/YI034801.MP4
d:/yi/ronaldo/YI058001.MP4
ffmpeg -i <filename> -ss 20 -vframes 1 ./png/thumb%%02d.png


# YI - extract all 40 frames
create a dir
/Users/geewiz/Desktop/180928_yi_sync/YI034901
ffmpeg -i <filename> -ss 37 -t 8.03 -vf fps=0.1 <dir>/thumb%%02d.png
