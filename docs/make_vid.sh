ffmpeg -r 30 -f image2 -i tmp/%06d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
