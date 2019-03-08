ffmpeg -r 30 -f image2 -i tmp/%*.jpg -vcodec libx264 -crf 23  -pix_fmt yuv420p test.mp4
