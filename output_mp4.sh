#!/bin/sh

ffmpeg -framerate 2 -i output/img%03d.jpg output.mp4

