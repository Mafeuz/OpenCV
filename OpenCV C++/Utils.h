#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void show_img(Mat img, double scale);

void showGrayImg(string path, double scale);

Mat canny_img(Mat img, float cthresh=50.0, int knrlX = 3, int knrlY = 3, int dilation = 0, int erosion = 0);

void display_video(string path, double scale);

Mat imgCropping(Mat img, int startX, int startY, int width, int height);

Mat warpImg(Mat img, Point2f src[4], Point2f dst[4], float w, float h);

void getContours(Mat img, Mat imgToDraw, int dilation=0, float min_area=20, int filter_corners=-1, bool draw_bboxes=false);

#endif