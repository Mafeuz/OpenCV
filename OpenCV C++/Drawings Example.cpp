#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Utils.h" 

using namespace cv;
using namespace std;

// Basic Drawing Functions:

int main() {
    
	Mat blank_img(512, 512, CV_8UC3, Scalar(100, 0, 0));

	int radius = 100;
	circle(blank_img, Point(256, 256), radius, Scalar(0, 69, 200), FILLED);
	circle(blank_img, Point(256, 256), radius, Scalar(0, 0, 100), 10);

	rectangle(blank_img, Point(130, 226), Point(382, 286), Scalar(0, 0, 0), FILLED);

	line(blank_img, Point(130, 226), Point(382, 286), Scalar(255, 255, 255), 3);

	putText(blank_img, "Text", Point(130, 226), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar(0, 0, 255), 2);

	imshow("Img", blank_img);
    
	waitKey(0);
	destroyAllWindows();
    
	return 0;
    
}