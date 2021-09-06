#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Utils.h" 

using namespace cv;
using namespace std;

int main() {

	string path = "ztest_media/cube1.png";
	Mat img = imread(path);
	resize(img, img, Size(300, 300));

	Mat imgToDraw(300, 300, CV_8UC3, Scalar(100, 0, 0));

	getContours(img, imgToDraw, 0, 10, -1, true);

	imshow("Img", img);
	imshow("Draw Contours", imgToDraw);

	waitKey(0);
	destroyAllWindows();
    
	return 0;
}