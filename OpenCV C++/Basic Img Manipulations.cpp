#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Utils.h" 

using namespace cv;
using namespace std;

int main() {
    
	string path1 = "ztest_media/cube1.png";
	string path2 = "ztest_media/cube2.png";
    
	Mat img1 = imread(path1);
	Mat img2 = imread(path2);
    
	// Canny:
	Mat cannyImg;
	cannyImg = canny_img(img1, 5, 5, 1, 0);
	imshow("Canny", cannyImg);

	// Croping:
	Mat imgCrop;
	imgCrop = imgCropping(img2, 30, 30, 100, 110);
	imshow("Crop", imgCrop);

	waitKey(0);
	destroyAllWindows();
    
	return 0;
}