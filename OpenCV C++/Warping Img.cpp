#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Utils.h" 

using namespace cv;
using namespace std;

int main() {
    
	Mat imgWarpTest = imread("ztest_media/imgForWarp.png");
	Mat imgWarped;

	// Points manually checked:
	//P1 215, 46
	//P2 345, 108
	//P3 87, 109
	//P4 218, 173

	// Warping:
	float w = 250;
	float h = 300;

	Point2f src[4] = { {215,45}, {345, 108}, {87, 109}, {218, 173} };
	Point2f dst[4] = { {0.0f,0.0f}, {w,0.0f}, {0.0f,h}, {w,h} };

	imgWarped = warpImg(imgWarpTest, src, dst, w, h);

	imshow("Orginal Img", imgWarpTest);
	imshow("Warped Img", imgWarped);

	waitKey(0);
	destroyAllWindows();
    
	return 0;
    
}