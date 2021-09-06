#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Utils.h" 

using namespace cv;
using namespace std;

// Filtering Colors with Trackbars (Using HSV Color Space)

	Mat orange_car = imread("ztest_media/orange_car.jpg");

int main(){
    
	resize(orange_car, orange_car, Size(200, 200));
	cvtColor(orange_car, orange_car, COLOR_BGR2HSV);

	Mat mask;

	// Initial Values for Hue~Saturation~Value:
	int hmin = 50, smin = 50, vmin = 50;
	int hmax = 170, smax = 240, vmax = 250;

	// Define Window and Create Trackbars:    
	namedWindow("Trackbars 1", (50, 50));
	namedWindow("Trackbars 2", (50, 50));
	createTrackbar("Hue Min", "Trackbars 1", &hmin, 179);
	createTrackbar("Hue Max", "Trackbars 1", &hmax, 179);

	createTrackbar("Sat Min", "Trackbars 1", &smin, 255);
	createTrackbar("Sat Max", "Trackbars 1", &smax, 255);

	createTrackbar("Val Min", "Trackbars 2", &vmin, 255);
	createTrackbar("Val Max", "Trackbars 2", &vmax, 255);

	while (true) {

		int key = waitKey(2);

		if (key == 'k') {
			cout << "K was pressed to close." << endl;
			break;
		}

		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);

		inRange(orange_car, lower, upper, mask);

		imshow("Filtered", mask);

	}
	
	destroyAllWindows();
    
	return 0;
    
}