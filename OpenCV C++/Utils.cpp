#include "Utils.h"

///////// OpenCV Utils //////////

void show_img(Mat img, double scale) {

	Mat imgToShow = img;
	resize(img, imgToShow, Size(), scale, scale, INTER_LINEAR);
	imshow("Image", imgToShow);
	waitKey(0);
}

void showGrayImg(string path, double scale) {

	Mat img = imread(path);
	Mat imgGray;

	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	resize(imgGray, imgGray, Size(), scale, scale, INTER_LINEAR);
	imshow("Image", imgGray);
	waitKey(0);
}

Mat canny_img(Mat img, int knrlX, int knrlY, int dilation, int erosion) {

	Mat canny_img;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(knrlX, knrlY));

	GaussianBlur(img, img, Size(3, 3), 5, 0);
	Canny(img, canny_img, 50, 50);

	if (dilation > 0) {
		dilate(canny_img, canny_img, kernel, Point(-1, -1), dilation);
	}

	if (erosion > 0) {
		erode(canny_img, canny_img, kernel, Point(-1, -1), erosion);
	}

	return canny_img;

}

void display_video(string path, double scale) {

	VideoCapture cap(path);
	Mat frame;

	while (true) {

		bool success = cap.read(frame);

		if (success == false) {
			cout << "No video ON" << endl;
			break;
		}

		int key = waitKey(20);

		if (key == 'k') {
			cout << "K was pressed, closing video." << endl;
			break;
		}

		resize(frame, frame, Size(), scale, scale, INTER_LINEAR);
		imshow("Video", frame);

	}

	cap.release();
	destroyAllWindows();

}

Mat imgCropping(Mat img, int startX, int startY, int width, int height) {

	Mat imgCrop;
	Rect roi(startX, startY, width, height);
	imgCrop = img(roi);

	return imgCrop;
}
