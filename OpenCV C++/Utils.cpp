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

Mat canny_img(Mat img, float cthresh, int knrlX, int knrlY, int dilation, int erosion) {

	Mat img_gray;
	Mat canny_img;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(knrlX, knrlY));

	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	GaussianBlur(img_gray, img_gray, Size(3, 3), 5, 0);
	Canny(img_gray, canny_img, cthresh, cthresh);

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

Mat warpImg(Mat img, Point2f src[4], Point2f dst[4], float w, float h) {

	Mat H = getPerspectiveTransform(src, dst);
	Mat img_warped;
	warpPerspective(img, img_warped, H, Point(w, h));

	return img_warped;

}

void getContours(Mat img, Mat imgToDraw, int dilation, float min_area, int filter_corners, bool draw_bboxes) {

	Mat cannyImg;

	cannyImg = canny_img(img, 50, 5, 5, dilation, 0);

	vector<vector<Point>> contours;
	vector<Vec4i> hiearchy;

	findContours(cannyImg, contours, hiearchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++){
	
		int area = contourArea(contours[i]);
		float perimeter = arcLength(contours[i], true);
		approxPolyDP(contours[i], conPoly[i], 0.02 * perimeter, true);
		boundRect[i] = boundingRect(conPoly[i]);

		if (area > min_area) {

			if ((filter_corners != -1) && (conPoly[i].size() == filter_corners)) {
				drawContours(imgToDraw, conPoly, i, Scalar(255, 0, 255), 2);

				if (draw_bboxes == true) {
					rectangle(imgToDraw, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2);
				}

			}

			if (filter_corners == -1) {
				drawContours(imgToDraw, contours, i, Scalar(255, 0, 255), 2);

				if (draw_bboxes == true) {
					rectangle(imgToDraw, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2);
				}

			}

		}

	}

}