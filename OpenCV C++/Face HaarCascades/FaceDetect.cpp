#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include "Utils.h"

using namespace cv;
using namespace std;

Mat img = imread("will_smith1.jpg");

int main() {

	CascadeClassifier faceHaarCascade;

	faceHaarCascade.load("haarcascades/haarcascade_frontalface_default.xml");

	vector<Rect> faces;

	faceHaarCascade.detectMultiScale(img, faces, 1.1, 10);

	for (int i = 0; i < faces.size(); i++) {

		rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 0), 3);

	}

	imshow("Detection", img);

	waitKey(0);

	destroyAllWindows();

	////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////// Testing with Video (SLOWWWWWW) ////////////////////////////////////

	string path = "ztest_media/people.mp4";

	VideoCapture cap(path);
	Mat frame;

	while (true) {

		bool success = cap.read(frame);

		if (success == false) {
			cout << "No video ON" << endl;
			break;
		}

		int key = waitKey(1);

		if (key == 'k') {
			cout << "K was pressed, closing video." << endl;
			break;
		}

		resize(frame, frame, Size(), 0.8, 0.8, INTER_LINEAR);

		vector<Rect> faces;

		faceHaarCascade.detectMultiScale(frame, faces, 1.1, 10);

		for (int i = 0; i < faces.size(); i++) {

			rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(255, 0, 0), 3);

		}

		imshow("Video", frame);

	}

	cap.release();
	destroyAllWindows();

	return 0;
}



