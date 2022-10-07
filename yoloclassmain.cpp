#include "ImageDetect.h"
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	ImageDetect Test;
	cv::Mat Img = imread("6.jpg");
	/*bool Pencil = Test.Two_Pencilhead_polished(Img);
	Mat roundness_output, residual_output;
	int roundness_score = Test.Two_Roundness(Img, roundness_output);
	int residual_score = Test.Two_Residual(Img, residual_output);*/

	ImageDetect test;
	test.GetImage(Img);

	bool flag = test.YoloInit("best.onnx");
	yoloresult st1;
	test.YoloDetect(st1);
	int a = test.Two_WhiteDot(st1);
	int b = test.Two_Protrude(st1);
	test.printyoloresult(st1);
	//test.printImage();
	

	return 0;
}

