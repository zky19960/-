#pragma once

#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <iostream>
#include <cmath>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\dnn\dnn.hpp>

/********************************
下面从yolo代码移植过来的头文件内容，实现放在了cpp中
********************************/

/********************************
下面从yolo代码移植过来的头文件内容，实现放在了cpp中
********************************/

typedef struct _Output
{
	int id;//结果类别id
	float confidence;//结果置信度
	cv::Rect box;//矩形框
}Output;

class Yolo
{
	public:
		Yolo() {}
		~Yolo() {}

		bool readModel(cv::dnn::Net &net, std::string &netPath, bool isCuda);
		bool Detect(cv::Mat &SrcImg, cv::dnn::Net &net, std::vector<Output> &output);
		void drawPred(cv::Mat &img, std::vector<Output> result, std::vector<cv::Scalar> color, std::vector<double> &stds);

	private:
		const float netAnchors[3][6] = { { 10.0, 13.0, 16.0, 30.0, 33.0, 23.0 },{ 30.0, 61.0, 62.0, 45.0, 59.0, 119.0 },{ 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 } };
		const float netStride[3] = { 8, 16.0,32 };
		const int netWidth = 640;
		const int netHeight = 640;
		float nmsThreshold = 0.45;
		float boxThreshold = 0.25;
		float classThreshold = 0.25;

	public:
		std::vector<std::string> className = { "White", "Black", "Raised", "Cavve", "Nick" };
};

/***************************************封装类开始**************************************/

/**********************************
@樊师兄:
1.实例化ImageDetect后通过GetImage函数将检测图片导入，如果导入为空则返回false，成功为true；
2.findModel函数用于查找yolo网络当前的数据权重路径,"NoAny"是没有加载任何路径；
3.实例化后，调用方法为GetImage->YoloInit->YoloDetect,需要定义一个yoloresult结构体传入;
@小川师兄:
1.调用方法为GetImage->Confirm，返回值为结果
2.调用方法为GetImage->roundness，返回值为结果
**********************************/

typedef struct _yoloresult_
{
	friend class ImageDetect;

	public:
		std::vector<Output> result;
		std::vector<double> stds;
		std::vector<std::string> className;
	
	private:
		bool flag = false;

}yoloresult;

class ImageDetect
{
	public:
		bool GetImage(cv::Mat Image);
		bool GetImage(std::string ImagePath);
		std::string findModel();
		bool YoloInit(std::string modelpath);
		bool YoloDetect(yoloresult &yoloresult);
		static bool printyoloresult(const yoloresult yoloresult);

		cv::Mat residual();  //main3算法,半成品,求半导电残留
		cv::Mat Confirm();   //main2算法，确认铅笔头是否完成
		double roundness(); //main算法，不圆度
		
		ImageDetect() {}
		~ImageDetect() {}

	private:
		Yolo *yolo = new Yolo;
		
		cv::Mat Image;
		cv::dnn::Net net;
		std::string model = "NoAny";//当前的网络权重保存路径

		std::vector<cv::Scalar> color;//图框颜色渲染所用

		bool HaveImage = false;//用于判断是否已经导入图像，false没有导入，true导入成功
};

#endif