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
�����yolo������ֲ������ͷ�ļ����ݣ�ʵ�ַ�����cpp��
********************************/

/********************************
�����yolo������ֲ������ͷ�ļ����ݣ�ʵ�ַ�����cpp��
********************************/

typedef struct _Output
{
	int id;//������id
	float confidence;//������Ŷ�
	cv::Rect box;//���ο�
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

/***************************************��װ�࿪ʼ**************************************/

/**********************************
@��ʦ��:
1.ʵ����ImageDetect��ͨ��GetImage���������ͼƬ���룬�������Ϊ���򷵻�false���ɹ�Ϊtrue��
2.findModel�������ڲ���yolo���統ǰ������Ȩ��·��,"NoAny"��û�м����κ�·����
3.ʵ�����󣬵��÷���ΪGetImage->YoloInit->YoloDetect,��Ҫ����һ��yoloresult�ṹ�崫��;
@С��ʦ��:
1.���÷���ΪGetImage->Confirm������ֵΪ���
2.���÷���ΪGetImage->roundness������ֵΪ���
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

		cv::Mat residual();  //main3�㷨,���Ʒ,��뵼�����
		cv::Mat Confirm();   //main2�㷨��ȷ��Ǧ��ͷ�Ƿ����
		double roundness(); //main�㷨����Բ��
		
		ImageDetect() {}
		~ImageDetect() {}

	private:
		Yolo *yolo = new Yolo;
		
		cv::Mat Image;
		cv::dnn::Net net;
		std::string model = "NoAny";//��ǰ������Ȩ�ر���·��

		std::vector<cv::Scalar> color;//ͼ����ɫ��Ⱦ����

		bool HaveImage = false;//�����ж��Ƿ��Ѿ�����ͼ��falseû�е��룬true����ɹ�
};

#endif