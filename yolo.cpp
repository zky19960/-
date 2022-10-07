//#include"stdafx.h"
#include"yolo.h";

using namespace std;
using namespace cv;
using namespace dnn;

bool Yolo::readModel(Net &net, string &netPath, bool isCuda = false) 
{
	try 
	{
		net = readNet(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

	}
	//cpu
	else 
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}

bool Yolo::Detect(Mat &SrcImg, Net &net, vector<Output> &output) 
{
	Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg = SrcImg.clone();
	if (maxLen > 1.2*col || maxLen > 1.2*row) {
		Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
		SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
		netInputImg = resizeImg;
	}
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(104, 117, 123), true, false);
	//�������������û�����������µ��ǽ��ƫ��ܴ󣬿��Գ������������������
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0,0), true, false);
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(114, 114,114), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	//vector<string> outputLayerName{"345","403", "461","output" };
	//net.forward(netOutputImg, outputLayerName[3]); //��ȡoutput�����
	net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	std::vector<int> classIds;//���id����
	std::vector<float> confidences;//���ÿ��id��Ӧ���Ŷ�����
	std::vector<cv::Rect> boxes;//ÿ��id���ο�
	float ratio_h = (float)netInputImg.rows / netHeight;
	float ratio_w = (float)netInputImg.cols / netWidth;
	int net_width = className.size() + 5;  //������������������+5
	float* pdata = (float*)netOutputImg[0].data;
	for (int stride = 0; stride < 3; stride++) {    //stride
		int grid_x = (int)(netWidth / netStride[stride]);
		int grid_y = (int)(netHeight / netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++) { //anchors
			const float anchor_w = netAnchors[stride][anchor * 2];
			const float anchor_h = netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++) {
				for (int j = 0; j < grid_x; j++) {
					float box_score = pdata[4]; //Sigmoid(pdata[4]);//��ȡÿһ�е�box���к���ĳ������ĸ���
					if (box_score > boxThreshold) {
						cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
						Point classIdPoint;
						double max_class_socre;
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = (float)max_class_socre; //Sigmoid((float)max_class_socre);
						if (max_class_socre > classThreshold) {
							//rect [x,y,w,h]
							float x = pdata[0];// (Sigmoid(pdata[0]) * 2.f - 0.5f + j) * netStride[stride];  //x
							float y = pdata[1];// (Sigmoid(pdata[1]) * 2.f - 0.5f + i) * netStride[stride];   //y
							float w = pdata[2];// powf(Sigmoid(pdata[2]) * 2.f, 2.f) * anchor_w;   //w
							float h = pdata[3];// powf(Sigmoid(pdata[3]) * 2.f, 2.f) * anchor_h;  //h
							int left = (x - 0.5*w)*ratio_w;
							int top = (y - 0.5*h)*ratio_h;
							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre*box_score);
							boxes.push_back(Rect(left, top, int(w*ratio_w), int(h*ratio_h)));
						}
					}
					pdata += net_width;//��һ��
				}
			}
		}
	}

	//ִ�з�����������������нϵ����Ŷȵ������ص���NMS��
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, classThreshold, nmsThreshold, nms_result);
	for (int i = 0; i < nms_result.size(); i++) {
		int idx = nms_result[i];
		Output result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}

	if (output.size())
		return true;
	else
		return false;
}

void Yolo::drawPred(Mat &img, vector<Output> result, vector<Scalar> color,vector<double> &stds)
{
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;

		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		double std1;
		std::cout << result[i].box <<std::endl;
		string label = className[result[i].id] + ":" + to_string(result[i].confidence);
		if (result[i].id == 2 || result[i].id == 3)//���ӻ�ͻ�� ���б�׼�����
		{
			Mat ROI = img(result[i].box);
			imshow("12", ROI);
			//imwrite("out.bmp", img);
			waitKey();
			//��ROI�����׼����� ������׼��
			Mat gauss_img;
			GaussianBlur(img, gauss_img, Size(5, 5), 5, 0);
			Mat scharr_img;
			Scharr(gauss_img, scharr_img, CV_64F, 0, 1);
			imshow("123", scharr_img);
			//imwrite("out.bmp", img);
			waitKey();
			Mat edeg_img;
			convertScaleAbs(scharr_img, edeg_img);
			Mat edge_gray;
			cvtColor(edeg_img, edge_gray, COLOR_BGR2GRAY);
			Mat thresh_img;
			threshold(edge_gray, thresh_img, 129, 0, THRESH_TOZERO);
			imshow("1234",thresh_img);
			//imwrite("out.bmp", img);
			waitKey();
			vector<int>row_coordinates;
			vector<int>col_coordinates;
			for (size_t i = 0; i < thresh_img.cols; i++)
			{
				for (size_t j = 0; j < thresh_img.rows; j++)
				{
					if (thresh_img.at<uchar>(j, i) > 0)
					{
						row_coordinates.push_back(j);
						col_coordinates.push_back(i);
						cv::circle(img,Point(i, j), 1, (255, 0, 0), -1);
						break;
					}
				}
			}
			double sum = 0;
			for (size_t i = 0; i < row_coordinates.size(); i++)
			{
				int m = row_coordinates[i];
				int n = col_coordinates[i];
				img.at<Vec3b>(m, n)[0] = 0;
				img.at<Vec3b>(m, n)[1] = 0;
				img.at<Vec3b>(m, n)[2] = 255;
				//��������ƽ��ֵ
				sum = row_coordinates[i] + sum;
			}
			double lengths = row_coordinates.size();
			double row_avg = sum / lengths;
			cout << row_avg << endl;
			double variance = 0;
			for (size_t i = 0; i < row_coordinates.size(); i++)
			{
				//���󴦷���
				double m = row_coordinates[i];
				variance = variance + (m - row_avg)*(m - row_avg);
			}
			//��׼��
			std1 = sqrtf(variance / lengths);
			stds.push_back(std1);


			

		}

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(img, label+" "+to_string(std1), Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	imshow("1", img);
	//imwrite("out.bmp", img);
	waitKey();
	//destroyAllWindows();
}