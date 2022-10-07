#include "ImageDetect.h"

//#include <cstdlib>

/********************************
下面从yolo代码移植过来的源代码内容
********************************/
bool Yolo::readModel(cv::dnn::Net &net, std::string &netPath, bool isCuda = false)
{
	try
	{
		net = cv::dnn::readNet(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	if (isCuda) 
	{
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

bool Yolo::Detect(cv::Mat &SrcImg, cv::dnn::Net &net, std::vector<Output> &output)
{
	cv::Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	cv::Mat netInputImg = SrcImg.clone();
	if (maxLen > 1.2*col || maxLen > 1.2*row) {
		cv::Mat resizeImg = cv::Mat::zeros(maxLen, maxLen, CV_8UC3);
		SrcImg.copyTo(resizeImg(cv::Rect(0, 0, col, row)));
		netInputImg = resizeImg;
	}
	
	cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(104, 117, 123), true, false);
	//如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0,0), true, false);
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(114, 114,114), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	//vector<string> outputLayerName{"345","403", "461","output" };
	//net.forward(netOutputImg, outputLayerName[3]); //获取output的输出
	net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	float ratio_h = (float)netInputImg.rows / netHeight;
	float ratio_w = (float)netInputImg.cols / netWidth;
	int net_width = className.size() + 5;  //输出的网络宽度是类别数+5
	float* pdata = (float*)netOutputImg[0].data;
	for (int stride = 0; stride < 3; stride++) {    //stride
		int grid_x = (int)(netWidth / netStride[stride]);
		int grid_y = (int)(netHeight / netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++) { //anchors
			const float anchor_w = netAnchors[stride][anchor * 2];
			const float anchor_h = netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++) {
				for (int j = 0; j < grid_x; j++) {
					float box_score = pdata[4]; //Sigmoid(pdata[4]);//获取每一行的box框中含有某个物体的概率
					if (box_score > boxThreshold) {
						cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
						cv::Point classIdPoint;
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
							boxes.push_back(cv::Rect(left, top, int(w*ratio_w), int(h*ratio_h)));
						}
					}
					pdata += net_width;//下一行
				}
			}
		}
	}

	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, classThreshold, nmsThreshold, nms_result);
	for (int i = 0; i < nms_result.size(); i++) 
	{
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

void Yolo::drawPred(cv::Mat &img, std::vector<Output> result, std::vector<cv::Scalar> color, std::vector<double> &stds)
{
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;

		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		double std1 = 0.0;
		std::cout << result[i].box << std::endl;
		std::string label = className[result[i].id] + ":" + std::to_string(result[i].confidence);
		if (result[i].id == 2 || result[i].id == 3)//凹坑或突刺 进行标准差计算
		{
			cv::Mat ROI = img(result[i].box);
			cv::imshow("12", ROI);
			//imwrite("out.bmp", img);
			cv::waitKey();
			//将ROI传入标准差计算 给出标准差
			cv::Mat gauss_img;
			GaussianBlur(img, gauss_img, cv::Size(5, 5), 5, 0);
			cv::Mat scharr_img;
			Scharr(gauss_img, scharr_img, CV_64F, 0, 1);
			imshow("123", scharr_img);
			//imwrite("out.bmp", img);
			cv::waitKey();
			cv::Mat edeg_img;
			convertScaleAbs(scharr_img, edeg_img);
			cv::Mat edge_gray;
			cvtColor(edeg_img, edge_gray, cv::COLOR_BGR2GRAY);
			cv::Mat thresh_img;
			cv::threshold(edge_gray, thresh_img, 129, 0, cv::THRESH_TOZERO);
			//cv::imshow("1234", thresh_img);
			//imwrite("out.bmp", img);
			cv::waitKey();
			std::vector<int>row_coordinates;
			std::vector<int>col_coordinates;
			
			for (size_t i = 0; i < thresh_img.cols; i++)
			{
				for (size_t j = 0; j < thresh_img.rows; j++)
				{
					if (thresh_img.at<uchar>(j, i) > 0)
					{
						row_coordinates.push_back(j);
						col_coordinates.push_back(i);
						cv::circle(img, cv::Point(i, j), 1, (255, 0, 0), -1);
						break;
					}
				}
			}
			
			double sum = 0;
			
			for (size_t i = 0; i < row_coordinates.size(); i++)
			{
				int m = row_coordinates[i];
				int n = col_coordinates[i];
				img.at<cv::Vec3b>(m, n)[0] = 0;
				img.at<cv::Vec3b>(m, n)[1] = 0;
				img.at<cv::Vec3b>(m, n)[2] = 255;
				//求行坐标平均值
				sum = row_coordinates[i] + sum;
			}
			double lengths = row_coordinates.size();
			double row_avg = sum / lengths;
			//std::cout << row_avg << std::endl;
			double variance = 0;
			for (size_t i = 0; i < row_coordinates.size(); i++)
			{
				//先求处方差
				double m = row_coordinates[i];
				variance = variance + (m - row_avg)*(m - row_avg);
			}
			//标准差
			std1 = sqrtf(variance / lengths);
			stds.push_back(std1);

		}

		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = std::max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(img, label + " " + std::to_string(std1), cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	//cv::imshow("1", img);
	//imwrite("out.bmp", img);
	//cv::waitKey();
	//destroyAllWindows();
}

/********************************
下面封装类源代码内容，主要调用该类
********************************/

bool ImageDetect::GetImage(cv::Mat Image)
{
	if (Image.empty())
	{
		return false;
	}

	else
	{
		ImageDetect::Image = Image.clone();//深拷贝输入图像到类中，与源图像独立
		HaveImage = true;
		return true;
	}
}

bool ImageDetect::GetImage(std::string ImagePath)
{
	cv::Mat srcImage = cv::imread(ImagePath);

	if (Image.empty())
	{
		return false;
	}

	else
	{
		Image = srcImage.clone();
		HaveImage = true;
		return true;
	}
}

std::string ImageDetect::findModel()
{
	return model;
}

bool ImageDetect::YoloInit(std::string modelpath)
{
	bool flag = yolo->readModel(net, modelpath, false);
	
	model = modelpath;

	if (flag == false)
	{
		model = "NoAny";
		return flag;
	}

	else
	{
		srand(time(0));

		for (int i = 0; i < 80; i++) 
		{
			int b = rand() % 256;
			int g = rand() % 256;
			int r = rand() % 256;
			color.push_back(cv::Scalar(b, g, r));
		}

		return flag;
	}
}

bool ImageDetect::YoloDetect(yoloresult &yoloresult)
{
	bool flag= yolo->Detect(Image, net, yoloresult.result);

	if (flag == false)
	{
		return flag;
	}

	else
	{
		yoloresult.className = yolo->className;
		yolo->drawPred(Image, yoloresult.result, color, yoloresult.stds);
		yoloresult.flag = true;
		return flag;
	}
}

bool ImageDetect::printyoloresult(const yoloresult yoloresult)
{
	if (yoloresult.flag == false)
	{
		return false;
	}
		
	else
	{
		for (int i = 0; i < yoloresult.result.size(); i++)
		{
			std::cout << yoloresult.className[yoloresult.result[i].id] << std::endl;
		}

		return true;
	}
}

/***************************************
Main3函数
****************************************/

cv::Mat patch(int i, int j, cv::Mat src)
{
	cv::Mat patch_img(src, cv::Range(i - 50, i + 50), cv::Range(j - 50, j + 50));
	//rectangle(src, Point(j - 50, i - 50), Point(j + 50, i + 50), Scalar(255), 1);
	//imshow("patch", src);
	//waitKey(1000);
	return patch_img;
}

double AreaContrast(cv::Mat patchs)//只需要传入一张图像就ok
{
	//将patch内的像素值全部存入容器Vpixel中
	std::vector<int> Vpixel;
	for (size_t ii = 0; ii < patchs.rows; ii++)
	{
		for (size_t jj = 0; jj < patchs.cols; jj++)
		{
			int value = patchs.at<uchar>(ii, jj);
			Vpixel.push_back(value);
		}
	}
	//对Vpixel中的像素值进行排序
	std::sort(Vpixel.begin(), Vpixel.end());
	//Vpiexl的索引是0---Vpixel.size()-1,故前500个为0-499，后500个为Vpixel.size()-501---Vpixel.size()-1
	//定义两个容器存储前500个和后500个
	std::vector<int>max_500, min_500;
	int min_start = Vpixel.size() - 501;
	
	for (size_t m = 0; m < 500; m++)
	{
		int minvalue = Vpixel[m];
		int maxvalue = Vpixel[min_start + m];
		max_500.push_back(maxvalue);
		min_500.push_back(minvalue);
	}

	//最大值和最小值的求和
	double max_sum = 0;
	double min_sum = 0;
	
	for (size_t n = 0; n < max_500.size(); n++)
	{
		max_sum = max_sum + max_500[n];
		min_sum = min_sum + min_500[n];

	}
	
	double dif = (max_sum - min_sum) / 500;

	Vpixel.clear();
	max_500.clear();
	min_500.clear();

	return dif;
}

cv::Mat ImageDetect::residual()
{
	cv::Mat img, edge_img;
	std::vector<int> j_edge;

	cv::cvtColor(Image, img, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(img, img, cv::Size(3, 3), 5, 0);
	cv::Canny(img, edge_img, 50, 100);

	for (size_t i = 0; i < edge_img.rows; i++)
	{
		for (size_t j = 0; j < edge_img.cols; j++)
		{
			if (edge_img.at<uchar>(i, j) == 255)
			{
				j_edge.push_back(j);
				break;
			}
		}
	}

	sort(j_edge.begin(), j_edge.end());//从小到大排序
	
	int j_edge_length = j_edge.size();

	int hang = 0;
	
	for (size_t i = 50; i < 1851; i = i + 100)//i这个值不用改
	{
		std::vector<double> Vavg, Vstd;
		int lie = 0;


		for (size_t j = j_edge[0] - 50; j > 50; j = j - 100)//j这个值根据j_edge[0]进行修改，以j_edge[0]-50作为起始点
		{
			cv::Mat patch_image = patch(i, j, img);//取小patch操作，得到100*100的图像块
											   //进行判断，查看patch内是否存在黑色小颗粒残留
											   //判断依据：取patch内前500个最大值的和与最后500个最小值的和的差值，因黑色颗粒残留只存在于
											   //白色区域，因此每隔patch内的差值在一个大致范围呢，这个范围可以根据工装进行调节设置
			double dif = AreaContrast(patch_image);
			//cout << "第" << hang << "的第" << lie << "个patch的的差值：" << dif << endl;
			lie++;
		}
		hang++;
	}
}

cv::Mat ImageDetect::Confirm()
{
	cv::Mat Input = Image.clone();
	cv::Mat img, gauss_img,edge_img;
	cv::cvtColor(Input, img, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(img, gauss_img, cv::Size(5, 5), 5, 0);
	cv::Canny(gauss_img, edge_img, 20,50,3,false);

	std::vector<cv::Vec4i> lines1, lines2;
	cv::HoughLinesP(edge_img, lines1, 1, CV_PI / 360, 30, 20, 5);
	cv::HoughLinesP(edge_img, lines1, 1, CV_PI / 360, 30, 20, 10);

	for (size_t i = 0; i < lines1.size(); i++)
	{
		double x1 = lines1[i][0];
		double y1 = lines1[i][1];
		double x2 = lines1[i][2];
		double y2 = lines1[i][3];
		double k = -(y2 - y1) / (x2 - x1);
		double angles = atan(k)*57.29577;
		//if (angles<20)
		//{
		//	line(color_img, Point(x1, y1), Point(x2, y2), Scalar(0,0,255), 1);
		//	cout << angles << endl;
		//}
		if (abs(angles) < 80 && abs(angles) > 70)
		{
			cv::line(Input, cv::Point(x1, y1), cv::Point(x2, y2),cv::Scalar(0, 0, 255), 1);
			std::string text;
			text = std::to_string(angles);
			cv::putText(Input, text, cv::Point(x1, y1), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);

		}
		
		else if (abs(angles) < -70 && abs(angles) > -80)
		{
			cv::line(Input, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1);
			std::string text;
			text = std::to_string(angles);
			cv::putText(Input, text, cv::Point(x1, y1), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
		}
	}

	return Input;
}

double ImageDetect::roundness()
{
	cv::Mat Input = Image.clone();
	cv::Mat img, gauss_img, scharr_img, edeg_img, edge_gray, thresh_img;
	cv::cvtColor(Input, img, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(img, gauss_img, cv::Size(5, 5), 5, 0);
	cv::Scharr(gauss_img, scharr_img, CV_64F, 0, 1);
	cv::convertScaleAbs(scharr_img, edeg_img);
	cv::cvtColor(edeg_img, edge_gray, cv::COLOR_BGR2GRAY);
	cv::threshold(edge_gray, thresh_img, 129, 0, cv::THRESH_TOZERO);

	std::vector<int>row_coordinates, col_coordinates;
	
	for (size_t i = 0; i < thresh_img.cols; i++)
	{
		for (size_t j = 0; j < thresh_img.rows; j++)
		{
			if (thresh_img.at<uchar>(j, i)>0)
			{
				row_coordinates.push_back(j);
				col_coordinates.push_back(i);
				break;
			}
		}
	}

	double sum = 0;
	for (size_t i = 0; i < row_coordinates.size(); i++)
	{
		int m = row_coordinates[i];
		int n = col_coordinates[i];
		Input.at<cv::Vec3b>(m, n)[0] = 0;
		Input.at<cv::Vec3b>(m, n)[1] = 0;
		Input.at<cv::Vec3b>(m, n)[2] = 255;
		//求行坐标平均值
		sum = row_coordinates[i] + sum;
	}
	double lengths = row_coordinates.size();
	double row_avg = sum / lengths;
	//cout << row_avg << endl;
	double variance = 0;
	for (size_t i = 0; i < row_coordinates.size(); i++)
	{
		//先求处方差
		double m = row_coordinates[i];
		variance = variance + (m - row_avg)*(m - row_avg);
	}
	//标准差
	double std = sqrtf(variance / lengths);
	return std;
}
