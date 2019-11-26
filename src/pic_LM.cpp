/*
	���������������������ͼ����й��������

*/

#include "pch.h"
#include <iostream>
#include <opencv2/core/core.hpp> 
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/ml/ml.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\imgproc\types_c.h>



#define MAX_CORNERS 1000

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	//��ȡ����ͼƬ
	vector<Mat> imgs, grayImgs;
	Mat img = imread("1458.jpg");
	imshow("yuantu��", img);
	imgs.push_back(img);
	img = imread("1462.jpg");
	imgs.push_back(img);
	

	//�ҶȻ�
	for (size_t i = 0; i < imgs.size(); i++) {
		//����ԭ����ͼƬ
		Mat temp;
		temp.create(imgs[i].rows, imgs[i].cols, CV_8UC1);

		cvtColor(imgs[i], temp, CV_RGB2GRAY);
		grayImgs.push_back(temp);
	}
	//�����Ƿ���ת��Ϊ�Ҷ�ͼ����Ϊopencv�����������ǻ��ڻҶ�ͼ�ģ�
	for (size_t i = 0; i < imgs.size() && i < grayImgs.size(); i++) {
		//imshow("origin",imgs[i]);
		//imshow("gray",grayImgs[i]);
		//waitKey(10000);
	}

	//��Ǵ����������㲢��ʾ
	vector<Point2f> point[2];
	double qualityLevel = 0.01;
	double minDistance = 5;
	/*
		void goodFeaturesToTrack( InputArray image, OutputArray corners,
								  int maxCorners, double qualityLevel, double minDistance,
								  InputArray mask=noArray(), int blockSize=3,
								  bool useHarrisDetector=false, double k=0.04 )
	*/
	//��imgs[0]�еļ�⵽�Ľǵ����point[0]��
	goodFeaturesToTrack(grayImgs[0], point[0], MAX_CORNERS, qualityLevel, minDistance);
	cout << point[0].size() << endl;
	/*
	  void circle(CV_IN_OUT Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0);
	*/
	//��ʾ�ǵ�
	//for(size_t i= 0;i<point[0].size();i++){
	   // circle(imgs[0], cvPoint(cvRound(point[0][i].x),cvRound(point[0][i].y)), 3, cvScalar(255, 0, 0), 1, CV_AA, 0);
	//}
	//imshow("detected corner", imgs[0]);
	/*
	   void cv::calcOpticalFlowFarneback( InputArray _prev0, InputArray _next0,
							   OutputArray _flow0, double pyr_scale, int levels, int winsize,
							   int iterations, int poly_n, double poly_sigma, int flags )
	   void line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
	*/
	//���ܹ���
	Mat flow;
	calcOpticalFlowFarneback(grayImgs[0], grayImgs[1], flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	cout << flow.size() << endl;  //��ԭͼ��ÿ�����ض��������

	for (size_t y = 0; y < imgs[0].rows; y += 10) {
		for (size_t x = 0; x < imgs[0].cols; x += 10) {
			Point2f fxy = flow.at<Point2f>(y, x);
			line(imgs[0], Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), CV_RGB(255, 0, 0), 1, 8);
		}
	}

	imshow("���ܹ�����", imgs[0]);
	//ϡ�����
	TermCriteria criteria = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	vector<uchar> status;
	vector<float> err;

	calcOpticalFlowPyrLK(grayImgs[0], grayImgs[1], point[0], point[1], status, err, Size(15, 15), 3, criteria);

	for (size_t i = 0; i < point[0].size() && i < point[1].size(); i++) {
		line(imgs[1], Point(cvRound(point[0][i].x), cvRound(point[0][i].y)), Point(cvRound(point[1][i].x),
			cvRound(point[1][i].y)), cvScalar(0, 0, 255), 1, 16);
	}
	cout << flow.size() << endl;  //��ԭͼ��ÿ�����ض��������
	//imshow("ϡ�������", imgs[1]);
	Mat res = imgs[1];
	imshow("ϡ�������", res);
	imwrite("save.jpg", res);
	waitKey(0);
	
	return 0;
}