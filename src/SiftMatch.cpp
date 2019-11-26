/*
	�����ģ�SIFT�ؼ���ƥ��

*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <vector>
#include <cmath>

#define PI 3.14159265358979323846264338327950288419716939937510582097
using namespace std;
using namespace cv;

void main()
{
	//��ȡԭʼ��׼ͼ�ʹ�ƥ��ͼ
	Mat srcImg1 = imread("1.JPG");      //����׼ͼ
	Mat srcImg2 = imread("2.JPG");      //��׼ͼ

	//��ʾ��׼�ʹ���׼ͼ
	imshow("����׼ͼ", srcImg1);
	imshow("��׼ͼ", srcImg2);

	//����SIFT������������
	SiftFeatureDetector siftDetector1;
	SiftFeatureDetector siftDetector2;

	//����KeyPoint����
	vector<KeyPoint>keyPoints1;
	vector<KeyPoint>keyPoints2;

	//��������
	siftDetector1.detect(srcImg1, keyPoints1);
	siftDetector2.detect(srcImg2, keyPoints2);

	//����������(�ؼ���)
	Mat feature_pic1, feature_pic2;
	drawKeypoints(srcImg1, keyPoints1, feature_pic1, Scalar::all(-1));
	drawKeypoints(srcImg2, keyPoints2, feature_pic2, Scalar::all(-1));

	drawKeypoints(srcImg1, keyPoints1, feature_pic1, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(srcImg2, keyPoints2, feature_pic2, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//��ʾԭͼ

	//��ʾ���
	imshow("feature1", feature_pic1);
	imshow("feature2", feature_pic2);

	//���������������� / ����������ȡ
	SiftDescriptorExtractor descriptor;
	Mat description1;
	descriptor.compute(srcImg1, keyPoints1, description1);
	Mat description2;
	descriptor.compute(srcImg2, keyPoints2, description2);

	cout << keyPoints1.size() << endl;
	cout << description1.cols << endl;      //����
	cout << description1.rows << endl;      //����


	//����BFMatch����ƥ��
	//BruteForceMatcher<L2<float>>matcher;    //ʵ��������ƥ����
	FlannBasedMatcher matcher;  //ʵ����FLANNƥ����
	vector<DMatch>matches;   //����ƥ��������
	matcher.match(description1, description2, matches);  //ʵ��������֮���ƥ��

	//�м����
	int i, j, k; double sum = 0; double b;

	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < matches.size(); i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}
	cout << "�����룺" << max_dist << endl;
	cout << "��С���룺" << min_dist << endl;

	//ɸѡ���Ϻõ�ƥ���  
	vector<DMatch> good_matches;
	double dThreshold = 0.5;    //ƥ�����ֵ��Խ��ƥ��ĵ���Խ��
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance < dThreshold * max_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	//RANSAC ������ƥ�������� ��Ҫ��Ϊ�������֣�
	//1������matches�����������,������ת��Ϊfloat����
	//2��ʹ����������󷽷�findFundamentalMat,�õ�RansacStatus
	//3������RansacStatus������ƥ��ĵ�Ҳ��RansacStatus[i]=0�ĵ�ɾ��

   //����matches�����������,������ת��Ϊfloat����
	vector<KeyPoint> R_keypoint01, R_keypoint02;
	for (i = 0; i < good_matches.size(); i++)
	{
		R_keypoint01.push_back(keyPoints1[good_matches[i].queryIdx]);
		R_keypoint02.push_back(keyPoints2[good_matches[i].trainIdx]);
		// �����仰����⣺R_keypoint1��Ҫ�洢img01������img02ƥ��������㣬
		// matches�д洢����Щƥ���Ե�img01��img02������ֵ
	}

	//����ת��
	vector<Point2f>p01, p02;
	for (i = 0; i < good_matches.size(); i++)
	{
		p01.push_back(R_keypoint01[i].pt);
		p02.push_back(R_keypoint02[i].pt);
	}

	//������������޳���ƥ���
	vector<uchar> RansacStatus;
	Mat Fundamental = findHomography(p01, p02, RansacStatus, RANSAC);
	Mat dst;
	warpPerspective(srcImg1, dst, Fundamental, Size(srcImg1.cols, srcImg1.rows));

	imshow("��׼���ͼ", dst);
	imwrite("dst.jpg", dst);

	//�޳���ƥ��ĵ��
	vector<KeyPoint> RR_keypoint01, RR_keypoint02;
	vector<DMatch> RR_matches;            //���¶���RR_keypoint ��RR_matches���洢�µĹؼ����ƥ�����
	int index = 0;
	for (i = 0; i < good_matches.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			RR_keypoint01.push_back(R_keypoint01[i]);
			RR_keypoint02.push_back(R_keypoint02[i]);
			good_matches[i].queryIdx = index;
			good_matches[i].trainIdx = index;
			RR_matches.push_back(good_matches[i]);
			index++;
		}
	}
	cout << "�ҵ���������ԣ�" << RR_matches.size() << endl;

	//����������ƥ����ͼ
	Mat img_RR_matches;
	drawMatches(srcImg1, RR_keypoint01, srcImg2, RR_keypoint02, RR_matches, img_RR_matches, Scalar(0, 255, 0), Scalar::all(-1));
	imshow("������ƥ����", img_RR_matches);
	imwrite("ƥ��ͼ.jpg", img_RR_matches);

	waitKey(0);
}
