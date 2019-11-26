/*
	任务一：在图像中选择坐标并打印 计算灭点 及相机fm 参数 

*/


#include "pch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <opencv2/highgui/highgui_c.h>

using namespace cv;
using namespace std;

struct LinePara
{
	float k;
	float b;

};


Mat src, dst;
vector<float> mouse;

// 获取直线参数  
void getLinePara(float& x1, float& y1, float& x2, float& y2, LinePara & LP)
{
	double m = 0;

	// 计算分子  
	m = x2 - x1;

	if (0 == m)
	{
		LP.k = 10000.0;
		LP.b = y1 - LP.k * x1;
	}
	else
	{
		LP.k = (y2 - y1) / (x2 - x1);
		LP.b = y1 - LP.k * x1;
	}
}
float sqrtmy(float x1, float y1, float x2, float y2) {
	double res = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
	cout << res<<endl;
	return res;
}
// 获取交点  
bool getCross(float & x1, float &y1, float & x2, float & y2, float & x3, float &y3, float & x4, float & y4, Point2f & pt) {

	LinePara para1, para2;
	getLinePara(x1, y1, x2, y2, para1);
	getLinePara(x3, y3, x4, y4, para2);

	// 判断是否平行  
	if (abs(para1.k - para2.k) > 0.001)
	{
		pt.x = (para2.b - para1.b) / (para1.k - para2.k);
		pt.y = para1.k * pt.x + para1.b;

		return true;

	}
	else
	{
		return false;
	}

}
void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Mat p;
		p = *(Mat*)ustc;
		Point  pt = Point(x, y);
		char temp[16];
		sprintf_s(temp, "(%d,%d)", pt.x, pt.y);
		putText(src, temp, pt, CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 2, 8);
		//printf("b=%d\t", p.at<Vec3b>(pt)[0]);
		//printf("g=%d\t", p.at<Vec3b>(pt)[1]);
		//printf("r=%d\n", p.at<Vec3b>(pt)[2]);
		mouse.push_back(pt.x);
		mouse.push_back(pt.y);
		circle(src, pt, 2, Scalar(0, 0, 255), 2, 8);
	}
}
void qudian() {
	
	//src = imread("test1.png");
	src = imread("3.png");
	Point2f miedian1, miedian2;
	//namedWindow("input image", CV_WINDOW_AUTOSIZE);
	namedWindow("input image", 1);
	resize(src, dst, Size(), 0.5, 0.5);
	//resize(src, dst, Size(), 1, 1);
	setMouseCallback("input image", on_mouse, &dst);
	for (int i = 0; i < 2; i++) {
		imshow("input image", dst);
		namedWindow("input image", 1);

		waitKey(0);
	}

	vector<float>::iterator it = mouse.begin();
	float xa, xb, xc, xd, ya, yb, yc, yd;

	xa = *it*2;
	xb = *(it + 2)*2;
	xc = *(it + 4) * 2;
	xd = *(it + 6) * 2;

	ya = *(it + 1) * 2;
	yb = *(it + 3) * 2;
	yc = *(it + 5) * 2;
	yd = *(it + 7) * 2;

	//xa = *it ;
	//xb = *(it + 2) ;
	//xc = *(it + 4) ;
	//xd = *(it + 6) ;

	//ya = *(it + 1) ;
	//yb = *(it + 3) ;
	//yc = *(it + 5) ;
	//yd = *(it + 7) ;
	//xa = 1, ya = 1, ya = 2, yb = 2, xc = 0, yc = 2, xd = 2, yc = 0;

	cout << "mouse.size=" << mouse.size() << endl;
	cout << "x1=" << xa << endl;
	cout << "y1=" << ya  << endl;
	cout << "x2=" << xb  << endl;
	cout << "y2=" << yb << endl;
	cout << "x3=" << xc  << endl;
	cout << "y3=" << yc<< endl;
	cout << "x4=" << xd << endl;
	cout << "y4=" << yd  << endl;
	//开始处理
	getCross(xa, ya, xb, yb, xc, xc, xd, yd, miedian1);
	getCross(xa, ya, xc, yc, xb, yb, xd, yd, miedian2);
	cout << miedian1.x << " , " << miedian1.y << endl;
	cout << miedian2.x << " , " << miedian2.y << endl;

	////////////////////////////
	LinePara plantline, plantline1;
	getLinePara(miedian1.x, miedian1.y, miedian2.x, miedian2.y, plantline);
	cout << dst.cols << "	" << dst.rows << endl;
	Point2f imgcenter(dst.cols, dst.rows);
	Point2f Puv;
	//获得垂点
	Puv.x = (imgcenter.x + plantline.k*imgcenter.y - plantline.k*plantline.b) / (plantline.k*plantline.k + 1);
	Puv.y = (plantline.k*imgcenter.x + plantline.k*plantline.k*imgcenter.y + plantline.b) / (plantline.k*plantline.k + 1);
	cout << "Puv.x=" << Puv.x << "		Puv.y=" << Puv.y << endl;

	getLinePara(miedian1.x, miedian1.y, Puv.x, Puv.y, plantline1);

	cout << "plantline.b=" << plantline.b << "	plantline.k=" << plantline.k << endl;
	cout << "plantline1.b=" << plantline1.b << "	plantline1.k=" << plantline1.k << endl;
	//计算距离
	float PPuv = sqrtmy(dst.cols, dst.rows, Puv.x, Puv.y);
	float MPuv1 = sqrtmy(miedian1.x, miedian1.y, Puv.x, Puv.y);
	float MPuv2 = sqrtmy(miedian2.x, miedian2.y, Puv.x, Puv.y);
	float M1M2 = sqrtmy(miedian1.x, miedian1.y, miedian2.x, miedian2.y);
	float OPuv = sqrt(MPuv1*MPuv2);
	cout << "MPuv1=" << MPuv1 << endl;
	cout << "MPuv2=" << MPuv2 << endl;
	cout << "M1M2=" << M1M2 << endl;
	cout << "OPuv=" << OPuv << endl;
	cout << "PPuv=" << PPuv << endl;

	float fm = sqrt(OPuv*OPuv - PPuv * PPuv);
	cout << "fm=" << fm << endl;
}
void shoudong(Point2f mypt1, Point2f mypt2, Point2f mypt3, Point2f mypt4) {
	Point2f pt1, pt2, pt3, pt4, pt;
	pt1.x = mypt1.x;
	pt1.y = mypt1.y;

	pt2.x = mypt2.x;
	pt2.y = mypt2.y;

	pt3.x = mypt3.x;
	pt3.y = mypt3.y;

	pt4.x = mypt4.x;
	pt4.y = mypt4.y;

	getCross(pt1.x, pt1.y, pt2.x, pt2.y, pt3.x, pt3.y, pt4.x, pt4.y, pt);

	cout << pt.x << " , " << pt.y << endl; 
}
int main(int argc, char* argv)
{
	//Point2f pt1(510.139,748.956), pt2(902, 1028), pt3(1412, 1016), pt4(646,863);
	//shoudong(pt1,pt2, pt3, pt4);
	//qudian();
	sqrtmy(648,863,1412,1016);
	return 0;
}


