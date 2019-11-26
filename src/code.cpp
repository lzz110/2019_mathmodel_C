/*
	该文件是 任务一 至 任务四 问题求解的 C++ 和 MATLAB 代码汇总
	
	仅有关键函数定义，完整任务一至四的代码文件参考对应的cpp文件

*/



//////////////////   任务1 C++求解     ///////////
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
//获取两点距离
float sqrtmy(float x1, float y1, float x2, float y2) {
	double res = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
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
//响应鼠标函数
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

//手动选择四个角点坐标输出焦距
void spgetfx() {
	//src = imread("test1.png");
	src = imread("4.png");
	Point2f miedian1, miedian2;
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

    //获得两个灭点坐标
	getCross(xa, ya, xb, yb, xc, xc, xd, yd, miedian1);
	getCross(xa, ya, xc, yc, xb, yb, xd, yd, miedian2);

	LinePara plantline;
	getLinePara(miedian1.x, miedian1.y, miedian2.x, miedian2.y, plantline);
	Point2f imgcenter(dst.cols, dst.rows);
	Point2f Puv;
    
	//获得垂点
	Puv.x = (imgcenter.x + plantline.k*imgcenter.y - plantline.k*plantline.b) / (plantline.k*plantline.k + 1);
	Puv.y = (plantline.k*imgcenter.x + plantline.k*plantline.k*imgcenter.y + plantline.b) / (plantline.k*plantline.k + 1);
	
    //计算距离fm
	float PPuv = sqrtmy(dst.cols, dst.rows, Puv.x, Puv.y);
	float MPuv1 = sqrtmy(miedian1.x, miedian1.y, Puv.x, Puv.y);
	float MPuv2 = sqrtmy(miedian2.x, miedian2.y, Puv.x, Puv.y);
	float M1M2 = sqrtmy(miedian1.x, miedian1.y, miedian2.x, miedian2.y);
	float OPuv = sqrt(MPuv1*MPuv2);
	float fm = sqrt(OPuv*OPuv - PPuv * PPuv);
	cout << "fm=" << fm << endl;
}


///////////    任务1 MATLAB求解   ////////////////
fvi = 657.9  ;  fvj = 1284.5  ;
fui = 1081.92  ;  fuj = 1752.83  ; %和D共线
px=750; py=1000;     %中心点横纵坐标 手动输入
xin_D_x=1024; xin_D_y=1756;     %D'图像坐标手动输入
xin_A_x = 1142; xin_A_y = 1908;%A' (xa,ya,f) 手动输入
AD_realLen = 0.8;

f1=strcat('(puvx-', num2str(double(fui)),')*(puvx-',num2str(px),')+(puvy-', num2str(double(fuj))...
    ,')*(puvy-', num2str(py),')=0');
f2=strcat('(puvx-',num2str(double(fvi)),')*(puvx-',num2str(px),')+(puvy-',num2str(double(fvj))...
    ,')*(puvy-',num2str(py),')=0');
% f1='(puvx+902.7315)*(puvx-352)+(puvy+388.0320)*(puvy-288)=0';
% f2='(puvx-1107.1190)*(puvx-352)+(puvy+405.6929)*(puvy-288)=0';
[puvx,puvy]=solve(f1,f2)

n=length(puvx);
puvx=puvx(1);
n=length(puvy); 
puvy=puvy(1);

%求焦距f
Lfupuv=longth(puvx,puvy,fui,fuj);
Lfvpuv=longth(puvx,puvy,fvi,fvj);
Lppuv=longth(puvx,puvy,px,py);
f=sqrt(Lfupuv*Lfvpuv-Lppuv^2)

%求旋转矩阵M
s1=sqrt(fui*fui+fuj*fuj+f*f);
s2=sqrt(fvi*fvi+fvj*fvj+f*f);
a1=fui/s1;
a2=fuj/s1;
a3=f/s1;
b1=fvi/s2;
b2=fvj/s2;
b3=f/s2;
m1=[a1,a2,a3];
m2=[b1,b2,b3];
% m3=[a2*b3-a3*b2,a3*b1-a1*b3, a1*b2-a2*b1]
m3=[-a2*b3+a3*b2,-a3*b1+a1*b3, -a1*b2+a2*b1];

M=[m1',m2',m3']
%计算平移向量
%A'D''平行于o-fu,由方向向量和一点确定直线方程（x-x0）/m=(y-y0)/n=(z-z0)/p  (m,n,p)为方向向量
%确定直线od'方程  由两点式方程 (x-x1)/(x2-x1)=(y-y1)/(y2-y1)=(z-z1)/(z2-z1)
%由C点和摄像机坐标系原点求直线方程 
f5=strcat('x/',num2str(xin_D_x),'=y/',num2str(xin_D_y));       %D'
f6=strcat('y/',num2str(xin_D_y),'=z/',num2str(double(f)));

%由o-fv向量和A'点求直线方程,

f7=strcat('(x-',num2str(xin_A_x),')/',num2str(double(fvi)),'=(y-'...
    ,num2str(xin_A_y),')/',num2str(double(fvj)),'=(z-'...
    ,num2str(double(f)),')/',num2str(double(f))); %A' (xa,ya,f)
%计算D''点坐标
[x,y,z]=solve(f5,f6,f7);

lAD=llongth(x,y,z,407,517,f);  %计算A'D''在摄像机坐标系下的长度

%计算平移向量To-c=Mc-o*OA  OA向量等于OA'向量乘上AD向量的模再除以A'D''的长度
u=[1,0,0];
%计算AD向量在摄像机坐标系下的长度
LAD=M*u'*AD_realLen;% AD 长度
n=length(LAD);
LAD=llongth(0,0,0,LAD(1),LAD(2),LAD(3));
%计算OA'向量
OA=[407,517,f];%A' (xa,ya,f)

T=inv(M)*OA'*LAD/lAD

Dis_len = (LAD / lAD) * llongth(0,0,0,OA(1),OA(2),OA(3))


/////////////////////////////////////////////////////////////////////
//////////////////   任务2  C++ 光流法测量超车速度差   ///////////
//获取透视变换矩阵
Mat PerspectiveTrans(Mat src, Point2f* scrPoints, Point2f* dstPoints) {
	Mat dst;
	Trans = getPerspectiveTransform(scrPoints, dstPoints);
	warpPerspective(src, dst, Trans, Size(src.cols, src.rows), INTER_CUBIC);    
	cout << Trans << endl;
	return dst;
}
Mat MyAffineTrans(Mat src,Mat trans) {
	Mat dst;
	//warpAffine(src, dst, trans, Size(src.cols, src.rows), INTER_CUBIC);
	warpPerspective(src, dst, Trans, Size(src.cols, src.rows), INTER_CUBIC);
	return dst;
}

//将光流图透视变换
void AffinePoints()
{
	Mat im_temp, im_res;
    im_temp=imread("save.jpg");
	//src = imread("20.jpg");
	src = imread("1.png");
	namedWindow("input image", WINDOW_AUTOSIZE);
	setMouseCallback("input image", on_mouse, &src);

	for(int i=0;i<2;i++)
	{
		imshow("input image", src);
		waitKey(0);
	}
	cout << pointset.size() << endl;
	vector<Point2f>::iterator it=pointset.begin();

	for (int i = 0; i < pointset.size(); i++)
		temp[i] = *(it + i);
	Point2f AffinePoints0[4] = { temp[0], temp[1],temp[2], temp[3] };
	Point2f AffinePoints1[4] = { temp[4], temp[5],temp[6], temp[7] };
	//Mat dst_affine = AffineTrans(src, AffinePoints0, AffinePoints1);
	Mat dst_perspective = PerspectiveTrans(src, AffinePoints0, AffinePoints1);
	for (int i = 0; i < 4; i++)
	{
		circle(src, AffinePoints0[i], 2, Scalar(0, 0, 255), 2);
		//circle(dst_affine, AffinePoints1[i], 2, Scalar(0, 0, 255), 2);
		circle(dst_perspective, AffinePoints1[i], 2, Scalar(0, 0, 255), 2);
	}
	cout <<Trans.type() <<endl;
	
	im_res = MyAffineTrans(im_temp, Trans);
    imshow("res", im_res);
	imwrite("src.jpg", src);
	imwrite("res.jpg", im_res);
	imshow("origin", src);
	//imshow("affine", dst_affine);
	imshow("perspective", dst_perspective);
	imwrite("perspective.jpg", dst_perspective);
	waitKey();
}

//////////////////   任务3   C++光流法测量高铁速度  MATLAB计算桥距  ///////////


//C++ 光流法测量高铁速度  同任务二方法

%% 线性增长
res = 0;
L1=0.3348
Li=0.0837
diff = 286
for i=1:286
%     res=res+(Li/L1)^(i/diff)*L1;
    res=res+(Li/L1)^(1/diff)*L1;
end
res

//////////////////   任务4   C++单目多图三维重建  ///////////
///使用SIFT进行特征点匹配
void Siftmatch(){
//读取原始基准图和待匹配图
	Mat srcImg1 = imread("1.JPG");      //待配准图
	Mat srcImg2 = imread("2.JPG");      //基准图

	//显示基准和待配准图
	imshow("待配准图", srcImg1);
	imshow("基准图", srcImg2);

	//定义SIFT特征检测类对象
	SiftFeatureDetector siftDetector1;
	SiftFeatureDetector siftDetector2;

	//定义KeyPoint变量
	vector<KeyPoint>keyPoints1;
	vector<KeyPoint>keyPoints2;

	//特征点检测
	siftDetector1.detect(srcImg1, keyPoints1);
	siftDetector2.detect(srcImg2, keyPoints2);

	//绘制特征点(关键点)
	Mat feature_pic1, feature_pic2;
	drawKeypoints(srcImg1, keyPoints1, feature_pic1, Scalar::all(-1));
	drawKeypoints(srcImg2, keyPoints2, feature_pic2, Scalar::all(-1));

	drawKeypoints(srcImg1, keyPoints1, feature_pic1, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(srcImg2, keyPoints2, feature_pic2, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//显示原图

	//显示结果
	imshow("feature1", feature_pic1);
	imshow("feature2", feature_pic2);

	//计算特征点描述符 / 特征向量提取
	SiftDescriptorExtractor descriptor;
	Mat description1;
	descriptor.compute(srcImg1, keyPoints1, description1);
	Mat description2;
	descriptor.compute(srcImg2, keyPoints2, description2);

	cout << keyPoints1.size() << endl;
	cout << description1.cols << endl;      //列数
	cout << description1.rows << endl;      //行数


	//进行BFMatch暴力匹配
	//BruteForceMatcher<L2<float>>matcher;    //实例化暴力匹配器
	FlannBasedMatcher matcher;  //实例化FLANN匹配器
	vector<DMatch>matches;   //定义匹配结果变量
	matcher.match(description1, description2, matches);  //实现描述符之间的匹配

	//中间变量
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
	cout << "最大距离：" << max_dist << endl;
	cout << "最小距离：" << min_dist << endl;

	//筛选出较好的匹配点  
	vector<DMatch> good_matches;
	double dThreshold = 0.5;    //匹配的阈值，越大匹配的点数越多
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance < dThreshold * max_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	//RANSAC 消除误匹配特征点 主要分为三个部分：
	//1）根据matches将特征点对齐,将坐标转换为float类型
	//2）使用求基础矩阵方法findFundamentalMat,得到RansacStatus
	//3）根据RansacStatus来将误匹配的点也即RansacStatus[i]=0的点删除

   //根据matches将特征点对齐,将坐标转换为float类型
	vector<KeyPoint> R_keypoint01, R_keypoint02;
	for (i = 0; i < good_matches.size(); i++)
	{
		R_keypoint01.push_back(keyPoints1[good_matches[i].queryIdx]);
		R_keypoint02.push_back(keyPoints2[good_matches[i].trainIdx]);
		// 这两句话的理解：R_keypoint1是要存储img01中能与img02匹配的特征点，
		// matches中存储了这些匹配点对的img01和img02的索引值
	}

	//坐标转换
	vector<Point2f>p01, p02;
	for (i = 0; i < good_matches.size(); i++)
	{
		p01.push_back(R_keypoint01[i].pt);
		p02.push_back(R_keypoint02[i].pt);
	}

	//计算基础矩阵并剔除误匹配点
	vector<uchar> RansacStatus;
	Mat Fundamental = findHomography(p01, p02, RansacStatus, RANSAC);
	Mat dst;
	warpPerspective(srcImg1, dst, Fundamental, Size(srcImg1.cols, srcImg1.rows));

	imshow("配准后的图", dst);
	imwrite("dst.jpg", dst);

	//剔除误匹配的点对
	vector<KeyPoint> RR_keypoint01, RR_keypoint02;
	vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
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
	cout << "找到的特征点对：" << RR_matches.size() << endl;

	//画出消除误匹配后的图
	Mat img_RR_matches;
	drawMatches(srcImg1, RR_keypoint01, srcImg2, RR_keypoint02, RR_matches, img_RR_matches, Scalar(0, 255, 0), Scalar::all(-1));
	imshow("消除误匹配点后", img_RR_matches);
	imwrite("匹配图.jpg", img_RR_matches);
	waitKey(0);
}

void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector <vector<Vec3b>>& colors_for_all
)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image;

	// 读取图像，获取图像特征点并保存
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	for (auto it = image_names.begin(); it != image_names.end(); ++it)
	{
		image = imread(*it);
		if (image.empty())
		{
			continue;
		}

		vector<KeyPoint> key_points;
		Mat descriptor;
		// 偶尔出现内存分配失败的错误  Detects keypoints and computes the descriptors
		// sift->detectAndCompute(image, noArray(), key_points, descriptor);
		sift->detect(image, key_points);
		sift->compute(image, key_points, descriptor);


		// 特征点过少，则排除该图像
		if (key_points.size() <= 10)
		{
			continue;
		}

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		vector<Vec3b> colors(key_points.size());	// 三通道 存放该位置三通道颜色
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& p = key_points[i].pt;
			/*cout << p.x << ", " << p.y << endl;
			if (i == 2653)
			{
				cout << p.x << ", " << p.y << endl;
				cout << image.rows << ", " << image.cols << endl;
			}*/
			if (p.x <= image.rows && p.y <= image.cols)
				colors[i] = image.at<Vec3b>(p.x, p.y);
		}

		colors_for_all.push_back(colors);
	}
}

void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);

	// 获取满足Ratio Test的最小匹配的距离
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		// Rotio Test
		if (knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance)
		{
			continue;
		}

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist)
		{
			min_dist = dist;
		}
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		// 排除不满足Ratio Test的点和匹配距离过大的点
		if (
			knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
		{
			continue;
		}

		// 保存匹配点
		matches.push_back(knn_matches[r][0]);
	}
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
	// 根据内参数矩阵获取相机的焦距和光心坐标（主点坐标）
	double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	// 根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty())
	{
		return false;
	}

	double feasible_count = countNonZero(mask);	// 得到非零元素，即数组中的有效点
	// cout << (int)feasible_count << " - in - " << p1.size() << endl;

	// 对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
	{
		return false;
	}

	// 分解本征矩阵，获取相对变换
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	// cout << "pass_count = " << pass_count << endl;

	// 同时位于两个相机前方的点的数量要足够大
	if (((double)pass_count) / feasible_count < 0.7)
	{
		return false;
	}
	return true;
}

void reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure)
{
	// 两个相机的投影矩阵[R T], triangulatePoints 只支持float型
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);	// 对角矩阵 为1
	proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);

	R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK * proj1;
	proj2 = fK * proj2;

	// 三角化重建
	triangulatePoints(proj1, proj2, p1, p2, structure);
}

void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2
)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		/*Point2f tmp1 = p1[matches[i].queryIdx].pt;
		Point2f tmp2 = p2[matches[i].trainIdx].pt;
		if (tmp1.x <= width && tmp1.y < height)*/
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		//if (tmp2.x <= width && tmp2.y < height)
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
		{
			p1.push_back(p1_copy[i]);
		}
	}
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
		{
			p1.push_back(p1_copy[i]);
		}
	}
}

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, Mat& structure, vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << structure.cols;

	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; i++)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.cols; ++i)
	{
		Mat_<float> c = structure.col(i);
		c /= c(3);	//齐次坐标，需要除以最后一个元素才是真正的坐标值
		fs << Point3f(c(0), c(1), c(2));
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();

}
void reconstruct(){
    find_transform();
    Siftmatch();
    maskout_colors();
    save_structure();
    
}
int main(int argc, char* argv)
{
	spgetfx(); 
    AffinePoints();
    reconstruct();
	return 0;
}


