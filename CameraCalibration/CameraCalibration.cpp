#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <fstream>

using namespace cv;
using namespace std;

string datapath = "../newdata/";
// 生成点云坐标后保存
static void saveXYZ(string filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	ofstream fp(filename);
	if (!fp.is_open())
	{
		std::cout << "打开点云文件失败" << endl;
		fp.close();
		return;
	}
	//遍历写入
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);   //三通道浮点型
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)
				continue;
			fp << point[0] << " " << point[1] << " " << point[2] << endl;
		}
	}
	fp.close();
}

// 存储视差数据
void saveDisp(const string filename, const Mat& mat)
{
	ofstream fp(filename, ios::out);
	fp << mat.rows << endl;
	fp << mat.cols << endl;
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			double disp = mat.at<short>(y, x); // 这里视差矩阵是CV_16S 格式的，故用 short 类型读取
			fp << disp << endl;       // 若视差矩阵是 CV_32F 格式，则用 float 类型读取
		}
	}
	fp.close();
}


void F_Gray2Color(Mat gray_mat, Mat& color_mat)
{
	color_mat = Mat::zeros(gray_mat.size(), CV_8UC3);
	int rows = color_mat.rows, cols = color_mat.cols;

	Mat red = Mat(gray_mat.rows, gray_mat.cols, CV_8U);
	Mat green = Mat(gray_mat.rows, gray_mat.cols, CV_8U);
	Mat blue = Mat(gray_mat.rows, gray_mat.cols, CV_8U);
	Mat mask = Mat(gray_mat.rows, gray_mat.cols, CV_8U);

	subtract(gray_mat, Scalar(255), blue);         // blue(I) = 255 - gray(I)
	red = gray_mat.clone();                        // red(I) = gray(I)
	green = gray_mat.clone();                      // green(I) = gray(I),if gray(I) < 128

	compare(green, 128, mask, CMP_GE);             // green(I) = 255 - gray(I), if gray(I) >= 128
	subtract(green, Scalar(255), green, mask);
	convertScaleAbs(green, green, 2.0, 2.0);

	vector<Mat> vec;
	vec.push_back(red);
	vec.push_back(green);
	vec.push_back(blue);
	cv::merge(vec, color_mat);
}

Mat F_mergeImg(Mat img1, Mat disp8) {
	Mat color_mat = Mat::zeros(img1.size(), CV_8UC3);

	Mat red = img1.clone();
	Mat green = disp8.clone();
	Mat blue = Mat::zeros(img1.size(), CV_8UC1);

	vector<Mat> vec;
	vec.push_back(red);
	vec.push_back(blue);
	vec.push_back(green);
	cv::merge(vec, color_mat);

	return color_mat;
}

//双目立体标定
static void StereoCalib(const vector<string>& imagelist, Size boardSize, float squareSize, bool displayCorners = false, bool useCalibrated = true, bool showRectified = true)
{
	if (imagelist.size() % 2 != 0)
	{
		cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
	}

	const int maxScale = 2;
	// ARRAY AND VECTOR STORAGE:

	vector<vector<Point2f> > imagePoints[2];
	vector<vector<Point3f> > objectPoints;
	Size imageSize;

	int i, j, k, nimages = (int)imagelist.size() / 2;

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	vector<string> goodImageList;

	for (i = j = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			const string& filename = imagelist[i * 2 + k];
			Mat img = imread(datapath + filename, 0);
			if (img.empty())
				break;
			if (imageSize == Size())
				imageSize = img.size();
			else if (img.size() != imageSize)
			{
				cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
				break;
			}
			bool found = false;
			vector<Point2f>& corners = imagePoints[k][j];
			for (int scale = 1; scale <= maxScale; scale++)
			{
				Mat timg;
				if (scale == 1)
					timg = img;
				else
					resize(img, timg, Size(), scale, scale);
				found = findChessboardCorners(timg, boardSize, corners,
					CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
				if (found)
				{
					if (scale > 1)
					{
						Mat cornersMat(corners);
						cornersMat *= 1. / scale;
					}
					break;
				}
			}
			if (displayCorners)
			{
				cout << filename << endl;
				Mat cimg, cimg1;
				cvtColor(img, cimg, COLOR_GRAY2BGR);
				drawChessboardCorners(cimg, boardSize, corners, found);
				double sf = 640. / MAX(img.rows, img.cols);
				resize(cimg, cimg1, Size(), sf, sf);
				imshow("corners", cimg1);
				char c = (char)waitKey(500);
				if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
					exit(-1);
			}
			else
				putchar('.');
			if (!found)
				break;
			cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
					30, 0.01));
		}
		if (k == 2)
		{
			goodImageList.push_back(imagelist[i * 2]);
			goodImageList.push_back(imagelist[i * 2 + 1]);
			j++;
		}
	}
	cout << j << " pairs have been successfully detected.\n";
	nimages = j;
	if (nimages < 2)
	{
		cout << "Error: too little pairs to run the calibration\n";
		return;
	}

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; i++)
	{
		for (j = 0; j < boardSize.height; j++)
			for (k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}

	cout << "Running stereo calibration ...\n";

	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
	cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);
	Mat R, T, E, F;

	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, E, F,
		CALIB_FIX_ASPECT_RATIO +
		CALIB_ZERO_TANGENT_DIST +
		CALIB_USE_INTRINSIC_GUESS +
		CALIB_SAME_FOCAL_LENGTH +
		CALIB_RATIONAL_MODEL +
		CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
	cout << "done with RMS error=" << rms << endl;

	// CALIBRATION QUALITY CHECK
	// because the output fundamental matrix implicitly
	// includes all the output information,
	// we can check the quality of calibration using the
	// epipolar geometry constraint: m2^t*F*m1=0
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];
	for (i = 0; i < nimages; i++)
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt[2];
		for (k = 0; k < 2; k++)
		{
			imgpt[k] = Mat(imagePoints[k][i]);
			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
			computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
		}
		for (j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
					imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average epipolar err = " << err / npoints << endl;

	// save intrinsic parameters
	FileStorage fs("intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";

	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	fs.open("extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	if (!showRectified)
		return;

	Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)
	if (useCalibrated)
	{
		// we already computed everything
	}
	// OR ELSE HARTLEY'S METHOD
	else
		// use intrinsic parameters of each camera, but
		// compute the rectification transformation directly
		// from the fundamental matrix
	{
		vector<Point2f> allimgpt[2];
		for (k = 0; k < 2; k++)
		{
			for (i = 0; i < nimages; i++)
				std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
		}
		F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
		Mat H1, H2;
		stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

		R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
		R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
		P1 = cameraMatrix[0];
		P2 = cameraMatrix[1];
	}

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);
		h = cvRound(imageSize.height*sf);
		canvas.create(h, w * 2, CV_8UC3);
	}
	else
	{
		sf = 300. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);
		h = cvRound(imageSize.height*sf);
		canvas.create(h * 2, w, CV_8UC3);
	}

	for (i = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			Mat img = imread(datapath + goodImageList[i * 2 + k], 0), rimg, cimg;
			remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
			cvtColor(rimg, cimg, COLOR_GRAY2BGR);
			Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
			resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
			if (useCalibrated)
			{
				Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
					cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
				rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
			}
		}

		if (!isVerticalStereo)
			for (j = 0; j < canvas.rows; j += 16)
				line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for (j = 0; j < canvas.cols; j += 16)
				line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
		imshow("rectified", canvas);
		char c = (char)waitKey();
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

// 双目立体匹配和测量
int stereoMatch(const vector<string>& imagelist,
	string intrinsic_filename = "intrinsics.yml",
	string extrinsic_filename = "extrinsics.yml",
	bool no_display = false,
	string point_cloud_filename = "输出/point3D.txt"
)
{
	//获取待处理的左右相机图像
	int color_mode = 0;
	Mat img1 = imread(datapath + imagelist[0], color_mode);
	Mat img2 = imread(datapath + imagelist[1], color_mode);
	
	imwrite("输出/原始左图像.jpg", img1);
	imwrite("输出/原始右图像.jpg", img2);

	Size img_size = img1.size();

	// reading intrinsic parameters
	FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
	if (!fs.isOpened())
	{
		std::cout << "Failed to open file " << intrinsic_filename << endl;
		return -1;
	}
	Mat M1, D1, M2, D2;      //左右相机的内参数矩阵和畸变系数
	fs["M1"] >> M1;
	fs["D1"] >> D1;
	fs["M2"] >> M2;
	fs["D2"] >> D2;

	// 读取双目相机的立体矫正参数
	fs.open(extrinsic_filename, CV_STORAGE_READ);
	if (!fs.isOpened())
	{
		std::cout << "Failed to open file  " << extrinsic_filename << endl;
		return -1;
	}

	// 立体矫正
	Rect roi1, roi2;
	Mat Q;
	Mat R, T, R1, P1, R2, P2;
	fs["R"] >> R;
	fs["T"] >> T;

	//Alpha取值为-1时，OpenCV自动进行缩放和平移
	cv::stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

	// 获取两相机的矫正映射
	Mat map11, map12, map21, map22;
	initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
	initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

	// 矫正原始图像
	Mat img1r, img2r;
	remap(img1, img1r, map11, map12, INTER_LINEAR);
	remap(img2, img2r, map21, map22, INTER_LINEAR);
	img1 = img1r;
	img2 = img2r;

	// 初始化 stereoBMstate 结构体
	Ptr<StereoBM> bm = StereoBM::create(16, 9);

	int unitDisparity = 6;//40
	int numberOfDisparities = unitDisparity * 16;

	bm->setROI1(roi1);
	bm->setROI2(roi2);
	bm->setPreFilterCap(31);
	bm->setBlockSize(9);
	bm->setMinDisparity(0);
	bm->setNumDisparities(numberOfDisparities);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(15);
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(1);

	// 计算
	Mat disp, disp8;
	int64 t = getTickCount();
	bm->compute(img1, img2, disp);
	t = getTickCount() - t;
	printf("立体匹配耗时: %fms\n", t * 1000 / getTickFrequency());

	// 将16位符号整形的视差矩阵转换为8位无符号整形矩阵
	disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));

	// 视差图转为彩色图
	Mat vdispRGB = disp8;
	F_Gray2Color(disp8, vdispRGB);
	// 将左侧矫正图像与视差图融合
	Mat merge_mat = F_mergeImg(img1, disp8);

	saveDisp("输出/视差数据.txt", disp);

	//显示
	if (!no_display) {
		imshow("左侧矫正图像", img1);
		imwrite("输出/left_undistortRectify.jpg", img1);
		imshow("右侧矫正图像", img2);
		imwrite("输出/right_undistortRectify.jpg", img2);
		imshow("视差图", disp8);
		imwrite("输出/视差图.jpg", disp8);
		imshow("视差图_彩色.jpg", vdispRGB);
		imwrite("输出/视差图_彩色.jpg", vdispRGB);
		imshow("左矫正图像与视差图合并图像", merge_mat);
		imwrite("输出/左矫正图像与视差图合并图像.jpg", merge_mat);
		cv::waitKey();
		std::cout << endl;
	}
	cv::destroyAllWindows();

	// 视差图转为深度图
	cout << endl << "计算深度映射... " << endl;
	Mat xyz;
	reprojectImageTo3D(disp, xyz, Q, true);    //获得深度图  disp: 720*1280 
	cv::destroyAllWindows();
	cout << endl << "保存点云坐标... " << endl;
	saveXYZ(point_cloud_filename, xyz);

	cout << endl << endl << "结束" << endl << "Press any key to end... ";

	getchar();
	return 0;
}

static bool readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}

int main()
{
	//指示参数
	bool COLLECTED_FLAG = true;
	int COUNT_FLAG = 0;
	bool CALIBRATION_FLAG = false;
	//打开摄像头
	const string WINDOW_NAME_R("CameraCalibration_R");
	const string WINDOW_NAME_L("CameraCalibration_L");
	VideoCapture camera_r(2);
	camera_r.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	camera_r.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	VideoCapture camera_l(0);
	camera_l.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	camera_l.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	if (!camera_r.isOpened()) {
		fprintf(stderr, "Error getting camera_r...\n");
		exit(1);
	}
	if (!camera_l.isOpened()) {
		fprintf(stderr, "Error getting camera_l...\n");
		exit(1);
	}
	//调整窗口大小
	namedWindow(WINDOW_NAME_R, WINDOW_KEEPRATIO | WINDOW_AUTOSIZE);
	namedWindow(WINDOW_NAME_L, WINDOW_KEEPRATIO | WINDOW_AUTOSIZE);
	if (!COLLECTED_FLAG)
	{
		cout << "请输入回车键采集数据..." << endl;
	}

	Mat frame_l;
	Mat frame_r;
	while (waitKey(25) != 27)
	{
		//显示实时图像
		camera_l >> frame_l;
		camera_r >> frame_r;
		imshow(WINDOW_NAME_L, frame_l);
		imshow(WINDOW_NAME_R, frame_r);
		//采集数据
		if (!COLLECTED_FLAG)
		{
			if (waitKey(25) == 13)
			{
				COUNT_FLAG++;
				imwrite(datapath + "left" + to_string(COUNT_FLAG) + ".jpg", frame_l);
				imwrite(datapath + "right" + to_string(COUNT_FLAG) + ".jpg", frame_r);
				cout << "已采集到照片" << "left" << COUNT_FLAG << "和" << "right" << COUNT_FLAG << endl;
				if (COUNT_FLAG == 13)
				{
					COLLECTED_FLAG = true;
					cout << "照片采集结束，按回车键开始标定..." << endl;
				}
			}
		}
		//立体标定
		if (!CALIBRATION_FLAG&&COLLECTED_FLAG)
		{
			if (waitKey(25) == 13)
			{
				Size boardSize(11, 8);
				string imagelistfn1 = datapath + "stereo_calib.xml";
				bool showRectified = true;
				float squareSize = 2.0;

				vector<string> imagelist1;
				bool ok1 = readStringList(imagelistfn1, imagelist1);
				if (!ok1 || imagelist1.empty())
				{
					cout << "can not open " << imagelistfn1 << " or the string list is empty" << endl;
					return 0;
				}
				StereoCalib(imagelist1, boardSize, squareSize, false, true, showRectified);
				CALIBRATION_FLAG = true;
				cout << "立体标定结束，按回车键开始匹配当前帧..." << endl;
			}
		}
		//立体匹配
		if (CALIBRATION_FLAG)
		{
			if (waitKey(25) == 13)
			{
				cout << "正在匹配当前帧..." << endl;
				string intrinsic_filename = "intrinsics.yml";
				string extrinsic_filename = "extrinsics.yml";
				string point_cloud_filename = "输出/point3D.txt";
				imwrite(datapath + "left.jpg", frame_l);
				imwrite(datapath + "right.jpg", frame_r);

				string imagelistfn2 = datapath + "stereo_match.xml";
				vector<string> imagelist2;
				bool ok2 = readStringList(imagelistfn2, imagelist2);
				if (!ok2 || imagelist2.empty())
				{
					cout << "can not open " << imagelistfn2 << " or the string list is empty" << endl;
					return 0;
				}
				stereoMatch(imagelist2, intrinsic_filename, extrinsic_filename, false, point_cloud_filename);
			}
		}
	}

	//Size boardSize(9, 6);
	//string imagelistfn1 = "../data/stereo_calib.xml";
	//bool showRectified = true;
	//float squareSize = 1.0;

	//string intrinsic_filename = "intrinsics.yml";
	//string extrinsic_filename = "extrinsics.yml";
	//string point_cloud_filename = "输出/point3D.txt";

	//vector<string> imagelist1;
	//bool ok = readStringList(imagelistfn1, imagelist1);
	//if (!ok || imagelist1.empty())
	//{
	//	cout << "can not open " << imagelistfn1 << " or the string list is empty" << endl;
	//	return 0;
	//}
	//StereoCalib(imagelist1, boardSize, squareSize, true, true, showRectified);

	//string imagelistfn2 = "../data/stereo_match.xml";
	//vector<string> imagelist2;
	//ok = readStringList(imagelistfn2, imagelist2);
	//if (!ok || imagelist2.empty())
	//{
	//	cout << "can not open " << imagelistfn2 << " or the string list is empty" << endl;
	//	return 0;
	//}
	//stereoMatch(imagelist2, intrinsic_filename, extrinsic_filename, false, point_cloud_filename);

	return 0;
}