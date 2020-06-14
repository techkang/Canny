#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <ctime>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#define PI 3.1415926


using namespace std;
using namespace cv;
class Canny2 {
public:
	int kernel, thresh_low, thresh_high, max_edge = 512;
	float sigma;
	bool interactive;
	Canny2(int kernel = 5, int thresh_low = 50, int thresh_high = 200, bool interactive = false, float sigma = 1.4) {
		this->kernel = kernel, this->thresh_low = thresh_low, this->thresh_high = thresh_high;
		this->interactive = interactive, this->sigma = sigma;
	}
	Mat read_image(const string image_name);
	void process(const string image_name);

private:
	Mat smooth(const Mat& image);
	void gradient(const Mat& image, Mat& dx, Mat& dy, Mat& magnitude);
	Mat nms(const Mat& dx, const Mat& dy, const Mat& magnitude);
	Mat double_threshold(const Mat& nms);
	void show(const Mat& thresh);
	void interact();
	Mat convolution(const Mat& src, const Mat& kernel);
	Mat hist_equalize(const Mat& image);
	void check_iterative(const Mat& nms, Mat& boundary, Mat& checked, queue<vector<int> >& lists);
	void convert_image(Mat& image);
	Mat main_part(Mat& image);
};
Mat Canny2::read_image(const string name) {
	return imread(name, IMREAD_COLOR);
}
void Canny2::convert_image(Mat& image) {
	if (max(image.rows, image.cols) > this->max_edge) {
		float ratio = this->max_edge * 1. / max(image.rows, image.cols);
		resize(image, image, Size((int)image.cols * ratio, (int)image.rows * ratio));
	}
	cvtColor(image, image, COLOR_BGR2GRAY);
	image.convertTo(image, CV_32FC1);
}
Mat Canny2::smooth(const Mat& image) {
	Mat gaussian_kernel(this->kernel, this->kernel, CV_32FC1);
	float coef = 1. / 2 / PI / this->sigma / this->sigma;
	for (int row = 0; row < gaussian_kernel.rows; row++) {
		for (int col = 0; col < gaussian_kernel.cols; col++) {
			gaussian_kernel.at<float>(row, col) = exp(-(pow(row - kernel / 2, 2) + pow(col - kernel / 2, 2)) / 2 / pow(sigma, 2)) * coef;
		}
	}
	return this->convolution(image, gaussian_kernel);
}
Mat Canny2::convolution(const Mat& src, const Mat& kernel) {
	Mat blurred(src.rows, src.cols, CV_32FC1, Scalar(0));
	int kernel_size = kernel.rows;
	for (int row = 0; row < src.rows - kernel_size; row++) {
		for (int col = 0; col < src.cols - kernel_size; col++) {
			float total = 0;
			for (int i = 0; i < kernel_size; i++) {
				for (int j = 0; j < kernel_size; j++) {
					total += kernel.at<float>(i, j) * src.at<float>(row + i, col + j);
				}
			}
			blurred.at<float>(row + kernel_size / 2, col + kernel_size / 2) = total;
		}
	}
	return blurred;
}
Mat Canny2::hist_equalize(const Mat& image) {
	Mat new_image(image.rows, image.cols, CV_32FC1, Scalar(0));
	Mat uint_image;
	image.convertTo(uint_image, CV_8UC1);
	vector<int> count(256, 0);
	vector<float> s(256, 0);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			count[uint_image.at<uint8_t>(i, j)]++;
		}
	}
	int sigma = 0;
	float zero_count = count[0];
	count[0] = 0;
	for (int i = 0; i < 256; i++) {
		sigma += count[i];
		s[i] = 255. * sigma * 1. / max(uint_image.rows * uint_image.cols - zero_count, 1.f);
	}
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			new_image.at<float>(i, j) = s[uint_image.at<uint8_t>(i, j)];
		}
	}
	return new_image;
}
void Canny2::gradient(const Mat& image, Mat& dx, Mat& dy, Mat& magnitude) {
	float kernel_x_array[3][3] = { {1,0,-1},{2,0,-2},{1,0,-1} };
	float kernel_y_array[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
	Mat kernel_x(3, 3, CV_32FC1, &kernel_x_array);
	Mat kernel_y(3, 3, CV_32FC1, &kernel_y_array);
	dx = this->convolution(image, kernel_x);
	dy = this->convolution(image, kernel_y);
	magnitude = image.clone();
	float max_magnitude = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			magnitude.at<float>(i, j) = sqrt(pow(dx.at<float>(i, j), 2) + pow(dy.at<float>(i, j), 2));
			max_magnitude = max(max_magnitude, magnitude.at<float>(i, j));
		}
	}
	for (int i = 0; i < magnitude.rows; i++) {
		for (int j = 0; j < magnitude.cols; j++) {
			magnitude.at<float>(i, j) = magnitude.at<float>(i, j) * 255. / max_magnitude;
		}
	}
	magnitude = this->hist_equalize(magnitude);
}
Mat Canny2::nms(const Mat& dx, const Mat& dy, const Mat& magnitude) {
	float grad_x, grad_y, grad, grad1, grad2, grad3, grad4, grad_temp1, grad_temp2, weight;
	int direction;
	Mat nms = magnitude.clone();
	int boundary = this->kernel;
	for (int i = 0; i < nms.rows; i++) {
		for (int j = 0; j < nms.cols; j++) {
			if (i < boundary || j < boundary || i >= nms.rows - boundary - 1 || j >= nms.cols - boundary - 1) {
				nms.at<float>(i, j) = 0;
			}
		}
	}
	for (int i = 1; i < nms.rows - 1; i++) {
		for (int j = 1; j < nms.cols - 1; j++) {
			if (magnitude.at<float>(i, j) > this->thresh_low) {
				grad_x = dx.at<float>(i, j);
				grad_y = dy.at<float>(i, j);
				grad = magnitude.at<float>(i, j);
				direction = ((int)grad_x * grad_y > 0) * 2 - 1;
				if (abs(grad_y) > abs(grad_x)) {
					weight = abs(grad_x) / max(abs(grad_y), 1e-6f);
					grad2 = magnitude.at<float>(i - 1, j);
					grad4 = magnitude.at<float>(i + 1, j);

					grad1 = magnitude.at<float>(i - 1, j - direction);
					grad3 = magnitude.at<float>(i + 1, j + direction);
				}
				else {
					weight = abs(grad_y) / max(abs(grad_x), 1e-6f);
					grad2 = magnitude.at<float>(i, j - 1);
					grad4 = magnitude.at<float>(i, j + 1);

					grad1 = magnitude.at<float>(i - direction, j - 1);
					grad3 = magnitude.at<float>(i + direction, j + 1);
				}
				grad_temp1 = (1 - weight) * grad1 + weight * grad2;
				grad_temp2 = (1 - weight) * grad3 + weight * grad4;
				if (grad <= grad_temp1 || grad < grad_temp2) {
					nms.at<float>(i, j) = 0;
				}
			}
		}
	}
	return nms;
}
void Canny2::check_iterative(const Mat& nms, Mat& boundary, Mat& checked, queue<vector<int> >& lists) {
	int i, j;
	vector<int> coor;
	while (!lists.empty()) {
		i = lists.front()[0];
		j = lists.front()[1];
		lists.pop();
		if (i < 0 || j < 0 || i >= boundary.rows || j >= boundary.cols)
			continue;
		if (checked.at<uint8_t>(i, j))
			continue;
		checked.at<uint8_t>(i, j) = 1;
		if (nms.at<float>(i, j) <= this->thresh_low)
			continue;
		boundary.at<uint8_t>(i, j) = 255;
		for (int row : {-1, 0, 1}) {
			for (int col : {-1, 0, 1}) {
				if (row || col)
					lists.push(vector<int>{i + row, j + col});
			}
		}
	}
}
Mat Canny2::double_threshold(const Mat& nms) {
	Mat boundary(nms.rows, nms.cols, CV_8UC1, Scalar(0));
	Mat checked(nms.rows, nms.cols, CV_8UC1, Scalar(0));
	Mat new_nms = nms.clone();
	for (int i = 0; i < new_nms.rows; i++) {
		for (int j = 0; j < new_nms.cols; j++) {
			if (new_nms.at<float>(i, j) < this->thresh_low) {
				new_nms.at<float>(i, j) = 0;
			}
		}
	}
	new_nms = this->hist_equalize(new_nms);
	queue<vector<int> > lists;
	for (int i = 0; i < new_nms.rows; i++) {
		for (int j = 0; j < new_nms.cols; j++) {
			if (new_nms.at<float>(i, j) > this->thresh_high) {
				lists.push(vector<int>{i, j});
			}
		}
	}
	this->check_iterative(new_nms, boundary, checked, lists);
	return boundary;
}
void Canny2::show(const Mat& boundary) {
	Mat disp_img;
	boundary.convertTo(disp_img, CV_8UC1);
	imshow("my version", disp_img);
	waitKey(0);
}
Mat Canny2::main_part(Mat& image) {
	this->convert_image(image);
	image = this->smooth(image);
	Mat dx, dy, magnitude;
	this->gradient(image, dx, dy, magnitude);
	Mat nms = this->nms(dx, dy, magnitude);
	Mat thresh = this->double_threshold(nms);
	return thresh;
}
void Canny2::process(const string name) {
	clock_t start, end;
	Mat temp, temp2;
	if (!this->interactive) {
		Mat image = this->read_image(name);
		temp2 = image.clone();
		start = clock();
		Mat thresh = this->main_part(image);
		end = clock();
		cout << "my version time used: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
		Canny(temp2, temp, 50, 200);
		end = clock();
		cout << "OpenCV time used: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
		this->show(thresh);
	}
	else
		this->interact();
}
void Canny2::interact() {
	VideoCapture cap;
	if (!cap.open(0))
		return;
	for (;;) {
		Mat image;
		cap >> image;
		flip(image, image, 1);
		if (image.empty())
			break;
		Mat thresh = this->main_part(image);
		imshow("边缘检测，按ESC退出", thresh);
		if (waitKey(10) == 27)
			break;
	}
}
int main() {
	int inter = 0;
	cout << "请选择： 使用摄像头（1）或直接处理图片（0）" << endl;
	cin >> inter;
	Canny2 canny(5, 100, 200, (bool) inter, 1.4);
	canny.process("lenna.png");
	return 0;
}