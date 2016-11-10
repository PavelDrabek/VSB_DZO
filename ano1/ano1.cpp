// ano1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string>

int ShowGradient(int width, int height) {
	cv::Mat src_8uc1a_img = cv::Mat(height, width, CV_8UC1);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			src_8uc1a_img.at<uchar>(y, x) = x;
		}
	}
	cv::imshow("gradient", src_8uc1a_img); // display image
	return 0;
}

int ShowRGB(int width, int height) {
	cv::Mat src_32fc3_img = cv::Mat(height, width, CV_8UC3);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			src_32fc3_img.at<cv::Vec3b>(y, x) = cv::Vec3b((uchar)x, (uchar)y, (uchar)255);
		}
	}
	cv::imshow("rgb", src_32fc3_img); // display image
	return 0;
}

int GammaCorrection() {
	float gamma = 0.5;

	cv::Mat src_8uc1a_img = cv::imread("images/moon.jpg", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale
	int height = src_8uc1a_img.rows;
	int width = src_8uc1a_img.cols;
	
	cv::Mat scr_32fc1a_img = cv::Mat(height, width, CV_32FC1);
	double minGamma = 0, maxGamma = 1;
	double nMinGamma = 0, nMaxGamma = 255;
	double nRange = maxGamma - minGamma;

	cv::minMaxLoc(src_8uc1a_img, &minGamma, &maxGamma);
	maxGamma /= 255.0;
	minGamma /= 255.0;
	double range = maxGamma - minGamma;
	printf("min: %f, max: %f, range: %f\n", minGamma, maxGamma, range);

	float min = 255;
	float max = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float o = src_8uc1a_img.at<uchar>(y, x) / 255.0;
			float gc = cv::pow(o, 1 / gamma);
			//float rc = ((o - minGamma) / (maxGamma - minGamma)) * (nMaxGamma - nMinGamma) + nMinGamma;
			float rc = (o - minGamma) / range;
			scr_32fc1a_img.at<float>(y, x) = gc;
		}
	}

	cv::minMaxLoc(scr_32fc1a_img, &minGamma, &maxGamma);
	range = maxGamma - minGamma;
	printf("min: %f, max: %f, range: %f\n", minGamma, maxGamma, range);

	cv::imshow("moon - original", src_8uc1a_img); // display image
	cv::imshow("moon - corrected", scr_32fc1a_img); // display image
	return 0;
}

cv::Mat RangeCorrection(cv::Mat source, float min, float max)
{
	int height = source.rows;
	int width = source.cols;
	cv::Mat result = cv::Mat(height, width, CV_32FC1);

	double minGamma = 0, maxGamma = 1;
	cv::minMaxLoc(source, &minGamma, &maxGamma);

	double range = maxGamma - minGamma;
	printf("min: %f, max: %f, range: %f\n", minGamma, maxGamma, range);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float o = source.at<float>(y, x);
			float rc = (o - minGamma) / range;
			result.at<float>(y, x) = rc;
		}
	}

	cv::minMaxLoc(result, &minGamma, &maxGamma);
	range = maxGamma - minGamma;
	printf("min: %f, max: %f, range: %f\n", minGamma, maxGamma, range);

	return result;
}


cv::Mat RangeCorrectionDouble(cv::Mat source, double min, double max)
{
	int height = source.rows;
	int width = source.cols;
	cv::Mat result = cv::Mat(height, width, CV_64FC1);

	double minGamma = 0, maxGamma = 1;
	cv::minMaxLoc(source, &minGamma, &maxGamma);

	double range = maxGamma - minGamma;
	printf("min: %f, max: %f, range: %f\n", minGamma, maxGamma, range);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			double o = source.at<double>(y, x);
			double rc = (o - minGamma) / range;
			result.at<double>(y, x) = rc;
		}
	}

	cv::minMaxLoc(result, &minGamma, &maxGamma);
	range = maxGamma - minGamma;
	printf("min: %f, max: %f, range: %f\n", minGamma, maxGamma, range);

	return result;
}

cv::Mat GammaCorrection(float gamma, cv::Mat source) 
{
	int height = source.rows;
	int width = source.cols;

	cv::Mat result = cv::Mat(height, width, CV_32FC1);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float o = source.at<float>(y, x);
			float gc = cv::pow(o, 1 / gamma);
			result.at<float>(y, x) = gc;
		}
	}

	return result;
}

cv::Mat ConvertTo32FC1(cv::Mat source)
{
	int height = source.rows;
	int width = source.cols;

	cv::Mat result = cv::Mat(height, width, CV_32FC1);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			result.at<float>(y, x) = source.at<uchar>(y, x) / 255.0;
		}
	}

	return result;
}

cv::Mat ConvertTo64FC1(cv::Mat source)
{
	int height = source.rows;
	int width = source.cols;

	cv::Mat result = cv::Mat(height, width, CV_64FC1);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			result.at<double>(y, x) = source.at<uchar>(y, x) / 255.0;
		}
	}

	return result;
}

cv::Mat uniformMat = cv::Mat::ones(3, 3, CV_32F);
cv::Mat gaussMat = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, 5, 1, 0, 1, 0);
cv::Mat testMat = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
cv::Mat testMat2 = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
cv::Mat testMat3 = (cv::Mat_<float>(3, 3) << 1, 0, 1, 1, 0, 1, 1, 0, 1);
cv::Mat testMat4 = (cv::Mat_<float>(3, 3) << 0, -1, 0, 1, 0, -1, 0, 1, 0);
cv::Mat testMat5 = (cv::Mat_<float>(3, 3) << 3, 2, 3, 2, 0, 2, 3, 2, 3);


cv::Mat ApplyMatrix(cv::Mat source, cv::Mat matrix, float matValue)
{
	int sHeight = source.rows;
	int sWidth = source.cols;
	int mHeight = matrix.rows;
	int mWidth = matrix.cols;

	cv::Mat result = cv::Mat(sHeight, sWidth, CV_32FC1);

	for (int sy = 0; sy < sHeight; sy++) {
		for (int sx = 0; sx < sWidth; sx++) {
			float pixel = 0;
			for (int my = 0; my < mHeight; my++) {
				for (int mx = 0; mx < mWidth; mx++) {
					int x = sx + mx - mWidth / 2;
					int y = sy + my - mHeight / 2;
					if (x < 0 || x >= sWidth || y < 0 || y >= sHeight) {
						//printf("x: %d, y: %d, pixel: %f, matValue: %f\n", x, y, pixel * matValue, matValue);
						//pixel = source.at<float>(sy, sx) / matValue;
						pixel = source.at<float>(sy, sx) / matValue;
						continue;
					}
					pixel += source.at<float>(y, x) * matrix.at<float>(my, mx);
				}
			}
			result.at<float>(sy, sx) = pixel * matValue;
		}
	}

	return result;
}


#define E 2.71828182846
#define sqr(a) ((a)*(a))

double GetConductance(double I, double Id, double sigma2) {
	double deltaI2 = sqr(Id - I);
	return pow(E, -deltaI2 / sigma2);
}

cv::Mat Anisotropic(cv::Mat source, double lambda, double sigma2)
{
	int height = source.rows;
	int width = source.cols;

	//double sigma2 = sqr(sigma);

	cv::Mat result = cv::Mat(height, width, CV_64FC1);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			double I = source.at<double>(y, x);

			if (x < 1 || x+1 >= width || y < 1 || y+1 >= height) {
				result.at<double>(y, x) = I;
				continue;
			}

			double IN = source.at<double>(y-1, x);
			double IS = source.at<double>(y+1, x);
			double IE = source.at<double>(y, x+1);
			double IW = source.at<double>(y, x-1);

			double cN = GetConductance(I, IN, sigma2);
			double cS = GetConductance(I, IS, sigma2);
			double cE = GetConductance(I, IE, sigma2);
			double cW = GetConductance(I, IW, sigma2);

			double In = I * (1 - lambda * (cN + cS + cE + cW)) + lambda * (cN*IN + cS*IS + cE*IE + cW*IW);

			result.at<double>(y, x) = In;
		}
	}

	return result;
}

cv::Mat Resize(cv::Mat src, int width, int height) {
	cv::Size size(width, height);//the dst image size,e.g.100x100
	cv::Mat dst;//dst image
	cv::resize(src, dst, size, 0, 0, cv::INTER_NEAREST);//resize image
	return dst;
}

cv::Mat InverseFourierTransformation(cv::Mat fourier) {
	int width = fourier.cols;
	int height = fourier.rows;

	cv::Mat result = cv::Mat(height, width, CV_64FC1);
	double norm = 1.0 / sqrt(width * height);
	double H_d = 1.0 / height;
	double W_d = 1.0 / width;

	printf("Transform from fourier\n");
	for (int m = 0; m < height; m++) {
		for (int n = 0; n < width; n++) {
			double Fmn_r = 0;
			for (int k = 0; k < height; k++) {
				for (int l = 0; l < width; l++) {
					cv::Vec2d c = fourier.at<cv::Vec2d>(k, l);
					double fkl_r = c.val[0];
					double fkl_i = c.val[1];
					double F_kl = sqrt(fkl_r * fkl_r + fkl_i * fkl_i);

					double X = (k * m * H_d) + (l * n * W_d);
					double alpha = 2 * M_PI * X;
					double phi_r = norm * cos(alpha);
					double phi_i = norm * sin(alpha);

					Fmn_r += (fkl_r * phi_r) - (fkl_i * phi_i);
				}
			}
			int y = (int)(m + height / 2) % height;
			int x = (int)(n + height / 2) % width;
			result.at<double>(m, n) = Fmn_r;
		}
		printf(".", m);
	}
	printf("\n");

	return result;
}

cv::Mat FourierTransformation(cv::Mat source) {
	int width = source.cols;
	int height = source.rows;
	double norm = 1.0 / sqrt(width * height);
	double H_d = 1.0 / height;
	double W_d = 1.0 / width;
	// normalizace
	source *= norm;
	
	printf("Transform to fourier\n");
	cv::Mat fourier = cv::Mat(height, width, CV_64FC2);
	for (int k = 0; k < height; k++) {
		for (int l = 0; l < width; l++) {
			double R_kl = 0;
			double I_kl = 0;
			for (int m = 0; m < height; m++) {
				for (int n = 0; n < width; n++) {
					double f = source.at<double>(m, n);
					double X = (k * m * H_d) + (l * n * W_d);
					double alpha = -2 * M_PI * X;
					double phi_r = f * cos(alpha);
				 	double phi_i = f * sin(alpha);

					R_kl += phi_r;
					I_kl += phi_i;
				}
			}

			cv::Vec2d c;
			c.val[0] = R_kl;
			c.val[1] = I_kl;
			fourier.at<cv::Vec2d>(k, l) = c;
		}
		printf(".");
	}
	printf("\n");

	return fourier;
}

void ShowFourier(cv::Mat fourier, bool rCorrectPP = false, bool rCorrectRI = false) {
	int width = fourier.cols;
	int height = fourier.rows;

	cv::Mat phase = cv::Mat(height, width, CV_64FC1);
	cv::Mat power = cv::Mat(height, width, CV_64FC1);
	cv::Mat R = cv::Mat(height, width, CV_64FC1);
	cv::Mat I = cv::Mat(height, width, CV_64FC1);

	for (int k = 0; k < height; k++) {
		for (int l = 0; l < width; l++) {
			cv::Vec2d c = fourier.at<cv::Vec2d>(k, l);
			double R_kl = c.val[0];
			double I_kl = c.val[1];
			double F_kl = sqrt(R_kl * R_kl + I_kl * I_kl);
			
			int y = (int)(k + height / 2) % height;
			int x = (int)(l + width / 2) % width;
			phase.at<double>(k, l) = atan2(I_kl, R_kl);
			power.at<double>(y, x) = log(F_kl * F_kl);
			R.at<double>(k, l) = R_kl;
			I.at<double>(k, l) = I_kl;
		}
	}

	if (rCorrectPP) {
		phase = RangeCorrectionDouble(phase, 0, 1);
		power = RangeCorrectionDouble(power, 0, 1);
	}
	if (rCorrectRI) {
		R = RangeCorrectionDouble(R, 0, 1);
		I = RangeCorrectionDouble(I, 0, 1);
	}
	cv::imshow("phase", Resize(phase, 256, 256));
	cv::imshow("power", Resize(power, 256, 256));
	cv::imshow("R", Resize(R, 256, 256));
	cv::imshow("I", Resize(I, 256, 256));
}

#define SQR( x ) ( ( x ) * ( x ) )
#define DEG2RAD( x ) ( ( x ) * M_PI / 180.0 )
#define MY_MIN( X, Y ) ( ( X ) < ( Y ) ? ( X ) : ( Y ) )

int LinearInterpolation(IplImage* img, int x1, int y1, int x2, int y2) {

}

void geom_dist(IplImage* src, IplImage* dst, bool bili, double K1 = 1.0, double K2 = 1.0)
{
	double cu = src->width * 0.5;
	double cv = src->height * 0.5;
	double R = sqrt(cu*cu + cv*cv);


	for (int yn = 0; yn < src->height; yn++) {
		uchar* ptr_dst = (uchar*)(dst->imageData + (int)yn * dst->widthStep);
		for (int xn = 0; xn < src->width; xn++) {

			double x = (xn - cu) / R;
			double y = (yn - cv) / R;
			double sqrR = x*x + y*y;
			double fi_sqrR = 1 + K1 * sqrR + K2 * sqrR*sqrR;

			double xd = (xn - cu) * (1 / fi_sqrR) + cu;
			double yd = (yn - cv) * (1 / fi_sqrR) + cv;

			uchar* ptr_src = (uchar*)(src->imageData + (int)yd * src->widthStep);


			ptr_dst[3 * (int)xn + 0] = ptr_src[3 * (int)xd + 0]; //Set B (BGR format)
			ptr_dst[3 * (int)xn + 1] = ptr_src[3 * (int)xd + 1]; //Set G (BGR format)
			ptr_dst[3 * (int)xn + 2] = ptr_src[3 * (int)xd + 2]; //Set R (BGR format)
		}
	}
}


int K1 = 3.0, K2 = 1.0;
IplImage *img, *img_geom;

void on_change(int id)
{
	geom_dist(img, img_geom, false, K1 / 100.0, K2 / 100.0);
	cvShowImage("Geom Dist", img_geom);
}

int runGeomDist()
{
	img = cvLoadImage("images/distorted_panorama.jpg", CV_LOAD_IMAGE_COLOR);
	if (!img)
	{
		printf("Unable to load image!\n");
		exit(-1);
	}

	cvNamedWindow("Original Image");
	cvShowImage("Original Image", img);

	img_geom = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
	geom_dist(img, img_geom, false, K1 / 100.0, K2 / 100.0);

	cvNamedWindow("Geom Dist");
	cvShowImage("Geom Dist", img_geom);

	cvCreateTrackbar("K1", "Geom Dist", &K1, 1000, on_change);
	cvCreateTrackbar("K2", "Geom Dist", &K2, 1000, on_change);

	cvWaitKey(0);

	return 0;
}

cv::Mat DeleteNoise(cv::Mat fourier, int radius) {
	int width = fourier.cols;
	int height = fourier.rows;
	int r2 = radius * radius;

	cv::Mat result = cv::Mat(height, width, CV_64FC2);

	for (int k = 0; k < height; k++) {
		for (int l = 0; l < width; l++) {
			int y = (int)(k + height * 0.5) % height - (int)(height * 0.5);
			int x = (int)(l + width * 0.5) % width - (int)(width * 0.5);

			double rkl2 = (x*x + y*y);

			cv::Vec2d c = fourier.at<cv::Vec2d>(k, l);
			if (rkl2 > r2) {
				c = cv::Vec2d(0, 0);
			}

			result.at<cv::Vec2d>(k, l) = c;
		}
	}

	return result;
}

cv::Mat DeleteBars(cv::Mat fourier, int barierY, int barierX) {
	int width = fourier.cols;
	int height = fourier.rows;

	cv::Mat result = cv::Mat(height, width, CV_64FC2);

	for (int k = 0; k < height; k++) {
		for (int l = 0; l < width; l++) {
			int y = (int)(k + height / 2) % height;
			int x = (int)(l + width / 2) % width;

			cv::Vec2d c = fourier.at<cv::Vec2d>(k, l);
			if (k < barierY && l > barierX && l < width - barierX) {
				c = cv::Vec2d(0, 0);
			}

			result.at<cv::Vec2d>(k, l) = c;
		}
	}

	return result;
}

void Histogram() {
	cv::Mat img = cv::imread("images/uneq.jpg", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale
	img.convertTo(img, CV_8UC1);

	imshow("Original Image", img);

	int width = img.cols;
	int height = img.rows;
	int res = width * height;
	const int L = 256;

	int colors[L] = { 0 };
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int i = img.at<uchar>(y, x);
			colors[img.at<uchar>(y, x)]++;
		}
	}

	int cdf[L] = { 0 };
	int cdfMin = (unsigned int)(~0) >> 1;
	int pMin = (unsigned int)(~0) >> 1;
	int pMax = 0;
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < i; j++) {
			cdf[i] += colors[j];
		}
		if (cdf[i] > 0 && cdf[i] < cdfMin) {
			cdfMin = cdf[i];
		}
		if (colors[i] < pMin) {
			pMin = colors[i];
		}
		if (colors[i] > pMax) {
			pMax = colors[i];
		}
	}
	printf("%d", cdfMin);

	cv::Mat result = cv::Mat(height, width, CV_8UC1);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int v = img.at<uchar>(y, x);
			int hv = (int)round( ((cdf[v] - cdfMin) / (double)(res - cdfMin)) * (L - 1) );

			result.at<uchar>(y, x) = hv;
		}
	}

	cv::Mat histBefore = cv::Mat(L, L, CV_8UC1);
	cv::Mat histCdf = cv::Mat(L, L, CV_8UC1);

	for (int y = 0; y < L; y++) {
		for (int x = 0; x < L; x++) {
			if (colors[x] == 0) {
				histBefore.at<uchar>(y, x) = 0;
			} else {
				histBefore.at<uchar>(y, x) = y >(res / colors[x]) ? 255 : 0;
			}
			if (cdf[x] == 0) {
				histCdf.at<uchar>(y, x) = 0;
			}
			else {
				histCdf.at<uchar>(y, x) = y >(res / cdf[x]) ? 255 : 0;
			}
		}
	}

	imshow("Result image", result);
	imshow("Hisogram before", histBefore);
	imshow("Hisogram cdf", histCdf);
}

int main(int argc, char* argv[])
{
	//cv::Mat src_8uc1a_img = cv::imread("images/moon.jpg", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale
	//cv::Mat src_32fc1_moon = ConvertTo32FC1(src_8uc1a_img);
	//cv::Mat gamma05 = GammaCorrection(0.1, src_32fc1_moon);
	//cv::Mat gamma25 = GammaCorrection(40.5, src_32fc1_moon);
	//cv::Mat range = RangeCorrection(src_32fc1_moon, 0, 1);
	//cv::Mat gammaRange05 = RangeCorrection(gamma05, 0, 1);
	//cv::Mat gammaRange25 = RangeCorrection(gamma25, 0, 1);

	//cv::imshow("moon - original", src_32fc1_moon); 
	//cv::imshow("moon - gamma corrected 05", gamma05);
	//cv::imshow("moon - gamma corrected 25", gamma25);
	//cv::imshow("moon - range corrected", range);
	//cv::imshow("moon - gamma and range corrected 05", gammaRange05);
	//cv::imshow("moon - gamma and range corrected 25", gammaRange25);


	//cv::Mat src_8uc1_lena = cv::imread("images/lena.png", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale
	//cv::Mat src_32fc1_lena; //= ConvertTo32FC1(src_8uc1_lena);
	//cv::Mat src_64fc1_lena;
	//src_8uc1_lena.convertTo(src_32fc1_lena, CV_32FC1, 1 / 255.0, 0);
	//src_8uc1_lena.convertTo(src_64fc1_lena, CV_64FC1, 1/255.0, 0);

	//cv::imshow("lena - orig", src_32fc1_lena);
	//cv::imshow("lena - matrix1", ApplyMatrix(src_32fc1_lena, testMat, 1.0 / 1.0));
	//cv::imshow("lena - matrix2", ApplyMatrix(src_32fc1_lena, testMat2, 1.0 / 1.0));
	//cv::imshow("lena - matrix3", ApplyMatrix(src_32fc1_lena, testMat3, 1.0 / 6.0));
	//cv::imshow("lena - matrix4", ApplyMatrix(src_32fc1_lena, testMat4, 1.0 / 4.0));
	//cv::imshow("lena - matrix5", ApplyMatrix(src_32fc1_lena, testMat5, 1.0 / 20.0));

	/*double lambda = 0.1;
	double sigma2 = sqr(0.015);
	cv::Mat output, tmp;
	src_8uc1_lena.convertTo(output, CV_64FC1, 1 / 255.0, 0);
	cv::imshow("lena - orig", output);
	for (int i = 0; i < 1000; i++)
	{
		printf("%d\n", i);
		tmp = output;
		output = Anisotropic(output, lambda, sigma2);
		cv::imshow("lena - anisotropic", output);
		cv::waitKey(10);
	}*/

	//cv::Mat src_8uc1_lena = cv::imread("images/lena64_noise.png", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale
	//cv::Mat orig; // = Resize(ConvertTo64FC1(src_8uc1_lena), 64, 64);
	//src_8uc1_lena.convertTo(orig, CV_64FC1, 1.0 / 255.0);
	//
	//cv::imshow("orig", Resize(orig, 256, 256));

	//cv::Mat fourier = FourierTransformation(orig);
	//
	//fourier = DeleteNoise(fourier, 28);
	//fourier = DeleteBars(fourier, 1, 10);
	//ShowFourier(fourier, true, false);
	//cv::waitKey(30);

	//cv::Mat reverted = InverseFourierTransformation(fourier);
	//cv::imshow("restored", Resize(reverted, 256, 256));

	//runGeomDist();
	Histogram();

	cv::waitKey(0); // press any key to exit
	return 0;
}



