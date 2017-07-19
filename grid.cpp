#include <opencv2/opencv.hpp>
#include <stdio.h>

int main(int argc, char** argv) {
	cv::Mat sudoku = cv::imread("sudoku.jpg", 0);

	if (sudoku.empty()) {
		std::cout << "Image not loaded";
		return 1;
	}
	else {
		std::cout << "Displaying image";
		cv::Mat box = cv::Mat(sudoku.size(), CV_8UC1);
		cv::GaussianBlur(sudoku, box, cv::Size(11, 11), 0);
		cv::adaptiveThreshold(box, box, 255, 
			cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 2);
		cv::bitwise_not(box, box);
		
		cv::Mat kernel = (cv::Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
		cv::dilate(box, box, kernel);
		cv::floodFill(box, cv::Point(205, 390), cv::Scalar(128));

		cv::Mat mask = cv::Mat::zeros(box.size(), CV_8UC1);
		mask.setTo(255, box == 128);
		cv::erode(mask, mask, kernel);

		int right = 0;
		int left = mask.cols;
		
		int bottom = 0;
		int top = mask.rows;

		for (int i = 0; i < mask.rows; i++) {
			for (int j = 0; j < mask.cols; j++) {
				if (mask.at<int>(i, j) == 255) {
					if (i < left)   left = i;
					if (i > right)  right = i;
					if (j < top)    top = j;
					if (j > bottom) bottom = j;
				}
			}
		}

		int width = right - left;
		int height = bottom - top;
		cv::Rect sudokuBoard(top, left, height, width);
		cv::Mat croppedImage = sudoku(sudokuBoard);

		cv::imshow("filtered", croppedImage);
		cv::waitKey(0);
		return 0;
	}
}