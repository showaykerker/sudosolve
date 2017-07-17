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
		cv::namedWindow("input", CV_WINDOW_AUTOSIZE);
		cv::imshow("input", sudoku);
		cv::waitKey(0);

		cv::Mat box = cv::Mat(sudoku.size(), CV_8UC1);
		return 0;
	}
}