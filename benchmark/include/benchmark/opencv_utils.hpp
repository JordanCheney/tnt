#ifndef TNT_OPENCV_UTILS_HPP
#define TNT_OPENCV_UTILS_HPP

#include <opencv2/core.hpp>

namespace tnt
{

template <typename DataType>
inline cv::Mat create_cv_mat(int rows, int cols, cv::Scalar value);

template <> inline cv::Mat create_cv_mat<uint8_t>(int rows, int cols, cv::Scalar value) { return cv::Mat(rows, cols, CV_8U, value); }
template <> inline cv::Mat create_cv_mat<uint16_t>(int rows, int cols, cv::Scalar value) { return cv::Mat(rows, cols, CV_16U, value); }
template <> inline cv::Mat create_cv_mat<int8_t>(int rows, int cols, cv::Scalar value) { return cv::Mat(rows, cols, CV_8S, value); }
template <> inline cv::Mat create_cv_mat<int16_t>(int rows, int cols, cv::Scalar value) { return cv::Mat(rows, cols, CV_16S, value); }
template <> inline cv::Mat create_cv_mat<int32_t>(int rows, int cols, cv::Scalar value) { return cv::Mat(rows, cols, CV_32S, value); }
template <> inline cv::Mat create_cv_mat<float>(int rows, int cols, cv::Scalar value) { return cv::Mat(rows, cols, CV_32F, value); }
template <> inline cv::Mat create_cv_mat<double>(int rows, int cols, cv::Scalar value) { return cv::Mat(rows, cols, CV_64F, value); }

}

#endif // TNT_OPENCV_UTILS_HPP
