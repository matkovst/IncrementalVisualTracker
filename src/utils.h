#pragma once

#include <vector>
#include <opencv2/core.hpp>

#include "defaults.h"

cv::Mat warpImg(
    const cv::Mat& image, const cv::Mat& state, cv::Size targetSize, bool flatten = false);

cv::Mat cumsum(const cv::Mat& image);

cv::Mat matRowIndexing(const cv::Mat& image, const cv::Mat& rowIds);

cv::Rect state2Rect(const cv::Mat& state, cv::Size targetSize);