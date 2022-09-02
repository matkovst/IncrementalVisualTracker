#pragma once

#include <opencv2/core.hpp>

void BrownianMotion(cv::Mat& states, const cv::Mat& noise, const cv::Mat& lowerBound);