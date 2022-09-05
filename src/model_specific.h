#pragma once

#include <opencv2/core.hpp>
#include "defaults.h"

void BrownianMotion(cv::Mat& states, const cv::Mat& noise, const cv::Mat& lowerBound);