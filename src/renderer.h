#pragma once

#include <opencv2/core.hpp>
#include "ivtracker.h"

cv::Mat renderTelemetry(int height, const cv::TickMeter& meter);

cv::Mat renderEigenbasis(
    int width, const ObjectTemplate& templ, const cv::Mat& warpImage);