#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include "ivtracker.h"
#include "defaults.h"

cv::Mat renderTelemetry(
    cv::Size imageSize, const cv::TickMeter& meter, const IncrementalVisualTracker::Ptr& tracker);

cv::Mat renderEigenbasis(
    int width, const IncrementalVisualTracker::Ptr& tracker);