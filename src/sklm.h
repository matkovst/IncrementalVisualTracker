#pragma once

#include <vector>
#include <opencv2/core.hpp>

/**
 * @brief Sequential Karhunen-Loeve Transform.
 * 
 * @param data new observations
 * @param U0 old eigenbasis
 * @param D0 old singular values
 * @param mu0 old sample mean
 * @param n number of previous data
 * @param ff forgetting factor
 * 
 * @param U new eigenbasis
 * @param D new singular values
 * @param mu new sample mean
 * @param neff new number of data
 */
void sklm(
    const std::vector<cv::Mat>& data, const cv::Mat& U0, const cv::Mat& D0, const cv::Mat& mu0, int n, float ff, 
    cv::Mat& U, cv::Mat& D, cv::Mat& mu, int& neff);