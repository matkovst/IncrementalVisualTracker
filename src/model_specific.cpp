#include <opencv2/imgproc.hpp>
#include "model_specific.h"

namespace
{

cv::RNG NormalSampler;

}

void BrownianMotion(cv::Mat& states, const cv::Mat& noise, const cv::Mat& lowerBound)
{
    const int nParams = noise.total();
    const int nSamples = states.rows;
    for (int i = 0; i < nParams; ++i)
    {
        cv::Mat perturbed = cv::Mat::zeros(nSamples, 1, CV_32F);
        NormalSampler.fill(perturbed, cv::RNG::NORMAL, 0, noise.at<float>(i));
        states.col(i) += perturbed;
        states.col(i) = cv::max(states.col(i), lowerBound.at<float>(i));
    }
}