#include <cmath>
#include <iostream>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include "utils.h"


cv::Mat warpImg(const cv::Mat& image, const cv::Mat& states, cv::Size targetSize, bool flatten)
{
    if (image.empty())
        throw std::runtime_error("warpImg: Given empty image");

    auto makeWarp = [](
        const cv::Mat& image, const cv::Mat& state, cv::Size targetSize, bool flatten)
    {
        const float cx = state.at<float>(0);
        const float cy = state.at<float>(1);
        const float scale = state.at<float>(2);
        const float aspectRatio = state.at<float>(3);
        
        const float width = scale * targetSize.width;
        const float height = aspectRatio * width;
        const float x1 = cx - width / 2.0f;
        const float y1 = cy - height / 2.0f;
        const float x2 = x1 + width;
        const float y2 = y1 + height;
        cv::Rect2f warpBox(x1, y1, width, height);
        if (x1 < 0)
            warpBox += cv::Point2f(std::abs(x1), 0);
        if (x2 >= image.cols)
            warpBox -= cv::Point2f(x2 - image.cols + 1, 0);
        if (y1 < 0)
            warpBox += cv::Point2f(0, std::abs(y1));
        if (y2 >= image.rows)
            warpBox -= cv::Point2f(0, y2 - image.rows + 1);

        cv::Mat crop = image(cv::Rect2i(warpBox));
        cv::resize(crop, crop, targetSize, 0.0, 0.0, cv::INTER_LINEAR);
        if (flatten)
            crop = crop.reshape(0, targetSize.area());
        return crop;
    };

    const int nStates = states.rows;
    const int nParams = states.cols;

    if (1 == nStates) // single warp
    {
        const auto warp8u = makeWarp(image, states, targetSize, flatten);
        cv::Mat output;
        warp8u.convertTo(output, CV_32F);
        return output;
    }

    if (flatten)
    {
        cv::Mat output(targetSize.area(), nStates, CV_32F);
        for (int i = 0; i < nStates; ++i)
            makeWarp(image, states.row(i), targetSize, flatten)
                .copyTo(output.col(i));

        return output;
    }

    return cv::Mat(); // TODO
}

cv::Mat cumsum(const cv::Mat& image)
{
    cv::Mat output(image.size(), image.type());
    cv::MatConstIterator_<float> imageIt = image.begin<float>();
    cv::MatIterator_<float> outputIt = output.begin<float>();
    for (float sum = 0.0f; imageIt != image.end<float>(); ++imageIt, ++outputIt)
        (*outputIt) = (sum += (*imageIt));

    return output;
}

cv::Mat matRowIndexing(const cv::Mat& image, const cv::Mat& rowIds)
{
    cv::Mat output = cv::Mat(image.size(), image.type());
    cv::MatConstIterator_<float> rowIdsIt = rowIds.begin<float>();
    for (int i = 0; i < image.rows; ++i, ++rowIdsIt)
    {
        const int rowId = static_cast<int>(std::floor(*rowIdsIt));
        image.row(rowId).copyTo(output.row(i));
    }

    return output;
}

cv::Rect state2Rect(const cv::Mat& state, cv::Size targetSize)
{
    const float cx = state.at<float>(0);
    const float cy = state.at<float>(1);
    const float scale = state.at<float>(2);
    const float aspectRatio = state.at<float>(3);

    const float width = scale * targetSize.width;
    const float height = aspectRatio * width;
    const float x1 = cx - width / 2.0f;
    const float y1 = cy - height / 2.0f;
    const float x2 = x1 + width;
    const float y2 = y1 + height;
    return cv::Rect(int(x1), int(y1), int(width), int(height));
}