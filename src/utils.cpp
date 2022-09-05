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
        const auto cx = state.at<PRECISION>(0);
        const auto cy = state.at<PRECISION>(1);
        const auto scale = state.at<PRECISION>(2);
        const auto aspectRatio = state.at<PRECISION>(3);
        
        const auto width = static_cast<PRECISION>(scale * targetSize.width);
        const auto height = static_cast<PRECISION>(aspectRatio * width);
        const auto x1 = cx - (width / PRECISION(2.0));
        const auto y1 = cy - (height / PRECISION(2.0));
        const auto x2 = x1 + width;
        const auto y2 = y1 + height;
        cv::Rect_<PRECISION> warpBox(x1, y1, width, height);
        if (x1 < 0)
            warpBox += cv::Point_<PRECISION>(std::abs(x1), 0);
        if (x2 >= image.cols)
            warpBox -= cv::Point_<PRECISION>(x2 - image.cols + 1, 0);
        if (y1 < 0)
            warpBox += cv::Point_<PRECISION>(0, std::abs(y1));
        if (y2 >= image.rows)
            warpBox -= cv::Point_<PRECISION>(0, y2 - image.rows + 1);

        cv::Mat crop = image(cv::Rect2i(warpBox)).clone();
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
        warp8u.convertTo(output, CV_PRECISION);
        return output;
    }

    if (flatten)
    {
        cv::Mat output(targetSize.area(), nStates, CV_PRECISION);
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
    cv::MatConstIterator_<PRECISION> imageIt = image.begin<PRECISION>();
    cv::MatIterator_<PRECISION> outputIt = output.begin<PRECISION>();
    for (PRECISION sum = PRECISION(0.0); imageIt != image.end<PRECISION>(); ++imageIt, ++outputIt)
        (*outputIt) = (sum += (*imageIt));

    return output;
}

cv::Mat matRowIndexing(const cv::Mat& image, const cv::Mat& rowIds)
{
    cv::Mat output = cv::Mat(image.size(), image.type());
    cv::MatConstIterator_<PRECISION> rowIdsIt = rowIds.begin<PRECISION>();
    for (int i = 0; i < image.rows; ++i, ++rowIdsIt)
    {
        const int rowId = static_cast<int>(std::floor(*rowIdsIt));
        image.row(rowId).copyTo(output.row(i));
    }

    return output;
}

cv::Rect state2Rect(const cv::Mat& state, cv::Size targetSize)
{
    const auto cx = state.at<PRECISION>(0);
    const auto cy = state.at<PRECISION>(1);
    const auto scale = state.at<PRECISION>(2);
    const auto aspectRatio = state.at<PRECISION>(3);

    const auto width = static_cast<PRECISION>(scale * targetSize.width);
    const auto height = static_cast<PRECISION>(aspectRatio * width);
    const auto x1 = cx - width / PRECISION(2.0);
    const auto y1 = cy - height / PRECISION(2.0);
    const auto x2 = x1 + width;
    const auto y2 = y1 + height;
    return cv::Rect(int(x1), int(y1), int(width), int(height));
}