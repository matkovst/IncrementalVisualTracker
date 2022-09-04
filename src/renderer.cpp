#include <iostream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "renderer.h"

namespace
{

constexpr int FontFace { cv::FONT_HERSHEY_SIMPLEX };
constexpr double FontScale { 0.7 };
constexpr int Thk { 1 };
const std::string MaxStr { "Wake up, Neo! The Matrix has you" };

const cv::Size EigvecDisplaySize { 128, 128 };

const cv::Scalar ColorGreen { 0, 255, 0 };
const cv::Scalar ColorRed { 0, 0, 255 };

}

cv::Mat renderTelemetry(int height, const cv::TickMeter& meter)
{
    const std::int64_t fno = meter.getCounter();
    const float latency = static_cast<float>(meter.getAvgTimeMilli());

    int baseline;
    const cv::Size textSize = cv::getTextSize(MaxStr, FontFace, FontScale, Thk, &baseline);
    cv::Mat output = cv::Mat::zeros(height, textSize.width, CV_8UC3);

    const cv::Point offset(0, 25);
    cv::Point pos = cv::Point(15, 0);
    cv::putText(output, "#: " + std::to_string(fno), pos += offset, FontFace, FontScale, ColorGreen, Thk);
    cv::putText(output, cv::format("latency: %.2f ms", latency), pos += offset, FontFace, FontScale, ColorGreen, Thk);

    return output;
}

cv::Mat renderEigenbasis(
    int width, const ObjectTemplate& templ, const cv::Mat& warpImage)
{
    // Post-process mean for displaying
    cv::Mat mean32f = templ.mean.reshape(0, templ.size.width).clone();
    cv::Mat mean8u;
    mean32f.convertTo(mean8u, CV_8U, 255.0);
    cv::cvtColor(mean8u, mean8u, cv::COLOR_GRAY2BGR);
    cv::resize(mean8u, mean8u, EigvecDisplaySize);
    cv::putText(mean8u, "Mean", cv::Point(5, 15), FontFace, FontScale - 0.1, ColorRed, Thk);

    // Post-process warp for displaying
    cv::Mat warpImage8u;
    warpImage.convertTo(warpImage8u, CV_8U, 255.0);
    cv::cvtColor(warpImage8u, warpImage8u, cv::COLOR_GRAY2BGR);
    cv::resize(warpImage8u, warpImage8u, EigvecDisplaySize);
    cv::putText(warpImage8u, "Warp", cv::Point(5, 15), FontFace, FontScale - 0.1, ColorRed, Thk);

    // Post-process error and reconstruction for displaying
    cv::Mat error8u, recon8u;
    if (!templ.eigbasis.empty())
    {
        const cv::Mat warpImageFlatten = warpImage.reshape(0, templ.size.area());
        const cv::Mat zeroMeanWarp = warpImageFlatten - templ.mean;
        cv::Mat UTDiff = templ.eigbasis.t() * zeroMeanWarp;
        cv::Mat error32f = zeroMeanWarp - (templ.eigbasis * (templ.eigbasis.t() * zeroMeanWarp));
        error32f = error32f.reshape(0, templ.size.width);
        
        error32f.convertTo(error8u, CV_8U, 255.0);
        cv::cvtColor(error8u, error8u, cv::COLOR_GRAY2BGR);
        cv::resize(error8u, error8u, EigvecDisplaySize);
        cv::putText(error8u, "Err", cv::Point(5, 15), FontFace, FontScale - 0.1, ColorRed, Thk);

        cv::Mat recon32f = warpImage + error32f;
        
        recon32f.convertTo(recon8u, CV_8U, 255.0);
        cv::cvtColor(recon8u, recon8u, cv::COLOR_GRAY2BGR);
        cv::resize(recon8u, recon8u, EigvecDisplaySize);
        cv::putText(recon8u, "Recon", cv::Point(5, 15), FontFace, FontScale - 0.1, ColorRed, Thk);
    }
    if (error8u.empty())
        error8u = cv::Mat::zeros(EigvecDisplaySize, CV_8UC3);
    if (recon8u.empty())
        recon8u = cv::Mat::zeros(EigvecDisplaySize, CV_8UC3);

    cv::Mat output = cv::Mat::zeros(EigvecDisplaySize.height, width, CV_8UC3);
    mean8u.copyTo(
        output(cv::Rect(
            0, 0, EigvecDisplaySize.width, EigvecDisplaySize.height)));
    warpImage8u.copyTo(
        output(cv::Rect(
            EigvecDisplaySize.width, 0, EigvecDisplaySize.width, EigvecDisplaySize.height)));
    error8u.copyTo(
        output(cv::Rect(
            EigvecDisplaySize.width*2, 0, EigvecDisplaySize.width, EigvecDisplaySize.height)));
    recon8u.copyTo(
        output(cv::Rect(
            EigvecDisplaySize.width*3, 0, EigvecDisplaySize.width, EigvecDisplaySize.height)));

    return output;
}