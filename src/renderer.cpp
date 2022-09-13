#include <iostream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "utils.h"
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

void renderEstimation(cv::Mat& image, const Estimation& est, double rejectThr)
{
    cv::rectangle(image, est.position, ColorRed, Thk+1);
    cv::putText(
        image, 
        cv::format("conf: %.2f", float(est.confidence)), 
        est.position.tl() - cv::Point(5, 20), 
        FontFace, 
        FontScale - 0.1, 
        ColorRed, 
        Thk);
}

cv::Mat renderTelemetry(
    cv::Size imageSize, const cv::TickMeter& meter, const IncrementalVisualTracker::Ptr& tracker)
{
    const auto& templ = tracker->objectTemplate();
    const auto& states = tracker->states();
    const auto& stateConfidences = tracker->stateConfidences();

    const std::int64_t fno = meter.getCounter();
    const auto latency = static_cast<float>(meter.getAvgTimeMilli());

    cv::Scalar probColor = ColorGreen;
    if (templ.prob <= 0.0 + std::numeric_limits<double>::epsilon())
        probColor = ColorRed;

    cv::Mat output = cv::Mat::zeros(imageSize, CV_8UC3);

    cv::Mat sortedStates;
    cv::sortIdx(states, sortedStates, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
    cv::Mat stateConfidences8u;
    stateConfidences.convertTo(stateConfidences8u, CV_8U, 255.0);
    cv::Mat coloredStateConfidences;
    cv::applyColorMap(stateConfidences8u, coloredStateConfidences, cv::COLORMAP_JET);
    for (int i = 0; i < tracker->Np; ++i)
    {
        const int idx = sortedStates.at<int>(i);
        const auto& state = states.row(idx);
        const auto& stateConf = stateConfidences.at<PRECISION>(idx);
        const auto cx = static_cast<int>(state.at<PRECISION>(0));
        const auto cy = static_cast<int>(state.at<PRECISION>(1));
        const auto stateRect = state2Rect(state, templ.size);
        if (stateConf > 0.0)
        {
            const auto& colorPix = coloredStateConfidences.at<cv::Vec3b>(idx);
            cv::rectangle(output, stateRect, cv::Scalar(colorPix));
            cv::circle(output, cv::Point(cx, cy), 1, cv::Scalar(colorPix));
        }
        else
        {
            cv::circle(output, cv::Point(cx, cy), 1, cv::Scalar::all(127));
        }
    }

    const cv::Point offset(0, 25);
    cv::Point pos = cv::Point(15, 0);
    cv::putText(output, "#: " + std::to_string(fno), pos += offset, FontFace, FontScale, ColorGreen, Thk);
    cv::putText(output, cv::format("latency: %.2f ms", latency), pos += offset, FontFace, FontScale, ColorGreen, Thk);
    cv::putText(output, cv::format("p(x): %.2f", float(templ.prob)), pos += offset, FontFace, FontScale - 0.1, probColor, Thk);

    return output;
}

cv::Mat renderEigenbasis(
    int width, const IncrementalVisualTracker::Ptr& tracker)
{
    const auto& templ = tracker->objectTemplate();
    const auto& warpImage32f = tracker->mostLikelyWarpImage();

    auto postprocessImage = [](const cv::Mat& image){
        cv::Mat output;
        image.convertTo(output, CV_8U, 255.0);
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
        cv::resize(output, output, EigvecDisplaySize);
        return output;
    };

    // Post-process mean for displaying
    cv::Mat mean32f = templ.mean.reshape(0, templ.size.width).clone();
    cv::Mat mean8u = postprocessImage(mean32f);
    cv::putText(mean8u, "Mean", cv::Point(5, 15), FontFace, FontScale - 0.1, ColorRed, Thk);

    // Post-process warp for displaying
    cv::Mat warpImage8u = postprocessImage(warpImage32f);
    cv::putText(warpImage8u, "Warp", cv::Point(5, 15), FontFace, FontScale - 0.1, ColorRed, Thk);

    // Post-process error and reconstruction for displaying
    cv::Mat error8u, recon8u;
    if (!templ.eigbasis.empty())
    {
        const cv::Mat warpImageFlatten = warpImage32f.reshape(0, templ.size.area());
        const cv::Mat zeroMeanWarp = warpImageFlatten - templ.mean;
        cv::Mat error32f = zeroMeanWarp - (templ.eigbasis * (templ.eigbasis.t() * zeroMeanWarp));
        error32f = error32f.reshape(0, templ.size.width);

        // cv::Mat errorSq;
        // cv::pow(error32f, 2, errorSq);
        // const double rmse = std::sqrt(cv::sum(errorSq)[0]);
        
        error8u = postprocessImage(error32f);
        cv::putText(error8u, "Err", cv::Point(5, 15), FontFace, FontScale - 0.1, ColorRed, Thk);

        cv::Mat recon32f = warpImage32f + error32f;
        
        recon8u = postprocessImage(recon32f);
        cv::putText(recon8u, "Recon", cv::Point(5, 15), FontFace, FontScale - 0.1, ColorRed, Thk);
    }
    if (error8u.empty())
        error8u = cv::Mat::zeros(EigvecDisplaySize, CV_8UC3);
    if (recon8u.empty())
        recon8u = cv::Mat::zeros(EigvecDisplaySize, CV_8UC3);

    // Stack all images
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