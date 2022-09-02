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

const cv::Scalar ColorGreen { 0, 255, 0 };

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
    cv::putText(output, cv::format("latency: %.2f", latency), pos += offset, FontFace, FontScale, ColorGreen, Thk);

    return output;
}