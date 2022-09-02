#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "src/ivtracker.h"
#include "src/renderer.h"


constexpr int Nparticles { 400 };
constexpr float Condenssig { 0.75f };
constexpr float ForgettingFactor { 0.99f };
constexpr int Batchsize { 5 };
constexpr int Maxbasis { 16 };
const cv::Mat Affsig { (cv::Mat_<float>(4, 1) << 5.0f, 5.0f, 0.02f, 0.002f) };
const cv::Size TemplSize { 32, 32 };

const std::string WinName = "Incremental Visual Tracker Demo";
const cv::String CommandLineParams =
    "{ help usage h ?   |         | print help }"
    "{ @input i         |  0      | input stream }"
    "{ input_scale      |  1.0    | input resolution scale }"

    // Tracker debug params
    "{ init_x           |  0.0    | initial object x-coord }"
    "{ init_y           |  0.0    | initial object y-coord }"
    "{ init_w           |  0.0    | initial object w-coord }"
    "{ init_h           |  0.0    | initial object h-coord }"
    ;


cv::Mat preprocessImage(const cv::Mat& image)
{
    cv::Mat grayImage, grayImage32f;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    grayImage.convertTo(grayImage32f, CV_32F, 0.003921569);
    return grayImage32f;
}


int main(int argc, char** argv)
{
    std::cout << "Program started" << std::endl;

    /* Check and parse cmd args */
    cv::CommandLineParser parser(argc, argv, CommandLineParams);
    parser.about(WinName);
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }
    if (!parser.check())
    {
        parser.printErrors();
        return EXIT_FAILURE;
    }
    const auto input = parser.get<std::string>("@input");
    const auto inputScale = parser.get<double>("input_scale");
    const auto initx = parser.get<float>("init_x");
    const auto inity = parser.get<float>("init_y");
    const auto initw = parser.get<float>("init_w");
    const auto inith = parser.get<float>("init_h");
    const cv::Rect2f initialBoxf(initx, inity, initw, inith);
    const bool hasInitialBox = (initx * inity * initw * inith) != 0;

    /* Capture input */    
    cv::VideoCapture capture;
    if ("0" == input)
        capture.open(0);
    else
        capture.open(input);
    if (!capture.isOpened())
    {
        std::cerr << "Could not open video" << std::endl;
        return EXIT_FAILURE;
    }
    cv::Mat frame0;
    capture >> frame0;
    if (frame0.empty())
    {
        std::cerr << "Empty frame" << std::endl;
        return EXIT_FAILURE;
    }
    if (1.0 != inputScale)
        cv::resize(frame0, frame0, cv::Size(), inputScale, inputScale);

    const cv::Rect initialBox(
        static_cast<int>(initialBoxf.x * frame0.cols), 
        static_cast<int>(initialBoxf.y * frame0.rows),
        static_cast<int>(initialBoxf.width * frame0.cols), 
        static_cast<int>(initialBoxf.height * frame0.rows)
    );

    /* Initialize tracker */
    IncrementalVisualTracker tracker(
        Affsig, Nparticles, Condenssig, ForgettingFactor, Batchsize, TemplSize, Maxbasis, 
        IncrementalVisualTracker::ErrorNorm::Robust);

    cv::Mat preprocFrame0 = preprocessImage(frame0);
    if (hasInitialBox)
    {
        std::cout << "Given initial box: " << initialBox << std::endl;
        if (!tracker.init(preprocFrame0, initialBox))
        {
            std::cerr << "Could not initialize tracker" << std::endl;
            return EXIT_FAILURE;
        }
    }

    /* Start main loop */
    cv::Mat frame = frame0;
    cv::Mat preprocFrame = preprocFrame0;
    std::int64_t frameNum = 1;
    cv::TickMeter meter;
    while (capture.isOpened())
    {
        /* Analytical core */
        meter.start();
        const auto estBoundingBox = tracker.track(preprocFrame);
        meter.stop();

        /* Render results */
        cv::rectangle(frame, estBoundingBox, cv::Scalar(0, 0, 255), 2);
        const auto telemetryPanel = renderTelemetry(frame.rows, meter);
        cv::Mat detailedFrame;
        cv::hconcat(frame, telemetryPanel, detailedFrame);
        cv::imshow(WinName, detailedFrame);
        // cv::waitKey(500);

        const auto key = static_cast<char>(cv::waitKey(15));
        if (27 == key || 'q' == key)
            break;

        /* Advance to the next frame */
        ++frameNum;
        capture >> frame;
        if (frame.empty())
            break;
        if (1.0 != inputScale)
            cv::resize(frame, frame, cv::Size(), inputScale, inputScale);
        preprocFrame = preprocessImage(frame);
    }

    capture.release();
    cv::destroyAllWindows();
    
    std::cout << "Program finished successfully" << std::endl;
    return 0;
}
