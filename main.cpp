#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "src/defaults.h"
#include "src/ivtracker.h"
#include "src/renderer.h"


constexpr PRECISION Condenssig { PRECISION(0.75) };
constexpr PRECISION ForgettingFactor { PRECISION(0.99) };
const cv::Mat Affsig {
    (cv::Mat_<PRECISION>(4, 1) << PRECISION(10.0), PRECISION(10.0), PRECISION(0.05), PRECISION(0.02))};
const cv::Size TemplSize { 32, 32 };

const std::string WinName = "Incremental Visual Tracker Demo";
const cv::String CommandLineParams =
    "{ help usage h ?   |         | print help }"
    "{ @input i         |  0      | input stream }"
    "{ input_scale      |  1.0    | input resolution scale }"

    // Tracker debug params
    "{ nparticles       |  600    | number of particles }"
    "{ maxbasis         |  16     | effective number of eigenvectors }"
    "{ batchsize        |  5      | observation batch size used for updating I-PCA }"

    // Tracker debug params
    "{ init_x           |  0.0    | initial object x-coord }"
    "{ init_y           |  0.0    | initial object y-coord }"
    "{ init_w           |  0.0    | initial object w-coord }"
    "{ init_h           |  0.0    | initial object h-coord }"
    "{ record           |         | record video }"
    ;


cv::Mat preprocessImage(const cv::Mat& image)
{
    cv::Mat grayImage, grayImagef;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    grayImage.convertTo(grayImagef, CV_PRECISION, 0.003921569);
    return grayImagef;
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
    const auto nParticles = parser.get<int>("nparticles");
    const auto maxbasis = parser.get<int>("maxbasis");
    const auto batchsize = parser.get<int>("batchsize");
    const auto initx = parser.get<float>("init_x");
    const auto inity = parser.get<float>("init_y");
    const auto initw = parser.get<float>("init_w");
    const auto inith = parser.get<float>("init_h");
    const auto recordName = parser.get<std::string>("record");
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
    
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (fps > 30.0)
        fps = 30.0;

    /* (optional) Initialize video writer */
    cv::VideoWriter writer;

    /* Create tracker */
    IncrementalVisualTracker::Ptr tracker = std::make_shared<IncrementalVisualTracker>(
        Affsig, nParticles, Condenssig, ForgettingFactor, batchsize, TemplSize, maxbasis, 
        IncrementalVisualTracker::ErrorNorm::Robust);

    /* Set initial box */
    cv::Rect initialBox;
    if (hasInitialBox)
    {
        initialBox = cv::Rect(
            static_cast<int>(initialBoxf.x * frame0.cols), 
            static_cast<int>(initialBoxf.y * frame0.rows),
            static_cast<int>(initialBoxf.width * frame0.cols), 
            static_cast<int>(initialBoxf.height * frame0.rows)
        );

        std::cout << "Given initial box: " << initialBox << std::endl;
    }
    else // Allow user to select initial box manually
    {
        initialBox = cv::selectROI(WinName, frame0);
        if (initialBox.empty())
        {
            std::cerr << "Program cancelled" << std::endl;
            return EXIT_SUCCESS;
        }
        std::cout << "Initial box selected manually: " << initialBox << std::endl;
    }

    /* Initialize tracker */
    cv::Mat preprocFrame0 = preprocessImage(frame0);
    if (!tracker->init(preprocFrame0, initialBox))
    {
        std::cerr << "Could not initialize tracker" << std::endl;
        return EXIT_FAILURE;
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
        const auto estBoundingBox = tracker->track(preprocFrame);
        meter.stop();

        /* Render results */
        cv::rectangle(frame, estBoundingBox, cv::Scalar(0, 0, 255), 2);
        const auto telemetryPanel = renderTelemetry(
            frame.size(), meter, tracker);
        const auto eigenPanel = renderEigenbasis(
            frame.cols + telemetryPanel.cols, tracker);
        cv::Mat detailedFrame, detailedFrameTop;
        cv::hconcat(frame, telemetryPanel, detailedFrameTop);
        cv::vconcat(detailedFrameTop, eigenPanel, detailedFrame);
        cv::imshow(WinName, detailedFrame);

        // (optional) Record video
        if (!recordName.empty() && !writer.isOpened())
        {
            writer.open(
                recordName + ".avi",
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                fps,
                detailedFrame.size(),
                true
            );
        }
        if (writer.isOpened())
            writer.write(detailedFrame);

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

    if (capture.isOpened())
        capture.release();
    if (writer.isOpened())
        writer.release();
    cv::destroyAllWindows();
    
    std::cout << "Program finished successfully" << std::endl;
    return 0;
}
