#pragma once

#include <opencv2/core.hpp>

struct ObjectTemplate final
{
    cv::Mat mean;       // sample mean of the images
    cv::Mat eigbasis;   // eigenbasis
    cv::Mat eigval;     // eigenvalues
    int neff { 0 };     // effective number of data
    cv::Size size;      // template image size
};

/**
 * @brief Incremental robust self-learning algorithm for visual tracking
 * 
 */
class IncrementalVisualTracker final
{
public:

    enum ErrorNorm { L2, Robust };

    /**
     * @brief Construct a new Incremental Visual Tracker object
     * 
     * @param affsig stdevs of dynamic process
     * @param nparticles number of particles
     * @param condenssig stdev of observation likelihood
     * @param forgetting forgetting factor for PCA
     * @param batchsize size of frames after which do PCA update
     * @param templShape size of object window for PCA
     * @param maxbasis number of eigenvectors for PCA
     * @param errfunc error function for minimizing the effect of noisy pixels
     */
    IncrementalVisualTracker(
        const cv::Mat& affsig, int nparticles = 100, float condenssig = 0.75f, float forgetting = 0.95f, 
        int batchsize = 5, cv::Size templShape = cv::Size(32, 32), int maxbasis = 16, int errfunc = ErrorNorm::L2);

    ~IncrementalVisualTracker();

    /**
     * @brief Initialize tracker with initial object location
     * 
     * @param image initial image
     * @param initialBox initial object location
     * 
     * @return initialized
     */
    bool init(const cv::Mat& image, cv::Rect initialBox);

    /**
     * @brief Track object location on given image
     * 
     * @param image input uchar grayscale image
     * 
     * @return estimated object location
     */
    cv::Rect track(const cv::Mat& image);

    /**
     * @brief Get object template
     */
    const ObjectTemplate& objectTemplate() const noexcept;

    /**
     * @brief Get most likely particle (warp image)
     */
    const cv::Mat& mostLikelyWarpImage() const noexcept;

private:

    /**
     * @brief CONDENSATION affine warp estimator. It looks for the most likely particle.
     * 
     * @param image input uchar grayscale image
     */
    void estimateWarpCondensation(const cv::Mat& image);

private:
    cv::Mat m_affsig;
    int m_nparticles;
    float m_condenssig;
    float m_forgetting;
    int m_batchsize;
    cv::Size m_templShape;
    int m_maxbasis;
    int m_errfunc;

    cv::Mat m_paramsLowerBound;
    int m_templDim;
    cv::Mat m_conf;
    bool m_trackerInitialized;
    ObjectTemplate m_templ;
    cv::Mat m_est;
    cv::Mat m_wimg;
    cv::Mat m_states;
    std::vector<cv::Mat> m_wimgs;

    /* Auxiliaries for optimization */
    cv::Mat m_diff;
    cv::Mat m_UTDiff;
    cv::RNG m_randomSampler;
};