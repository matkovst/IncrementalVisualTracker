#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>

#include "defaults.h"

/**
 *
 * Some helpful nomenclature:
 *  d - dimentionality of input data
 *  k - effective number of eigenvectors
 *  Np - number of particles
 *  Na - number of state parameters
 *  Nb - observation batch size used for updating I-PCA
 * 
 *  I - image patch     (d x 1)
 *  mu - sample mean    (d x 1)
 *  U - eigenvectors    (d x k)
 * 
 */

struct ObjectTemplate final
{
    cv::Mat mean;       // sample mean of the images    (d x 1)
    cv::Mat eigbasis;   // eigenbasis                   (d x k)
    cv::Mat eigval;     // eigenvalues                  (k x 1)
    int neff { 0 };     // effective number of data
    cv::Size size;      // template image size
    double prob { 0.0 };// probability under the observation model
};

struct Estimation final
{
    cv::Rect position;
    double confidence { 0.0 };
};

/**
 * @brief Incremental robust self-learning algorithm for visual tracking
 * 
 */
class IncrementalVisualTracker final
{
public:

    enum ErrorNorm { L2, Robust, Ppca };
    using Ptr = std::shared_ptr<IncrementalVisualTracker>;

    const int d;    // dimentionality of input data
    const int Np;   // number of particles
    const int k;    // effective number of eigenvectors (the first top k eigenvectors)

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
     * @param robustThr reject region for robust norm
     */
    IncrementalVisualTracker(
        const cv::Mat& affsig, 
        int nparticles = 100, 
        PRECISION condenssig = PRECISION(0.75), 
        PRECISION forgetting = PRECISION(0.95), 
        int batchsize = 5, 
        cv::Size templShape = cv::Size(32, 32), 
        int maxbasis = 16, 
        double robustThr = 0.1);

    ~IncrementalVisualTracker();

    /**
     * @brief Initialize tracker with initial object location
     * 
     * @param image initial image (grayscaled, float32/64, normalized from 0 to 1)
     * @param initialBox initial object location
     * 
     * @return initialized
     */
    bool init(const cv::Mat& image, cv::Rect initialBox);

    /**
     * @brief Track object location on given image
     * 
     * @param image input image (grayscaled, float32/64, normalized from 0 to 1)
     * 
     * @return estimated object location + confidence
     */
    Estimation track(const cv::Mat& image);


    const cv::Mat& stateConfidences() const noexcept;

    const cv::Mat& states() const noexcept;

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

    /* Parameters governing the algorithm */

    cv::Mat m_affsig;               // stdevs of affine parameters (Na x 1)
    PRECISION m_condenssig;         // stdev of the observation likelihood
    PRECISION m_forgetting;         // forgetting factor
    int m_batchsize;                // number of observations used for eigenbasis learning
    cv::Size m_templShape;          // 2d shape of object template
    double m_robustThr;             // reject region for robust norm
    int m_errfunc;                  // error function used for distance-to-subspace estimation

    /* Program data */

    bool m_trackerInitialized;          // tracker init code
    cv::Mat m_paramsLowerBound;         // min possible values for state parameters
    cv::Mat m_stateConfidences;         // particle confidences (Nb x 1)
    cv::Mat m_mostLikelyState;          // the state/particle with highest probability under the observation model
    cv::Mat m_mostLikelyWarpImage;      // warp image corresponding to the most likely state
    cv::Mat m_states;                   // states (Np x Na)
    ObjectTemplate m_templ;             // predicted object template
    std::vector<cv::Mat> m_warpBatch;   // batch of last Nb observation

    /* Auxilary data for optimization */

    cv::Mat m_residual;         // residual/error                               (d x Np)
    cv::Mat m_backProj;         // back projection (projected on U and back)    (k x Nb)
    cv::RNG m_randomSampler;    // particle random generator
};