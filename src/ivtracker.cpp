#include <iostream>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "sklm.h"
#include "utils.h"
#include "ivtracker.h"
#include "model_specific.h"

IncrementalVisualTracker::IncrementalVisualTracker(
    const cv::Mat& affsig, int nparticles, float condenssig, float forgetting, 
    int batchsize, cv::Size templShape, int maxbasis, int errfunc)
        : m_affsig(affsig.clone())
        , m_nparticles(nparticles)
        , m_condenssig(condenssig)
        , m_forgetting(forgetting)
        , m_batchsize(batchsize)
        , m_templShape(templShape)
        , m_maxbasis(maxbasis)
        , m_errfunc(errfunc)
        , m_templDim(0)
        , m_trackerInitialized(false)
{
    m_templDim = m_templShape.area();
    m_diff = cv::Mat::zeros(m_templDim, m_nparticles, CV_32F);
    m_conf = cv::Mat(m_nparticles, 1, CV_32F, cv::Scalar::all(1.0 / m_nparticles));
    
    m_templ.mean = cv::Mat::zeros(m_templDim, 1, CV_32F);

    const float mincx = 0.001f;
    const float mincy = 0.001f;
    const float minscale = (2.0f + std::numeric_limits<float>::epsilon()) / m_templShape.width;
    const float minar = 0.00001f;
    m_paramsLowerBound = (cv::Mat_<float>(4, 1) << mincx, mincy, minscale, minar);
}

IncrementalVisualTracker::~IncrementalVisualTracker() = default;

bool IncrementalVisualTracker::init(const cv::Mat& image, cv::Rect initialBox)
{
    if (m_trackerInitialized)
        return true;

    if (initialBox.empty())
        throw std::runtime_error("IncrementalVisualTracker::init: Given empty box");

    // Make initial state parameters
    const float cx = initialBox.x + initialBox.width / 2.0f;
    const float cy = initialBox.y + initialBox.height / 2.0f;
    const float scale = initialBox.width / float(m_templShape.width);
    const float aspectRatio = initialBox.height / float(initialBox.width);
    const cv::Mat state0 = (cv::Mat_<float>(1, 4) << cx, cy, scale, aspectRatio);

    const auto mean2d = warpImg(image, state0, m_templShape);
    m_templ.mean = mean2d.reshape(0, m_templDim).clone();
    m_est = state0.clone();
    m_wimg = mean2d.clone();
    m_trackerInitialized = true;

    return true;
}

cv::Rect IncrementalVisualTracker::track(const cv::Mat& image)
{
    if (!m_trackerInitialized)
        return cv::Rect();

    // Do the condensation magic and find the most likely location
    estimateWarpCondensation(image);

    // Do incremental update when we accumulate enough data
    if (m_wimgs.size() >= m_batchsize)
    {
        if (m_UTDiff.empty())
        {
            cv::Mat eigbasis, eigval, mean;
            int neff;
            sklm(
                m_wimgs, 
                m_templ.eigbasis, 
                m_templ.eigval, 
                m_templ.mean, 
                m_templ.neff, 
                m_forgetting, 
                eigbasis, 
                eigval, 
                mean, 
                neff);
            cv::swap(m_templ.eigbasis, eigbasis);
            cv::swap(m_templ.eigval, eigval);
            cv::swap(m_templ.mean, mean);
            m_templ.neff = neff;
        }
        else
        {
            cv::Mat recon = m_templ.eigbasis * m_UTDiff;
            for (int i = 0; i < recon.cols; ++i)
                recon.col(i) += m_templ.mean;

            cv::Mat eigbasis, eigval, mean;
            int neff;
            sklm(
                m_wimgs, 
                m_templ.eigbasis, 
                m_templ.eigval, 
                m_templ.mean, 
                m_templ.neff, 
                m_forgetting, 
                eigbasis, 
                eigval, 
                mean, 
                neff);
            cv::swap(m_templ.eigbasis, eigbasis);
            cv::swap(m_templ.eigval, eigval);
            cv::swap(m_templ.mean, mean);
            m_templ.neff = neff;

            for (int i = 0; i < recon.cols; ++i)
                recon.col(i) -= m_templ.mean;
            m_UTDiff = m_templ.eigbasis.t() * recon;
        }

        m_wimgs.clear();

        const int nCurrentEigenvectors = m_templ.eigbasis.cols;
        if (nCurrentEigenvectors > m_maxbasis)
        {
            m_templ.eigbasis = m_templ.eigbasis.colRange(0, m_maxbasis).clone();
            m_templ.eigval = m_templ.eigval.rowRange(0, m_maxbasis).clone();
            if (!m_UTDiff.empty())
                m_UTDiff = m_UTDiff.rowRange(0, m_maxbasis).clone();
        }
    }

    return state2Rect(m_est, m_templShape);
}

void IncrementalVisualTracker::estimateWarpCondensation(const cv::Mat& image)
{
    /* Propagate density */
    if (m_states.empty()) // the first iteration. Just tile initial template
    {
        cv::repeat(m_est, m_nparticles, 1, m_states);
    }
    else
    {
        const cv::Mat cumconf = cumsum(m_conf);
        const cv::Mat cumconfNN = cv::repeat(cumconf, 1, m_nparticles);

        cv::Mat uniformN(1, m_nparticles, CV_32F);
        cv::randu(uniformN, 0.0f, 1.0f);
        const cv::Mat uniformNN = cv::repeat(uniformN, m_nparticles, 1);

        cv::Mat sumMask, cdfIds;
        cv::compare(uniformNN, cumconfNN, sumMask, cv::CMP_GT);
        cv::reduce(sumMask, cdfIds, 0, cv::REDUCE_SUM, CV_32F);
        cv::multiply(cdfIds, 0.003921569f, cdfIds, 1.0, CV_32F);

        cv::Mat cdfSamples = matRowIndexing(m_states, cdfIds);
        cv::swap(m_states, cdfSamples);
    }

    /* Apply dynamical model */
    BrownianMotion(m_states, m_affsig, m_paramsLowerBound); // (in-place)

    /* Apply observation model */

    // Retrieve image patches It predicated by Xt
    const auto wimgsFlatten = warpImg(image, m_states, m_templShape, true);
    for (int i = 0; i < m_nparticles; ++i)
    {
        const cv::Mat zeroMean = wimgsFlatten.col(i) - m_templ.mean;
        zeroMean.copyTo(m_diff.col(i));
    }

    // Compute likelihood under the observation model for each patch
    const int nCurrentEigenvectors = m_templ.eigbasis.cols;
    if (nCurrentEigenvectors > 0)
    {
        // Compute (I - mu) - UU.T(I - mu)
        cv::Mat UTDiff = m_templ.eigbasis.t() * m_diff;
        m_diff -= (m_templ.eigbasis * UTDiff);
        cv::swap(UTDiff, m_UTDiff);
    }

    cv::Mat diff2;
    cv::pow(m_diff, 2, diff2);
    const float prec = 1.0f / m_condenssig;
    switch (m_errfunc)
    {
    case ErrorNorm::Robust:
        {
            const float rsig = 0.1f;
            cv::Mat rho = diff2 / (diff2 + rsig);
            cv::Mat diff2sum;
            cv::reduce(rho, diff2sum, 0, cv::REDUCE_SUM, CV_32F);
            cv::multiply(diff2sum, -prec, diff2sum, 1.0, CV_32F);
            cv::exp(diff2sum.t(), m_conf);
        }
        break;
    case ErrorNorm::L2:
        {
            cv::Mat diff2sum;
            cv::reduce(diff2, diff2sum, 0, cv::REDUCE_SUM, CV_32F);
            cv::multiply(diff2sum, -prec, diff2sum, 1.0, CV_32F);
            cv::exp(diff2sum.t(), m_conf);
        }
        break;
    }

    /* Store most likely particle */
    const float normCoeff = 1.0 / cv::sum(m_conf)[0];
    cv::multiply(m_conf, normCoeff, m_conf, 1.0, CV_32F);
    double maxProb;
    int maxProbIdx;
    cv::minMaxIdx(m_conf, nullptr, &maxProb, nullptr, &maxProbIdx);
    m_est = m_states.row(maxProbIdx).clone();
    const cv::Mat maxProbWimgFlatten = wimgsFlatten.col(maxProbIdx).clone();
    m_wimg = maxProbWimgFlatten.reshape(0, m_templShape.width).clone();
    cv::transpose(m_wimg, m_wimg);

    m_wimgs.emplace_back(maxProbWimgFlatten.clone());
}