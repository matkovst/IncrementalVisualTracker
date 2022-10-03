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
    const cv::Mat& affsig, int nparticles, PRECISION condenssig, PRECISION forgetting, 
    int batchsize, cv::Size templShape, int maxbasis, double robustThr)
        : m_affsig(affsig.clone())
        , m_condenssig(condenssig)
        , m_forgetting(forgetting)
        , m_batchsize(batchsize)
        , m_templShape(templShape)
        , m_robustThr(robustThr)
        , m_errfunc((robustThr > 0.0) ? Robust : L2)
        , m_trackerInitialized(false)
        , d(templShape.area())
        , Np(nparticles)
        , k(maxbasis)
{
    m_residual = cv::Mat::zeros(d, Np, CV_PRECISION);
    m_stateConfidences = cv::Mat(Np, 1, CV_PRECISION, cv::Scalar::all(1.0 / Np));
    
    m_templ.mean = cv::Mat::zeros(d, 1, CV_PRECISION);
    m_templ.size = m_templShape;

    const auto mincx = static_cast<PRECISION>(0.001);
    const auto mincy = static_cast<PRECISION>(0.001);
    const auto minscale = (static_cast<PRECISION>(2.0) + std::numeric_limits<PRECISION>::epsilon()) / m_templShape.width;
    const auto minar = static_cast<PRECISION>(0.00001);
    m_paramsLowerBound = (cv::Mat_<PRECISION>(4, 1) << mincx, mincy, minscale, minar);
}

IncrementalVisualTracker::~IncrementalVisualTracker() = default;

bool IncrementalVisualTracker::init(const cv::Mat& image, cv::Rect initialBox)
{
    if (m_trackerInitialized)
        return true;

    if (initialBox.empty())
        throw std::runtime_error("IncrementalVisualTracker::init: Given empty box");

    // Make initial state parameters
    const auto cx = initialBox.x + initialBox.width / static_cast<PRECISION>(2.0);
    const auto cy = initialBox.y + initialBox.height / static_cast<PRECISION>(2.0);
    const auto scale = initialBox.width / static_cast<PRECISION>(m_templShape.width);
    const auto aspectRatio = initialBox.height / static_cast<PRECISION>(initialBox.width);
    const cv::Mat state0 = (cv::Mat_<PRECISION>(1, 4) << cx, cy, scale, aspectRatio);

    const auto mean2d = warpImg(image, state0, m_templShape);
    m_templ.mean = mean2d.reshape(0, d).clone();
    m_mostLikelyState = state0.clone();
    m_mostLikelyWarpImage = mean2d.clone();
    m_trackerInitialized = true;

    return true;
}

Estimation IncrementalVisualTracker::track(const cv::Mat& image)
{
    if (!m_trackerInitialized)
        return Estimation();
    // Do the condensation magic and find the most likely location
    estimateWarpCondensation(image);

    // Do incremental update when we accumulate enough data
    if (m_warpBatch.size() >= m_batchsize)
    {
        if (m_backProj.empty())
        {
            cv::Mat eigbasis, eigval, mean;
            int neff;
            sklm(
                m_warpBatch, 
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
            cv::Mat recon = m_templ.eigbasis * m_backProj;
            for (int i = 0; i < recon.cols; ++i)
                recon.col(i) += m_templ.mean;

            cv::Mat eigbasis, eigval, mean;
            int neff;
            sklm(
                m_warpBatch, 
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
            m_backProj = m_templ.eigbasis.t() * recon;
        }

        m_warpBatch.clear();

        const int nCurrentEigenvectors = m_templ.eigbasis.cols;
        if (nCurrentEigenvectors > k)
        {
            m_templ.eigbasis = m_templ.eigbasis.colRange(0, k).clone();
            m_templ.eigval = m_templ.eigval.rowRange(0, k).clone();
            if (!m_backProj.empty())
                m_backProj = m_backProj.rowRange(0, k).clone();
        }
    }

    Estimation est;
    est.position = state2Rect(m_mostLikelyState, m_templShape);
    est.confidence = m_templ.prob;
    return est;
}

void IncrementalVisualTracker::estimateWarpCondensation(const cv::Mat& image)
{
    /* Propagate density */
    if (m_states.empty()) // the first iteration. Just tile initial template
    {
        cv::repeat(m_mostLikelyState, Np, 1, m_states);
    }
    else
    {
        const cv::Mat cumconf = cumsum(m_stateConfidences);
        const cv::Mat cumconfNN = cv::repeat(cumconf, 1, Np);

        cv::Mat uniformN(1, Np, CV_PRECISION);
        m_randomSampler.fill(
            uniformN, cv::RNG::UNIFORM, static_cast<PRECISION>(0.0), static_cast<PRECISION>(0.99));
        const cv::Mat uniformNN = cv::repeat(uniformN, Np, 1);

        cv::Mat sumMask, cdfIds;
        cv::compare(uniformNN, cumconfNN, sumMask, cv::CMP_GT);
        cv::reduce(sumMask, cdfIds, 0, cv::REDUCE_SUM, CV_PRECISION);
        cv::multiply(cdfIds, static_cast<PRECISION>(0.003921569), cdfIds, 1.0, CV_PRECISION);

        cv::Mat cdfSamples = matRowIndexing(m_states, cdfIds);
        cv::swap(m_states, cdfSamples);
    }

    /* Apply dynamical model */
    BrownianMotion(m_states, m_affsig, m_paramsLowerBound); // (in-place)

    /* Apply observation model */

    // Retrieve image patches It predicated by Xt
    const auto wimgsFlatten = warpImg(image, m_states, m_templShape, true);
    for (int i = 0; i < Np; ++i)
    {
        const cv::Mat zeroMean = wimgsFlatten.col(i) - m_templ.mean;
        zeroMean.copyTo(m_residual.col(i));
    }

    // Compute likelihood under the observation model for each patch
    const int nCurrentEigenvectors = m_templ.eigbasis.cols;
    if (nCurrentEigenvectors > 0)
    {
        // Compute (I - mu) - UU.T(I - mu)
        cv::Mat backProj = m_templ.eigbasis.t() * m_residual;
        m_residual -= (m_templ.eigbasis * backProj);
        cv::swap(backProj, m_backProj);
    }

    cv::Mat residual2;
    cv::pow(m_residual, 2, residual2);
    const auto prec = PRECISION(1.0) / m_condenssig;
    cv::Mat residual2sum;
    switch (m_errfunc)
    {
    case ErrorNorm::Ppca: // TODO
        break;
    case ErrorNorm::Robust:
        {
            const auto scaleParam = static_cast<PRECISION>(m_robustThr);
            cv::Mat rho = residual2 / (residual2 + scaleParam);
            cv::reduce(rho, residual2sum, 0, cv::REDUCE_SUM, CV_PRECISION);
            cv::multiply(residual2sum, -prec, residual2sum, 1.0, CV_PRECISION);
            cv::exp(residual2sum.t(), m_stateConfidences);
        }
        break;
    case ErrorNorm::L2:
        {
            cv::reduce(residual2, residual2sum, 0, cv::REDUCE_SUM, CV_PRECISION);
            cv::multiply(residual2sum, -prec, residual2sum, 1.0, CV_PRECISION);
            cv::exp(residual2sum.t(), m_stateConfidences);
        }
        break;
    }

    /* Store most likely particle */
    const auto normCoeff = static_cast<PRECISION>(1.0 / cv::sum(m_stateConfidences)[0]);
    cv::multiply(m_stateConfidences, normCoeff, m_stateConfidences, 1.0, CV_PRECISION);
    double maxProb = 0;
    int maxProbIdx[2] = {0,0};
    cv::minMaxIdx(m_stateConfidences, nullptr, &maxProb, nullptr, maxProbIdx);
    m_mostLikelyState = m_states.row(maxProbIdx[0]).clone();
    m_templ.prob = maxProb;
    const cv::Mat maxProbWimgFlatten = wimgsFlatten.col(maxProbIdx[0]).clone();
    m_mostLikelyWarpImage = maxProbWimgFlatten.reshape(0, m_templShape.width).clone();

    m_warpBatch.emplace_back(maxProbWimgFlatten.clone());       
}


const cv::Mat& IncrementalVisualTracker::stateConfidences() const noexcept
{
    return m_stateConfidences;
}

const cv::Mat& IncrementalVisualTracker::states() const noexcept
{
    return m_states;
}

const ObjectTemplate& IncrementalVisualTracker::objectTemplate() const noexcept
{
    return m_templ;
}

const cv::Mat& IncrementalVisualTracker::mostLikelyWarpImage() const noexcept
{
    return m_mostLikelyWarpImage;
}