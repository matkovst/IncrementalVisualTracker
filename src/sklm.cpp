#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

// order is important https://github.com/opencv/opencv/issues/17366
#include <Eigen/QR>
#include <Eigen/SVD>
#include <opencv2/core/eigen.hpp>

#include "sklm.h"

void sklm(
    const std::vector<cv::Mat>& data, const cv::Mat& U0, const cv::Mat& D0, const cv::Mat& mu0, int n, float ff, 
    cv::Mat& U, cv::Mat& D, cv::Mat& mu, int& neff)
{
    if (data.empty())
        throw std::runtime_error("sklm: data is empty");

    const int m = data.size();
    const int d = data[0].total();

    cv::Mat dataMat(d, m, CV_32F);
    for (int di = 0; di < d; ++di)
        for (int mi = 0; mi < m; ++mi)
            dataMat.at<std::float_t>(di, mi) = data[mi].at<std::float_t>(di, 0);

    auto zeroMean = [](const cv::Mat& data, const cv::Mat& mean, int d, int m) {
        cv::Mat zeroMeanData(d, m, CV_32F);
        for (int mi = 0; mi < m; ++mi)
            zeroMeanData.col(mi) = data.col(mi) - mean;
        return zeroMeanData;
    };

    if (U0.empty()) // first eigenbasis calculation
    {
        cv::reduce(dataMat, mu, 1, cv::REDUCE_AVG);
        const cv::Mat zeroMeanData = zeroMean(dataMat, mu, d, m);

        // SvdMachine.compute(zeroMeanData, D, U, cv::noArray());
        Eigen::MatrixXf zeroMeanData_eigen;
        cv::cv2eigen(zeroMeanData, zeroMeanData_eigen);
        Eigen::BDCSVD<Eigen::MatrixXf> svd(zeroMeanData_eigen, Eigen::ComputeThinU);
        cv::eigen2cv(svd.matrixU(), U);
        cv::eigen2cv(svd.singularValues(), D);

        neff = m;
    }
    else // incremental update of eigenbasis
    {
        cv::Mat mu1;
        cv::reduce(dataMat, mu1, 1, cv::REDUCE_AVG);
        const cv::Mat zeroMeanData = zeroMean(dataMat, mu1, d, m);

        // Compute new mean Ic = (fn/(fn+m))Ia + (m/(fn+m))Ib
        const float acoeff = (ff*n) / (ff*n + m);
        const float bcoeff = m / (ff*n + m);
        mu = acoeff*mu0 + bcoeff*mu1;

        // Compute B{^} = [ (I_m+1 - Ib) ... (I_n+m - Ib) sqrt(nm/(n+m))(Ib - Ia) ]
        cv::Mat B = cv::Mat::zeros(d, m+1, CV_32F);
        zeroMeanData.copyTo(B.colRange(0, m));
        const double harmean = (m * n) / static_cast<double>(m + n);
        cv::Mat diff = std::sqrt(harmean) * (mu1 - mu0);
        diff.copyTo(B.col(m));
        neff = m + ff*n;

        const cv::Mat Bproj = U0.t() * B;
        cv::Mat Bdiff = B - (U0 * Bproj);
        cv::Mat Borth;
        Eigen::MatrixXf A;
        cv::cv2eigen(Bdiff, A);
        Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
        auto thinQ = qr.householderQ() * Eigen::MatrixXf::Identity(A.rows(), A.cols());
        cv::eigen2cv(thinQ, Borth);
        cv::Mat Q;
        cv::hconcat(U0, Borth, Q);

        // Compute R
        cv::Mat R_tl = cv::Mat::eye(D0.rows, D0.rows, CV_32F);
        cv::Mat D0ff = D0 * ff;
        for (int i = 0; i < D0ff.rows; ++i)
            R_tl.at<float>(i, i) = D0ff.at<float>(i);
        cv::Mat R_tr = Bproj;
        cv::Mat R_bl = cv::Mat::zeros(B.cols, D0.rows, CV_32F);
        cv::Mat R_br = Borth.t() * Bdiff;

        cv::Mat R, R_top, R_bottom;
        cv::hconcat(R_tl, R_tr, R_top);
        cv::hconcat(R_bl, R_br, R_bottom);
        cv::vconcat(R_top, R_bottom, R);

        // Compute the SVD of R
        cv::Mat Uraw, Draw;
        Eigen::MatrixXf R_eigen;
        cv::cv2eigen(R, R_eigen);
        Eigen::BDCSVD<Eigen::MatrixXf> svd(R_eigen, Eigen::ComputeThinU);
        cv::eigen2cv(svd.matrixU(), Uraw);
        cv::eigen2cv(svd.singularValues(), Draw);

        const float cutoff = cv::norm(Draw) * 0.001f;
        const cv::Mat keepMask = Draw >= cutoff;
        const int nKeep = cv::sum(keepMask)[0] / 255.0;
        U = cv::Mat(Uraw.rows, nKeep, Uraw.type());
        D = cv::Mat(nKeep, 1, Draw.type());
        int rowcolIter = 0;
        for (int i = 0; i < keepMask.rows; ++i)
        {
            if (0 == keepMask.at<std::uint8_t>(i))
                continue;
            Uraw.col(i).copyTo(U.col(rowcolIter));
            D.at<float>(i) = Draw.at<float>(rowcolIter);
            ++rowcolIter;
        }
        U = Q * U;
    }
}