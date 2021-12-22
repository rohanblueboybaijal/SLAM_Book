#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

int main(int argc, char **argv) {
    double ar = 1.0, br = 2.0, cr = 1.0; //True parameter value
    double ae = 2.0, be = -1.0, ce = 5.0; // Estimated parameter value
    int N = 100; // Number of data points
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; // opencv Random Number Generator

    std::vector<double> x_data, y_data;
    for(int i=0; i<N; i++) {
        double x = i/100.0;
        x_data.push_back(x);
        y_data.push_back(std::exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // Gauss Newton Method
    int iterations = 100;
    double cost = 0, lastCost = 0;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    for(int iter = 0; iter < iterations; iter++) {
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        cost = 0;

        for (int i = 0; i < N; i++) {
            double xi = x_data[i], yi = y_data[i];  // 第i个数据点
            double error = yi - std::exp(ae * xi * xi + be * xi + ce);
            Eigen::Vector3d J; // 雅可比矩阵
            J[0] = -xi * xi * std::exp(ae * xi * xi + be * xi + ce);  // de/da
            J[1] = -xi * std::exp(ae * xi * xi + be * xi + ce);  // de/db
            J[2] = -std::exp(ae * xi * xi + be * xi + ce);  // de/dc

            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * inv_sigma * error * J;

            cost += error * error;
        }

        // Solve Linear Equation Hx = b
        Eigen::Vector3d dx = H.ldlt().solve(b);
        if(std::isnan(dx[0])) {
            std::cout << "result is nan" << std::endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            std::cout<<"cost: "<<cost<<" >= last cost: "<<lastCost<<", break."<<std::endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;

        std::cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
         "\t\testimated params: " << ae << "," << be << "," << ce << std::endl;
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << "solve time cost = "<<time_used.count() << " seconds." << std::endl;

    std::cout<<"estimated abc = " << ae << ", " << be << ", " << ce << std::endl;

    return 0;
}