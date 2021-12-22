#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>


// The calculation model of the cost function
struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // Calculation of residuals
    template<typename T>
    bool operator()(
        const T *const abc,
        T *residual ) const {
        residual[0] = T(_y) - ceres::exp(abc[0]*T(_x)*T(_x) + abc[1]*T(_x) + abc[2]);
        return true;
    }

    const double _x, _y;
};


int main(int argc, char **argv) {
    double ar = 1.0, br = 2.0, cr = 1.0; //True parameter value
    double ae = 2.0, be = -1.0, ce = 5.0; // Estimated parameter value
    int N = 100; // Number of data points
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; // opencv Random Number Generator

    std::vector<double> x_data, y_data;
    for (int i=0; i<N; i++) {
        double x = i/100.0;
        x_data.push_back(x);
        y_data.push_back(std::exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma*w_sigma));
    }

    double abc[3] = {ae, be, ce};

    //Constructing the Least Squares Problem
    ceres::Problem problem;
    for (int i=0; i<N; i++) {
        problem.AddResidualBlock(  //Constructing the Least Squares Problem
            // Use automatic derivation, template parameters: error type, output dimension, input dimension, dimension should be consistent with the previous struct

            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                new CURVE_FITTING_COST(x_data[i], y_data[i])
            ),
            nullptr, // Kernel function, not used here, empty
            abc      // Parameters to be estimated
        );
    }

    // Configure solver
    ceres::Solver::Options options;  // There are many configuration items to fill in
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // How to solve the incremental equation
    options.minimizer_progress_to_stdout = true;  // Output to cout

    ceres::Solver::Summary summary; // Optimization information

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);  // Start to optimize
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve time cost = " << time_used.count() << " seconds. " << std::endl;

    // output result
    std::cout<< summary.BriefReport() << std::endl;
    std::cout << "estimated a,b,c = ";
    for (auto a : abc) std::cout<<a<<" ";
    std::cout<<std::endl;
    return 0;
}