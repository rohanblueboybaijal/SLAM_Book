#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

// Vertex of the curve model, template parameters: optimizing variable dimensions and data types

class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // reset
    virtual void setToOriginImpl() override {
      _estimate << 0, 0, 0;
    }

    // renew
    virtual void oplusImpl(const double *update) override {
      _estimate += Eigen::Vector3d(update);
    }

    // Save and read functions : they are left blank
    virtual bool read(std::istream &in) {}

    virtual bool write(std::ostream &out) const {}
};

// Error model template parameters: observation dimension, type, connection vertex type
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    // Calculate curve model error
    virtual void computeError() override {
      const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
      const Eigen::Vector3d abc = v->estimate();
      _error(0,0) = _measurement - std::exp(abc(0,0)*_x*_x + abc(1,0)*_x + abc(2,0));
    }

    // Calculate the Jacobian matrix
    virtual void linearizeOplus() override {
      const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
      const Eigen::Vector3d abc = v->estimate();
      double y = std::exp(abc[0]*_x*_x + abc[1]*_x + abc[2]);
      _jacobianOplusXi[0] = -_x*_x*y;
      _jacobianOplusXi[1] = -_x*y;
      _jacobianOplusXi[2] = -y;
    }

    virtual bool read(std::istream &in) {}

    virtual bool write(std::ostream &out) const {}

    double _x;
};

int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0; // real parameters
  double ae = 2.0, be = -1.0, ce = 5.0; // estimated parameters
  int N = 100;
  double w_sigma = 1.0;
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;

  std::vector<double> x_data, y_data;
  for (int i=0; i<N; i++) {
    double x = i/100.0;
    x_data.push_back(x);
    y_data.push_back(std::exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma*w_sigma));
  }

  // Build graph optimization, first set g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType; // The dimension of the optimization variable for each error term is 3, and the dimension of the error value is 1

  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // Linear solver type

  // Gradient descent method, you can choose from GN, LM, DogLeg
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

  g2o::SparseOptimizer optimizer; // Graph model
  optimizer.setAlgorithm(solver); // Set up the solver
  optimizer.setVerbose(true);     // Turn on debug output

  // add vertices to the graph
  CurveFittingVertex *v = new CurveFittingVertex();
  v->setEstimate(Eigen::Vector3d(ae, be, ce));
  v->setId(0);
  optimizer.addVertex(v);

  // add edges to the graph
  // In graph optimization, nodes are the variables to be optimized and edges represent the error terms or some kind of constraints
  for (int i=0; i<N; i++) {
    CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
    edge->setId(i);
    edge->setVertex(0,v); // Set the connected vertices
    edge->setMeasurement(y_data[i]);    // Observed value
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma*w_sigma));  // Information matrix: the inverse of the covariance matrix
    optimizer.addEdge(edge);
  }

  // Perform optimization
  std::cout << "Start Optimization " << std::endl;
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

  std::cout<<"Solve time cost = "<< time_used.count() << " seconds. " << std::endl;

  // Output optimized value
  Eigen::Vector3d abc_estimate = v->estimate();
  std::cout<<"Estimated model : " << abc_estimate.transpose() << std::endl;

  return 0;

}