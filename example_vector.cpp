#include <Eigen/Core>
#include <iostream>
#include <cmath>
#include "SCA.h"


using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace SCASpp;


// log(1 + x^T x); grad = 2x / (1 + x^T x)
double vector_test(const VectorXd& x)
{
    double x_square = x.squaredNorm();
    double f = std::log(1.0 + x_square);
    return f;
}

VectorXd grad_fun(const VectorXd& x)
{
    double x_square = x.squaredNorm();
    VectorXd grad = 2.0 * x / (1.0 + x_square);
    return grad;
}

// H(x) = 2I/(1+x^T x) - 4x x^T/(1+x^T x)^2
MatrixXd H_fun(const VectorXd& x)
{
    double xtx = x.squaredNorm();
    MatrixXd I = MatrixXd::Identity(x.size(), x.size());
    MatrixXd xxT = x * x.transpose();
    MatrixXd M = (2.0 / (1.0 + xtx)) * I
                 - (4.0 / std::pow(1.0 + xtx, 2)) * xxT;
    return M;
}

int main()
{
    SCAParam<double> param({
        .epsilon = 1e-6,
        .step_size = 0.5,
        .H_type = 2,
        .is_log = true
        // .max_iterations = 1000 // 如有需要可加
    });

    SCASpp::SCAsolver<double, VectorXd> solver(param);

    VectorXd x(2);
    x << 1.0, 2.0;

    solver.set_objfun(vector_test);
    solver.set_Hfun(H_fun);
    solver.set_gradfun(grad_fun);
    solver.minimize(x);

    std::cout << solver.final_iteration() << " iterations" << std::endl;
    std::cout << "x = \n" << x << std::endl;
    std::cout << "f(x) = " << solver.final_obj_value() << std::endl;

    return 0;
}
