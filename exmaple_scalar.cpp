#include <Eigen/Core>
#include <iostream>
#include <cmath>
#include "SCA.h"


using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace SCASpp;

// scalar test: log(1+x^2), grad: 2x/(1+x^2)
double scalar_test(const double& x)
{
    double f = std::log(1.0 + x * x);
    return f;
}

double grad_fun(const double& x){
    double g = 2.0 * x / (1.0 + x * x);
    return g;
}

int main()
{
    SCAParam<double> param({
        .epsilon = 1e-6,
        .step_size = 0.5,
        .is_log = true
    });

    //param.print();

    SCAsolver<double> solver(param);

    

    double x = 1.0;
    solver.set_objfun(scalar_test);
    solver.set_gradfun(grad_fun);
    solver.minimize(x);

    std::cout << "iteration number: " << solver.final_iteration() << std::endl;
    std::cout << "x = " << x << std::endl;
    std::cout << "f(x) = " << solver.final_obj_value() << std::endl;

    return 0;
}