#include <Eigen/Core>
#include <iostream>
#include <cmath>
#include <random>
#include <optional>
#include "SCA.h" 

// unitest for higher dimension problem where the problem parameters need to be 
// transfered as a function parameters


using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace SCASpp;


constexpr int n = 10;
constexpr int seed = 42;
std::mt19937 rng(seed);
std::uniform_real_distribution<double> dist(0.0, 1.0);


// Step 1: define the problem parameter structure
struct obj_params_struct {
    VectorXd C;
    VectorXd y;
};

// Step 2: define the objective function
// x: variable
// obj_params: problem parameters
double vector_test(const VectorXd& x,const obj_params_struct& obj_params)
{
    const VectorXd& C = obj_params.C;
    const VectorXd& y = obj_params.y;
    double ctx = C.dot(x);
    double f = (x - y).squaredNorm() + std::log(1 + ctx * ctx);
    return f;
}

// Step 3: define the gradient of the objective function
// x: variable
// obj_params: problem parameters
VectorXd grad_fun(const VectorXd& x, const obj_params_struct& obj_params){
    const VectorXd& C = obj_params.C;
    const VectorXd& y = obj_params.y;
    double ctx = C.dot(x);
    VectorXd grad = 2.0 * (x - y) + 2.0 * C * ctx / (1 + ctx * ctx);
    return grad;
}

// Step 3.2 (optional): define Hessian matrix.
// It is only needed when using some types of options.
MatrixXd H_fun(const VectorXd& x, const obj_params_struct& obj_params)
{
    const VectorXd& C = obj_params.C;
    double ctx = C.dot(x);
    int n = x.size();

    MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd CCt = C * C.transpose();

    // 2 * I + 2CC^T / (1+ctx^2) - 4 ctx^2 CC^T / (1+ctx^2)^2
    double denom = 1 + ctx * ctx;
    MatrixXd M = 2.0 * I
        + 2.0 * CCt / denom
        - 4.0 * ctx * ctx * CCt / (denom * denom);
    return M;
}

int main()
{
    VectorXd y(n), C(n);
    // generate parameters for the test
    for (int i = 0; i < n; ++i) {
        C(i) = dist(rng);
        y(i) = dist(rng);
    }
    // problem parameters
    obj_params_struct obj_params{C, y};

    VectorXd x = VectorXd::Zero(n);

    // solver parameters
    SCAParam<double> param({
        .epsilon = 1e-6,
        .step_size = 0.5,
        .H_type = 2,
        .is_log = true
        // .max_iterations = 1000 // 
    });

    // SCAsolver, the first parameter is the variable type of the ; 
    // the second parameter is the variable type.
    SCASpp::SCAsolver<double, VectorXd> solver(param);

    // warp objective function, gradiant function and Hessian
    // with obj_params
    auto objfun = [&](const VectorXd& x) {
        return vector_test(x, obj_params);
    };
    // set objective function
    solver.set_objfun(objfun);

    auto gfun = [&](const VectorXd& x) {
        return grad_fun(x, obj_params);
    };
    // set gradient function
    solver.set_gradfun(gfun);

    auto hfun = [&](const VectorXd& x) {
        return H_fun(x, obj_params);
    };
    // set matrix H function (optional)
    solver.set_Hfun(hfun);

    solver.minimize(x);

    std::cout << solver.final_iteration() << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << solver.final_obj_value() << std::endl;// should be 0.497449

    return 0;
}