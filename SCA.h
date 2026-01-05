#ifndef SCASpp_SCA_H
#define SCASpp_SCA_H

#include <Eigen/Core>
#include "include/param.h"
#include "include/cal_H.h"
#include "include/logger.h"

namespace SCASpp {
/**
 * @brief Successive Convex Approximation solver
 * 
 * @tparam Scalar Scalar type used to represent objective function values, step sizes, and other numerical values.
 *                Supports floating-point types such as float, double, etc.
 *                Default: double
 * 
 * @tparam XType  Decision variable type representing the input variables of the optimization problem.
 *                - Scalar optimization: XType = Scalar (default)
 *                  Example: SCAsolver<double> for solving f(x): R -> R
 *                - Vector optimization: XType = Eigen::VectorXd or other vector types
 *                  Example: SCAsolver<double, Eigen::VectorXd> for solving f(x): R^n -> R
 *                Default: Scalar (scalar optimization)
 */
template <typename Scalar = double, typename XType = Scalar>
class SCAsolver
{
private:

    static constexpr bool is_vector = !std::is_same<XType, Scalar>::value;

    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    int iteration;
    const SCAParam<Scalar>& m_param;
    Scalar obj_value;

    std::function<Scalar(const XType&)> objfun;
    std::function<XType(const XType&)> gradfun;
    std::function<Matrix(const XType&)> Hfun;
    cal_H<Scalar> H_calculator; // 

    XType m_grad; // 
    XType m_xp;   //

    std::unique_ptr<Logger<Scalar>> logger;

public:
    SCAsolver(const SCAParam<Scalar>& param) :
        m_param(param),
        logger(nullptr)
    {
        m_param.check_param();

        if (m_param.is_log) {
            logger = std::make_unique<Logger<Scalar>>();
        }
    }

    void set_objfun(const std::function<Scalar(const XType&)>& f) { objfun = f; }
    void set_gradfun(const std::function<XType(const XType&)>& f) { gradfun = f; }

    template<typename T = XType>
    typename std::enable_if<!std::is_same<T, Scalar>::value>::type
    set_Hfun(const std::function<Matrix(const XType&)>& f) { Hfun = f; }

    inline void minimize(XType& x) {
        using std::abs;
        
        if (logger) {
            logger->start();
        }

        iteration = 1;
        obj_value = objfun(x);
        XType cur_grad = gradfun(x);
        XType old_x = x;
        XType direction_x;

        Scalar step = m_param.step_size;
        Scalar convergence_metric = 0;

        for (;;) {
            old_x = x;

            if constexpr (is_vector) {
                //auto inv_H = H_calculator.update(m_param.H_type, len, m_param.tau, Hfun, x,cur_grad);
                //auto inv_H = H_calculator.calculate_inv_H(m_param.H_type, len, m_param.tau, Hfun, x);
                Vector new_x = H_calculator.update(m_param.H_type, x.size(), m_param.tau, Hfun, x,cur_grad);
                //direction_x = old_x - inv_H*cur_grad;
                direction_x = old_x - new_x;
                x = step * direction_x + (1 - step) * old_x;
            } else {
                direction_x = (old_x - cur_grad) / m_param.tau;
                x = step * direction_x + (1 - step) * old_x;
            }

            obj_value = objfun(x);
            cur_grad = gradfun(x);
            step = step * (1 - m_param.decay * step);

            if constexpr (is_vector) {
                convergence_metric = (x - old_x).norm();
            } else {
                convergence_metric = abs(x - old_x);
            }

            if (logger) {
                logger->log(iteration, obj_value, convergence_metric);
            }

        
            if (iteration > 0 && convergence_metric <= m_param.epsilon) {
                std::cout << "Success." << std::endl;
                if (logger) {
                logger->print_summary();  
            }
                return;
            }
            

            if (iteration >= m_param.max_iterations) {
                std::cout << "Fail.Reach maximum iteration limit." << std::endl;
                return;
            }
            iteration++;
        }
    }

    const int final_iteration() const { return iteration; }
    const Scalar final_obj_value() const { return obj_value; }
    const Logger<Scalar>* get_logger() const { return logger.get(); }
};


}

#endif