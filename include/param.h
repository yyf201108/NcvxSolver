#ifndef SCASpp_PARAM_H
#define SCASpp_PARAM_H

#include <Eigen/Core>
#include <stdexcept>  // std::invalid_argument
#include <optional>

namespace SCASpp {
///
/// Parameters to control the algorithm.
///
template <typename Scalar = double>
class SCAParam
{
private:
    /// Default values
    static constexpr Scalar default_epsilon = Scalar(1e-5);
    static constexpr Scalar default_step_size = Scalar(1);
    static constexpr Scalar default_decay = Scalar(0.99);
    static constexpr Scalar default_tau = Scalar(1.0);
    static constexpr int default_max_iterations = 10000;
    static constexpr int default_H_type = 1;
    static constexpr bool default_is_log = false;

public:

///Allow users to set only the desired parameters, with others using default values.
   struct Options {
        /// Convergence tolerance (must be non-negative)
        std::optional<Scalar> epsilon = std::nullopt;

        /// Step size for updates (must be in (0,1])
        std::optional<Scalar> step_size = std::nullopt;

        /// Decay factor (must be in [0,1])
        std::optional<Scalar> decay = std::nullopt;

        /// Regularization parameter (must be positive)
        std::optional<Scalar> tau = std::nullopt;

        /// Maximum number of iterations
        std::optional<int> max_iterations = std::nullopt;

        /// Hessian approximation type (1 or 2)
        std::optional<int> H_type = std::nullopt;

        /// Enable logging
        std::optional<bool> is_log = std::nullopt;
    };

    Scalar epsilon;
    Scalar step_size;
    Scalar decay;
    Scalar tau;
    int max_iterations;
    int H_type;
    bool is_log;
  
    explicit SCAParam(const Options& opts = {}) : 
        epsilon(opts.epsilon.value_or(default_epsilon)),
        step_size(opts.step_size.value_or(default_step_size)),
        decay(opts.decay.value_or(default_decay)),
        tau(opts.tau.value_or(default_tau)),
        max_iterations(opts.max_iterations.value_or(default_max_iterations)),
        H_type(opts.H_type.value_or(default_H_type)),
        is_log(opts.is_log.value_or(default_is_log))
    { 
        check_param();
    }

    inline void check_param() const
    {
        if (epsilon < 0)
            throw std::invalid_argument("'epsilon' must be non-negative");
        if (step_size <= 0 || step_size > 1)
            throw std::invalid_argument("'step_size' must be (0,1]");
        if (tau <= 0)
            throw std::invalid_argument("'tau' must be positive");
        if (decay < 0 || decay > 1)
            throw std::invalid_argument("'decay' must be [0,1]");
        if (max_iterations < 0)
            throw std::invalid_argument("'max_iterations' must be non-negative");
    }

    void print() const {
        std::cout << "Solver Parameters:\n"
                  << "  epsilon: " << epsilon << "\n"
                  << "  step_size: " << step_size << "\n"
                  << "  decay: " << decay << "\n"
                  << "  tau: " << tau << "\n"
                  << "  max_iterations: " << max_iterations << "\n"
                  << "  H_type: " << H_type << "\n"
                  << "  is_log: " << std::boolalpha << is_log << "\n";
    }
};


}  

#endif  // 

