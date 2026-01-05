#ifndef SCASpp_LOG_H
#define SCASpp_LOG_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <stdexcept>

namespace SCASpp {
template <typename Scalar = double>
class Logger {
private:
    struct LogEntry {
        int iteration;
        Scalar obj_value;
        Scalar convergence_metric;  
        double elapsed_time;   
    };

    std::vector<LogEntry> log_data;
    std::chrono::high_resolution_clock::time_point start_time;
    bool is_started;
public:
    Logger() : is_started(false) {}

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        is_started = true;
        log_data.clear();
    }

    void log(int iteration, Scalar obj_value, Scalar convergence_metric) {
        if (!is_started) {
            std::cerr << "Warning: Logger not started. Call start() first." << std::endl;
            return;
        }

        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;

        LogEntry entry;
        entry.iteration = iteration;
        entry.obj_value = obj_value;
        entry.convergence_metric = convergence_metric;
        entry.elapsed_time = elapsed.count();

        log_data.push_back(entry);
    }

    void print_summary() const {
        if (log_data.empty()) {
            std::cout << "No log data available." << std::endl;
            return;
        }

    const auto& last = log_data.back();
        std::cout << "\n========== Optimization Summary ==========\n";
        std::cout << "Final iteration:     " << last.iteration << std::endl;
        std::cout << "Final obj value:     " << std::scientific << std::setprecision(8) 
                  << last.obj_value << std::endl;
        std::cout << "Final convergence:   " << std::scientific << std::setprecision(6) 
                  << last.convergence_metric << std::endl;
        std::cout << "Total time:          " << std::fixed << std::setprecision(4) 
                  << last.elapsed_time << " s" << std::endl;
        std::cout << "==========================================\n" << std::endl;
    }


    const std::vector<LogEntry>& get_data() const {
        return log_data;
    }
};
}
#endif
