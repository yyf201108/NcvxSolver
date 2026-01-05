# Nonconvex Problem Solver
### 📖 Overview

It is a C++ library for solving nonconvex optimization problems. It supports both scalar and vector optimization with flexible parameter wrapping.

### ✨ Features

- 📦 **Header-Only Library**: Easy integration into any project
- 🎯 **Generic Template Design**: Supports arbitrary scalar types (`float`, `double`, etc.) 
- 🔧 **Flexible Configuration**: Comprehensive parameter control through `SCAParam` class
- 📈 **Customizable**: Easy to extend with custom approximation methods
- 🚀 **Arbitrary Dimensional Problems**: works with any dimension from 1D scalar problems to N-dimensional vector problems


### 🔧 Core Components

#### 1. **SCA.h** - Main Solver
The core solver class `SCAsolver<Scalar, XType>` implementing the SCA algorithm. Template parameters allow it to work with both scalar (1D) and vector (N-D) problems.

#### 2. **param.h** - Configuration Parameters
`SCAParam` struct controls solver behavior including convergence tolerance (`epsilon`), maximum iterations, step size parameters (`alpha`, `beta`), and logging options.

#### 3. **logger.h** - Iteration Logger
`Logger` class tracks optimization progress, including objective values, convergence measures, and timing information.

#### 4. **cal_H.h** - Quadratic coefficient Approximation
Provides built-in methods for approximating the quadratic coefficient matrix , including two options.

#### 5. **example_scalar.cpp**, **example_vector.cpp**, **example_high.cpp**
Three examples showing basic steps for setting solver configurations and getting results.
