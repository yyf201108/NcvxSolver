#ifndef SCASpp_CALH_H
#define SCASpp_CALH_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
//#include <variant>
#include <Eigen/Dense>
#include <stdexcept>

namespace SCASpp {
template <typename Scalar = double>
class cal_H
{
    public:
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        //using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using SparseMatrix = Eigen::SparseMatrix<Scalar>;
        //using Matrix = std::variant<DenseMatrix, SparseMatrix>;

        template <typename HessianFunc>
        Vector update(int type, int n, Scalar tau, HessianFunc& f,Vector& x, Vector& grad){
            if (type == 1) {
                Eigen::SparseMatrix<Scalar> I_sparse(n, n);
                I_sparse.setIdentity();
                return 1/tau*I_sparse*grad;}
            else if (type == 2){
                if (!f) {
                    throw std::runtime_error("Error: Hessian function is not provided for type 2!");
                 }

                DenseMatrix init_H = f(x);
                if (!init_H.isApprox(init_H.transpose(), 1e-8)) {
                    std::cout << init_H << std::endl;
                    throw std::runtime_error("Error: Hessian matrix is not symmetric!");
                }
                Eigen::LLT<DenseMatrix> llt(init_H);
                if (llt.info() == Eigen::Success) {
               //std::cout << "Hessian is symmetric positive definite.\n";

                return llt.solve(grad);//init_H.inverse()*grad;
                } else {
                  //  std::cout << "Hessian is NOT positive definite. Modifying...\n";
                }
                Eigen::SelfAdjointEigenSolver<DenseMatrix> eig(init_H);
                if (eig.info() != Eigen::Success) {
                    throw std::runtime_error("Eigen decomposition failed!");
                }

                const Vector& eigvals = eig.eigenvalues();
                const DenseMatrix& eigvecs = eig.eigenvectors();

                // Vector eigvals_fixed = eigvals.unaryExpr([&](double v) { return std::max(v, tau); });
                // Vector eigvals_inv = eigvals_fixed.unaryExpr([](double v) { return 1.0 / v; }); 
                Vector eigvals_inv = eigvals.unaryExpr([&](Scalar v) { return Scalar(1) / std::max(v, tau);});
                //DenseMatrix H_inv = eigvecs * eigvals_inv.asDiagonal() * eigvecs.transpose();
                // std::cout << " modified eigen values"<< eigvals_fixed << std::endl;
                // std::cout << eigvecs << std::endl;
                //return H_inv*grad;

                Vector result = eigvecs.transpose() * grad; 
                result = eigvals_inv.cwiseProduct(result);  
                return eigvecs * result;                    
               

                }else{
                    throw std::runtime_error("wrong H type");
                }
                

            

                
            }
        
        // DenseMatrix update(int type, int n,Scalar tau){
        //     if (type == 1)
        //     {
        //         Eigen::SparseMatrix<Scalar> I_sparse(n, n);
        //         I_sparse.setIdentity();
        //         return 1/tau*I_sparse;}else if(type == 2){
        //             throw std::runtime_error("Lack function");
        //         }else{
        //             throw std::runtime_error("wrong H type");
        //         }
        //     }
        

};
}


#endif 