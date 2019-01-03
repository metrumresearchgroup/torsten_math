#ifndef STAN_MATH_TORSTEN_DSOLVE_CVODES_FWD_SYSTEM_HPP
#define STAN_MATH_TORSTEN_DSOLVE_CVODES_FWD_SYSTEM_HPP

#include <stan/math/prim/arr/fun/value_of.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/torsten/dsolve/pk_cvodes_system.hpp>
#include <stan/math/torsten/dsolve/cvodes_sens_rhs.hpp>
#include <stan/math/torsten/pk_csda.hpp>
#include <stan/math/rev/mat/functor/jacobian.hpp>
#include <stan/math/torsten/dsolve/cvodes_service.hpp>

namespace torsten {

  /**
   * Choose among three methods to calculate the
   * sensitivities:
   * CSDA: complex step derivative approximation
   * AD: automatic differentiation by Stan
   * differential quotient provided by CVODES
   **/
  enum PkCvodesSensMethod {
    CSDA, AD, DQ
  };
}

namespace torsten {
  namespace dsolve {
    /**
     * CVODES ODE system with forward sensitivity calculation
     *
     * @tparam F type of functor for ODE residual.
     * @tparam Tts type of time
     * @tparam Ty0 type of initial unknown values.
     * @tparam Tpar type of parameters.
     * @tparam Lmm method of integration(CV_ADAMS or CV_BDF)
     * @tparam Sm method of sensitivity calculatioin, choose among @c PkCvodesSensMethod.
     */
    template <typename F, typename Tts, typename Ty0, typename Tpar, int Lmm, PkCvodesSensMethod Sm>  // NOLINT
    class PKCvodesFwdSystem;

    /**
     * when use CSDA to calculate sensitivity, the
     * user-supplied RHS function of the ODE system must be
     * able to operate on complex numbers, most of current Stan
     * math functions do not support this.
     */
    template <typename F, typename Tts, typename Ty0, typename Tpar, int Lmm>
    class PKCvodesFwdSystem<F, Tts, Ty0, Tpar, Lmm, CSDA> : public PKCvodesSystem<F, Tts, Ty0, Tpar, Lmm> {  // NOLINT
    public:
      using Ode = PKCvodesSystem<F, Tts, Ty0, Tpar, Lmm>;
    private:
      N_Vector* nv_ys_;
      std::vector<std::complex<double> >& yy_cplx_;
      std::vector<std::complex<double> >& theta_cplx_;
      std::vector<std::complex<double> >& fval_cplx_;
    public:
      /**
       * Construct CVODES ODE system from initial condition and parameters
       *
       * @param[in] f ODE residual functor
       * @param[in] y0 initial condition
       * @param[in] theta parameters of the base ODE
       * @param[in] x_r continuous data vector for the ODE
       * @param[in] x_i integer data vector for the ODE
       * @param[in] msgs stream to which messages are printed
       */
      PKCvodesFwdSystem(PKCvodesService<Ode>& serv,
                           const F& f,
                           double t0,
                           const std::vector<Tts>& ts,
                           const std::vector<Ty0>& y0,
                           const std::vector<Tpar>& theta,
                           const std::vector<double>& x_r,
                           const std::vector<int>& x_i,
                           std::ostream* msgs) :
        Ode(serv, f, t0, ts, y0, theta, x_r, x_i, msgs),
        nv_ys_(Ode::serv_.nv_ys),
        yy_cplx_(Ode::serv_.yy_cplx),
        theta_cplx_(Ode::serv_.theta_cplx),
        fval_cplx_(Ode::serv_.fval_cplx)
      {}

      /**
       * Dummy destructor. Deallocation of CVODES memory is done
       * in @c PKCvodesService.
       */
      ~PKCvodesFwdSystem() {
      }

      /**
       * return N_Vector pointer array of sensitivity
       */
      N_Vector* nv_ys() { return nv_ys_; }

      /**
       * convert to void pointer for CVODES callbacks
       */
      void* to_user_data() {  // prepare to inject ODE info
        return static_cast<void*>(this);
      }

      /**
       * Calculate sensitivity rhs using CVODES vectors. The
       * internal workspace is allocated by @c PKCvodesService.
       * We use CSDA to compute senstivity, so we need to
       * generate complex version of parameters.
       */
      void eval_sens_rhs(int ns, double t, N_Vector y, N_Vector ydot,
                         N_Vector* ys, N_Vector* ysdot,
                         N_Vector temp1, N_Vector temp2) {
        using std::complex;
        using cplx = complex<double>;
        using B = PKCvodesSystem<F, Tts, Ty0, Tpar, Lmm>;
        const int n = B::N_;
        const double h = 1.E-20;
        for (int i = 0; i < ns; ++i) {
          for (int j = 0; j < n; ++j) {
            yy_cplx_[j] = cplx(NV_Ith_S(y, j), h * NV_Ith_S(ys[i], j));
          }

          /* if y0 is the only parameter, use tangent linear
           * model(TLM). Otherwise use full forward sensitivity model.
           * Note that when both y0 and theta are
           * parameters, the first n vector of ys are for y0 sensitivity.
           */
          if (B::is_var_y0 && i < n) {
            fval_cplx_ =
              B::f_(t, yy_cplx_, B::theta_dbl_, B::x_r_, B::x_i_, B::msgs_);
          } else {
            std::transform(B::theta_dbl_.begin(),
                           B::theta_dbl_.end(),
                           theta_cplx_.begin(),
                           [](double r) -> cplx {return cplx(r, 0.0); });
            theta_cplx_.at(i - B::is_var_y0 * n) += cplx(0.0, h);
            fval_cplx_ =
              B::f_(t, yy_cplx_, theta_cplx_, B::x_r_, B::x_i_, B::msgs_);
          }

          std::transform(fval_cplx_.begin(),
                         fval_cplx_.end(),
                         B::fval_.begin(),
                         [&h](cplx x) -> double { return std::imag(x)/h; });
          for (int j = 0; j < n; ++j) NV_Ith_S(ysdot[i], j) = B::fval_[j];
        }
      }
    };

    /**
     * use autodiff to calculate sensitivity
     */
    template <typename F, typename Tts, typename Ty0, typename Tpar, int Lmm>
    class PKCvodesFwdSystem<F, Tts, Ty0, Tpar, Lmm, AD> : public PKCvodesSystem<F, Tts, Ty0, Tpar, Lmm> {  // NOLINT
    public:
      using Ode = PKCvodesSystem<F, Tts, Ty0, Tpar, Lmm>;
    private:
      N_Vector* nv_ys_;
      std::vector<stan::math::var> ysv;
    public:
      /**
       * Construct CVODES ODE system from initial condition and parameters
       *
       * @param[in] f ODE residual functor
       * @param[in] y0 initial condition
       * @param[in] theta parameters of the base ODE
       * @param[in] x_r continuous data vector for the ODE
       * @param[in] x_i integer data vector for the ODE
       * @param[in] msgs stream to which messages are printed
       */
      PKCvodesFwdSystem(PKCvodesService<Ode>& serv,
                           const F& f,
                           double t0,
                           const std::vector<Tts>& ts,
                           const std::vector<Ty0>& y0,
                           const std::vector<Tpar>& theta,
                           const std::vector<double>& x_r,
                           const std::vector<int>& x_i,
                           std::ostream* msgs) :
        Ode(serv, f, t0, ts, y0, theta, x_r, x_i, msgs),
        nv_ys_(Ode::serv_.nv_ys),
        ysv(Ode::N_)
      {}

      /**
       * Dummy destructor. Deallocation of CVODES memory is done
       * in @c PKCvodesService.
       */
      ~PKCvodesFwdSystem() {
      }

      /**
       * return N_Vector pointer array of sensitivity
       */
      N_Vector* nv_ys() { return nv_ys_; }

      /**
       * convert to void pointer for CVODES callbacks
       */
      void* to_user_data() {  // prepare to inject ODE info
        return static_cast<void*>(this);
      }

      /**
       * Calculate sensitivity rhs using CVODES vectors. The
       * internal workspace is allocated by @c PKCvodesService.
       */
      void eval_sens_rhs(int ns, double t, N_Vector y, N_Vector ydot,
                         N_Vector* ys, N_Vector* ysdot,
                         N_Vector temp1, N_Vector temp2) {
        using Eigen::Matrix;
        using Eigen::MatrixXd;
        using Eigen::VectorXd;
        using Eigen::Dynamic;
        using stan::math::matrix_v;
        using stan::math::vector_v;
        using stan::math::var;

        using B = PKCvodesSystem<F, Tts, Ty0, Tpar, Lmm>;

        const int& n = B::N_;
        const int& m = B::M_;
        auto& f = B::f_;
        const std::vector<double> & theta_dbl = B::theta_dbl_;
        const std::vector<double> & x_r       = B::x_r_;
        const std::vector<int>    & x_i       = B::x_i_;
        std::ostream* msgs              = B::msgs_;

        static std::vector<double> g;
        std::vector<double>& yv_{B::y_vec_};
        for (int i = 0; i < n; ++i) yv_[i] = NV_Ith_S(y, i);

        // initialize ysdot
        for (int i = 0; i < ns; ++i) {
          auto nvp = N_VGetArrayPointer(ysdot[i]);
          for (int j = 0; j < n; ++j) nvp[j] = 0.0;
        }

        try {
          stan::math::start_nested();

          for (int j = 0; j < n; ++j) ysv[j] = yv_[j];
          std::vector<stan::math::var> pars;
          std::vector<stan::math::var> fy(n);

          if (B::is_var_par) {
            std::vector<var> theta(theta_dbl.begin(), theta_dbl.end());
            pars.reserve(n + m);
            pars.insert(pars.end(), ysv.begin(), ysv.end());
            pars.insert(pars.end(), theta.begin(), theta.end());
            fy = f(t, ysv, theta, x_r, x_i, msgs);
          } else {
            pars.reserve(n);
            pars.insert(pars.end(), ysv.begin(), ysv.end());
            fy = f(t, ysv, theta_dbl, x_r, x_i, msgs);
          }


          for (int j = 0; j < n; ++j) {
            stan::math::set_zero_all_adjoints_nested();
            fy[j].grad(pars, g);

            // df/dy*s_i term, for i = 1...ns
            for (int i = 0; i < ns; ++i) {
              auto ysp = N_VGetArrayPointer(ys[i]);
              auto nvp = N_VGetArrayPointer(ysdot[i]);
              for (int k = 0; k < n; ++k) nvp[j] += g[k] * ysp[k];
            }

            // df/dp_i term, for i = n...n+m-1
            if (B::is_var_par) {
              for (int i = 0; i < m; ++i) {
                auto nvp = N_VGetArrayPointer(ysdot[ns - m + i]);
                nvp[j] += g[n + i];
              }
            }
          }
        } catch (const std::exception& e) {
          stan::math::recover_memory_nested();
          throw;
        }
        stan::math::recover_memory_nested();
      }
    };
  }  // namespace dsolve
}  // namespace torsten

#endif
