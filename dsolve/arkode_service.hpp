#ifndef STAN_MATH_TORSTEN_DSOLVE_ARKODE_SERVICE_HPP
#define STAN_MATH_TORSTEN_DSOLVE_ARKODE_SERVICE_HPP

#include <stan/math/rev/core/recover_memory.hpp>
#include <stan/math/rev/meta/is_var.hpp>
#include <stan/math/torsten/dsolve/sundials_check.hpp>
#include <stan/math/torsten/dsolve/ode_func_type.hpp>
#include <stan/math/torsten/dsolve/cvodes_service.hpp>
#include <arkode/arkode.h>
#include <arkode/arkode_erkstep.h>
#include <arkode/arkode_butcher_erk.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <ostream>
#include <vector>
#include <algorithm>

namespace torsten {
  namespace dsolve {

    /** 
     * User data to be fed to ARKode RHS function. The object will be
     * static as it's designed to be a member of <code>servce</code>
     * class that will be static in actuall ODE solver call.
     * It provides internal RHS evaluation callback and related memory allocation.
     * 
     * @tparam Ode type of <code>PMXArkodeSystem</code> template
     */
    template<typename Ode>
    struct arkode_user_data {
      using F = typename ode_func<Ode>::type;

      const F* f;
      const size_t N;
      const size_t M;
      const size_t ns;
      const size_t fwd_ode_dim;
      std::vector<double> y;
      std::vector<double> fval;
      std::vector<double> theta_d;
      const std::vector<double>* px_r;
      const std::vector<int>* px_i;
      std::ostream* msgs;

      /** 
       * constructor
       * 
       * @param n dim of original ODE
       * @param m # of theta params
       * 
       */
      arkode_user_data(int n, int m) :
        f(nullptr), N(n), M(m),
        ns((Ode::is_var_y0 ? n : 0) + (Ode::is_var_par ? m : 0)),
        fwd_ode_dim(N + N * ns),
        y(n), fval(n), theta_d(m),
        px_r(nullptr), px_i(nullptr), msgs(nullptr)
      {}

      /** 
       * evaluate RHS function using current state, store
       * the result in internal <code>fval</code>.
       * 
       * @param t time
       * @param nv_y vector for state, may be longer than original
       * system dimension but the first <code>n</code> slots must
       * contain original system state.
       */
      inline void eval_rhs(double t, const N_Vector& nv_y) {
        for (size_t i = 0; i < N; ++i) {
          y[i] = NV_Ith_S(nv_y, i); 
        }
        fval = (*f)(t, y, theta_d, *px_r, *px_i, msgs);
      }

      /** 
       * evaluate RHS function using current state, store
       * the result in <code>ydot</code>. As ARKode doesn't support
       * sensitivity equation, the RHS system is the original system
       * appended with forward sensitivity system.
       * 
       * @param t time
       * @param nv_y vector for state (may include sensitivity) components.
       * @param ydot sensitivity
       */
      inline void eval_rhs(double t, N_Vector& nv_y, N_Vector& ydot) {
        using stan::math::var;        

        N_VConst(RCONST(0.0), ydot);

        // data-only
        if (!(Ode::is_var_y0 || Ode::is_var_par)) {
          eval_rhs(t, nv_y);
          for (size_t i = 0; i < N; ++i) {
            NV_Ith_S(ydot, i) = fval[i];
          }
          return;
        }

        // fwd sensivity
        try {
          stan::math::start_nested();

          std::vector<var> yv_work(NV_DATA_S(nv_y), NV_DATA_S(nv_y) + N);
          std::vector<var> theta_work(theta_d.begin(), theta_d.end());
          std::vector<var> fyv_work(Ode::is_var_par ?
                                    (*f)(t, yv_work, theta_work, *px_r, *px_i, msgs) :
                                    (*f)(t, yv_work, theta_d, *px_r, *px_i, msgs));

          stan::math::check_size_match("PMXArkodeSystem", "dz_dt", fyv_work.size(), "states", N);

          for (size_t i = 0; i < N; ++i) {
            stan::math::set_zero_all_adjoints_nested();
            NV_Ith_S(ydot, i) = fyv_work[i].val();
            fyv_work[i].grad();

            // df/dy*s_i term, for i = 1...ns
            for (size_t j = 0; j < ns; ++j) {
              for (size_t k = 0; k < N; ++k) {
                // std::cout << "taki test: " << j << " " << k << " " << NV_Ith_S(nv_y, N + N * j + k)  << "\n";
                NV_Ith_S(ydot, N + N * j + i) += NV_Ith_S(nv_y, N + N * j + k) * yv_work[k].adj();
              }
            }

            // df/dp_i term, for i = n...n+m-1
            if (Ode::is_var_par) {
              for (size_t j = 0; j < M; ++j) {
                NV_Ith_S(ydot, N + N * (ns - M + j) + i) += theta_work[j].adj();
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

    template <typename F, typename Tts, typename Ty0, typename Tpar>
    class PMXArkodeSystem;

    /** 
     * Specialization of <code>PMXOdeService</code> for ARKode.
     * 
     * @tparam Ode type of <code>PMXArkodeSystem</code> template
     */
    template <typename... Ts>
    struct PMXOdeService<PMXArkodeSystem<Ts...>> {
      using Ode = PMXArkodeSystem<Ts...>;

      arkode_user_data<Ode> user_data;
      N_Vector nv_y;
      void* mem;

      /**
       * Construct ARKODE ODE mem & workspace
       *
       * @param[in] n ODE system size
       * @param[in] m length of parameter theta
       * @param[in] f ODE RHS function
       */
      PMXOdeService(int n, int m) :
        user_data(n, m),
        nv_y(N_VNew_Serial(user_data.fwd_ode_dim)),
        mem(ERKStepCreate(arkode_rhs, 0.0, nv_y))
      {
        const double t0 = 0.0;
        N_VConst(RCONST(0.0), nv_y);

        CHECK_SUNDIALS_CALL(ERKStepSetUserData(mem, static_cast<void*>(&user_data)));
      }

      ~PMXOdeService() {
        ERKStepFree(&mem);
        N_VDestroy(nv_y);
      }

      static int arkode_rhs(double t, N_Vector y, N_Vector ydot, void* user_data) {
          arkode_user_data<Ode>* ode = static_cast<arkode_user_data<Ode>*>(user_data);
          ode -> eval_rhs(t, y, ydot);
          return 0;
      }
    };
  }
}

#endif
