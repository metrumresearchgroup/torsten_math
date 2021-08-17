#ifndef STAN_MATH_TORSTEN_DSOLVE_ARKODE_INTEGRATOR_HPP
#define STAN_MATH_TORSTEN_DSOLVE_ARKODE_INTEGRATOR_HPP

#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/torsten/dsolve/sundials_check.hpp>
#include <stan/math/torsten/dsolve/cvodes_service.hpp>
#include <arkode/arkode_butcher_erk.h>
#include <arkode/arkode_erkstep.h>
#include <arkode/arkode_arkstep.h>
#include <arkode/arkode.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>      // Newton nonlinear solver
#include <type_traits>

#ifdef TORSTEN_BRAID
#include <arkode/arkode_xbraid.h>
#include <braid.h>
#include <boost/mpi.hpp>
#endif

namespace torsten {
namespace dsolve {

/**
 * ARKODE ODE integrator.
 */
  template<int butcher_tab>
  struct PMXArkodeIntegrator {
    static constexpr int ARKODE_MAX_STEPS = 500;

    const double rtol_;
    const double atol_;
    const int64_t max_num_steps_;

    /**
     * constructor
     * @param[in] rtol relative tolerance
     * @param[in] atol absolute tolerance
     * @param[in] max_num_steps max nb. of times steps
     */
    PMXArkodeIntegrator(const double rtol, const double atol,
                        const int64_t max_num_steps = ARKODE_MAX_STEPS)
      : rtol_(rtol), atol_(atol), max_num_steps_(max_num_steps) {}
      
    /**
     * Return the solutions for the specified ODE
     * given the specified initial state,
     * initial times, times of desired solution, and parameters and
     * data, writing error and warning messages to the specified
     * stream contained in the ODE system.
     *
     * @tparam ODE type of ODE system
     * @param[in] ode ODE system
     * increasing order, all greater than the initial time.
     * @return a vector of states, each state being a vector of the
     * same size as the state variable, corresponding to a time in ts.
     */
    template <typename Ode, typename Observer>
    void integrate(Ode& ode, Observer& observer) {
      ArkodeService<Ode> serv(ode.N, ode.M, ode.ns, ode);
      N_Vector& y = serv.nv_y;
      void* mem = serv.mem;

      for (size_t i = 0; i < ode.system_size; ++i) {
        NV_Ith_S(y, i) = ode.y0_fwd_system[i];
      }

      const size_t n = ode.N;
      const size_t ns = ode.ns;

#ifdef TORSTEN_BRAID
      braid_data const& bd = ode.braid;

      CHECK_SUNDIALS_CALL(ARKStepReInit(mem, Ode::arkode_combined_rhs, NULL, ode.t0_, y));
      CHECK_SUNDIALS_CALL(ARKStepSStolerances(mem, rtol_, atol_));
      CHECK_SUNDIALS_CALL(ARKStepSetMaxNumSteps(mem, max_num_steps_));
      CHECK_SUNDIALS_CALL(ARKStepSetUserData(mem, static_cast<void*>(&ode)));
      CHECK_SUNDIALS_CALL(ARKStepSetTableNum(mem, butcher_tab, -1));
      // ARKStepSetOrder(mem, 3);
      // CHECK_SUNDIALS_CALL(ARKStepSetAdaptivityMethod(mem, ARK_ADAPT_PI, SUNTRUE, SUNFALSE, NULL));

      MPI_Comm comm_w = MPI_COMM_WORLD;
      braid_Core core    = NULL; // XBraid memory structure
      SUNBraidApp app      = NULL; // ARKode + XBraid interface structure        

      // Create the ARKStep + XBraid interface
      CHECK_SUNDIALS_CALL(ARKBraid_Create(mem, &app));
      CHECK_SUNDIALS_CALL(ARKBraid_SetInitFn(app, Ode::braid_init));
      CHECK_SUNDIALS_CALL(ARKBraid_SetAccessFn(app, Ode::braid_access));

      CHECK_SUNDIALS_CALL(ARKBraid_BraidInit(comm_w, comm_w, ode.t0_, ode.ts_.back(), bd.x_nt, app, &core));
      CHECK_SUNDIALS_CALL(braid_SetAbsTol(core, bd.x_tol));
      CHECK_SUNDIALS_CALL(braid_SetCFactor(core, -1, bd.x_cfactor));
      CHECK_SUNDIALS_CALL(braid_SetPrintLevel(core, bd.x_print_level));
      CHECK_SUNDIALS_CALL(braid_SetMaxLevels(core, bd.x_max_levels));
      braid_SetAccessLevel(core, bd.x_access_level);
      braid_SetMaxIter(core, bd.x_max_iter);
      // braid_SetSeqSoln(core, 1);

      braid_Drive(core);

      braid_Destroy(core);
      ARKBraid_Free(&app);
#else      
      CHECK_SUNDIALS_CALL(ARKStepReInit(mem, NULL, Ode::arkode_combined_rhs, ode.t0_, y));
      CHECK_SUNDIALS_CALL(ARKStepSStolerances(mem, rtol_, atol_));
      CHECK_SUNDIALS_CALL(ARKStepSetMaxNumSteps(mem, max_num_steps_));
      CHECK_SUNDIALS_CALL(ARKStepSetUserData(mem, static_cast<void*>(&ode)));
      CHECK_SUNDIALS_CALL(ARKStepSetTableNum(mem, butcher_tab, -1));

      // CHECK_SUNDIALS_CALL(ARKStepSetAdaptivityMethod(mem, ARK_ADAPT_PID, SUNTRUE, SUNFALSE, NULL));
      // CHECK_SUNDIALS_CALL(ERKStepSetMaxGrowth(mem, 10));
      // CHECK_SUNDIALS_CALL(ERKStepSetMinReduction(mem, 0.2));

      double t1 = ode.t0_;
      for (size_t i = 0; i < ode.ts_.size(); ++i) {
        CHECK_SUNDIALS_CALL(ARKStepEvolve(mem, stan::math::value_of(ode.ts_[i]), y, &t1, ARK_NORMAL));
        for (size_t j = 0; j < ode.system_size; ++j) {
          ode.y0_fwd_system[j] = NV_Ith_S(y, j);
        }
        observer(ode.y0_fwd_system, t1);
      }
#endif
    }
  };

}  // namespace dsolve
}  // namespace torsten

#endif
