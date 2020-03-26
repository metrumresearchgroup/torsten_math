#ifndef STAN_MATH_TORSTEN_EVENT_HPP
#define STAN_MATH_TORSTEN_EVENT_HPP

#include<stan/math/torsten/torsten_def.hpp>
#include<stan/math/torsten/pmx_ode_integrator.hpp>
#include <stan/math/torsten/mpi/precomputed_gradients.hpp>
#include<vector>

namespace torsten {
  template<typename time_t, typename jump_t, typename force_t>
  struct Event {
    /// event id:
    /// 0. observation
    /// 1. reset
    /// 2. reset + evolve
    /// 3. steady state
    /// 4. reset + steady statea
    const int id;
    time_t t0;
    time_t t1;
    PKRec<jump_t> jump;
    std::vector<force_t> force;

    Event(int id_, time_t t0_, time_t t1_,
          PKRec<jump_t> jump_, std::vector<force_t> force_) :
      id(id_), t0(t0_), t1(t1_), jump(jump_), force(force_)
    {}

    template<typename T, typename model_t, PMXOdeIntegratorId It>
    inline void operator()(PKRec<T>& y, const model_t& model,
                           const PMXOdeIntegrator<It>& integ) {
      model.solve(y, t0, t1, force, integ);
    }

    template<typename T, typename model_t, PMXOdeIntegratorId It>
    inline void operator()(Eigen::VectorXd& yd, PKRec<T>& y,
                           const model_t& model,
                           const PMXOdeIntegrator<It>& integ) {
      std::vector<stan::math::var> vt(model.vars(t1));
      model.solve_d(yd, y, t0, t1, force, integ);
      y = torsten::mpi::precomputed_gradients(yd, vt);
    }

  };
}


#endif
