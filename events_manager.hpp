#ifndef STAN_MATH_TORSTEN_EVENTS_MANAGER_HPP
#define STAN_MATH_TORSTEN_EVENTS_MANAGER_HPP

#include <Eigen/Dense>
#include <stan/math/torsten/PKModel/PKModel.hpp>
#include <boost/math/tools/promotion.hpp>
#include <string>
#include <vector>

namespace torsten {
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
struct EventsManager {
  using T_scalar = typename stan::return_type<T0, T1, T2, T3, typename stan::return_type<T4, T5, T6>::type >::type;
  using T_time = typename stan::return_type<T0, T1, T6, T2>::type;
  using T_rate = typename stan::return_type<T2, T5>::type;
  using T_amt = typename stan::return_type<T1, T5>::type;

  EventHistory<T_time, T1, T2, T3> eh;
  ModelParameterHistory<T_time, T4, T5, T6> ph;
  std::vector<std::vector<T_rate> > rh;
  std::vector<T_amt> ah;
  int nKeep;

  EventsManager(int nCmt,
                const std::vector<T0>& time,
                const std::vector<T1>& amt,
                const std::vector<T2>& rate,
                const std::vector<T3>& ii,
                const std::vector<int>& evid,
                const std::vector<int>& cmt,
                const std::vector<int>& addl,
                const std::vector<int>& ss,
                const std::vector<std::vector<T4> >& pMatrix,
                const std::vector<std::vector<T5> >& biovar,
                const std::vector<std::vector<T6> >& tlag,
                const std::vector<Eigen::Matrix<T4, -1, -1> >& systems)
  : eh(time, amt, rate, ii, evid, cmt, addl, ss),
    ph(time, pMatrix, biovar, tlag, systems),
    rh(),
    ah(),
    nKeep(0)
  {
    eh.Sort();
    ph.Sort();
    nKeep = eh.size();

    eh.AddlDoseEvents();
    ph.CompleteParameterHistory(eh);

    eh.AddLagTimes(ph, nCmt);
    RateHistory<T_time, T2> rate_history(eh, nCmt);
    ph.CompleteParameterHistory(eh);

    int iRate = 0;
    for (int i = 0; i < eh.size(); i++) {

      // Use index iRate instead of i to find rate at matching time, given there
      // is one rate per time, not per event.
      if (rate_history.time(iRate) != eh.time(i)) iRate++;
      std::vector<T_rate> rate_i(nCmt);
      for (int j = 0; j < nCmt; ++j) {
        rate_i[j] = rate_history.rate(iRate, j) * ph.GetValueBio(i, j);
      }
      rh.push_back(rate_i);

      ah.push_back(ph.GetValueBio(i, eh.cmt(i) - 1) * eh.amt(i));
    }
  }

  EventHistory<T_time, T1, T2, T3>& events() {
    return eh;
  }

  ModelParameterHistory<T_time, T4, T5, T6>& parameters() {
    return ph;
  }

  std::vector<std::vector<T_rate> >& rates() {
    return rh;
  }

  std::vector<T_amt>& amts() {
    return ah;
  }
};

}

#endif
