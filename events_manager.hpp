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
  using T_par = T4;

  EventHistory<T_time, T1, T2, T3> event_his;
  ModelParameterHistory<T_time, T4, T5, T6> param_his;
  std::vector<std::vector<T_rate> > rate_v;
  std::vector<T_amt> amt_v;
  std::vector<std::vector<T_par> > par_v;
  int nKeep;
  int ncmt;

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
                const std::vector<std::vector<T6> >& tlag)
  : event_his(time, amt, rate, ii, evid, cmt, addl, ss),
    param_his(time, pMatrix, biovar, tlag),
    rate_v(),
    amt_v(),
    par_v(),
    nKeep(0),
    ncmt(nCmt)
  {
    event_his.Sort();
    param_his.Sort();
    nKeep = event_his.size();

    event_his.AddlDoseEvents();
    param_his.CompleteParameterHistory(event_his);

    event_his.AddLagTimes(param_his, nCmt);
    RateHistory<T_time, T2> rate_history(event_his, nCmt);
    param_his.CompleteParameterHistory(event_his);

    int iRate = 0;
    for (size_t i = 0; i < event_his.size(); i++) {

      // Use index iRate instead of i to find rate at matching time, given there
      // is one rate per time, not per event.
      if (rate_history.time(iRate) != event_his.time(i)) iRate++;
      std::vector<T_rate> rate_i(nCmt);
      for (int j = 0; j < nCmt; ++j) {
        rate_i[j] = rate_history.rate(iRate, j) * param_his.GetValueBio(i, j);
      }
      rate_v.push_back(rate_i);

      amt_v.push_back(param_his.GetValueBio(i, event_his.cmt(i) - 1) * event_his.amt(i));
    }

    par_v.resize(event_his.size());
    for (size_t i = 0; i < event_his.size(); ++i) {
      auto p = param_his.GetModelParameters(i);
      par_v[i] = p.get_RealParameters();
    }
  }

  EventsManager(int nCmt,
                const std::vector<T0>& time,
                const std::vector<T1>& amt,
                const std::vector<T2>& rate,
                const std::vector<T3>& ii,
                const std::vector<int>& evid,
                const std::vector<int>& cmt,
                const std::vector<int>& addl,
                const std::vector<int>& ss,
                const std::vector<std::vector<T5> >& biovar,
                const std::vector<std::vector<T6> >& tlag,
                const std::vector<Eigen::Matrix<T4, -1, -1> >& systems)
  : event_his(time, amt, rate, ii, evid, cmt, addl, ss),
    param_his(time, biovar, tlag, systems),
    rate_v(),
    amt_v(),
    nKeep(0),
    ncmt(nCmt)
  {
    event_his.Sort();
    param_his.Sort();
    nKeep = event_his.size();

    event_his.AddlDoseEvents();
    param_his.CompleteParameterHistory(event_his);

    event_his.AddLagTimes(param_his, nCmt);
    RateHistory<T_time, T2> rate_history(event_his, nCmt);
    param_his.CompleteParameterHistory(event_his);

    int iRate = 0;
    for (size_t i = 0; i < event_his.size(); i++) {

      // Use index iRate instead of i to find rate at matching time, given there
      // is one rate per time, not per event.
      if (rate_history.time(iRate) != event_his.time(i)) iRate++;
      std::vector<T_rate> rate_i(nCmt);
      for (int j = 0; j < nCmt; ++j) {
        rate_i[j] = rate_history.rate(iRate, j) * param_his.GetValueBio(i, j);
      }
      rate_v.push_back(rate_i);

      amt_v.push_back(param_his.GetValueBio(i, event_his.cmt(i) - 1) * event_his.amt(i));
    }

    par_v.resize(event_his.size());
    for (size_t i = 0; i < event_his.size(); ++i) {
      auto p = param_his.GetModelParameters(i);
      auto k = p.get_K();
      std::vector<T_par> par(k.size());
      for (size_t j = 0; j < par.size(); ++j) par[j] = k(j);
      par_v[i] = par;
    }
  }

  EventHistory<T_time, T1, T2, T3>& events() {
    return event_his;
  }

  ModelParameterHistory<T_time, T4, T5, T6>& parameters() {
    return param_his;
  }

  std::vector<std::vector<T_rate> >& rates() {
    return rate_v;
  }

  std::vector<T_amt>& amts() {
    return amt_v;
  }

  std::vector<std::vector<T_par> >& pars() {
    return par_v;
  }
};

}

#endif
