#ifndef STAN_MATH_TORSTEN_EVENTS_MANAGER_HPP
#define STAN_MATH_TORSTEN_EVENTS_MANAGER_HPP

#include <Eigen/Dense>
#include <stan/math/torsten/PKModel/PKModel.hpp>
#include <stan/math/torsten/dsolve/pk_vars.hpp>
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

  // const std::vector<T0>& time_;
  // const std::vector<T1>& amt_;
  // const std::vector<T2>& rate_;
  // const std::vector<T3>& ii_;
  // const std::vector<int>& evid_;
  // const std::vector<int>& cmt_;
  // const std::vector<int>& addl_;
  // const std::vector<int>& ss_;
  // const std::vector<std::vector<T4> >& pMatrix_;
  // const std::vector<std::vector<T5> >& biovar_;
  // const std::vector<std::vector<T6> >& tlag_;

  EventHistory<T_time, T1, T2, T3> event_his;
  std::vector<std::vector<T_rate> > rate_v;
  std::vector<T_amt> amt_v;
  std::vector<std::vector<T_par> > par_v;
  std::vector<int> keep_ev;

  const int nKeep;
  const int ncmt;

  template <typename T0_, typename T1_, typename T2_, typename T3_, typename T4_, typename T5_, typename T6_>
  EventsManager(int nCmt,
                const std::vector<T0_>& time,
                const std::vector<T1_>& amt,
                const std::vector<T2_>& rate,
                const std::vector<T3_>& ii,
                const std::vector<int>& evid,
                const std::vector<int>& cmt,
                const std::vector<int>& addl,
                const std::vector<int>& ss,
                const std::vector<std::vector<T4_> >& pMatrix,
                const std::vector<std::vector<T5_> >& biovar,
                const std::vector<std::vector<T6_> >& tlag) :
    event_his(time, amt, rate, ii, evid, cmt, addl, ss),
    nKeep(event_his.size()),
    ncmt(nCmt)
  {
    event_his.Sort();

    ModelParameterHistory<T_time, T4, T5, T6> param_his(time, pMatrix, biovar, tlag);
    param_his.Sort();

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

    keep_ev.reserve(nKeep);
    for (size_t i = 0; i < event_his.size(); ++i) {    
      if (event_his.keep(i)) keep_ev.push_back(i);
    }
  }

  /*
   * For population models, we need generate events using
   * ragged arrays.
   */
  template <typename T0_, typename T1_, typename T2_, typename T3_, typename T4_, typename T5_, typename T6_>
  EventsManager(int nCmt,
                int ibegin, int isize,
                const std::vector<T0_>& time,
                const std::vector<T1_>& amt,
                const std::vector<T2_>& rate,
                const std::vector<T3_>& ii,
                const std::vector<int>& evid,
                const std::vector<int>& cmt,
                const std::vector<int>& addl,
                const std::vector<int>& ss,
                int ibegin_theta, int isize_theta,
                const std::vector<std::vector<T4_> >& pMatrix,
                int ibegin_biovar, int isize_biovar,
                const std::vector<std::vector<T5_> >& biovar,
                int ibegin_tlag, int isize_tlag,
                const std::vector<std::vector<T6_> >& tlag) :
    event_his(ibegin, isize, time, amt, rate, ii, evid, cmt, addl, ss),
    nKeep(event_his.size()),
    ncmt(nCmt)
  {
    event_his.Sort();

    ModelParameterHistory<T_time, T4, T5, T6>
      param_his(ibegin, isize, time,
                ibegin_theta, isize_theta, pMatrix,
                ibegin_biovar, isize_biovar, biovar,
                ibegin_tlag, isize_tlag, tlag);
    param_his.Sort();

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

    keep_ev.reserve(nKeep);
    for (size_t i = 0; i < event_his.size(); ++i) {    
      if (event_his.keep(i)) keep_ev.push_back(i);
    }
  }


  template <typename T0_, typename T1_, typename T2_, typename T3_, typename T4_, typename T5_, typename T6_>
  EventsManager(int nCmt,
                const std::vector<T0_>& time,
                const std::vector<T1_>& amt,
                const std::vector<T2_>& rate,
                const std::vector<T3_>& ii,
                const std::vector<int>& evid,
                const std::vector<int>& cmt,
                const std::vector<int>& addl,
                const std::vector<int>& ss,
                const std::vector<std::vector<T5_> >& biovar,
                const std::vector<std::vector<T6_> >& tlag,
                const std::vector<Eigen::Matrix<T4_, -1, -1>>& systems) :
    event_his(time, amt, rate, ii, evid, cmt, addl, ss),
    nKeep(event_his.size()),
    ncmt(nCmt)
  {
    event_his.Sort();

    ModelParameterHistory<T_time, T4, T5, T6> param_his(time, biovar, tlag, systems);
    param_his.Sort();

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

    keep_ev.reserve(nKeep);
    for (size_t i = 0; i < event_his.size(); ++i) {    
      if (event_his.keep(i)) keep_ev.push_back(i);
    }
  }

  const EventHistory<T_time, T1, T2, T3>& events() const {
    return event_his;
  }

  const std::vector<std::vector<T_rate> >& rates() const {
    return rate_v;
  }

  const std::vector<T_amt>& amts() const {
    return amt_v;
  }

  const std::vector<std::vector<T_par> >& pars() const {
    return par_v;
  }

  /*
   * check the exisitence of SS dosing events
   */
  static bool has_ss_dosing(const std::vector<int>& evid,
                            const std::vector<int>& ss) {
    if (ss.size() == 1) {
      return false;
    } else {
      bool res = false;
      for (size_t i = 0; i < evid.size(); ++i) {
        if ((evid[i] == 1 || evid[i] == 4) && ss[i] != 0) {
          res = true;
          break;
        }
      }
      return res;
    }
  }

  /*
   * calculate the total nb. of events without generating
   * events history.
   */
  template <typename T0_, typename T1_, typename T2_, typename T3_, typename T4_, typename T5_, typename T6_>
  static int nevents(const std::vector<T0_>& time,
                     const std::vector<T1_>& amt,
                     const std::vector<T2_>& rate,
                     const std::vector<T3_>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     const std::vector<std::vector<T4_> >& pMatrix,
                     const std::vector<std::vector<T5_> >& biovar,
                     const std::vector<std::vector<T6_> >& tlag) {
    return nevents(0, time.size(), time, amt, rate, ii, evid, cmt, addl, ss,
                   0, pMatrix.size(), pMatrix,
                   0, biovar.size(), biovar,
                   0, tlag.size(), tlag);
  }

  template <typename T0_, typename T1_, typename T2_, typename T3_, typename T4_, typename T5_, typename T6_>
  static int nevents(int ibegin, int isize,
                     const std::vector<T0_>& time,
                     const std::vector<T1_>& amt,
                     const std::vector<T2_>& rate,
                     const std::vector<T3_>& ii,
                     const std::vector<int>& evid,
                     const std::vector<int>& cmt,
                     const std::vector<int>& addl,
                     const std::vector<int>& ss,
                     int ibegin_pMatrix, int isize_pMatrix,
                     const std::vector<std::vector<T4_> >& pMatrix,
                     int ibegin_biovar, int isize_biovar,
                     const std::vector<std::vector<T5_> >& biovar,
                     int ibegin_tlag, int isize_tlag,
                     const std::vector<std::vector<T6_> >& tlag) {
    using stan::math::value_of;

    int res;
    bool has_lag = std::any_of(tlag.begin() + ibegin_tlag, tlag.begin() + ibegin_tlag + isize_tlag,
                              [](const std::vector<T6_>& v) {
                                 return std::any_of(v.begin(), v.end(), [](const T6_& x) { return std::abs(value_of(x)) > 1.E-10; });
                              });

    if (!has_lag) {
      int n = isize;
      for (int i = ibegin; i < ibegin + isize; ++i) {
        if (evid[i] == 1 || evid[i] == 4) {      // is dosing event
          if (addl[i] > 0 && ii[i] > 0) {        // has addl doses
            if (rate[i] > 0 && amt[i] > 0) {
              n++;                               // end event for original IV dose
              n += 2 * addl[i];                  // end event for addl IV dose
            } else {
              n += addl[i];
            }
          } else if (rate[i] > 0 && amt[i] > 0) {
            n++;                                 // end event for IV dose
          }
        }
      }
      res = n;
    } else if (isize_tlag == 1) {
      int n = isize;
      std::vector<std::tuple<double, int>> dose;
      dose.reserve(isize);
      for (int i = ibegin; i < ibegin + isize; ++i) {
        if (evid[i] == 1 || evid[i] == 4) {      // is dosing event
          if (tlag[ibegin_tlag][cmt[i] - 1] > 0.0) {       // tlag dose
            n++;
          }
          if (addl[i] > 0 && ii[i] > 0) {        // has addl doses
            if (rate[i] > 0 && amt[i] > 0) {
              n++;                               // end ev for IV dose
              n += 2 * addl[i];                  // end ev for addl IV dose
            } else {
              n += addl[i];
            }
            if (tlag[ibegin_tlag][cmt[i] - 1] > 0.0) {     // tlag dose
              n += addl[i];
            }
          } else if (rate[i] > 0 && amt[i] > 0) {
            n++;                                 // end event for IV dose
          }
        }
      }
      res = n;
    } else {
      // int n = time.size();
      // std::vector<std::tuple<double, int>> addl_dose;
      // for (size_t i = 0; i < time.size(); i++) {
      //   if (evid[i] == 1 || evid[i] == 4) {      // is dosing event
      //     if (tlag[0][cmt[i] - 1] > 0.0) {       // tlag dose
      //       n++;
      //     }
      //     if (addl[i] > 0 && ii[i] > 0) {        // has addl doses
      //       if (rate[i] > 0 && amt[i] > 0) {
      //         n++;                               // end ev for IV dose
      //         n += 2 * addl[i];                  // end ev for addl IV dose
      //       } else {
      //         n += addl[i];
      //       }
      //       for (int j = 0; j < addl[i]; ++j) {
      //         addl_dose.push_back(std::make_tuple(value_of(time[i]) + (1+j) * value_of(ii[i]), cmt[i]));
      //       }
      //     } else if (rate[i] > 0 && amt[i] > 0) {
      //       n++;                                 // end event for IV dose
      //     }
      //   }
      // }
      // std::sort(addl_dose.begin(), addl_dose.end(),
      //           [](std::tuple<double, int>& a, std::tuple<double, int>& b)
      //           {
      //             return std::get<0>(a) < std::get<0>(b);
      //           });
    }

    return res;
  }
};

}

#endif
