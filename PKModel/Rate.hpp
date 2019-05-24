#ifndef STAN_MATH_TORSTEN_PKMODEL_RATE_HPP
#define STAN_MATH_TORSTEN_PKMODEL_RATE_HPP

#include <stan/math/torsten/PKModel/Event.hpp>
#include <stan/math/torsten/PKModel/functions.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <vector>

namespace torsten {

/**
 * The RateHistory class defines objects that contain a vector of rates,
 * along with a series of functions that operate on them.
 */
template <typename T_rate>
struct RateHistory {

  template<typename T2>
  struct Rate {
    double t;
    std::vector<T2> r;

    Rate(int ncmt) : t(0), r(ncmt, 0.0)
    {}
  };

  std::vector<Rate<T_rate> > Rates;

  /*
   * generate rates using event history
   */
  template<typename T0, typename T1, typename T2, typename T3, typename T4_container, typename T5, typename T6>
  RateHistory(EventHistory<T0, T1, T2, T3, T4_container, T5, T6>& events, int nCmt) {
    using T_time = typename EventHistory<T0, T1, T2, T3, T4_container, T5, T6>::T_time;
    using std::vector;
    using stan::math::value_of;

    const int n = events.size();
    for (size_t i = 0; i < n; ++i) {
      if ((events.is_dosing(i)) && (events.rate(i) > 0 && events.amt(i) > 0)) {
        T_time endTime = events.time(i) + events.amt(i)/events.rate(i);
        Event<T_time, T1, T2, T3> newEvent(endTime, 0, 0, 0, 2, events.cmt(i), 0, 0, false, true);
        events.InsertEvent(newEvent);
      }
    }
    if (!events.Check()) events.Sort();

    Rate<T_rate> newRate(nCmt);
    // unique_times is sorted
    std::vector<int> unique_times(events.unique_times());
    for (auto i : unique_times) {
      newRate.t = value_of(events.time(i));
      Rates.push_back(newRate);
    }
    sort();

    for (size_t i = 0; i < events.size(); ++i) {
      if ((events.is_dosing(i)) && (events.rate(i) > 0 && events.amt(i) > 0)) {
        double t0 = value_of(events.time(i));
        double t1 = t0 + value_of(events.amt(i)/events.rate(i));
        for (auto&& r : Rates) {
          if (r.t > t0 && r.t <= t1) {
            r.r[events.cmt(i) - 1] += events.rate(i);
          }
        }
      }
    }
  }

  double time(int i) { return Rates[i].t; }

  const T_rate& rate(int i, int j) { return Rates[i].r[j]; }

  struct by_time {
    bool operator()(Rate<T_rate> const &a, Rate<T_rate> const &b) {
      return a.t < b.t;
    }
  };

  void sort() {
    if (!std::is_sorted(Rates.begin(), Rates.end(), by_time()))
      std::sort(Rates.begin(), Rates.end(), by_time());
  }
};

}

#endif
