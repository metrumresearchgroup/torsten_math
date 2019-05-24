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
template<typename T0, typename T1, typename T2, typename T3, typename T4_container, typename T5, typename T6>
struct RateHistory {
    using T_time = typename EventHistory<T0, T1, T2, T3, T4_container, T5, T6>::T_time;
    using T_rate = T2;

  using rate_t = std::pair<double, std::vector<T_rate> >;

  EventHistory<T0, T1, T2, T3, T4_container, T5, T6>& events;
  std::vector<rate_t> Rates;

  /*
   * generate rates using event history
   */
  RateHistory(EventHistory<T0, T1, T2, T3, T4_container, T5, T6>& events_in, int nCmt)
    : events(events_in)
  {
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

    rate_t newRate = std::pair<double, std::vector<T_rate> >(0.0, std::vector<T_rate>(nCmt, 0.0));
    // unique_times is sorted
    std::vector<int> unique_times(events.unique_times());
    for (auto i : unique_times) {
      newRate.first = value_of(events.time(i));
      Rates.push_back(newRate);
    }
    sort();

    for (size_t i = 0; i < events.size(); ++i) {
      if ((events.is_dosing(i)) && (events.rate(i) > 0 && events.amt(i) > 0)) {
        double t0 = value_of(events.time(i));
        double t1 = t0 + value_of(events.amt(i)/events.rate(i));
        for (auto&& r : Rates) {
          if (time(r) > t0 && time(r) <= t1) {
            r.second[events.cmt(i) - 1] += events.rate(i);
          }
        }
      }
    }
  }

  double time(int i) { return Rates[i].first; }

  double time(const std::pair<double, std::vector<T_rate>>& r) { return r.first; }

  const T_rate& rate(int i, int j) { return Rates[i].second[j]; }
  const T_rate& rate(const std::pair<double, std::vector<T_rate>>& r, int j) { return r.second[j]; }

  struct by_time {
    bool operator()(rate_t const &a, rate_t const &b) {
      return a.first < b.first;
    }
  };

  void sort() {
    if (!std::is_sorted(Rates.begin(), Rates.end(), by_time()))
      std::sort(Rates.begin(), Rates.end(), by_time());
  }
};

}

#endif
