#ifndef STAN_MATH_TORSTEN_PKMODEL_RATE_HPP
#define STAN_MATH_TORSTEN_PKMODEL_RATE_HPP

#include <Eigen/Dense>
#include <stan/math/torsten/PKModel/functions.hpp>
#include <algorithm>
#include <vector>

namespace torsten {

/**
 * The Rate class defines objects that contain the rate in each compartment
 * at each time of the event schedule (but not nescessarily at each event,
 * since two events may happen at the same time).
 */
template<typename T_time, typename T_rate>
struct Rate {
  T_time time;
  std::vector<T_rate> rate;  // rate for each compartment

  Rate() {
    std::vector<T_rate> v(1, 0);
    time = 0;
    rate = v;
  }

  Rate(T_time p_time, std::vector<T_rate> p_rate) {
    time = p_time;
    rate = p_rate;
  }

  // access functions
  T_time get_time() const { return time; }
  std::vector<T_rate> get_rate() const { return rate; }

  // Overload = operator
  // Allows us to construct a rate of var from a rate of double
  template <typename T0, typename T1>
  void copy(const Rate<T0, T1>& rate1) {
    time = rate1.get_time();
    rate.resize(rate1.get_rate().size());
    for (size_t i = 0; i < rate.size(); i++)
      rate[i] = rate1.get_rate()[i];
  }
};

/**
 * The RateHistory class defines objects that contain a vector of rates,
 * along with a series of functions that operate on them.
 */
template <typename T_time, typename T_rate>
struct RateHistory {
  std::vector<Rate<T_time, T_rate> > Rates;

  /*
   * generate rates using event history
   */
  template <typename T_amt, typename T_ii>
  RateHistory(torsten::EventHistory<T_time, T_amt, T_rate, T_ii>& events, int nCmt) {
    using std::vector;

    if (!events.Check()) events.Sort();

    vector<T_rate> rate_init(nCmt, 0);
    Rate<T_time, T_rate> newRate(0, rate_init);
    for (size_t j = 0; j < events.size(); j++)
      if (j == 0 || events.time(j) != events.time(j - 1)) {
        newRate.time = events.time(j);
        Rates.push_back(newRate);
      }

    // RemoveRate(0);  // remove rate created by default constructor.

    if (!Check()) sort();

    // Create time vector for rates
    vector<T_time> RateTimes(Rates.size(), 0);
    for (size_t j = 0; j < Rates.size(); j++) RateTimes[j] = Rates[j].time;

    // Create time vector for events
    vector<T_time> EventTimes(events.size(), 0);
    for (size_t j = 0; j < events.size(); j++) EventTimes[j] = events.time(j);

    size_t i = 0, k, l;
    T_time endTime;
    torsten::Event<T_time, T_amt, T_rate, T_ii> newEvent;
    while (i < events.size()) {
      if ((events.evid(i) == 1 || events.evid(i) == 4) && (events.rate(i) > 0 && events.amt(i) > 0)) {
          endTime = events.time(i) + events.amt(i)/events.rate(i);
          newEvent = newEvent(endTime, 0, 0, 0, 2, events.cmt(i), 0, 0, false, true);
          events.InsertEvent(newEvent);
          if (!events.Check()) events.Sort();
          EventTimes.push_back(endTime);
          std::sort(EventTimes.begin(), EventTimes.end());

          // Only create a new Rate if endTime does not correspond to a time
          // that is already in RateHistory. - CHECK
          if (!find_time(RateTimes, endTime)) {
            newRate.time = endTime;
            Rates.push_back(newRate);
            // InsertRate(newRate);
            if (!Check()) sort();
            RateTimes.push_back(endTime);
            std::sort(RateTimes.begin(), RateTimes.end());
          }

          // Find indexes at which time of event and endtime occur.
          l = SearchReal(RateTimes, events.size(), events.time(i));
          k = SearchReal(RateTimes, events.size(), endTime);

          // Compute Rates for each element between the two times
          for (size_t iRate = l ; iRate < k; iRate++)
            Rates[iRate].rate[events.cmt(i) - 1] += events.rate(i);
        }
        i++;
    }

    // Sort events and rates
    if (!Check()) sort();
    if (!events.Check()) events.Sort();
  }

  T_time time(int i) { return Rates[i].time; }

  T_rate rate(int i, int j) { return Rates[i].rate[j]; }

  bool Check() {
    int i = Rates.size() - 1;
    bool ordered = true;

    while ((i > 0) && (ordered)) {
      ordered = (Rates[i].time >= Rates[i - 1].time);
      i--;
    }
    return ordered;
  }

  struct by_time {
    bool operator()(Rate<T_time, T_rate> const &a, Rate<T_time, T_rate>
      const &b) {
      return a.time < b.time;
    }
  };

  void sort() { std::sort(Rates.begin(), Rates.end(), by_time()); }
};

}

#endif
