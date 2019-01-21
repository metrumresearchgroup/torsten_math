#ifndef STAN_MATH_TORSTEN_PKMODEL_EVENT_HPP
#define STAN_MATH_TORSTEN_PKMODEL_EVENT_HPP

#include <Eigen/Dense>
#include <stan/math/torsten/PKModel/functions.hpp>
#include <iostream>
#include <algorithm>
#include <vector>

namespace torsten {

template<typename T_time, typename T_parameters, typename T_biovar,
  typename T_tlag> struct ModelParameterHistory;

/**
 * The Event class defines objects that contain the elements of an event,
 * following NONMEM conventions:
 *    time
 *    amt: amount
 *    rate
 *    ii: interdose interval
 *    evid: event identity
 *      (0) observation
 *      (1) dosing
 *      (2) other
 *      (3) reset
 *      (4) reset and dosing
 *    cmt: compartment in which the event occurs
 *    addl: additional doses
 *    ss: steady state approximation (0: no, 1: yes)
 *    keep: if TRUE, save the predicted amount at this event
 *          in the final output of the pred function.
 *    isnew: if TRUE, event was created when pred augmented
 *           the input data set
 */
template <typename T_time, typename T_amt, typename T_rate, typename T_ii>
struct Event{

  T_time time;
  T_amt amt;
  T_rate rate;
  T_ii ii;
  int evid, cmt, addl, ss;
  bool keep, isnew;

  Event() : time(0), amt(0), rate(0), ii(0), cmt(0), addl(0), ss(0), keep(false), isnew(false)
  {}

  Event(T_time p_time, T_amt p_amt, T_rate p_rate, T_ii p_ii, int p_evid,
        int p_cmt, int p_addl, int p_ss, bool p_keep, bool p_isnew) :
    time  (p_time ),
    amt   (p_amt  ),
    rate  (p_rate ),
    ii    (p_ii   ),
    evid  (p_evid ),
    cmt   (p_cmt  ),
    addl  (p_addl ),
    ss    (p_ss   ),
    keep  (p_keep ),
    isnew (p_isnew)
  {}

  /*
   * copy constructor
   */
  Event(const Event& other) :
    time  (other.time ),
    amt   (other.amt  ),
    rate  (other.rate ),
    ii    (other.ii   ),
    evid  (other.evid ),
    cmt   (other.cmt  ),
    addl  (other.addl ),
    ss    (other.ss   ),
    keep  (other.keep ),
    isnew (other.isnew)
  {}
    
  /**
   * The function operator is handy when we need to define the same event
   * multiple times, as we might in a FOR loop.
   */
  Event operator()(T_time p_time, T_amt p_amt, T_rate p_rate, T_ii p_ii,
                   int p_evid, int p_cmt, int p_addl, int p_ss, bool p_keep,
                   bool p_isnew) {
    Event newEvent;
    newEvent.time = p_time;
    newEvent.amt = p_amt;
    newEvent.rate = p_rate;
    newEvent.ii = p_ii;
    newEvent.evid = p_evid;
    newEvent.cmt = p_cmt;
    newEvent.addl = p_addl;
    newEvent.ss = p_ss;
    newEvent.keep = p_keep;
    newEvent.isnew = p_isnew;
    return newEvent;
  }

  // Access functions
  T_time get_time() { return time; }
  T_amt get_amt() { return amt; }
  T_rate get_rate() { return rate; }
  T_ii get_ii() { return ii; }
  int get_evid() { return evid; }
  int get_cmt() { return cmt; }
  int get_addl() { return addl; }
  int get_ss() { return ss; }
  bool get_keep() { return keep; }
  bool get_isnew() { return isnew; }

  // declare friends
};

/**
 * The EventHistory class defines objects that contain a vector of Events,
 * along with a series of functions that operate on them.
 */
template<typename T_time, typename T_amt, typename T_rate, typename T_ii>
struct EventHistory {
  std::vector<Event<T_time, T_amt, T_rate, T_ii> > Events;

  EventHistory() : Events() {}

  template<typename T0, typename T1, typename T2, typename T3>
  EventHistory(const std::vector<T0>& p_time, const std::vector<T1>& p_amt,
               const std::vector<T2>& p_rate, const std::vector<T3>& p_ii,
               const std::vector<int>& p_evid, const std::vector<int>& p_cmt,
               const std::vector<int>& p_addl, const std::vector<int>& p_ss)
    : Events(p_evid.size())
  {
    for (size_t i = 0; i < p_evid.size(); i++) {
      if (p_ii.size() == 1) {   // no additional dosing
        if (p_ss.size() == 1) {
          Events[i] = Event<T_time, T_amt, T_rate, T_ii>(p_time[i], p_amt[i], p_rate[i], 0, p_evid[i], p_cmt[i], 0, 0, true, false);
        } else {
          Events[i] = Event<T_time, T_amt, T_rate, T_ii>(p_time[i], p_amt[i], p_rate[i], 0, p_evid[i], p_cmt[i], 0, p_ss[i], true, false);
        }
      } else {
        if (p_ss.size() == 1) {
          Events[i] = Event<T_time, T_amt, T_rate, T_ii>(p_time[i], p_amt[i], p_rate[i], p_ii[i], p_evid[i], p_cmt[i], p_addl[i], 0, true, false);
        } else {
          Events[i] = Event<T_time, T_amt, T_rate, T_ii>(p_time[i], p_amt[i], p_rate[i], p_ii[i], p_evid[i], p_cmt[i], p_addl[i], p_ss[i], true, false);
        }
      }
    }
  }
  /*
   * copy constructor
   */
  template<typename T0, typename T1, typename T2, typename T3>
  EventHistory(const EventHistory<T0, T1, T2, T3>& other) : Events() {
    Events.reserve(other.size());
    for (size_t i = 0; i < other.size(); ++i) {
      Events.push_back(other.Events[i]);
    }
  }
  /*
   * Check if the events are in chronological order
   */
  bool Check() {
    int i = Events.size() - 1;
    bool ordered = true;

    while ((i > 0) && (ordered)) {
      // note: evid = 3 and evid = 4 correspond to reset events
      ordered = (((Events[i].time >= Events[i - 1].time)
        || (Events[i].evid == 3)) || (Events[i].evid == 4));
      i--;
    }
    return ordered;
  }

  Event<T_time, T_amt, T_rate, T_ii> GetEvent(int i) {
  Event<T_time, T_amt, T_rate, T_ii>
    newEvent(Events[i].time, Events[i].amt, Events[i].rate, Events[i].ii,
      Events[i].evid, Events[i].cmt, Events[i].addl, Events[i].ss,
      Events[i].keep, Events[i].isnew);
    return newEvent;
  }

  void InsertEvent(Event<T_time, T_amt, T_rate, T_ii> p_Event) {
    Events.push_back(p_Event);
  }

  void RemoveEvent(int i) {
    assert(i >= 0);
    Events.erase(Events.begin() + i);
  }

  void CleanEvent() {
    int nEvent = Events.size();
    for (int i = 0; i < nEvent; i++)
      if (Events[i].keep == false) RemoveEvent(i);
   }

  bool is_reset(int i) const {
    return evid(i) == 3 || evid(i) == 4;
  }

  bool is_dosing(int i) const {
    return evid(i) == 1 || evid(i) == 4;
  }

  static bool is_dosing(const std::vector<int>& event_id, int i) {
    return event_id[i] == 1 || event_id[i] == 4;
  }

  bool is_bolus_dosing(int i) const {
    const double eps = 1.0E-12;
    return is_dosing(i) && rate(i) < eps;
  }

  /**
   * Add events to EventHistory object, corresponding to additional dosing,
   * administered at specified inter-dose interval. This information is stored
   * in the addl and ii members of the EventHistory object.
   *
   * Events is sorted at the end of the procedure.
   */
  void AddlDoseEvents() {
    for (size_t i = 0; i < Events.size(); i++) {
      if (is_dosing(i) && ((Events[i].addl > 0) && (Events[i].ii > 0))) {
        Event<T_time, T_amt, T_rate, T_ii> newEvent = GetEvent(i);
        newEvent.addl = 0;
        newEvent.ii = 0;
        newEvent.ss = 0;
        newEvent.keep = false;
        newEvent.isnew = true;

        for (int j = 1; j <= addl(i); j++) {
          newEvent.time = time(i) + j * ii(i);
          InsertEvent(newEvent);
        }
      }
    }
  }

  struct by_time {
    bool operator()(const Event<T_time, T_amt, T_rate, T_ii> &a,
      const Event<T_time, T_amt, T_rate, T_ii> &b) {
        return a.time < b.time;
    }
  };

  void Sort() { std::stable_sort(Events.begin(), Events.end(), by_time()); }

  // Access functions
  T_time time (int i) const { return Events[i].time; }
  T_amt amt   (int i) const { return Events[i].amt; }
  T_rate rate (int i) const { return Events[i].rate; }
  T_ii ii     (int i) const { return Events[i].ii; }
  int evid    (int i) const { return Events[i].evid; }
  int cmt     (int i) const { return Events[i].cmt; }
  int addl    (int i) const { return Events[i].addl; }
  int ss      (int i) const { return Events[i].ss; }
  bool keep   (int i) const { return Events[i].keep; }
  bool isnew  (int i) const { return Events[i].isnew; }
  size_t size()       const { return Events.size(); }


  /**
   * Implement absorption lag times by modifying the times of the dosing events.
   * Two cases: parameters are either constant or vary with each event.
   * Function sorts events at the end of the procedure.
   *
   * @tparam T_parameters type of scalar model parameters
   * @param[in] ModelParameterHistory object that stores parameters for each event
   * @param[in] nCmt
   * @return - modified events that account for absorption lag times
   */
  template<typename T_parameters, typename T_biovar, typename T_tlag>
  void AddLagTimes(ModelParameterHistory<T_time, T_parameters, T_biovar,
                   T_tlag> Parameters, int nCmt) {
    int nEvent = Events.size(), pSize = Parameters.get_size();
    assert((pSize = nEvent) || (pSize == 1));

    int iEvent = nEvent - 1, cmt, ipar;
    Event<T_time, T_amt, T_rate, T_ii> newEvent;
    while (iEvent >= 0) {
      cmt = Events[iEvent].cmt;

      if (is_dosing(iEvent)) {
        ipar = std::min(iEvent, pSize - 1);  // ipar is the index of the ith
                                             // event or 0, if the parameters
                                             // are constant.
        if (Parameters.GetValueTlag(ipar, cmt - 1) != 0) {
          newEvent = GetEvent(iEvent);
          newEvent.time += Parameters.GetValueTlag(ipar, cmt - 1);
          newEvent.keep = false;
          newEvent.isnew = true;
          // newEvent.evid = 2  // CHECK
          InsertEvent(newEvent);

          Events[iEvent].evid = 2;  // Check
          // The above statement changes events so that CleanEvents does
          // not return an object identical to the original. - CHECK
        }
      }
      iEvent--;
    }
    Sort();
  }
};

}    // torsten namespace
#endif
