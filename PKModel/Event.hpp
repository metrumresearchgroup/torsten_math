#ifndef STAN_MATH_TORSTEN_PKMODEL_EVENT_HPP
#define STAN_MATH_TORSTEN_PKMODEL_EVENT_HPP

#include <iomanip>
#include <stan/math/torsten/return_type.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <stan/math/torsten/PKModel/functions.hpp>
#include <stan/math/torsten/pk_nsys.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <vector>

namespace torsten {

template<typename T_time, typename T_parameters_container, typename T_biovar,
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
  template<typename T0, typename T1, typename T2, typename T3, typename T4_container, typename T5, typename T6>
struct EventHistory {
    using T4 = typename stan::math::value_type<T4_container>::type;
    using T_scalar = typename torsten::return_t<T0, T1, T2, T3, T4, T5, T6>::type;
  using T_time = typename torsten::return_t<T0, T1, T6, T2>::type;
  using T_rate = typename torsten::return_t<T2, T5>::type;
  using T_amt = typename torsten::return_t<T1, T5>::type;

    const std::vector<T0>& time_;
    const std::vector<T1>& amt_;
    const std::vector<T2>& rate_;
    const std::vector<T3>& ii_;
    const std::vector<int>& evid_;
    const std::vector<int>& cmt_;
    const std::vector<int>& addl_;
    const std::vector<int>& ss_;
    const std::vector<T4_container>& theta_;
    const std::vector<std::vector<T5> >& biovar_;
    const std::vector<std::vector<T6> >& tlag_;

    // internally generated events
    const size_t events_size;
    std::vector<T_time> gen_time;
    std::vector<T1> gen_amt;
    std::vector<T2> gen_rate;
    std::vector<T3> gen_ii;
    std::vector<int> gen_evid;
    std::vector<int> gen_cmt;
    std::vector<int> gen_addl;
    std::vector<int> gen_ss;

    using IDVec = std::array<int, 7>;
    // 0: original(0)/generated(1)
    // 1: index in original/generated arrays
    // 2: evid
    // 3: is new?(0/1)
    // 4: theta id
    // 5: biovar id
    // 6: tlag id
    std::vector<IDVec> index;

    inline bool keep(const IDVec& id)  const { return id[0] == 0; }
    inline bool isnew(const IDVec& id) const { return id[3] == 1; }
    inline bool keep(int i)  const { return keep(index[i]); }
    inline bool isnew(int i) const { return isnew(index[i]); }
    int evid (int i) const { return index[i][2] ; }


  // EventHistory() : Events() {}

  EventHistory(const std::vector<T0>& p_time, const std::vector<T1>& p_amt,
               const std::vector<T2>& p_rate, const std::vector<T3>& p_ii,
               const std::vector<int>& p_evid, const std::vector<int>& p_cmt,
               const std::vector<int>& p_addl, const std::vector<int>& p_ss,
               const std::vector<T4_container>& theta,
               const std::vector<std::vector<T5> >& biovar,
               const std::vector<std::vector<T6> >& tlag)
    :
    time_(p_time),
    amt_(p_amt),
    rate_(p_rate),
    ii_(p_ii),
    evid_(p_evid),
    cmt_(p_cmt),
    addl_(p_addl),
    ss_(p_ss),
    theta_(theta),
    biovar_(biovar),
    tlag_(tlag),
    events_size(time_.size()),
    index(time_.size(), {0, 0, 0, 0, 0, 0, 0})
  {
    for (size_t i = 0; i < time_.size(); ++i) {
      index[i][1] = i;
      index[i][2] = evid_[i];
      if (theta.size() > 1)  index[i][4] = i;
      if (biovar.size() > 1) index[i][5] = i;
      if (tlag.size() > 1)   index[i][6] = i;
    }
    Sort();
    AddlDoseEvents();
  }

  /*
   * for a population with data in ragged array form, we
   * form the events history using the population data and
   * the location of the individual in the ragged arrays.
   * In this constructor we assume @c p_ii.size() > 1 and
   * @c p_ss.size() > 1.
   */
  EventHistory(int ibegin, int isize,
               const std::vector<T0>& p_time, const std::vector<T1>& p_amt,
               const std::vector<T2>& p_rate, const std::vector<T3>& p_ii,
               const std::vector<int>& p_evid, const std::vector<int>& p_cmt,
               const std::vector<int>& p_addl, const std::vector<int>& p_ss,
               int ibegin_theta, int isize_theta,
               const std::vector<T4_container>& theta,
               int ibegin_biovar, int isize_biovar,
               const std::vector<std::vector<T5> >& biovar,
               int ibegin_tlag, int isize_tlag,
               const std::vector<std::vector<T6> >& tlag) :
    time_(p_time),
    amt_(p_amt),
    rate_(p_rate),
    ii_(p_ii),
    evid_(p_evid),
    cmt_(p_cmt),
    addl_(p_addl),
    ss_(p_ss),
    theta_(theta),
    biovar_(biovar),
    tlag_(tlag),
    events_size(isize),
    index(isize, {0, 0, 0, 0, ibegin_theta, ibegin_biovar, ibegin_tlag})
    {
      const int iend = ibegin + isize;
      using stan::math::check_greater_or_equal;
      static const char* caller = "EventHistory::EventHistory";
      check_greater_or_equal(caller, "isize", isize , 1);
      check_greater_or_equal(caller, "time size", p_time.size() , size_t(iend));
      check_greater_or_equal(caller, "amt size", p_amt.size()   , size_t(iend));
      check_greater_or_equal(caller, "rate size", p_rate.size() , size_t(iend));
      check_greater_or_equal(caller, "ii size", p_ii.size()     , size_t(iend));
      check_greater_or_equal(caller, "evid size", p_evid.size() , size_t(iend));
      check_greater_or_equal(caller, "cmt size", p_cmt.size()   , size_t(iend));
      check_greater_or_equal(caller, "addl size", p_addl.size() , size_t(iend));
      check_greater_or_equal(caller, "ss size", p_ss.size()     , size_t(iend));
      for (size_t i = ibegin; i < iend; ++i) {
        index[i - ibegin][1] = i;
        index[i - ibegin][2] = evid_[i];
        if (isize_theta > 1)  index[i-ibegin][4] = ibegin_theta + i-ibegin;
        if (isize_biovar > 1) index[i-ibegin][5] = ibegin_biovar + i-ibegin;
        if (isize_tlag > 1)   index[i-ibegin][6] = ibegin_tlag + i-ibegin;
      }
      Sort();
      AddlDoseEvents();
    }

  /*
   * Check if the events are in chronological order
   */
  bool Check() {
    int i = size() - 1;
    bool ordered = true;

    while ((i > 0) && (ordered)) {
      // note: evid = 3 and evid = 4 correspond to reset events
      ordered = (((time(i) >= time(i-1)) || (evid(i) == 3)) || (evid(i) == 4));
      i--;
    }
    return ordered;
  }

  Event<T_time, T1, T2, T3> GetEvent(int i) {
  Event<T_time, T1, T2, T3>
    newEvent(time(i), amt(i), rate(i), ii(i), evid(i), cmt(i), addl(i), ss(i), keep(i), isnew(i));
    return newEvent;
  }

  void InsertEvent(Event<T_time, T1, T2, T3> p_Event) {
    index.push_back({1, int(gen_time.size()), p_Event.evid, 1, 0, 0, 0});
    gen_time.push_back(p_Event.time);
    gen_amt.push_back(p_Event.amt);
    gen_rate.push_back(p_Event.rate);
    gen_ii.push_back(p_Event.ii);
    gen_evid.push_back(p_Event.evid);
    gen_cmt.push_back(p_Event.cmt);
    gen_addl.push_back(p_Event.addl);
    gen_ss.push_back(p_Event.ss);
    // Events.push_back(p_Event);
  }

  // void RemoveEvent(int i) {
  //   assert(i >= 0);
  //   Events.erase(Events.begin() + i);
  // }

  // void CleanEvent() {
  //   int nEvent = Events.size();
  //   for (int i = 0; i < nEvent; i++)
  //     if (Events[i].keep == false) RemoveEvent(i);
  //  }

  bool is_reset(int i) const {
    return evid(i) == 3 || evid(i) == 4;
  }

  bool is_dosing(int i) const {
    return evid(i) == 1 || evid(i) == 4;
  }

  /*
   * if an event is steady-state dosing event.
   */
  bool is_ss_dosing(int i) const {
    return (is_dosing(i) && (ss(i) == 1 || ss(i) == 2)) || ss(i) == 3;
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
    for (size_t i = 0; i < size(); i++) {
      if (is_dosing(i) && ((addl(i) > 0) && (ii(i) > 0))) {
        Event<T_time, T1, T2, T3> newEvent = GetEvent(i);
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

    // bool by_time(const IDVec &a, const IDVec &b) const {
    //   using stan::math::value_of;
    //   double ta = keep(a) ? value_of(time_[a[1]]) : value_of(gen_time[a[1]]);
    //   double tb = keep(b) ? value_of(time_[b[1]]) : value_of(gen_time[b[1]]);
    //   return ta < tb;
    // }

    void Sort() { std::stable_sort(index.begin(), index.end(),
                                   [this](const IDVec &a, const IDVec &b) {
                                     using stan::math::value_of;
                                     double ta = keep(a) ? value_of(time_[a[1]]) : value_of(gen_time[a[1]]);
                                     double tb = keep(b) ? value_of(time_[b[1]]) : value_of(gen_time[b[1]]);
                                     return ta < tb;
                                   });
    }

  // Access functions
  T_time time (int i) const { return keep(index[i]) ? time_[index[i][1]] : gen_time[index[i][1]] ; }
  T1 amt   (int i)    const { return keep(index[i]) ? amt_[index[i][1]] : gen_amt[index[i][1]] ; }
  T2 rate (int i)     const { return keep(index[i]) ? rate_[index[i][1]] : gen_rate[index[i][1]] ; }
  T3 ii     (int i)   const { return keep(index[i]) ? ii_[index[i][1]] : gen_ii[index[i][1]] ; }
  int cmt     (int i) const { return keep(index[i]) ? cmt_[index[i][1]] : gen_cmt[index[i][1]] ; }
  int addl    (int i) const { return keep(index[i]) ? addl_[index[i][1]] : gen_addl[index[i][1]] ; }
  int ss      (int i) const { return keep(index[i]) ? ss_[index[i][1]] : gen_ss[index[i][1]] ; }
  size_t size()       const { return index.size(); }


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
  void AddLagTimes(const ModelParameterHistory<T_time, T4_container, T5, T6>& Parameters, int nCmt) {
    int nEvent = size(), pSize = Parameters.get_size();
    assert((pSize = nEvent) || (pSize == 1));

    int iEvent = nEvent - 1, ipar;
    Event<T_time, T1, T2, T3> newEvent;
    while (iEvent >= 0) {
      // cmt = ;

      if (is_dosing(iEvent)) {
        ipar = std::min(iEvent, pSize - 1);  // ipar is the index of the ith
                                             // event or 0, if the parameters
                                             // are constant.
        if (Parameters.GetValueTlag(ipar, cmt(iEvent) - 1) != 0) {
          newEvent = GetEvent(iEvent);
          newEvent.time += Parameters.GetValueTlag(ipar, cmt(iEvent) - 1);
          newEvent.keep = false;
          newEvent.isnew = true;
          // newEvent.evid = 2  // CHECK
          InsertEvent(newEvent);

          // Events[iEvent].evid = 2;  // Check
          index[iEvent][2] = 2;
          // The above statement changes events so that CleanEvents does
          // not return an object identical to the original. - CHECK
        }
      }
      iEvent--;
    }
    Sort();
  }

  /*
   * Overloading the << Operator
   */
  friend std::ostream& operator<<(std::ostream& os, const EventHistory& ev) {
    const int w = 6;
    os << "\n";
    os << std::setw(w) << "time" <<
      std::setw(w) << "amt" <<
      std::setw(w) << "rate" <<
      std::setw(w) << "ii" <<
      std::setw(w) << "evid" <<
      std::setw(w) << "cmt" <<
      std::setw(w) << "addl" <<
      std::setw(w) << "ss" <<
      std::setw(w) << "keep" <<
      std::setw(w) << "isnew" << "\n";
    for (size_t i = 0; i < ev.size(); ++i) {
      os <<
        std::setw(w)   << ev.time(i) << " " <<
        std::setw(w-1) << ev.amt(i) << " " <<
        std::setw(w-1) << ev.rate(i) << " " <<
        std::setw(w-1) << ev.ii(i) << " " <<
        std::setw(w-1) << ev.evid(i) << " " <<
        std::setw(w-1) << ev.cmt(i) << " " <<
        std::setw(w-1) << ev.addl(i) << " " <<
        std::setw(w-1) << ev.ss(i) << " " <<
        std::setw(w-1) << ev.keep(i) << " " <<
        std::setw(w-1) << ev.isnew(i) << "\n";
    }
    return os;
  }
};

}    // torsten namespace
#endif
