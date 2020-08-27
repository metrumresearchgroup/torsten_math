#ifndef STAN_MATH_TORSTEN_EVENT_HISTORY_HPP
#define STAN_MATH_TORSTEN_EVENT_HISTORY_HPP

#include <iomanip>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/prim/meta/return_type.hpp>
#include <stan/math/prim/err/check_greater_or_equal.hpp>
#include <stan/math/torsten/PKModel/functions.hpp>
#include <stan/math/torsten/pk_nsys.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

namespace torsten {

  template<typename T0, typename T4,
           template<typename...> class theta_container,
           int npar, typename... Ts>
  struct NonEventParameters;

  template<typename T0, typename T4,
           template<typename...> class theta_container,
           typename T5, typename T6>
  struct NonEventParameters<T0, T4, theta_container, 3, T5, T6> {
    /// <time, <theta index, F index, lag index> >
    static constexpr int npar = 3;
    using par_t = std::pair<double, std::array<int, npar> >;
    using lag_t = T6;
    using biovar_t = T5;

    std::vector<par_t> pars;
    const std::vector<T0>& time_;
    const std::vector<theta_container<T4>>& theta_;
    const std::vector<std::vector<T5> >& biovar_;
    const std::vector<std::vector<T6> >& tlag_;
    // const std::vector<std::vector<double> >& x_r_;
    // const std::vector<std::vector<int> >& x_i_;

    NonEventParameters(int ibegin, int isize,
                       const std::vector<T0>& time,
                       int ibegin_theta, int isize_theta,
                       const std::vector<theta_container<T4>>& theta,
                       int ibegin_biovar, int isize_biovar,
                       const std::vector<std::vector<T5> >& biovar,
                       int ibegin_tlag, int isize_tlag,
                       const std::vector<std::vector<T6> >& tlag) :
      pars(isize),
      time_(time),
      theta_(theta),
      biovar_(biovar),
      tlag_(tlag)
    {
      for (int i = 0; i < isize; ++i) {
        int j = isize_theta   > 1 ? ibegin_theta  + i : ibegin_theta;
        int k = isize_biovar  > 1 ? ibegin_biovar + i : ibegin_biovar;
        int l = isize_tlag    > 1 ? ibegin_tlag   + i : ibegin_tlag;
        pars[i] = std::make_pair<double, std::array<int, npar> >(stan::math::value_of(time_[ibegin + i]), {j, k, l });
      }
      sort();
    }

  template <typename rec_t>
  NonEventParameters(int id, const rec_t& rec,
                     int ibegin_theta, int isize_theta,
                     int ibegin_biovar, int isize_biovar,
                     int ibegin_tlag, int isize_tlag,
                     const std::vector<theta_container<T4>>& theta,
                     const std::vector<std::vector<T5> >& biovar,
                     const std::vector<std::vector<T6> >& tlag) :
    NonEventParameters(rec.begin_[id], rec.len_[id], rec.time_,
                       ibegin_theta, isize_theta, theta,
                       ibegin_biovar, isize_biovar, biovar,
                       ibegin_tlag, isize_tlag, tlag)
    {}

  template <typename rec_t>
  NonEventParameters(int id, const rec_t& rec,
                     const std::vector<theta_container<T4>>& theta,
                     const std::vector<std::vector<T5> >& biovar,
                     const std::vector<std::vector<T6> >& tlag) :
    NonEventParameters(rec.begin_[id], rec.len_[id], rec.time_,
                       rec.begin_param(id, theta), rec.len_param(id, theta), theta,
                       rec.begin_param(id, biovar), rec.len_param(id, biovar), biovar,
                       rec.begin_param(id, tlag), rec.len_param(id, tlag), tlag)
    {}

    inline void set_par_time(int i, double t) {
      std::get<0>(pars[i]) = t;
    }

    inline void set_par_array(int i, const std::array<int, npar>& a) {
      std::get<1>(pars[i]) = a;
    }

    inline double get_par_time(int i) const {
      return std::get<0>(pars[i]);
    }

    inline const std::array<int, npar>& get_par_array(int i) const {
      return std::get<1>(pars[i]);
    }

    inline const theta_container<T4>& theta(int i) const {
      return theta_[pars[i].second[0]];
    }

    inline const T5& bioavailability(int iEvent, int iParameter) const {
      return biovar_[get_par_array(iEvent)[1]][iParameter];
    }

    inline const T6& GetValueTlag(int iEvent, int iParameter) const {
      return tlag_[get_par_array(iEvent)[2]][iParameter];
    }


    inline int size() { return pars.size(); }
    /*
     * return if an event is a "reset" event(evid=3 or 4)
     */
    void sort() {
      std::sort(pars.begin(), pars.end(),
                [](const par_t& a, const par_t& b)
                { return a.first < b.first; });
    }

    bool is_ordered() {
      // check that elements are in chronological order.
      int i = pars.size() - 1;
      bool ordered = true;

      while (i > 0 && ordered) {
        ordered = (pars[i].first >= pars[i-1].first);
        i--;
      }
      return ordered;
    }
  };

  template<typename T0, typename T4,
           template<typename...> class theta_container>
  struct NonEventParameters<T0, T4, theta_container, 1> {
    /// <time, <theta index, F index, lag index> >
    static constexpr int npar = 1;
    using par_t = std::pair<double, std::array<int, npar> >;

    std::vector<par_t> pars;
    const std::vector<T0>& time_;
    const std::vector<theta_container<T4>>& theta_;

    NonEventParameters(int ibegin, int isize,
                       const std::vector<T0>& time,
                       int ibegin_theta, int isize_theta,
                       const std::vector<theta_container<T4>>& theta) :
      pars(isize),
      time_(time),
      theta_(theta)
    {
      for (int i = 0; i < isize; ++i) {
        int j = isize_theta   > 1 ? ibegin_theta  + i : ibegin_theta;
        pars[i] = std::make_pair<double, std::array<int, npar> >(stan::math::value_of(time_[ibegin + i]), {j});
      }
      sort();
    }

    inline void set_par_time(int i, double t) {
      std::get<0>(pars[i]) = t;
    }

    inline void set_par_array(int i, const std::array<int, npar>& a) {
      std::get<1>(pars[i]) = a;
    }

    inline double get_par_time(int i) const {
      return std::get<0>(pars[i]);
    }

    inline const std::array<int, npar>& get_par_array(int i) const {
      return std::get<1>(pars[i]);
    }

    inline const theta_container<T4>& theta(int i) const {
      return theta_[pars[i].second[0]];
    }

    inline const double bioavailability(int iEvent, int iParameter) const {
      return 1.0;
    }

    inline const double GetValueTlag(int iEvent, int iParameter) const {
      return 0.0;
    }

    inline int size() { return pars.size(); }
    /*
     * return if an event is a "reset" event(evid=3 or 4)
     */
    void sort() {
      std::sort(pars.begin(), pars.end(),
                [](const par_t& a, const par_t& b)
                { return a.first < b.first; });
    }

    bool is_ordered() {
      // check that elements are in chronological order.
      int i = pars.size() - 1;
      bool ordered = true;

      while (i > 0 && ordered) {
        ordered = (pars[i].first >= pars[i-1].first);
        i--;
      }
      return ordered;
    }
  };

  /**
   * The EventHistory class defines objects that contain a vector of Events,
   * along with a series of functions that operate on them.
   */
  template<typename T0, typename T1, typename T2, typename T3, typename T_lag>
  struct EventHistory {
    // using T_scalar = typename stan::return_type_t<T0, T1, T2, T3, T4, T5, T6>;
    using T_time = typename stan::return_type_t<T0, T1, T2, T3, T_lag>;
    // using T_rate = typename stan::return_type_t<T2, T5>;
    // using T_amt = typename stan::return_type_t<T1, T5>;
    /// <time, <theta index, F index, lag index> >
    using Param = std::pair<double, std::array<int, 3> >;
    using rate_t = std::pair<double, std::vector<T2> >;

    const std::vector<T0>& time_;
    const std::vector<T1>& amt_;
    const std::vector<T2>& rate_;
    const std::vector<T3>& ii_;
    const std::vector<int>& evid_;
    const std::vector<int>& cmt_;
    const std::vector<int>& addl_;
    const std::vector<int>& ss_;

    // internally generated events
    // these events are not from user input but required for event
    // specification. Such examples include those from tlag or
    // infusion dosing.
    const size_t num_event_times;
    std::vector<T_time> gen_time;
    std::vector<T1> gen_amt;
    std::vector<T2> gen_rate;
    std::vector<T3> gen_ii;
    std::vector<int> gen_cmt;
    std::vector<int> gen_addl;
    std::vector<int> gen_ss;

    // 0: original(0)/generated(1)
    // 1: index in original/generated arrays
    // 2: evid
    // 3: is new?(0/1)
    using IDVec = std::array<int, 4>;
    std::vector<IDVec> idx;

    // rate at distinct time
    std::vector<rate_t> rates;
    std::vector<int> rate_index;

    inline bool keep(const IDVec& id)  const { return id[0] == 0; }
    inline bool isnew(const IDVec& id) const { return id[3] == 1; }
    inline int evid (const IDVec& id) const { return id[2] ; }

    inline bool keep(int i)  const { return keep(idx[i]); }
    inline bool isnew(int i) const { return isnew(idx[i]); }
    inline int evid (int i) const { return evid(idx[i]); }

    /*
     * for a population with data in ragged array form, we
     * form the events history using the population data and
     * the location of the individual in the ragged arrays.
     * In this constructor we assume @c p_ii.size() > 1 and
     * @c p_ss.size() > 1.
     *
     * @param ncmt nb. of compartments
     * @param ibegin beginning index of the subject in data array
     * @param isize # of indices for current subject in data array
     * @param p_time event time
     * @param p_amt dosing amount
     * @param p_rate dosing rate
     * @param p_ii dosing interval
     * @param p_evid event id
     * @param p_cmt event compartment
     * @param p_addl additional events
     * @param p_ss steady states flag
     */
    EventHistory(int ncmt, int ibegin, int isize,
                 const std::vector<T0>& p_time, const std::vector<T1>& p_amt,
                 const std::vector<T2>& p_rate, const std::vector<T3>& p_ii,
                 const std::vector<int>& p_evid, const std::vector<int>& p_cmt,
                 const std::vector<int>& p_addl, const std::vector<int>& p_ss) :
      time_(p_time),
      amt_(p_amt),
      rate_(p_rate),
      ii_(p_ii),
      evid_(p_evid),
      cmt_(p_cmt),
      addl_(p_addl),
      ss_(p_ss),
      num_event_times(isize),
      idx(isize, {0, 0, 0, 0})
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
      for (size_t i = 0; i < isize; ++i) {
        idx[i][1] = ibegin + i;
        idx[i][2] = evid_[ibegin + i];
      }
      insert_addl_dose();
      sort_state_time();
    }

    /**
     * EventHistory constructor for an invidual, based on population
     * version.
     *
     * @param ncmt nb. of compartments
     * @param p_time event time
     * @param p_amt dosing amount
     * @param p_rate dosing rate
     * @param p_ii dosing interval
     * @param p_evid event id
     * @param p_cmt event compartment
     * @param p_addl additional events
     * @param p_ss steady states flag
     * @param theta parameter vector
     * @param biovar bioavailability
     * @param tlag lag time
     */
    EventHistory(int ncmt,
                 const std::vector<T0>& p_time, const std::vector<T1>& p_amt,
                 const std::vector<T2>& p_rate, const std::vector<T3>& p_ii,
                 const std::vector<int>& p_evid, const std::vector<int>& p_cmt,
                 const std::vector<int>& p_addl, const std::vector<int>& p_ss)
    : EventHistory(ncmt,
                   0, p_time.size(), p_time, p_amt, p_rate, p_ii, p_evid, p_cmt, p_addl, p_ss)
    {}

    bool is_reset(int i) const {
      return evid(i) == 3 || evid(i) == 4;
    }

    bool is_reset_event(const IDVec& id) {
      return evid(id) == 3 || evid(id) == 4;
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

    /*
     * use current event #i as template to @c push_back
     * another event.
     */
    void insert_event(int i) {
      idx.push_back({1, int(gen_time.size()), idx[i][2], 1});
      gen_time. push_back(time (i));
      gen_amt.  push_back(amt  (i));
      gen_rate. push_back(rate (i));
      gen_ii.   push_back(ii   (i));
      gen_cmt.  push_back(cmt  (i));
      gen_addl. push_back(addl (i));
      gen_ss.   push_back(ss   (i));
    }

    /**
     * Add events to EventHistory object, corresponding to additional dosing,
     * administered at specified inter-dose interval. This information is stored
     * in the addl and ii members of the EventHistory object.
     *
     * Events is sorted at the end of the procedure.
     */
    void insert_addl_dose() {
      for (int i = 0; i < size(); i++) {
        if (is_dosing(i) && ((addl(i) > 0) && (ii(i) > 0))) {
          for (int j = 1; j <= addl(i); j++) {
            insert_event(i);
            gen_time.back() += j * ii(i);
            gen_ii.back() = 0;
            gen_addl.back() = 0;
            gen_ss.back() = 0;
          }
        }
      }
    }

    /*
     * sort PMX events and nonevents times
     */
    void sort_state_time() { std::stable_sort(idx.begin(), idx.end(),
                                              [this](const IDVec &a, const IDVec &b) {
                                                using stan::math::value_of;
                                                double ta = keep(a) ? value_of(time_[a[1]]) : value_of(gen_time[a[1]]);
                                                double tb = keep(b) ? value_of(time_[b[1]]) : value_of(gen_time[b[1]]);
                                                return ta < tb;
                                              });
    }

    /** 
     * Generate end-of-infusion event to indicate the range of the infusion.
     * The new event has a special EVID = 9 and it introduces no action.
     * 
     * @param nCmt nb of compartments for the model
     */
    void generate_rates(int nCmt) {
      using std::vector;
      using stan::math::value_of;

      const int n = size();
      for (size_t i = 0; i < n; ++i) {
        if ((is_dosing(i)) && (rate(i) > 0 && amt(i) > 0)) {
          insert_event(i);
          idx.back()[2] = 9;    // set evid to a special type "9"
          gen_time. back() += amt(i)/rate(i);
          gen_amt.  back() = 0;
          gen_rate. back() = 0;
          gen_ii.   back() = 0;
          gen_addl. back() = 0;
          gen_ss.   back() = 0;
        }
      }
      sort_state_time();

      rate_t newRate{0.0, std::vector<T2>(nCmt, 0.0)};
      // unique_times is sorted
      std::vector<int> ut(unique_times());
      for (auto i : ut) {
        newRate.first = value_of(time(i));
        rates.push_back(newRate);
      }
      // check sorted?
      std::sort(rates.begin(), rates.end(), [](rate_t const &a, rate_t const &b) {
          return a.first < b.first;
        });

      for (size_t i = 0; i < size(); ++i) {
        if ((is_dosing(i)) && (rate(i) > 0 && amt(i) > 0)) {
          double t0 = value_of(time(i));
          double t1 = t0 + value_of(amt(i)/rate(i));
          for (auto&& r : rates) {
            if (r.first > t0 && r.first <= t1) {
              r.second[cmt(i) - 1] += rate(i);
            }
          }
        }
      }

      /*
       * rate index points to the rates for each state time,
       * since there is one rates vector per time, not per event.
       */
      rate_index.resize(idx.size());
      int iRate = 0;
      for (size_t i = 0; i < idx.size(); ++i) {
        if (rates[iRate].first != time(i)) iRate++;
        rate_index[i] = iRate;
      }
    }

    // Access functions
    inline T_time time (const IDVec& id) const { return keep(id) ? time_[id[1]] : gen_time[id[1]] ; }
    inline const T1& amt (const IDVec& id) const { return keep(id) ? amt_[id[1]] : gen_amt[id[1]] ; }
    inline const T2& rate (const IDVec& id) const { return keep(id) ? rate_[id[1]] : gen_rate[id[1]] ; }
    inline const T3& ii (const IDVec& id) const { return keep(id) ? ii_[id[1]] : gen_ii[id[1]] ; }
    inline int cmt (const IDVec& id) const { return keep(id) ? cmt_[id[1]] : gen_cmt[id[1]] ; }
    inline int addl (const IDVec& id) const { return keep(id) ? addl_[id[1]] : gen_addl[id[1]] ; }
    inline int ss (const IDVec& id) const { return keep(id) ? ss_[id[1]] : gen_ss[id[1]] ; }

    inline T_time time (int i) const { return time(idx[i]); }
    inline const T1& amt (int i) const { return amt (idx[i]); }
    inline const T2& rate (int i) const { return rate(idx[i]); }
    inline const T3& ii (int i) const { return ii  (idx[i]); }
    inline int cmt (int i) const { return cmt (idx[i]); }
    inline int addl (int i) const { return addl(idx[i]); }
    inline int ss (int i) const { return ss  (idx[i]); }

    inline size_t size() const { return idx.size(); }


    std::vector<int> unique_times() {
      std::vector<int> t(idx.size());
      std::iota(t.begin(), t.end(), 0);
      auto last = std::unique(t.begin(), t.end(),
                              [this](const int& i, const int& j) {return time(i) == time(j);});
      t.erase(last, t.end());
      return t;
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
