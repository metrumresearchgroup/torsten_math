#ifndef STAN_MATH_TORSTEN_PKMODEL_MODELPARAMETERS_HPP
#define STAN_MATH_TORSTEN_PKMODEL_MODELPARAMETERS_HPP

#include <stan/math/prim/mat/fun/to_array_1d.hpp>
#include <stan/math/torsten/PKModel/Event.hpp>
#include <stan/math/torsten/PKModel/ExtractVector.hpp>
#include <stan/math/torsten/PKModel/SearchReal.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <vector>

namespace torsten {

template<typename T_time, typename T_parameters, typename T_biovar, typename T_tlag>
  struct ModelParameterHistory;
/**
 * The ModelParameters class defines objects that contain the parameters of
 * a model at a given time.
 */
template<typename T_time, typename T_parameters, typename T_biovar, typename T_tlag>
struct ModelParameters {
  double time_;
  std::vector<T_parameters> theta_;
  int nrow, ncol;
  std::vector<T_biovar> biovar_;
  std::vector<T_tlag> tlag_;

  ModelParameters() {}

  ModelParameters(const T_time& time,
                  const std::vector<T_biovar>& biovar,
                  const std::vector<T_tlag>& tlag,
                  const Eigen::Matrix<T_parameters, Eigen::Dynamic, Eigen::Dynamic>& K)
    : time_(stan::math::value_of(time)), theta_(K.size()), nrow(K.rows()), ncol(K.cols()), biovar_(biovar), tlag_(tlag)
  {
    Eigen::Matrix<T_parameters, -1, -1>::Map(theta_.data(), nrow, ncol) = K;
  }

  ModelParameters(const T_time& time,
                  const std::vector<T_parameters>& theta,
                  const std::vector<T_biovar>& biovar,
                  const std::vector<T_tlag>& tlag)
    : time_(stan::math::value_of(time)), theta_(theta), biovar_(biovar), tlag_(tlag) {}

  /**
   * Adds parameters. Useful for the mixed solver, where
   * we want to augment the parameters with the intial PK
   * states when calling the numerical integrator.
   */
  template <typename T>
  ModelParameters<T_time, T, T_biovar, T_tlag>
  augment(const std::vector<T>& thetaAdd) const {
    std::vector<T> theta(theta_.size());
    for (size_t i = 0; i < theta.size(); i++) theta[i] = theta_[i];
    for (size_t i = 0; i < thetaAdd.size(); i++) theta.push_back(thetaAdd[i]);
    return
      ModelParameters<T_time, T, T_biovar, T_tlag>
        (time_, theta, biovar_, tlag_);
  }

  template <typename T>
  ModelParameters<T_time, T, T_biovar, T_tlag>
  augment(const Eigen::Matrix<T, Eigen::Dynamic, 1>& thetaAdd)
  const {
    return augment(stan::math::to_array_1d(thetaAdd));
  }

  /**
   * Edit time stored in parameter object.
   */
  void time(double time) {
    time_ = time;
  }

  int CountParameters() const {
    return theta_.size();
  }

  // access functions   // FIX ME - name should be get_theta.
  double get_time() const { return time_; }
  std::vector<T_parameters> get_RealParameters(bool return_matrix) const {
    if (return_matrix) {
      auto k = get_K();
      std::vector<T_parameters> par(k.size());
      for (size_t j = 0; j < par.size(); ++j) par[j] = k(j);
      return par;
    } else {
      return theta_;
    }
  }
  std::vector<T_biovar> get_biovar() const {
    return biovar_;
  }
  std::vector<T_tlag> get_tlag() const {
    return tlag_;
  }
  Eigen::Matrix<T_parameters, Eigen::Dynamic, Eigen::Dynamic> get_K() const {
    Eigen::Matrix<T_parameters, -1, -1> res(nrow, ncol);
    res = Eigen::Matrix<T_parameters, -1, -1>::Map(theta_.data(), nrow, ncol);
    return res;
  }
};

/**
 * The ModelParameterHistory class defines objects that contain a vector
 * of ModelParameters, along with a series of functions that operate on
 * them.
 */
template<typename T_time, typename T1, typename T2, typename T3>
struct ModelParameterHistory {
  static const std::vector<Eigen::Matrix<T1, -1, -1>> dummy_param_matrix;
  static const std::vector<std::vector<T1> > dummy_param_vector;

  const bool has_matrix_param;
  std::vector<std::pair<double, std::array<int, 3> > > time_;
  const std::vector<std::vector<T1> >& theta_;
  const std::vector<Eigen::Matrix<T1, -1, -1> >& K_;
  const std::vector<std::vector<T2> >& biovar_;
  const std::vector<std::vector<T3> >& tlag_;

  template<typename T0>
  ModelParameterHistory(const std::vector<T0>& time,
                        const std::vector<std::vector<T1> >& theta,
                        const std::vector<std::vector<T2> >& biovar,
                        const std::vector<std::vector<T3> >& tlag) :
    has_matrix_param(false),
    time_(time.size()),
    theta_(theta),
    K_(ModelParameterHistory<T_time, T1, T2, T3>::dummy_param_matrix),
    biovar_(biovar),
    tlag_(tlag)
  {
    for (int i = 0; i < time_.size(); ++i) {
      int j = theta.size()  > 1 ? i : 0;
      int k = biovar.size() > 1 ? i : 0;
      int l = tlag.size()   > 1 ? i : 0;
      time_[i] = std::make_pair<double, std::array<int, 3> >(stan::math::value_of(time[i]), {j, k, l} );
    }
  }

  template<typename T0>
  ModelParameterHistory(const std::vector<T0>& time,
                        const std::vector< Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> >& K,
                        const std::vector<std::vector<T2> >& biovar,
                        const std::vector<std::vector<T3> >& tlag) :
    has_matrix_param(true),
    time_(time.size()),
    theta_(ModelParameterHistory<T_time, T1, T2, T3>::dummy_param_vector),
    K_(K),
    biovar_(biovar),
    tlag_(tlag)
  {
    for (int i = 0; i < time_.size(); ++i) {
      int j = K.size()      > 1 ? i : 0;
      int k = biovar.size() > 1 ? i : 0;
      int l = tlag.size()   > 1 ? i : 0;
      time_[i] = std::make_pair<double, std::array<int, 3> >(stan::math::value_of(time[i]), {j, k, l} );
    }
  }

  template<typename T0>
  ModelParameterHistory(const std::vector<T0>& time,
                        const std::vector<std::vector<T1> >& theta,
                        const std::vector< Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> >& K,
                        const std::vector<std::vector<T2> >& biovar,
                        const std::vector<std::vector<T3> >& tlag) :
    has_matrix_param(theta.empty()),
    time_(time.size()),
    theta_(has_matrix_param ? ModelParameterHistory<T_time, T1, T2, T3>::dummy_param_vector : theta),
    K_(has_matrix_param ? K : ModelParameterHistory<T_time, T1, T2, T3>::dummy_param_matrix),
    biovar_(biovar),
    tlag_(tlag)
  {
    for (int i = 0; i < time_.size(); ++i) {
      int j = has_matrix_param ? (K.size() > 1 ? i : 0) : (theta.size() > 1 ? i : 0);
      int k = biovar.size() > 1 ? i : 0;
      int l = tlag.size()   > 1 ? i : 0;
      time_[i] = std::make_pair<double, std::array<int, 3> >(stan::math::value_of(time[i]), {j, k, l} );
    }    
  }

  /*
   * For population data in form of ragged array, we need to
   * generate individual parameter history given the entire
   * population data and the location of the
   * inidividual. However, @c theta, @c biovar and @c tlag
   * could have different lengths, so for each variable we
   * need an index that points to the range that belongs to the individual.
   * Note that if all three variables are of size 1, their
   * time is set to be the first entry of the @c time vector
   */
  template<typename T0>
  ModelParameterHistory(int ibegin, int isize,
                        const std::vector<T0>& time,
                        int ibegin_theta, int isize_theta,
                        const std::vector<std::vector<T1> >& theta,
                        const std::vector< Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> >& K,
                        int ibegin_biovar, int isize_biovar,
                        const std::vector<std::vector<T2> >& biovar,
                        int ibegin_tlag, int isize_tlag,
                        const std::vector<std::vector<T3> >& tlag) :
    has_matrix_param(theta.empty()),
    time_(isize),
    theta_(has_matrix_param ? ModelParameterHistory<T_time, T1, T2, T3>::dummy_param_vector : theta),
    K_(has_matrix_param ? K : ModelParameterHistory<T_time, T1, T2, T3>::dummy_param_matrix),
    biovar_(biovar),
    tlag_(tlag)
  {
    for (int i = 0; i < isize; ++i) {
      int j = isize_theta   > 1 ? ibegin_theta  + i : ibegin_theta;
      int k = isize_biovar  > 1 ? ibegin_biovar + i : ibegin_biovar;
      int l = isize_tlag    > 1 ? ibegin_tlag   + i : ibegin_tlag;
      time_[i] = std::make_pair<double, std::array<int, 3> >(stan::math::value_of(time[ibegin + i]), {j, k, l });
    }
  }

  ModelParameters<T_time, T1, T2, T3> GetModelParameters(int i) const {
    if (has_matrix_param) {
      return { std::get<0>(time_[i]), biovar_[std::get<1>(time_[i])[1]], tlag_[std::get<1>(time_[i])[2]], K_[std::get<1>(time_[i])[0]] };
    } else {
      return { std::get<0>(time_[i]), theta_[std::get<1>(time_[i])[0]], biovar_[std::get<1>(time_[i])[1]], tlag_[std::get<1>(time_[i])[2]] };
    }
  }

  /**
   * MPV.size gives us the number of events.
   * MPV[i].RealParameters.size gives us the number of
   * ODE parameters for the ith event.
   * 
   * FIX ME - rename this GetValueTheta
   */
  const T1& GetValue(int iEvent, int iParameter) const {
    return theta_[std::get<1>(time_[iEvent])[0]][iParameter];
  }

  const T2& GetValueBio(int iEvent, int iParameter) const {
    return biovar_[std::get<1>(time_[iEvent])[1]][iParameter];
  }

  const T3& GetValueTlag(int iEvent, int iParameter) const {
    return tlag_[std::get<1>(time_[iEvent])[2]][iParameter];
  }

  int get_size() const {
    return time_.size();
  }

  void Sort() {
    std::sort(time_.begin(), time_.end(),
              [](const std::pair<double, std::array<int, 3> >& a, const std::pair<double, std::array<int, 3> >& b)
              { return std::get<0>(a) < std::get<0>(b); });
  }

  bool Check() {
  // check that elements are in chronological order.
    int i = time_.size() - 1;
    bool ordered = true;

    while (i > 0 && ordered) {
      ordered = (std::get<0>(time_[i]) >= std::get<0>(time_[i-1]));
      i--;
    }
    return ordered;
  }

  /**
   * COMPLETE MODEL PARAMETERS
   *
   * Completes parameters so that it contains model parameters for each event 
   * in events. If parameters contains only one set of parameters (case where
   * the parameters are constant), this set is replicated for each event in
   * events. Otherwise a new parameter vector is added for each new event 
   * (isnew = true). This parameter vector is identical to the parameter vector
   * at the subsequent event. If the new event occurs at a time posterior to
   * the time of the last event, than the new vector parameter equals the
   * parameter vector of the last event. This amounts to doing an LOCF
   * (Last Observation Carried Forward).
   *
   * Events and Parameters are sorted at the end of the procedure.
   *
   * @param[in] parameters at each event
   * @param[in] events elements (following NONMEM convention) at each event
   * @return - modified parameters and events.
   */
  template<typename T0, typename T_p1, typename T_p2, typename T_p3>
  void CompleteParameterHistory(torsten::EventHistory<T0, T_p1, T_p2, T_p3, T1, T2, T3>& events) {
    int nEvent = events.size();
    assert(nEvent > 0);
    int len_Parameters = time_.size();  // numbers of events for which parameters are determined
    assert(len_Parameters > 0);

    // if (!Check()) Sort();
    Sort();
    if (!events.Check()) events.Sort();
    time_.resize(nEvent);

    int iEvent = 0;
    for (int i = 0; i < len_Parameters - 1; i++) {
      while (events.isnew(iEvent)) iEvent++;  // skip new events
      assert(std::get<0>(time_[i]) == events.time(iEvent));  // compare time of "old' events to time of parameters.
      iEvent++;
    }

    if (len_Parameters == 1)  {
      for (int i = 0; i < nEvent; i++) {
        // FIX ME - inefficient data storage
        time_[i] = std::make_pair<double, std::array<int, 3> >(stan::math::value_of(events.time(i)) , std::array<int,3>(std::get<1>(time_[0])));
        events.index[i][3] = 0;
      }
    } else {  // parameters are event dependent.
      std::vector<T_time> times(nEvent, 0);
      for (int i = 0; i < nEvent; i++) times[i] = time_[i].first;
      iEvent = 0;

      int k, j = 0;
      std::pair<double, std::array<int, 3> > newParameter;
      // ModelParameters<T_time, T1, T2, T3> newParameter;

      for (int i = 0; i < nEvent; i++) {
        while (events.isnew(iEvent)) {
          /* Three cases:
           * (a) The time of the new event is higher than the time of the last
           *     parameter vector in parameters (k = len_parameters).
           *     Create a parameter vector at the the time of the new event,
           *     with the parameters of the last parameter vector.
           *     (Last Observation Carried Forward)
           * (b) The time of the new event matches the time of a parameter vector
           *     in parameters. This parameter vector gets replicated.
           * (c) (a) is not verified and no parameter vector occurs at the time
           *     of the new event. A new parameter vector is created at the time
           *     of the new event, and its parameters are equal to the parameters
           *     of the subsequent parameter vector in parameters.
           */
          // Find the index corresponding to the time of the new event in the
          // times vector.
          k = SearchReal(times, len_Parameters - 1, events.time(iEvent));

          if ((k == len_Parameters) ||
              (stan::math::value_of(events.time(iEvent)) == std::get<0>(time_[k - 1])))
            // newParameter = GetModelParameters(k - 1);
            newParameter = time_[k-1];
          else
            newParameter = time_[k];
            // newParameter = GetModelParameters(k);

          // newParameter.time_ = stan::math::value_of(events.time(iEvent));
          // MPV_[len_Parameters + j] = newParameter;
          newParameter.first = stan::math::value_of(events.time(iEvent));
          time_[len_Parameters + j] = newParameter;
          events.index[iEvent][3] = 0;
          if (iEvent < nEvent - 1) iEvent++;
          j++;
        }

        if (iEvent < nEvent - 1) iEvent++;
      }
    }
    // if (!Check()) Sort();
    Sort();
  }
};

template<typename T_time, typename T1, typename T2, typename T3>
const std::vector<Eigen::Matrix<T1, -1, -1>>
ModelParameterHistory<T_time, T1, T2, T3>::dummy_param_matrix = {};

template<typename T_time, typename T1, typename T2, typename T3>
const std::vector<std::vector<T1> >
ModelParameterHistory<T_time, T1, T2, T3>::dummy_param_vector = {};

} 

#endif
