#ifndef STAN_MATH_TORSTEN_SOLVE_ODE_HPP
#define STAN_MATH_TORSTEN_SOLVE_ODE_HPP

#include <Eigen/Dense>
#include <stan/math/torsten/to_array_2d.hpp>
#include <stan/math/torsten/ev_manager.hpp>
#include <stan/math/torsten/pmx_population_check.hpp>
#include <stan/math/torsten/ev_solver.hpp>
#include <stan/math/torsten/pmx_ode_model.hpp>
#include <stan/math/torsten/pmx_check.hpp>
#include <stan/math/torsten/nonmem_events_args.hpp>
#include <vector>

namespace torsten {

  /** 
   * helper expression used for actual ODE solver functions to
   * detect if the last arg is <code>ostream*</code>
   * 
   * @tparam Ts all of the args for pmx solvers
   */
template<typename... Ts>
bool constexpr last_is_ostream_ptr =
  std::is_same<NthTypeOf<(sizeof...(Ts) - 1), Ts...>, std::ostream*>::value;

template<>
bool constexpr last_is_ostream_ptr<> = false;

  template<PMXOdeIntegratorId It>
  struct PMXSolveODE {
    
    /**
     * Computes the predicted amounts in each compartment at each event
     * for a general compartment model, defined by a system of ordinary
     * differential equations. 
     *
     * <b>Warning:</b> This prototype does not handle steady state events. 
     *
     * @tparam T0 type of scalar for time of events. 
     * @tparam T1 type of scalar for amount at each event.
     * @tparam T2 type of scalar for rate at each event.
     * @tparam T3 type of scalar for inter-dose inteveral at each event.
     * @tparam T4 type of scalars for the model parameters.
     * @tparam T5 type of scalars for the bio-variability parameters.
     * @tparam T6 type of scalars for the model tlag parameters.
     * @tparam F type of ODE system function.
     * @param[in] f functor for base ordinary differential equation that defines 
     *            compartment model.
     * @param[in] nCmt number of compartments in model
     * @param[in] pMatrix parameters at each event
     * @param[in] time times of events  
     * @param[in] amt amount at each event
     * @param[in] rate rate at each event
     * @param[in] ii inter-dose interval at each event
     * @param[in] evid event identity: 
     *                    (0) observation 
     *                    (1) dosing
     *                    (2) other 
     *                    (3) reset 
     *                    (4) reset AND dosing 
     * @param[in] cmt compartment number at each event 
     * @param[in] addl additional dosing at each event 
     * @param[in] ss steady state approximation at each event (0: no, 1: yes)
     * @param[in] rel_tol relative tolerance for the Boost ode solver 
     * @param[in] abs_tol absolute tolerance for the Boost ode solver
     * @param[in] max_num_steps maximal number of steps to take within 
     *            the Boost ode solver 
     * @param[in] as_rel_tol relative tolerance for the algebra solver
     * @param[in] as_abs_tol absolute tolerance for the algebra solver
     * @param[in] as_max_num_steps maximal number of steps to take within 
     *            the algebra solver
     * @return a matrix with predicted amount in each compartment 
     *         at each event. 
     *
     * FIX ME: currently have a dummy msgs argument. Makes it easier
     * to expose to stan grammar files, because I can follow more closely
     * what was done for the ODE integrator. Not ideal.
     */
    template <typename T0, typename T1, typename T2, typename T3, typename T4,
              typename T5, typename T6, typename F>
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3, T4, T5, T6>,
                   Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<std::vector<T4> >& pMatrix,
               const std::vector<std::vector<T5> >& biovar,
               const std::vector<std::vector<T6> >& tlag,
               double rel_tol,
               double abs_tol,
               long int max_num_steps,
               double as_rel_tol,
               double as_abs_tol,
               long int as_max_num_steps,
               std::ostream* msgs) {
      using std::vector;
      using Eigen::Dynamic;
      using Eigen::Matrix;

      // check arguments
      static const char* function("PMX SOLVE ODE");
      torsten::pmx_check(time, amt, rate, ii, evid, cmt, addl, ss,
                         pMatrix, biovar, tlag, function);

      using ER = NONMENEventsRecord<T0, T1, T2, T3>;
      using EM = EventsManager<ER, NonEventParameters<T0, T4, std::vector, std::tuple<T5, T6> >>;
      const ER events_rec(nCmt, time, amt, rate, ii, evid, cmt, addl, ss);

      Matrix<typename EM::T_scalar, Dynamic, Dynamic> pred =
        Matrix<typename EM::T_scalar, Dynamic, Dynamic>::Zero(events_rec.num_event_times(), EM::nCmt(events_rec));

      using model_type = torsten::PKODEModel<typename EM::T_par, F>;

      PMXOdeIntegrator<It> integrator(rel_tol, abs_tol, max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps, msgs);
      EventSolver<model_type, NonEventParameters<T0, T4, std::vector, std::tuple<T5, T6> >> pr;

      pr.pred(0, events_rec, pred, integrator, pMatrix, biovar, tlag, nCmt, f);
      return pred;
    }

    /*
     * Overload with default ODE & algebra solver controls 
     */
    template <typename T0, typename T1, typename T2, typename T3, typename T4,
              typename T5, typename T6, typename F>
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3, T4, T5, T6>,
                   Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<std::vector<T4> >& pMatrix,
               const std::vector<std::vector<T5> >& biovar,
               const std::vector<std::vector<T6> >& tlag,
               std::ostream* msgs) {
      return solve(f, nCmt,
                        time, amt, rate, ii, evid, cmt, addl, ss,
                        pMatrix, biovar, tlag,
                        1.e-6, 1.e-6, 1e6,
                        1.e-6, 1.e-6, 1e2,                        
                        msgs);
    }

    /*
     * Overload with default algebra solver controls 
     */
    template <typename T0, typename T1, typename T2, typename T3, typename T4,
              typename T5, typename T6, typename F>
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3, T4, T5, T6>,
                   Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<std::vector<T4> >& pMatrix,
               const std::vector<std::vector<T5> >& biovar,
               const std::vector<std::vector<T6> >& tlag,
               double rel_tol,
               double abs_tol,
               long int max_num_steps,
               std::ostream* msgs) {
      return solve(f, nCmt,
                        time, amt, rate, ii, evid, cmt, addl, ss,
                        pMatrix, biovar, tlag,
                        rel_tol, abs_tol, max_num_steps,
                        1.e-6, 1.e-6, 1e2,                        
                        msgs);
    }

    /**
     * Overload function to allow user to pass an std::vector for 
     * pMatrix/bioavailability/tlag
     */
    template <typename T0, typename T1, typename T2, typename T3,
              typename T_par, typename T_biovar, typename T_tlag,
              typename F,
              typename std::enable_if_t<!(torsten::is_std_vector<T_par, T_biovar, T_tlag>::value)>* = nullptr> //NOLINT
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3,
                                                       typename torsten::value_type<T_par>::type,
                                                       typename torsten::value_type<T_biovar>::type,
                                                       typename torsten::value_type<T_tlag>::type>,
                          Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<T_par>& pMatrix,
               const std::vector<T_biovar>& biovar,
               const std::vector<T_tlag>& tlag,
               double rel_tol,
               double abs_tol,
               long int max_num_steps,
               double as_rel_tol,
               double as_abs_tol,
               long int as_max_num_steps,
               std::ostream* msgs) {
      auto param_ = torsten::to_array_2d(pMatrix);
      auto biovar_ = torsten::to_array_2d(biovar);
      auto tlag_ = torsten::to_array_2d(tlag);

      return solve(f, nCmt,
                        time, amt, rate, ii, evid, cmt, addl, ss,
                        param_, biovar_, tlag_,
                        rel_tol, abs_tol, max_num_steps,
                        as_rel_tol, as_abs_tol, as_max_num_steps,
                        msgs);
    }

    /**
     * Overload function to allow user to pass an std::vector for 
     * pMatrix/bioavailability/tlag, with default ODE &
     * algebra solver controls.
     */
    template <typename T0, typename T1, typename T2, typename T3,
              typename T_par, typename T_biovar, typename T_tlag,
              typename F,
              typename std::enable_if_t<!(torsten::is_std_vector<T_par, T_biovar, T_tlag>::value)>* = nullptr> //NOLINT
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3,
                                                typename torsten::value_type<T_par>::type,
                                                typename torsten::value_type<T_biovar>::type,
                                                typename torsten::value_type<T_tlag>::type>,
                   Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<T_par>& pMatrix,
               const std::vector<T_biovar>& biovar,
               const std::vector<T_tlag>& tlag,
               std::ostream* msgs) {
      return solve(f, nCmt,
                        time, amt, rate, ii, evid, cmt, addl, ss,
                        pMatrix, biovar, tlag,
                        1.e-6, 1.e-6, 1e6,
                        1.e-6, 1.e-6, 1e2,
                        msgs);
    }

    /**
     * Overload function to allow user to pass an std::vector for 
     * pMatrix/bioavailability/tlag, with default
     * algebra solver controls.
     */
    template <typename T0, typename T1, typename T2, typename T3,
              typename T_par, typename T_biovar, typename T_tlag,
              typename F,
              typename std::enable_if_t<!(torsten::is_std_vector<T_par, T_biovar, T_tlag>::value)>* = nullptr> //NOLINT
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3,
                                                typename torsten::value_type<T_par>::type,
                                                typename torsten::value_type<T_biovar>::type,
                                                typename torsten::value_type<T_tlag>::type>,
                   Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<T_par>& pMatrix,
               const std::vector<T_biovar>& biovar,
               const std::vector<T_tlag>& tlag,
               double rel_tol,
               double abs_tol,
               long int max_num_steps,
               std::ostream* msgs) {
      return solve(f, nCmt,
                        time, amt, rate, ii, evid, cmt, addl, ss,
                        pMatrix, biovar, tlag,
                        rel_tol, abs_tol, max_num_steps,
                        1.e-6, 1.e-6, 1e2,
                        msgs);
    }

    // no tlag version
    template <typename T0, typename T1, typename T2, typename T3, typename T4,
              typename T5, typename F>
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3, T4, T5>,
                   Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<std::vector<T4> >& pMatrix,
               const std::vector<std::vector<T5> >& biovar,
               double rel_tol,
               double abs_tol,
               long int max_num_steps,
               double as_rel_tol,
               double as_abs_tol,
               long int as_max_num_steps,
               std::ostream* msgs) {
      // check arguments
      static const char* function("solve");
      const std::vector<std::vector<double> > tlag{{0.0}};
      torsten::pmx_check(time, amt, rate, ii, evid, cmt, addl, ss,
                         pMatrix, biovar, tlag, function);

      using ER = NONMENEventsRecord<T0, T1, T2, T3>;
      using EM = EventsManager<ER, NonEventParameters<T0, T4, std::vector, std::tuple<T5>>>;
      const ER events_rec(nCmt, time, amt, rate, ii, evid, cmt, addl, ss);

      Matrix<typename EM::T_scalar, Dynamic, Dynamic> pred =
        Matrix<typename EM::T_scalar, Dynamic, Dynamic>::Zero(events_rec.num_event_times(), EM::nCmt(events_rec));

      using model_type = torsten::PKODEModel<typename EM::T_par, F>;

      PMXOdeIntegrator<It> integrator(rel_tol, abs_tol, max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps, msgs);
      EventSolver<model_type, NonEventParameters<T0, T4, std::vector, std::tuple<T5>>> pr;

      pr.pred(0, events_rec, pred, integrator, pMatrix, biovar, nCmt, f);
      return pred;
    }

    // no tlag version
    template <typename T0, typename T1, typename T2, typename T3, typename T4,
              typename T5, typename F>
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3, T4, T5>,
                   Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
          const int nCmt,
          TORSTEN_PMX_FUNC_EVENTS_ARGS,
          const std::vector<std::vector<T4> >& pMatrix,
          const std::vector<std::vector<T5> >& biovar,
          std::ostream* msgs) {
      return solve(f, nCmt,
                   time, amt, rate, ii, evid, cmt, addl, ss,
                   pMatrix, biovar,
                   1.e-6, 1.e-6, 1e6,
                   1.e-6, 1.e-6, 1e2,                        
                   msgs);
    }

    // no tlag version
    template <typename T0, typename T1, typename T2, typename T3, typename T4,
              typename T5, typename F>
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3, T4, T5>,
                          Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
          const int nCmt,
          TORSTEN_PMX_FUNC_EVENTS_ARGS,
          const std::vector<std::vector<T4> >& pMatrix,
          const std::vector<std::vector<T5> >& biovar,
          double rel_tol,
          double abs_tol,
          long int max_num_steps,
          std::ostream* msgs) {
      return solve(f, nCmt,
                   time, amt, rate, ii, evid, cmt, addl, ss,
                   pMatrix, biovar,
                   rel_tol, abs_tol, max_num_steps,
                   1.e-6, 1.e-6, 1e2,                        
                   msgs);
    }

    /**
     * no tlag version: overload array 2d function
     */
    template <typename T0, typename T1, typename T2, typename T3,
              typename T_par, typename T_biovar, typename F,
              typename std::enable_if_t<!(torsten::is_std_vector<T_par, T_biovar>::value)>* = nullptr> //NOLINT
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3,
                                                       typename torsten::value_type<T_par>::type,
                                                       typename torsten::value_type<T_biovar>::type>,
                          Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<T_par>& pMatrix,
               const std::vector<T_biovar>& biovar,
               double rel_tol,
               double abs_tol,
               long int max_num_steps,
               double as_rel_tol,
               double as_abs_tol,
               long int as_max_num_steps,
               std::ostream* msgs) {
      auto param_ = torsten::to_array_2d(pMatrix);
      auto biovar_ = torsten::to_array_2d(biovar);

      return solve(f, nCmt,
                   time, amt, rate, ii, evid, cmt, addl, ss,
                   param_, biovar_,
                   rel_tol, abs_tol, max_num_steps,
                   as_rel_tol, as_abs_tol, as_max_num_steps,
                   msgs);
    }

    /**
     * no tlag version: overload array 2d function
     */
    template <typename T0, typename T1, typename T2, typename T3,
              typename T_par, typename T_biovar, typename F,
              typename std::enable_if_t<!(torsten::is_std_vector<T_par, T_biovar>::value)>* = nullptr> //NOLINT
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3,
                                                       typename torsten::value_type<T_par>::type,
                                                       typename torsten::value_type<T_biovar>::type>,
                          Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<T_par>& pMatrix,
               const std::vector<T_biovar>& biovar,
               std::ostream* msgs) {
      return solve(f, nCmt,
                   time, amt, rate, ii, evid, cmt, addl, ss,
                   pMatrix, biovar,
                   1.e-6, 1.e-6, 1e6,
                   1.e-6, 1.e-6, 1e2,
                   msgs);
    }

    /**
     * no tlag version: overload array 2d function
     */
    template <typename T0, typename T1, typename T2, typename T3,
              typename T_par, typename T_biovar, typename F,
              typename std::enable_if_t<!(torsten::is_std_vector<T_par, T_biovar>::value)>* = nullptr> //NOLINT
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3,
                                                       typename torsten::value_type<T_par>::type,
                                                       typename torsten::value_type<T_biovar>::type>,
                          Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<T_par>& pMatrix,
               const std::vector<T_biovar>& biovar,
               double rel_tol,
               double abs_tol,
               long int max_num_steps,
               std::ostream* msgs) {
      return solve(f, nCmt,
                   time, amt, rate, ii, evid, cmt, addl, ss,
                   pMatrix, biovar,
                   rel_tol, abs_tol, max_num_steps,
                   1.e-6, 1.e-6, 1e2,
                   msgs);
    }

    // no tlag/biovar version
    template <typename T0, typename T1, typename T2, typename T3, typename T4,
              typename F>
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3, T4>,
                   Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<std::vector<T4> >& pMatrix,
               double rel_tol,
               double abs_tol,
               long int max_num_steps,
               double as_rel_tol,
               double as_abs_tol,
               long int as_max_num_steps,
               std::ostream* msgs) {
      // check arguments
      static const char* function("solve");
      const std::vector<std::vector<double> > tlag{{0.0}};
      const std::vector<std::vector<double> > biovar{{1.0}};
      torsten::pmx_check(time, amt, rate, ii, evid, cmt, addl, ss,
                         pMatrix, biovar, tlag, function);

      using ER = NONMENEventsRecord<T0, T1, T2, T3>;
      using EM = EventsManager<ER, NonEventParameters<T0, T4, std::vector, std::tuple<>>>;
      const ER events_rec(nCmt, time, amt, rate, ii, evid, cmt, addl, ss);

      Matrix<typename EM::T_scalar, Dynamic, Dynamic> pred =
        Matrix<typename EM::T_scalar, Dynamic, Dynamic>::Zero(events_rec.num_event_times(), EM::nCmt(events_rec));

      using model_type = torsten::PKODEModel<typename EM::T_par, F>;

      PMXOdeIntegrator<It> integrator(rel_tol, abs_tol, max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps, msgs);
      EventSolver<model_type, NonEventParameters<T0, T4, std::vector, std::tuple<>>> pr;

      pr.pred(0, events_rec, pred, integrator, pMatrix, nCmt, f);
      return pred;
    }

    // no tlag/biovar version
    template <typename T0, typename T1, typename T2, typename T3, typename T4,
              typename F>
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3, T4>,
                   Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
          const int nCmt,
          TORSTEN_PMX_FUNC_EVENTS_ARGS,
          const std::vector<std::vector<T4> >& pMatrix,
          std::ostream* msgs) {
      return solve(f, nCmt,
                   time, amt, rate, ii, evid, cmt, addl, ss,
                   pMatrix,
                   1.e-6, 1.e-6, 1e6,
                   1.e-6, 1.e-6, 1e2,                        
                   msgs);
    }

    // no tlag/biovar version
    template <typename T0, typename T1, typename T2, typename T3, typename T4,
              typename F>
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3, T4>,
                          Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
          const int nCmt,
          TORSTEN_PMX_FUNC_EVENTS_ARGS,
          const std::vector<std::vector<T4> >& pMatrix,
          double rel_tol,
          double abs_tol,
          long int max_num_steps,
          std::ostream* msgs) {
      return solve(f, nCmt,
                   time, amt, rate, ii, evid, cmt, addl, ss,
                   pMatrix,
                   rel_tol, abs_tol, max_num_steps,
                   1.e-6, 1.e-6, 1e2,                        
                   msgs);
    }

    /**
     * no tlag/biovar version: overload array 2d function
     */
    template <typename T0, typename T1, typename T2, typename T3,
              typename T_par, typename F,
              typename std::enable_if_t<!(torsten::is_std_vector<T_par>::value)>* = nullptr> //NOLINT
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3,
                                                       typename torsten::value_type<T_par>::type>,
                          Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<T_par>& pMatrix,
               double rel_tol,
               double abs_tol,
               long int max_num_steps,
               double as_rel_tol,
               double as_abs_tol,
               long int as_max_num_steps,
               std::ostream* msgs) {
      auto param_ = torsten::to_array_2d(pMatrix);

      return solve(f, nCmt,
                   time, amt, rate, ii, evid, cmt, addl, ss,
                   param_,
                   rel_tol, abs_tol, max_num_steps,
                   as_rel_tol, as_abs_tol, as_max_num_steps,
                   msgs);
    }

    /**
     * no tlag version: overload array 2d function
     */
    template <typename T0, typename T1, typename T2, typename T3,
              typename T_par,typename F,
              typename std::enable_if_t<!(torsten::is_std_vector<T_par>::value)>* = nullptr> //NOLINT
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3,
                                                       typename torsten::value_type<T_par>::type>,
                          Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<T_par>& pMatrix,
               std::ostream* msgs) {
      return solve(f, nCmt,
                   time, amt, rate, ii, evid, cmt, addl, ss,
                   pMatrix,
                   1.e-6, 1.e-6, 1e6,
                   1.e-6, 1.e-6, 1e2,
                   msgs);
    }

    /**
     * no tlag version: overload array 2d function
     */
    template <typename T0, typename T1, typename T2, typename T3,
              typename T_par, typename F,
              typename std::enable_if_t<!(torsten::is_std_vector<T_par>::value)>* = nullptr> //NOLINT
    static Eigen::Matrix <typename stan::return_type_t<T0, T1, T2, T3,
                                                       typename torsten::value_type<T_par>::type>,
                          Eigen::Dynamic, Eigen::Dynamic>
    solve(const F& f,
               const int nCmt,
               TORSTEN_PMX_FUNC_EVENTS_ARGS,
               const std::vector<T_par>& pMatrix,
               double rel_tol,
               double abs_tol,
               long int max_num_steps,
               std::ostream* msgs) {
      return solve(f, nCmt,
                   time, amt, rate, ii, evid, cmt, addl, ss,
                   pMatrix,
                   rel_tol, abs_tol, max_num_steps,
                   1.e-6, 1.e-6, 1e2,
                   msgs);
    }

  };
}  
#endif
