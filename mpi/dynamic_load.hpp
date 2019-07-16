#ifndef STAN_MATH_TORSTEN_MPI_DYNAMIC_LOAD_HPP
#define STAN_MATH_TORSTEN_MPI_DYNAMIC_LOAD_HPP

#include <stan/math/torsten/dsolve/group_functor.hpp>
#include <stan/math/torsten/dsolve/pmx_ode_vars.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <vector>

namespace torsten {
  namespace mpi {
    
  // lock slaves in waiting-loop 
  template<typename... Args>
  struct PMXDynamicLoad;

  template<>
  struct PMXDynamicLoad<torsten::dsolve::pmx_ode_group_mpi_functor> {

    static const int init_buf_size = 9;
    static const int i_y           = 2;
    static const int i_p           = 3;
    static const int i_x_r         = 4;
    static const int i_x_i         = 5;
    static const int i_t_var       = 6;
    static const int i_y_var       = 7;
    static const int i_p_var       = 8;

    static const int up_tag        = 1;
    static const int work_tag      = 2;
    static const int down_tag      = 3;
    static const int res_tag       = 4;
    static const int err_tag       = 5;
    static const int kill_tag      = 6;

    const torsten::mpi::Communicator& pmx_comm;
    const MPI_Comm comm;
    std::vector<int> init_buf;
    std::vector<double> work_r;
    std::vector<int> work_i;

    /*
     * constructor only allocates @c init_buf
     */
    PMXDynamicLoad(const torsten::mpi::Communicator& comm_in) :
      pmx_comm(comm_in), comm(pmx_comm.comm), init_buf(init_buf_size, 0)
    {
      init_buf[0] = -1;

      static const char* caller = "PMXDynamicLoad::constructor";
      stan::math::check_greater(caller, "MPI comm size", pmx_comm.size, 1);
    }

    /*
     * is @c ts of var type?
     */
    inline bool is_var_ts() {
      return init_buf[i_t_var];
    }

    /*
     * is @c y0 of var type?
     */
    inline bool is_var_y0() {
      return init_buf[i_y_var];
    }

    /*
     * is @c theta of var type?
     */
    inline bool is_var_theta() {
      return init_buf[i_p_var];
    }

    /*
     * get first and last iterator of time step vector @c ts
     * for subject @c i.
     */
    template<typename T, typename Ti>
    inline std::vector<typename std::vector<T>::const_iterator >
    ts_range(int i, const std::vector<Ti>& len, const std::vector<T>& ts) {
      typename std::vector<T>::const_iterator begin =
        ts.begin() + std::accumulate(len.begin(), len.begin() + i, 0);
      typename std::vector<T>::const_iterator end = begin + len[i];
      return {begin, end};
    }

    /*
     * assemble work space data to be sent to slaves
     * @c work_r layout: t0, y0, theta, x_r, rtol, atol, ts
     * @c work_i layout: x_i, max_num_step
     */
    template<typename Ti, typename Tt, typename Ty, typename Tp>
    inline void set_work(int ip,
                         const std::vector<std::vector<Ty> >& y0,
                         double t0,
                         const std::vector<Ti>& len,
                         const std::vector<Tt>& ts,
                         const std::vector<std::vector<Tp> >& theta,
                         const std::vector<std::vector<double> >& x_r,
                         const std::vector<std::vector<int> >& x_i,
                         double rtol, double atol, int max_num_step) {
      int work_r_count0 = 1 + init_buf[i_y] + init_buf[i_p] + init_buf[i_x_r] + 2;
      auto ts_iters = ts_range(ip, len, ts);
      work_r.resize(work_r_count0 + std::distance(ts_iters[0], ts_iters[1]));
      work_r[0] = t0;
      std::vector<double>::iterator ptr = work_r.begin() + 1;
      auto to_val = [](const auto& x) { return stan::math::value_of(x); };
      ptr = std::transform(y0[ip].begin(), y0[ip].end(), ptr, to_val);
      ptr = std::transform(theta[ip].begin(), theta[ip].end(), ptr, to_val);
      ptr = std::copy(x_r[ip].begin(), x_r[ip].end(), ptr);
      *ptr = rtol;
      ptr++;
      *ptr = atol;        
      ptr++;
      ptr = std::transform(ts_iters[0], ts_iters[1], ptr, to_val);

      work_i.resize(x_i[0].size() + 1);
      std::copy(x_i[ip].begin(), x_i[ip].end(), work_i.begin());
      work_i.back() = max_num_step;
    }

    /*
     * disassemble work space data into ODE solver inputs
     * @c work_r layout: t0, y0, theta, x_r, rtol, atol, ts
     * @c work_i layout: x_i, max_num_step
     */
    template<typename Tt, typename Ty, typename Tp>
    inline void use_work(std::vector<Ty>& y0,
                         double & t0,
                         std::vector<Tt>& ts,
                         std::vector<Tp>& theta,
                         std::vector<double>& x_r,
                         std::vector<int>& x_i,
                         double& rtol, double& atol, int& max_num_step) {
      y0.resize(init_buf[i_y]);
      theta.resize(init_buf[i_p]);
      x_r.resize(init_buf[i_x_r]);
      x_i.resize(init_buf[i_x_i]);

      t0 = work_r[0];
      std::vector<double>::const_iterator first = work_r.begin() + 1;
      std::vector<double>::const_iterator end = work_r.begin() + 1 + y0.size();
      std::copy(first, end, y0.begin());
      first = end;
      end += theta.size();
      std::copy(first, end, theta.begin());
      first = end;
      end += x_r.size();
      std::copy(first, end, x_r.begin());
      first = end;
      rtol = *first;
      first++;
      atol = *first;
      first++;
      end = work_r.end();
      ts.resize(std::distance(first, end));
      std::copy(first, end, ts.begin());

      std::copy(work_i.begin(), work_i.begin() + x_i.size(), x_i.begin());
      max_num_step = work_i.back();
    }

    /*
     * master node (rank = 0) prompts slaves to break the
     * waiting-working cycle by sending a kill tag.
     */
    inline void kill_slaves() {
      if (pmx_comm.rank == 0) {
        for (int i = 1; i < pmx_comm.size; ++i) {
          MPI_Send(work_r.data(), 0, MPI_DOUBLE, i, kill_tag, comm);
        }
      }
    }

    /*
     * helper function to master node (rank = 0) to recv
     * results.
     */
    inline void
    master_recv(Eigen::MatrixXd& res,
                Eigen::MatrixXd& res_d,
                const std::vector<int>& task_map,
                const std::vector<std::vector<double>>& y0,
                const std::vector<int>& len,
                const std::vector<double>& ts,
                const std::vector<std::vector<double>>& theta,
                MPI_Status& stat) {
      int source = stat.MPI_SOURCE;
      int ires = task_map.at(source);
      auto its = ts_range(ires, len, ts);
      int n = y0[0].size();
      MPI_Recv(&res(n * std::distance(ts.begin(), its[0])),
               n * std::distance(its[0], its[1]),
               MPI_DOUBLE, source, MPI_ANY_TAG, comm, &stat);
    }
    
    /*
     * helper function to master node (rank = 0) to recv
     * results and generated @c var results if parameters
     * are present.
     */
    template<typename Tt, typename Ty, typename Tp,
             typename std::enable_if_t<torsten::is_var<Tt, Ty, Tp>::value >* = nullptr> //NOLINT
    inline void
    master_recv(Eigen::Matrix<typename torsten::return_t<Tt, Ty, Tp>::type, -1, -1>& res,
                Eigen::MatrixXd& res_d,
                const std::vector<int>& task_map,
                const std::vector<std::vector<Ty> >& y0,
                const std::vector<int>& len,
                const std::vector<Tt>& ts,
                const std::vector<std::vector<Tp> >& theta,
                MPI_Status& stat) {
      int source = stat.MPI_SOURCE;
      int ires = task_map.at(source);
      auto its = ts_range(ires, len, ts);

      int res_count;
      int n = y0[0].size();
      MPI_Get_count(&stat, MPI_DOUBLE, &res_count);
      res_d.resize(res_count/len[ires], len[ires]);
      MPI_Recv(res_d.data(), res_count, MPI_DOUBLE, source, MPI_ANY_TAG, comm, &stat);
      Eigen::Matrix<typename torsten::return_t<Tt, Ty, Tp>::type, -1, -1>::
        Map(&res(n * std::distance(ts.begin(), its[0])), n, len[ires]) =
        torsten::precomputed_gradients(res_d,
                                       torsten::dsolve::pmx_ode_vars(y0[ires], theta[ires], its[0], its[1]));
    }
    
    /*
     * master node (rank = 0) recv results and send
     * available tasks to vacant slaves.
     */
    template<typename Tt, typename Ty, typename Tp>
    inline Eigen::Matrix<typename torsten::return_t<Tt, Ty, Tp>::type, -1, -1>
    master(const torsten::dsolve::pmx_ode_group_mpi_functor& f,
           int integ_id,
           const std::vector<std::vector<Ty> >& y0,
           double t0,
           const std::vector<int>& len,
           const std::vector<Tt>& ts,
           const std::vector<std::vector<Tp> >& theta,
           const std::vector<std::vector<double> >& x_r,
           const std::vector<std::vector<int> >& x_i,
           double rtol, double atol, int max_num_step) {
      using Eigen::MatrixXd;
      using Eigen::Matrix;


      using scalar_t = typename torsten::return_t<Tt, Ty, Tp>::type;

      static const char* caller = "PMXDynamicLoad::master";
      stan::math::check_less(caller, "MPI comm rank", pmx_comm.rank, 1);

      init_buf[0]           = f.id;
      init_buf[1]           = integ_id;
      init_buf[i_y]         = y0[0].size();
      init_buf[i_p]         = theta[0].size();
      init_buf[i_x_r]       = x_r[0].size();
      init_buf[i_x_i]       = x_i[0].size();
      init_buf[i_t_var]     = torsten::is_var<Tt>::value;
      init_buf[i_y_var]     = torsten::is_var<Ty>::value;
      init_buf[i_p_var]     = torsten::is_var<Tp>::value;

      for (int i = 1; i < pmx_comm.size; ++i) {
        MPI_Send(init_buf.data(), init_buf.size(), MPI_INT, i, up_tag, comm);
      }

      const int m = theta[0].size();
      const int n = y0[0].size();
      const int np = theta.size(); // population size

      int begin_id = 0;
      std::vector<int> task_map(pmx_comm.size, -1);

      // initial task distribution
      int ip = 0, n_recv = 0;
      for (int i = 1; i < pmx_comm.size && ip < np; ++i, ++ip) {
        set_work(ip, y0, t0, len, ts, theta, x_r, x_i, rtol, atol, max_num_step);
        MPI_Send(work_r.data(), work_r.size(), MPI_DOUBLE, i, work_tag, comm);
        MPI_Send(work_i.data(), work_i.size(), MPI_INT, i, work_tag, comm);
        task_map[i] = ip;
      }

      // recv & dispatch new task
      MPI_Status stat;
      bool is_invalid = false;
      int npar0 = 1 + stan::is_var<Ty>::value ? n : 0 + stan::is_var<Tp>::value ? m : 0;
      Eigen::Matrix<scalar_t, -1, -1> res = Eigen::Matrix<scalar_t, -1, -1>::Zero(n, ts.size());
      MatrixXd res_d;
      
      for (;;) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &stat);
        int source = stat.MPI_SOURCE;
        if (stat.MPI_TAG == err_tag) {
          is_invalid = true;
          n_recv = np;
        } else {
          master_recv(res, res_d, task_map, y0, len, ts, theta, stat);
          n_recv++;
        }

        // more work to do?
        if (ip < np && (!is_invalid)) {
          set_work(ip, y0, t0, len, ts, theta, x_r, x_i, rtol, atol, max_num_step);          
          MPI_Send(work_r.data(), work_r.size(), MPI_DOUBLE, source, work_tag, comm);
          MPI_Send(work_i.data(), work_i.size(), MPI_INT, source, work_tag, comm);
          task_map.at(source) = ip++;
        }

        // all results recved?
        if (n_recv == np) {
          for (int i = 1; i < pmx_comm.size; ++i) {
            MPI_Send(work_r.data(), 0, MPI_DOUBLE, i, down_tag, comm);
          }
          break;
        }
      }

      return res;
    }

    /*
     * slaves solve ODE according to @c integrator_id
     */
    template<typename Tt, typename Ty, typename Tp>
    inline Eigen::MatrixXd slave_solve(int functor_id,
                                       int integ_id,
                                       const std::vector<Ty>& y0,
                                       double t0,
                                       const std::vector<Tt>& ts,
                                       const std::vector<Tp>& theta,
                                       const std::vector<double>& x_r,
                                       const std::vector<int>& x_i,
                                       double rtol, double atol, int max_num_step) {
      Eigen::MatrixXd res;
      torsten::dsolve::pmx_ode_group_mpi_functor f(functor_id);
      switch(integ_id) {
      case 1 : {
        using Ode = torsten::dsolve::PMXCvodesFwdSystem<torsten::dsolve::pmx_ode_group_mpi_functor, Tt, Ty, Tp, CV_ADAMS, AD>;
        torsten::dsolve::PMXOdeService<Ode> serv(y0.size(), theta.size());
        Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, NULL};
        torsten::dsolve::PMXCvodesIntegrator solver(rtol, atol, max_num_step);
        res = solver.template integrate<Ode, false>(ode);
        break;
      }
      case 2 : {
        using Ode = torsten::dsolve::PMXCvodesFwdSystem<torsten::dsolve::pmx_ode_group_mpi_functor, Tt, Ty, Tp, CV_BDF, AD>;
        torsten::dsolve::PMXOdeService<Ode> serv(y0.size(), theta.size());
        Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, NULL};
        torsten::dsolve::PMXCvodesIntegrator solver(rtol, atol, max_num_step);
        res = solver.template integrate<Ode, false>(ode);
        break;
      }
      default : {
        using scheme_t = boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>, double, std::vector<double>, double>;
        using Ode = dsolve::PMXOdeintSystem<torsten::dsolve::pmx_ode_group_mpi_functor, Tt, Ty, Tp>;
        dsolve::PMXOdeService<Ode> serv(y0.size(), theta.size());
        Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, NULL};
        dsolve::PMXOdeintIntegrator<scheme_t> solver(rtol, atol, max_num_step);
        res = solver.template integrate<Ode, false>(ode);
        break;
      }
      }
      return res;
    }

    /*
     * Slave nodes recv work and send back results.
     * Depends on the tag recved, the slaves do
     * - initialize model data
     * - recv, do, and send back work
     * - go back to waiting
     * - kill(break wait loop)
     */
    inline void slave() {
      using Eigen::MatrixXd;
      using stan::math::var;

      static const char* caller = "PMXDynamicLoad::slave";
      stan::math::check_greater(caller, "MPI comm rank", pmx_comm.rank, 0);

      MPI_Status stat;

      double t0;
      std::vector<double> x_r;
      std::vector<int> x_i;
      double rtol;
      double atol;
      int max_num_step;
      int functor_id;
      int integrator_id;

      for (;;) {
        MPI_Probe(0, MPI_ANY_TAG, comm, &stat);

        if (stat.MPI_TAG == up_tag) {
          int i;
          MPI_Get_count(&stat, MPI_INT, &i);
          MPI_Recv(init_buf.data(), init_buf.size(), MPI_INT, 0, up_tag, comm, &stat);          
        } else if (stat.MPI_TAG == work_tag) {
          int work_r_count;
          MPI_Get_count(&stat, MPI_DOUBLE, &work_r_count);

          // work_r layout: t0, y0, theta, x_r, rtol, atol, ts
          // work_i layout: x_r, max_step
          work_r.resize(work_r_count);
          work_i.resize(init_buf[i_x_i] + 1);
          MPI_Recv(work_r.data(), work_r.size(), MPI_DOUBLE, 0, work_tag, comm, &stat);
          MPI_Recv(work_i.data(), work_i.size(), MPI_INT, 0, work_tag, comm, &stat);

          functor_id = init_buf[0];
          integrator_id = init_buf[1];

          int tag_ = work_tag;
          MatrixXd res;
          try {
            if (!is_var_ts() && !is_var_y0() && !is_var_theta()) {
              std::vector<double> y0;
              std::vector<double> ts;
              std::vector<double> theta;
              use_work(y0, t0, ts, theta, x_r, x_i, rtol, atol, max_num_step);
              res = slave_solve(functor_id, integrator_id,
                                y0, t0, ts, theta, x_r, x_i, rtol, atol, max_num_step);
            } else if (is_var_ts() && !is_var_y0() && !is_var_theta()) {
              std::vector<double> y0;
              std::vector<var> ts;
              std::vector<double> theta;
              use_work(y0, t0, ts, theta, x_r, x_i, rtol, atol, max_num_step);
              res = slave_solve(functor_id, integrator_id,
                                y0, t0, ts, theta, x_r, x_i, rtol, atol, max_num_step);
            }
          } catch (const std::exception& e) {
            tag_ = err_tag;
          }

          MPI_Send(res.data(), tag_ == err_tag ? 0 : res.size(), MPI_DOUBLE, 0, tag_, comm);
        } else if (stat.MPI_TAG == kill_tag) {
          MPI_Recv(work_r.data(), 0, MPI_DOUBLE, 0, kill_tag, comm, &stat);
          break;
        } else if (stat.MPI_TAG == down_tag) {
          MPI_Recv(work_r.data(), 0, MPI_DOUBLE, 0, down_tag, comm, &stat);
        }
      }
    }
  };

  }  
}

#endif
