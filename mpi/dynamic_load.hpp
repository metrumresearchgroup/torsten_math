#ifndef STAN_MATH_TORSTEN_MPI_DYNAMIC_LOAD_HPP
#define STAN_MATH_TORSTEN_MPI_DYNAMIC_LOAD_HPP

#include <stan/math/torsten/dsolve/group_functor.hpp>
#include <vector>

namespace torsten {
  namespace mpi {
    
  // lock slaves in waiting-loop 
  template<typename T>
  struct PMXDynamicLoad;

  template<>
  struct PMXDynamicLoad<torsten::dsolve::pmx_ode_group_mpi_functor> {

    static const int init_buf_size = 9;
    static const int i_y           = 2;
    static const int i_theta       = 3;
    static const int i_x_r         = 4;
    static const int i_x_i         = 5;
    static const int i_t_var       = 6;
    static const int i_y_var       = 7;
    static const int i_theta_var   = 8;

    static const int up_tag        = 1;
    static const int work_tag      = 2;
    static const int down_tag      = 3;
    static const int res_tag       = 4;
    static const int err_tag       = 5;
    static const int kill_tag      = 6;

    const torsten::mpi::Communicator& pmx_comm;
    const MPI_Comm comm;
    std::vector<size_t> init_buf;
    std::vector<double> work_r;
    std::vector<int> work_i;

    PMXDynamicLoad(const torsten::mpi::Communicator& comm_in) :
      pmx_comm(comm_in), comm(pmx_comm.comm), init_buf(init_buf_size, 0)
    {
      init_buf[0] = -1;
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
    template<typename Ti>
    inline void set_work(int ip,
                         const std::vector<std::vector<double> >& y0,
                         double t0,
                         const std::vector<Ti>& len,
                         const std::vector<double>& ts,
                         const std::vector<std::vector<double> >& theta,
                         const std::vector<std::vector<double> >& x_r,
                         const std::vector<std::vector<int> >& x_i,
                         double rtol, double atol, int max_num_step) {
      int work_r_count0 = 1 + init_buf[i_y] + init_buf[i_theta] + init_buf[i_x_r] + 2;
      auto ts_iters = ts_range(ip, len, ts);
      work_r.resize(work_r_count0 + std::distance(ts_iters[0], ts_iters[1]));
      work_r[0] = t0;
      std::vector<double>::iterator ptr = work_r.begin() + 1;
      std::copy(y0[ip].begin(), y0[ip].end(), ptr);
      ptr += y0[ip].size();
      std::copy(theta[ip].begin(), theta[ip].end(), ptr);
      ptr += theta[ip].size();
      std::copy(x_r[ip].begin(), x_r[ip].end(), ptr);
      ptr += x_r[ip].size();
      *ptr = rtol;
      ptr++;
      *ptr = atol;        
      ptr++;
      std::copy(ts_iters[0], ts_iters[1], ptr);

      work_i.resize(x_i[0].size() + 1);
      std::copy(x_i[ip].begin(), x_i[ip].end(), work_i.begin());
      work_i.back() = max_num_step;
    }

    /*
     * disassemble work space data into ODE solver inputs
     * @c work_r layout: t0, y0, theta, x_r, rtol, atol, ts
     * @c work_i layout: x_i, max_num_step
     */
    inline void use_work(std::vector<double>& y0,
                         double & t0,
                         std::vector<double>& ts,
                         std::vector<double>& theta,
                         std::vector<double>& x_r,
                         std::vector<int>& x_i,
                         double& rtol, double& atol, int& max_num_step) {
      y0.resize(init_buf[i_y]);
      theta.resize(init_buf[i_theta]);
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
     * master node (rank = 0) recv results and send
     * available tasks to vacant slaves.
     */
    inline Eigen::MatrixXd master(const torsten::dsolve::pmx_ode_group_mpi_functor& f,
                                  int integ_id,
                                  const std::vector<std::vector<double> >& y0,
                                  double t0,
                                  const std::vector<int>& len,
                                  const std::vector<double>& ts,
                                  const std::vector<std::vector<double> >& theta,
                                  const std::vector<std::vector<double> >& x_r,
                                  const std::vector<std::vector<int> >& x_i,
                                  double rtol, double atol, int max_num_step) {
      using Eigen::MatrixXd;

      init_buf[0]           = f.id;
      init_buf[1]           = integ_id;
      init_buf[i_y]         = y0[0].size();
      init_buf[i_theta]     = theta[0].size();
      init_buf[i_x_r]       = x_r[0].size();
      init_buf[i_x_i]       = x_i[0].size();
      init_buf[i_t_var]     = 0;
      init_buf[i_y_var]     = 0;
      init_buf[i_theta_var] = 0;

      for (int i = 1; i < pmx_comm.size; ++i) {
        MPI_Send(init_buf.data(), init_buf_size, MPI_INT, i, up_tag, comm);
      }

      const int m = theta[0].size();
      const int n = y0[0].size();
      const int np = theta.size(); // population size


      int work_r_count0 = 1 + init_buf[i_y] + init_buf[i_theta] + init_buf[i_x_r] + 2;

      std::vector<double>::const_iterator iter = ts.begin();
      int begin_id = 0;

      std::vector<int> task_map(pmx_comm.size);

      // initial task distribution
      int ip = 0;
      for (int i = 0; i < pmx_comm.size && ip < np; ++i, ++ip) {
        set_work(ip, y0, t0, len, ts, theta, x_r, x_i, rtol, atol, max_num_step);
        MPI_Send(work_r.data(), work_r.size(), MPI_DOUBLE, i, work_tag, comm);
        MPI_Send(work_i.data(), work_i.size(), MPI_INT, i, work_tag, comm);
        task_map[i] = ip;
      }

      // recv & dispatch new task
      MPI_Status stat;
      bool is_invalid;
      while ((ip < np - 1) && (!is_invalid)) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &stat);
        if (stat.MPI_TAG == err_tag) {
          is_invalid = true;
        } else {
          int source = stat.MPI_SOURCE;
          auto ts_range_iter = ts_range(task_map[source], len, ts);
          std::vector<double> ts_i(ts_range_iter[0], ts_range_iter[1]);
          MatrixXd res(n, ts_i.size());
          MPI_Recv(res.data(), res.size(), MPI_DOUBLE, source, MPI_ANY_TAG, comm, &stat);
          set_work(++ip, y0, t0, len, ts, theta, x_r, x_i, rtol, atol, max_num_step);          
          MPI_Send(work_r.data(), work_r.size(), MPI_DOUBLE, source, work_tag, comm);
          MPI_Send(work_i.data(), work_i.size(), MPI_INT, source, work_tag, comm);
          task_map[source] = ip;
        }
      }

      for (int i = 0; i < pmx_comm.size; ++i) {
        MPI_Send(work_r.data(), 0, MPI_DOUBLE, i, down_tag, comm);
      }

      // FIME
      MatrixXd res;
      return res;
    }

    /*
     * Slave nodes recv work and send back results
     */
    inline void slave() {
      using Eigen::MatrixXd;

      MPI_Status stat;

      // recv activation data
      double t0;
      std::vector<double> y0;
      std::vector<double> ts;
      std::vector<double> theta;
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
          MPI_Recv(init_buf.data(), init_buf_size, MPI_INT, 0, up_tag, comm, &stat);          
        } else if(stat.MPI_TAG == work_tag) {
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
          torsten::dsolve::pmx_ode_group_mpi_functor f(functor_id);
          use_work(y0, t0, ts, theta, x_r, x_i, rtol, atol, max_num_step);

          int tag_ = work_tag;
          MatrixXd res;
          try {
            switch(integrator_id) {
            case 1 : {
              using Ode = torsten::dsolve::PMXCvodesFwdSystem<torsten::dsolve::pmx_ode_group_mpi_functor, double, double, double, CV_ADAMS, AD>;
              torsten::dsolve::PMXOdeService<Ode> serv(y0.size(), theta.size());
              Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, NULL};
              torsten::dsolve::PMXCvodesIntegrator solver(rtol, atol, max_num_step);
              res = solver.template integrate<Ode, false>(ode);
              break;
            }
            case 2 : {
              using Ode = torsten::dsolve::PMXCvodesFwdSystem<torsten::dsolve::pmx_ode_group_mpi_functor, double, double, double, CV_BDF, AD>;
              torsten::dsolve::PMXOdeService<Ode> serv(y0.size(), theta.size());
              Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, NULL};
              torsten::dsolve::PMXCvodesIntegrator solver(rtol, atol, max_num_step);
              res = solver.template integrate<Ode, false>(ode);
              break;
            }
            default : {
              using scheme_t = boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>, double, std::vector<double>, double>;
              using Ode = dsolve::PMXOdeintSystem<torsten::dsolve::pmx_ode_group_mpi_functor, double, double, double>;
              dsolve::PMXOdeService<Ode> serv(y0.size(), theta.size());
              Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, NULL};
              dsolve::PMXOdeintIntegrator<scheme_t> solver(rtol, atol, max_num_step);
              res = solver.template integrate<Ode, false>(ode);
              break;
            }
            }
          } catch (const std::exception& e) {
            tag_ = err_tag;
          }

          // send data
          int send_size = tag_ == err_tag ? 0 : res.size();
          MPI_Send(res.data(), send_size, MPI_DOUBLE, 0, tag_, comm);
        }
      }
    }
  };

  }  
}

#endif
