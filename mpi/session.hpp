#ifndef STAN_MATH_TORSTEN_MPI_SESSION_HPP
#define STAN_MATH_TORSTEN_MPI_SESSION_HPP

#ifdef TORSTEN_MPI

#include <stan/math/mpi/envionment.hpp>

/*
 * Given a Torsten communicator comm that has m workers and solves a group of
 * size n, the total nb. of processes n_proc involved is
 *
 * n_proc = m * b
 *
 * with b the nb. of cores used in the braid for a single subject/ode.
 * The population/group is evenly distributed
 * among m workers, with each worker employs b
 * processes/cores to solve a single subject/ode.
 * This implies there is effectively no braid(in-time-parallel) when b = 1.
 */

namespace torsten {
  namespace mpi {
    /*
     * MPI communicator wrapper for RAII. Note that no
     * MPI's predfined comm sich as @c MPI_COMM_WOLRD are allowed.
     */
    struct Session {
      static stan::math::mpi::Communicator pmx_parm_comm;
      static stan::math::mpi::Communicator pmx_data_comm;
      static stan::math::mpi::Communicator ode_parm_comm;
      static stan::math::mpi::Communicator ode_data_comm;
    };

    stan::math::mpi::Communicator Session::pmx_parm_comm(stan::math::mpi::Session::intra_chain_comm(1));
    stan::math::mpi::Communicator Session::pmx_data_comm(stan::math::mpi::Session::intra_chain_comm(1));
    stan::math::mpi::Communicator Session::ode_parm_comm(stan::math::mpi::Session::intra_chain_comm(1));
    stan::math::mpi::Communicator Session::ode_data_comm(stan::math::mpi::Session::intra_chain_comm(1));
  }
}

#endif

#endif
