#ifndef STAN_MATH_TORSTEN_MPI_IS_MINE_HPP
#define STAN_MATH_TORSTEN_MPI_IS_MINE_HPP

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

namespace torsten {
#ifdef TORSTEN_MPI
namespace mpi {

  /*
   * Round-Robin style allocation of individules to MPI
   * nodes.
   *
   * @param i ID of the individule
   * @param np size of the population
   * @return true if @c i is current node's responsiblity
   * for computation, false otherwise.
   */
  bool is_mine(boost::mpi::communicator comm,
               const int& i, const int& np) {
    int n = np / comm.size();
    int r = np % comm.size();
    int a = comm.rank() * n + (comm.rank() >= r ? r : comm.rank());
    int b = a + n + (comm.rank() >= r ? 0 : 1);
    return (i >= a && i < b);
  }
}
#endif
}

#endif
