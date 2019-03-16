#ifndef STAN_MATH_TORSTEN_MPI_COMMUNICATOR_HPP
#define STAN_MATH_TORSTEN_MPI_COMMUNICATOR_HPP

#ifdef TORSTEN_MPI

#include <boost/mpi.hpp>

namespace torsten {
  namespace mpi {
    /*
     * MPI communicator wrapper for RAII. Note that no
     * MPI's predfined comm sich as @c MPI_COMM_WOLRD are allowed.
     */
    struct Communicator {
    private:
      /*
       * Disable default constructor.
       */
      Communicator();

    public:
      MPI_Comm comm;
      int size;
      int rank;
      Communicator(MPI_Comm other) :
        comm(MPI_COMM_NULL) {
        MPI_Comm_dup(other, &comm);
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);        
      }
      ~Communicator() {
        MPI_Comm_free(&comm);
      }
    };
  }
}

#endif

#endif
