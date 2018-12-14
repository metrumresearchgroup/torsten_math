#ifdef TORSTEN_MPI

#ifndef STAN_MATH_TORSTEN_MPI_INIT_HPP
#define STAN_MATH_TORSTEN_MPI_INIT_HPP

#include <boost/mpi.hpp>

namespace torsten {
  namespace mpi {

    void init() {
      int flag;
      MPI_Initialized(&flag);
      if(!flag) MPI_Init(NULL, NULL);
    }
  }
}

#endif

#endif
