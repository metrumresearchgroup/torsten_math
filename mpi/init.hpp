#ifndef STAN_MATH_TORSTEN_MPI_INIT_HPP
#define STAN_MATH_TORSTEN_MPI_INIT_HPP

#include <boost/mpi.hpp>

namespace torsten {
  namespace mpi {

    void init() {
#ifdef TORSTEN_MPI
      int flag;
      MPI_Initialized(&flag);
      if(!flag) MPI_Init(NULL, NULL);
#endif
    }
  }
}

#endif
