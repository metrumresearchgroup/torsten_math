#ifndef STAN_MATH_TORSTEN_MPI_INIT_HPP
#define STAN_MATH_TORSTEN_MPI_INIT_HPP

#ifdef TORSTEN_MPI
#include <boost/mpi.hpp>
#endif

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
