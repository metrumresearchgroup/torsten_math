#ifndef STAN_MATH_TORSTEN_DSOLVE_ARKODE_DEF_HPP
#define STAN_MATH_TORSTEN_DSOLVE_ARKODE_DEF_HPP

#include <arkode/arkode.h>
#include <arkode/arkode_direct.h>
#include <nvector/nvector_serial.h>

#ifndef TORSTEN_ARK_ISM
#define TORSTEN_ARK_ISM CV_STAGGERED
#endif

#ifndef TORSTEN_CV_SENS
#define TORSTEN_CV_SENS AD
#endif

namespace torsten {
  namespace dsolve {

    /**
     * define CVODES constants
     * 
     */
    template<PMXCvodesSensMethod Sm, int Lmm, int ism>
    struct cvodes_def {
      static constexpr PMXCvodesSensMethod cv_sm = Sm; 
      static constexpr int cv_lmm = Lmm; 
      static constexpr int cv_ism = ism; 
    };
  }
}

#endif
