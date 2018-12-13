#ifndef STAN_MATH_TORSTEN_DSOLVE_PK_VARS_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PK_VARS_HPP

#include <stan/math/rev/scal/meta/is_var.hpp>

namespace torsten {
  namespace dsolve {
    template<typename T1, typename T2, typename T3>
    std::vector<typename stan::return_type<T1, T2, T3>::type>
    pk_vars(const std::vector<T1>& v1, const std::vector<T2>& v2, const std::vector<T3>& v3) // NOLINT
    {
      using stan::is_var;
      std::vector<typename stan::return_type<T1, T2, T3>::type> res;
      if (is_var<T1>::value) res.insert(res.end(), v1.begin(), v1.end());
      if (is_var<T2>::value) res.insert(res.end(), v2.begin(), v2.end());        
      if (is_var<T3>::value) res.insert(res.end(), v3.begin(), v3.end());        
      return res;
    }
  }
}

#endif
