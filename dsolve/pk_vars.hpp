#ifndef STAN_MATH_TORSTEN_DSOLVE_PK_VARS_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PK_VARS_HPP

#include <stan/math/rev/scal/meta/is_var.hpp>

namespace torsten {
  namespace dsolve {
    template<typename T1, typename T2, typename T3>
    std::vector<stan::math::var>
    pk_vars(const std::vector<T1>& v1, const std::vector<T2>& v2, const std::vector<T3>& v3) // NOLINT
    {
      using stan::is_var;
      std::vector<stan::math::var> res;
      if (is_var<T1>::value) res.insert(res.end(), v1.begin(), v1.end());
      if (is_var<T2>::value) res.insert(res.end(), v2.begin(), v2.end());        
      if (is_var<T3>::value) res.insert(res.end(), v3.begin(), v3.end());        
      return res;
    }


    template<typename T0, typename T1, typename T2, typename T3>
    std::vector<stan::math::var>
    pk_vars(const T0& x0, const Eigen::Matrix<T1, 1, -1>& m1, const std::vector<T2>& v2, const std::vector<T3>& v3) // NOLINT
    {
      using stan::is_var;
      std::vector<stan::math::var> res;
      if (is_var<T0>::value) res.push_back(x0);
      if (is_var<T1>::value) {
        std::vector<stan::math::var> v1(m1.data(), m1.data() + m1.size());
        res.insert(res.end(), v1.begin(), v1.end());
      }
      if (is_var<T2>::value) res.insert(res.end(), v2.begin(), v2.end());
      if (is_var<T3>::value) res.insert(res.end(), v3.begin(), v3.end());
      return res;
    }
  }
}

#endif
