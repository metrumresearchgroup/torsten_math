#ifndef STAN_MATH_TORSTEN_DUPLICATE_HPP
#define STAN_MATH_TORSTEN_DUPLICATE_HPP

#include <iostream>
#include <sstream>
#include <vector>

namespace torsten {

  template<typename T, typename T_out>
    std::vector<std::vector<T_out> > duplicate(const std::vector<std::vector<T> >& d)
    {
      std::vector<std::vector<T_out> > res(d.size());
      for (size_t i = 0; i < d.size(); ++i) {
        res[i].resize(d[i].size());
        for (size_t j = 0; j < d[i].size(); ++j) {
          res[i][j] = stan::math::value_of(d[i][j]);
        }
      }
      return res;
    }
    
  template<typename T, typename T_out>
    std::vector<T_out> duplicate(const std::vector<T>& d)
    {
      std::vector<T_out> res(d.size());
      for (size_t i = 0; i < d.size(); ++i) {
        res[i] = stan::math::value_of(d[i]);
      }
      return res;
    }
}

#endif
