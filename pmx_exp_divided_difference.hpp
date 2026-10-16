#ifndef STAN_MATH_TORSTEN_PMX_EXP_DIVIDED_DIFFERENCE_HPP
#define STAN_MATH_TORSTEN_PMX_EXP_DIVIDED_DIFFERENCE_HPP

#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/expm1.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/rev/fun/exp.hpp>
#include <stan/math/rev/fun/expm1.hpp>
#include <array>
#include <cmath>
#include <stdexcept>

namespace torsten {
namespace internal {

  /**
   * phi1(z) = (exp(z) - 1) / z, extended continuously by phi1(0) = 1.
   *
   * Near zero <code>expm1(z) / z</code> is 0/0 at z = 0 and its
   * derivative loses precision, so there the Taylor series
   * sum_{n=0}^{11} z^n / (n + 1)! is used; for |z| < 0.1 its
   * truncation error is below 1e-21 in value and derivative.
   *
   * @tparam T scalar type
   * @param z argument
   * @return phi1(z)
   */
  template <typename T>
  inline T pmx_phi1(const T& z) {
    using stan::math::expm1;
    if (std::abs(stan::math::value_of_rec(z)) < 0.1) {
      T s = 1.0;
      for (int n = 12; n >= 2; --n) {
        s = 1.0 + z * s / n;
      }
      return s;
    }
    return expm1(z) / z;
  }

  /**
   * Divided differences of g(x) = exp(x t) over the nodes x = -l, for
   * subsets S of K rates l_0, ..., l_{K-1} >= 0. A subset is a bitmask
   * over the rates; each value is computed on first use and memoized.
   *
   * For S of n + 1 rates, G(S) = g[-l_i : i in S] is the response of a
   * chain of compartments with those elimination rates and unit
   * transfer rates: G({l}) = exp(-l t), G({a, k}) = (exp(-k t) -
   * exp(-a t)) / (a - k), and adding the rate 0 integrates over [0, t],
   * e.g. G({0, l}) = (1 - exp(-l t)) / l. G is positive, symmetric and
   * smooth in the rates, including where rates coincide, so a linear
   * compartment model written in terms of G has no removable
   * singularities.
   *
   * A subset whose rates span at least 1 / t uses the recurrence
   *
   *   G(S) = (G(S \ {l_max}) - G(S \ {l_min})) / (l_max - l_min),
   *
   * which divides by the largest separation, so the difference loses at
   * most a few bits. A narrower subset of two rates uses pmx_phi1; one of
   * three or more uses the Taylor series about the mid-rate c,
   *
   *   G(S) = t^n exp(-c t) sum_m h_m(w) / (m + n)!,  w_i = (c - l_i) t,
   *
   * with h_m the complete homogeneous symmetric polynomial of degree m.
   * As |w_i| <= 1/2, the terms past m = 15 are below 2e-18 relative.
   *
   * @tparam T scalar type
   * @tparam K number of rates
   */
  template <typename T, int K>
  class PMXExpDividedDifferences {
    static constexpr int n_taylor = 15;
    std::array<T, K> l_;
    T t_;
    std::array<T, (1 << K)> g_;
    std::array<bool, (1 << K)> done_;

    T compute(unsigned s) {
      using stan::math::exp;
      using stan::math::value_of_rec;

      // the first smallest and the last largest rate: two distinct rates
      // even when they are equal, so that l_max - l_min carries the
      // derivatives with respect to both
      int i_min = -1, i_max = -1, n = -1;
      for (int i = 0; i < K; ++i) {
        if (s & (1u << i)) {
          ++n;
          if (i_min < 0 || value_of_rec(l_[i]) < value_of_rec(l_[i_min])) i_min = i;
          if (i_max < 0 || value_of_rec(l_[i]) >= value_of_rec(l_[i_max])) i_max = i;
        }
      }
      if (n < 0) {
        throw std::domain_error("PMXExpDividedDifferences: empty set of rates");
      }
      if (n == 0) {
        return exp(-l_[i_min] * t_);
      }

      const T dl = l_[i_max] - l_[i_min];
      if (value_of_rec(dl) * value_of_rec(t_) >= 1.0) {
        return ((*this)(s & ~(1u << i_max)) - (*this)(s & ~(1u << i_min))) / dl;
      }
      if (n == 1) {
        return t_ * (*this)(1u << i_min) * pmx_phi1(-dl * t_);
      }

      const T c = 0.5 * (l_[i_min] + l_[i_max]);
      std::array<T, n_taylor + 1> h;
      h[0] = 1.0;
      for (int m = 1; m <= n_taylor; ++m) {
        h[m] = 0.0;
      }
      for (int i = 0; i < K; ++i) {
        if (s & (1u << i)) {
          const T w = (c - l_[i]) * t_;
          for (int m = 1; m <= n_taylor; ++m) {
            h[m] += w * h[m - 1];
          }
        }
      }
      double f = 1.0;  // 1 / (m + n)!
      for (int j = 2; j <= n; ++j) {
        f /= j;
      }
      T sum = f;
      for (int m = 1; m <= n_taylor; ++m) {
        f /= m + n;
        sum += h[m] * f;
      }
      T tn = t_;
      for (int j = 1; j < n; ++j) {
        tn = tn * t_;
      }
      return tn * exp(-c * t_) * sum;
    }

  public:
    /**
     * @param l rates, all non-negative
     * @param t time span, non-negative
     */
    template <typename T_l, typename T_t>
    PMXExpDividedDifferences(const std::array<T_l, K>& l, const T_t& t) : t_(t) {
      for (int i = 0; i < K; ++i) {
        l_[i] = l[i];
      }
      done_.fill(false);
    }

    /**
     * @param s nonempty subset of the rates, as a bitmask
     * @return G(s)
     */
    const T& operator()(unsigned s) {
      if (!done_[s]) {
        g_[s] = compute(s);
        done_[s] = true;
      }
      return g_[s];
    }
  };

}  // namespace internal
}  // namespace torsten

#endif
