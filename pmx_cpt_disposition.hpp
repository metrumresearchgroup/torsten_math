#ifndef STAN_MATH_TORSTEN_PMX_CPT_DISPOSITION_HPP
#define STAN_MATH_TORSTEN_PMX_CPT_DISPOSITION_HPP

#include <stan/math/torsten/torsten_def.hpp>
#include <stan/math/torsten/pmx_exp_divided_difference.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/rev/fun/sqrt.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/meta/return_type.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

namespace torsten {
namespace internal {

  /*
   * Closed-form solutions of the linear compartment models y' = A y + r
   * with first-order absorption from a depot (compartment 0) into a one-
   * or two-compartment disposition, optionally followed by an effect
   * compartment.
   *
   * Every entry of exp(A t) is a positive combination of divided
   * differences G(S) of exp(-l t) over sets S of the model's rates (see
   * PMXExpDividedDifferences), so the solutions stay finite and accurate
   * where rates coincide, e.g. ka == CL / V, where an eigendecomposition
   * of A breaks down. Adding the rate 0 (bit Z) to every set turns exp(A t)
   * into its integral over [0, t], the response to a constant infusion.
   * Likewise an effect compartment, which integrates the central amount
   * at rate ke, sees ke times the central row with ke added to every set.
   *
   * A disposition provides:
   *   Ncmt, K, Z          compartments, rates, the bit of the rate 0
   *   rates()             the K rates
   *   apply(g, v, x, out) out += M v, where M is exp(A t) with the
   *                       rates x added to every set
   *   central(g, v, x)    the central row (index 1) of M v
   *   periodic(g, u)      u <- (I - exp(A ii))^{-1} u, by forward
   *                       substitution over the triangular block structure
   *   constant(u, r, cmt) u <- -A^{-1} r e_cmt
   * where g holds the divided differences at the time span t or ii.
   */

  /**
   * One-compartment disposition: depot -> central at ka, eliminated at k10.
   */
  template <typename T_par>
  struct PMXOneCptDisposition {
    using par_type = T_par;
    static constexpr int Ncmt = 2;
    static constexpr int K = 3;
    static constexpr unsigned Z = 1, KA = 2, K10 = 4;

    const T_par ka, k10;

    PMXOneCptDisposition(const T_par& ka_, const T_par& k10_) : ka(ka_), k10(k10_) {}

    std::array<T_par, K> rates() const { return {T_par(0.0), ka, k10}; }

    template <typename G, typename V>
    auto central(G& g, const V& v, unsigned x) const {
      return g(K10 | x) * v[1] + ka * g(KA | K10 | x) * v[0];
    }

    template <typename G, typename V, typename T>
    void apply(G& g, const V& v, unsigned x, PKRec<T>& out) const {
      out(0) += g(KA | x) * v[0];
      out(1) += central(g, v, x);
    }

    template <typename G, typename T>
    void periodic(G& g, PKRec<T>& u) const {
      // 1 - exp(-l ii) = l G({0, l}), accurate for small l ii
      u(0) = u(0) / (ka * g(Z | KA));
      u(1) = (u(1) + ka * g(KA | K10) * u(0)) / (k10 * g(Z | K10));
    }

    template <typename T, typename T_r>
    void constant(PKRec<T>& u, const T_r& r, int cmt) const {
      if (cmt == 1) {
        u(0) = r / ka;
      }
      u(1) = r / k10;
    }
  };

  /**
   * Two-compartment disposition: depot -> central at ka, central <->
   * peripheral at k12 and k21, eliminated from central at k10.
   *
   * The central/peripheral block has the distinct eigenvalues -alpha and
   * -beta, alpha - beta = s = sqrt(w^2 + 4 k12 k21) > 0 with w = k10 + k12 -
   * k21. Its spectral projectors have the entries (wp, -k21; -k12, wm) / s
   * and (wm, k21; k12, wp) / s with wp = (s + w) / 2, wm = (s - w) / 2, so
   * each entry of exp(A t) is a positive combination of divided
   * differences over {ka, alpha, beta}: the depot rate may equal either
   * disposition rate.
   */
  template <typename T_par>
  struct PMXTwoCptDisposition {
    using par_type = T_par;
    static constexpr int Ncmt = 3;
    static constexpr int K = 4;
    static constexpr unsigned Z = 1, KA = 2, A = 4, B = 8;

    T_par ka, k10, k12, k21;
    T_par s, alpha, beta, wp, wm;

    PMXTwoCptDisposition(const T_par& ka_, const T_par& k10_,
                         const T_par& k12_, const T_par& k21_)
      : ka(ka_), k10(k10_), k12(k12_), k21(k21_) {
      using stan::math::sqrt;
      const T_par w = k10 + k12 - k21;
      s = sqrt(w * w + 4.0 * k12 * k21);
      alpha = 0.5 * (k10 + k12 + k21 + s);
      beta = k10 * k21 / alpha;  // alpha beta = k10 k21, no cancellation
      if (w >= 0) {              // wp wm = k12 k21
        wp = 0.5 * (s + w);
        wm = k12 * k21 / wp;
      } else {
        wm = 0.5 * (s - w);
        wp = k12 * k21 / wm;
      }
    }

    std::array<T_par, K> rates() const { return {T_par(0.0), ka, alpha, beta}; }

    template <typename G, typename V>
    auto central(G& g, const V& v, unsigned x) const {
      return (wp * g(A | x) + wm * g(B | x)) / s * v[1] + k21 * g(A | B | x) * v[2]
        + ka * (wp * g(KA | A | x) + wm * g(KA | B | x)) / s * v[0];
    }

    template <typename G, typename V, typename T>
    void apply(G& g, const V& v, unsigned x, PKRec<T>& out) const {
      out(0) += g(KA | x) * v[0];
      out(1) += central(g, v, x);
      out(2) += k12 * g(A | B | x) * v[1] + (wm * g(A | x) + wp * g(B | x)) / s * v[2]
        + ka * k12 * g(KA | A | B | x) * v[0];
    }

    template <typename G, typename T>
    void periodic(G& g, PKRec<T>& u) const {
      // 1 - exp(-l ii) = l G({0, l}), accurate for small l ii
      u(0) = u(0) / (ka * g(Z | KA));
      const T w1 = u(1) + ka * (wp * g(KA | A) + wm * g(KA | B)) / s * u(0);
      const T w2 = u(2) + ka * k12 * g(KA | A | B) * u(0);
      // central/peripheral block: I - exp(B ii) has eigenvalues qa, qb
      const auto qa = alpha * g(Z | A);
      const auto qb = beta * g(Z | B);
      const auto det = qa * qb;
      u(1) = ((wm * qa + wp * qb) / s * w1 + k21 * g(A | B) * w2) / det;
      u(2) = (k12 * g(A | B) * w1 + (wp * qa + wm * qb) / s * w2) / det;
    }

    template <typename T, typename T_r>
    void constant(PKRec<T>& u, const T_r& r, int cmt) const {
      if (cmt == 1) {
        u(0) = r / ka;
      }
      u(1) = r / k10;
      u(2) = (cmt == 3 ? k10 + k12 : k12) * r / (k10 * k21);
    }
  };

  /**
   * A disposition D followed by an effect compartment, y_e' = ke (y_1 - y_e).
   */
  template <typename D>
  struct PMXEffCptDisposition {
    using par_type = typename D::par_type;
    static constexpr int Ncmt = D::Ncmt + 1;
    static constexpr int K = D::K + 1;
    static constexpr unsigned Z = D::Z, KE = 1u << D::K;

    const D cpt;
    const par_type ke;

    std::array<par_type, K> rates() const {
      std::array<par_type, K> l;
      const std::array<par_type, D::K> l_cpt = cpt.rates();
      std::copy(l_cpt.begin(), l_cpt.end(), l.begin());
      l[K - 1] = ke;
      return l;
    }

    template <typename G, typename V, typename T>
    void apply(G& g, const V& v, unsigned x, PKRec<T>& out) const {
      cpt.apply(g, v, x, out);
      out(Ncmt - 1) += g(KE | x) * v[Ncmt - 1] + ke * cpt.central(g, v, x | KE);
    }

    template <typename G, typename T>
    void periodic(G& g, PKRec<T>& u) const {
      cpt.periodic(g, u);
      u(Ncmt - 1) = (u(Ncmt - 1) + ke * cpt.central(g, u, KE)) / (ke * g(Z | KE));
    }

    template <typename T, typename T_r>
    void constant(PKRec<T>& u, const T_r& r, int cmt) const {
      if (cmt < Ncmt) {
        cpt.constant(u, r, cmt);
        u(Ncmt - 1) = u(1);
      } else {
        u(Ncmt - 1) = r / ke;
      }
    }
  };

  template <typename D, typename T_t>
  PMXExpDividedDifferences<stan::return_type_t<typename D::par_type, T_t>, D::K>
  pmx_cpt_divided_differences(const D& d, const T_t& t) {
    return PMXExpDividedDifferences<stan::return_type_t<typename D::par_type, T_t>, D::K>(d.rates(), t);
  }

  /**
   * Transient solution: y <- exp(A t) y + (integral of exp(A s) over [0, t]) rate.
   */
  template <typename D, typename T, typename T_t, typename T1>
  void pmx_cpt_solve(const D& d, PKRec<T>& y, const T_t& t, const std::vector<T1>& rate) {
    auto g = pmx_cpt_divided_differences(d, t);
    PKRec<T> pred = PKRec<T>::Zero(D::Ncmt);
    d.apply(g, y, 0, pred);
    // a constant zero rate adds nothing; an autodiff rate keeps its gradient
    if (!std::is_arithmetic<T1>::value
        || std::any_of(rate.begin(), rate.end(), [](const T1& r) { return r != 0; })) {
      d.apply(g, rate, D::Z, pred);
    }
    y = pred;
  }

  /**
   * Steady state at the start of a dosing interval, before that
   * interval's dose: the periodic solution u = exp(A ii) u + v, with v
   * the amounts one interval's dosing alone leaves at its end, or for a
   * constant infusion (ii = 0) the equilibrium u = -A^{-1} rate.
   */
  template <typename D, typename T_amt, typename T_r, typename T_ii>
  PKRec<stan::return_type_t<typename D::par_type, T_amt, T_r, T_ii>>
  pmx_cpt_steady_state(const D& d, const T_amt& amt, const T_r& rate, const T_ii& ii, int cmt) {
    using stan::math::value_of;
    using scalar = stan::return_type_t<typename D::par_type, T_amt, T_r, T_ii>;

    PKRec<scalar> u = PKRec<scalar>::Zero(D::Ncmt);
    if (rate == 0 || ii > 0) {
      if (rate == 0) {  // bolus
        u(cmt - 1) = amt;
        pmx_cpt_solve(d, u, ii, std::vector<double>(D::Ncmt, 0.0));
      } else {  // truncated infusions
        // Infusions longer than ii overlap: in each interval the rate
        // is (n + 1) * rate for t1 and n * rate for the remaining t2.
        const stan::return_type_t<T_amt, T_r> dt = amt / rate;
        const int n = int(std::floor(value_of(dt) / value_of(ii)) + 0.1);
        std::vector<T_r> r(D::Ncmt, 0.0);
        r[cmt - 1] = (n + 1) * rate;
        pmx_cpt_solve(d, u, dt - n * ii, r);
        r[cmt - 1] = n * rate;
        pmx_cpt_solve(d, u, (n + 1) * ii - dt, r);
      }
      auto g = pmx_cpt_divided_differences(d, ii);
      d.periodic(g, u);
    } else {  // constant infusion
      d.constant(u, rate, cmt);
    }
    return u;
  }

}  // namespace internal
}  // namespace torsten

#endif
