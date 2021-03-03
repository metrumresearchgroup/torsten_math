#ifndef STAN_MATH_TORSTEN_DSOLVE_ODE_TUPLE_FUNCTOR_HPP
#define STAN_MATH_TORSTEN_DSOLVE_ODE_TUPLE_FUNCTOR_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/err.hpp>
#include <stdexcept>
#include <ostream>
#include <vector>

namespace torsten {
namespace dsolve {
  
  /** 
   * A functor that forward a tuple of parameters
   * to functor callback that accepts a parameter pack.
   * @tparam F original functor callback
   * 
   */
  template<typename F>
  struct UnpackTupleFunc {
    F const& f_;
    UnpackTupleFunc(F const& f) : f_(f) {}

    template<typename Tuple>
    auto operator()(Tuple const& t) {
      static constexpr auto size = std::tuple_size<Tuple>::value;
      return call(t, std::make_index_sequence<size>{});
    }

  private:
    template<typename Tuple, size_t ... I>
    auto call(Tuple const& t, std::index_sequence<I ...>) {
      return f_(std::get<I>(t) ...);
    }
  };

  /** 
   * ODE RHS Functor that accepts parameter pack
   * 
   * @tparam F original ODE RHS functor
   * @tparam Tt time type
   * @tparam T_init unknown variables' type
   */
  template<typename F, typename Tt, typename T_init>
  struct VariadicOdeFunc {
    F const& f_;
    const Tt& t_;
    const Eigen::Matrix<T_init, -1, 1>& y_;
    std::ostream* msgs_;
    
    VariadicOdeFunc(F const& f, const Tt& t,
                    const Eigen::Matrix<T_init, -1, 1>& y,
                    std::ostream* msgs) :
      f_(f), t_(t), y_(y), msgs_(msgs)
    {}

    template<typename... T_par>
    auto operator()(const T_par&... args) const {
      return f_(t_, y_, msgs_, args...);
    }
  };

  /** 
   * ODE RHS functor that accepts tuple in place of parameter pack
   * 
   * @tparam F original ODE RHS functor type
   * 
   */
  template<typename F>
  struct TupleOdeFunc {
    const F& f_;
    
    TupleOdeFunc(F const& f) : f_(f) {}

    template<typename Tt, typename T_init, typename Tuple>
    auto operator()(const Tt& t,
                    const Eigen::Matrix<T_init, -1, 1>& y,
                    std::ostream* msgs,
                    const Tuple& args) const {
      VariadicOdeFunc<F, Tt, T_init> f(f_, t, y, msgs);
      UnpackTupleFunc<VariadicOdeFunc<F, Tt, T_init> > f1(f);
      return f1(args);
    }
  };

}
}

#endif
