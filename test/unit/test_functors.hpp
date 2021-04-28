#ifndef TORSTEN_PMX_TEST_FUNCTORS_HPP
#define TORSTEN_PMX_TEST_FUNCTORS_HPP

#include <stan/math/torsten/torsten.hpp>

#define PMX_ADD_FUNCTOR(name, func)             \
  struct name##_functor {                       \
    template <typename... Ts>                   \
    auto operator()(Ts... args) {               \
      return func(args...);                     \
    }                                           \
  };

PMX_ADD_FUNCTOR(pmx_solve_onecpt, torsten::pmx_solve_onecpt);
PMX_ADD_FUNCTOR(pmx_solve_linode, torsten::pmx_solve_linode);
PMX_ADD_FUNCTOR(pmx_solve_adams, torsten::pmx_solve_adams);
PMX_ADD_FUNCTOR(pmx_solve_bdf, torsten::pmx_solve_bdf);
PMX_ADD_FUNCTOR(pmx_solve_rk45, torsten::pmx_solve_rk45);


template<typename S>
struct is_linode_solver : std::false_type {};

template<>
struct is_linode_solver<pmx_solve_linode_functor> : std::true_type {};

template<typename S>
struct is_ode_solver : std::false_type {};

template<>
struct is_ode_solver<pmx_solve_rk45_functor> : std::true_type {};

template<>
struct is_ode_solver<pmx_solve_bdf_functor> : std::true_type {};

template<>
struct is_ode_solver<pmx_solve_adams_functor> : std::true_type {};

#endif
