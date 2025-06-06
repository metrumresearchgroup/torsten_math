#include <stan/math/rev.hpp>
#include <boost/mp11.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/functor/test_fixture_ode.hpp>
#include <test/unit/math/rev/functor/test_fixture_ode_lorenz.hpp>
#include <stan/math/torsten/test/unit/dsolve/ode_test_functors.hpp>

/**
 *
 * Use same solver functor type for both w & w/o tolerance control
 */
template <typename solve_type, typename... Ts>
using pmx_ode_test_tuple = std::tuple<solve_type, solve_type, Ts...>;

/**
 * Outer product of test types
 */
using lorenz_test_types = boost::mp11::mp_product<
    pmx_ode_test_tuple,
    ::testing::Types<pmx_ode_adams_functor, pmx_ode_bdf_functor, pmx_ode_ckrk_functor,
                     pmx_ode_rk45_functor>>;

TYPED_TEST_SUITE_P(lorenz_test);
TYPED_TEST_P(lorenz_test, param_and_data_finite_diff) {
  if (std::is_same<TypeParam,
                   std::tuple<pmx_ode_rk45_functor, pmx_ode_rk45_functor>>::value) {
    this->test_fd_vd(1.e-6, 3e-2);
    this->test_fd_dv(1.e-6, 3e-2);
    this->test_fd_vv(1.e-6, 3e-2);
  } else if (std::is_same<TypeParam, std::tuple<pmx_ode_ckrk_functor,
                                                pmx_ode_ckrk_functor>>::value) {
    this->test_fd_vd(1.e-6, 5e-2);
    this->test_fd_dv(1.e-6, 5e-2);
    this->test_fd_vv(1.e-6, 5e-2);
  } else {
    this->test_fd_vd(1.e-6, 1e-2);
    this->test_fd_dv(1.e-6, 1e-2);
    this->test_fd_vv(1.e-6, 1e-2);
  }
}
REGISTER_TYPED_TEST_SUITE_P(lorenz_test, param_and_data_finite_diff);
INSTANTIATE_TYPED_TEST_SUITE_P(TorstenODE, lorenz_test, lorenz_test_types);
