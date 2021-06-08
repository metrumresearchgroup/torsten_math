#ifndef STAN_MATH_TORSTEN_APPLY_HPP
#define STAN_MATH_TORSTEN_APPLY_HPP

namespace torsten {
  namespace detail {

    template<typename T, typename F, int... Is>
    void for_each_in_tuple(T&& t, F&& f, std::integer_sequence<int, Is...>) {
      auto l = { (f(std::get<Is>(t)), 0)... };
    }
  } // namespace torsten_detail

  template<typename... Ts, typename F>
  void apply(F&& f, std::tuple<Ts...>& t) {
    detail::for_each_in_tuple(t, f, std::make_integer_sequence<int, sizeof...(Ts)>());
  }
}

#endif
