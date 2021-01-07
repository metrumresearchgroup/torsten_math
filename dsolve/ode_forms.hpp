#ifndef STAN_MATH_TORSTEN_DSOLVE_ODE_FORMS_HPP
#define STAN_MATH_TORSTEN_DSOLVE_ODE_FORMS_HPP

namespace torsten {
  namespace dsolve {

    template <typename F, typename Tt, typename T_init, typename T_par>
    struct PMXOdeintSystem;

    template <typename F, typename Tts, typename Ty0, typename Tpar, typename cv_def>
    class PMXCvodesSystem;

    template <typename F, typename Tts, typename Ty0, typename Tpar>
    class PMXArkodeSystem;

    enum PMXOdeForms { Odeint, Cvodes, Arkode };

    template<typename T>
    struct OdeForm {
      static const PMXOdeForms value = Cvodes;
    };

    template<typename... Ts>
    struct OdeForm<PMXOdeintSystem<Ts...>> {
      static const PMXOdeForms value = Odeint;
    };

    template<typename... Ts>
    struct OdeForm<PMXArkodeSystem<Ts...>> {
      static const PMXOdeForms value = Arkode;
    };

  }
}


#endif
