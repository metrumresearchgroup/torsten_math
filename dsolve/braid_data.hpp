#ifndef STAN_MATH_TORSTEL_DSOLVE_BRAID_DATA_HPP
#define STAN_MATH_TORSTEL_DSOLVE_BRAID_DATA_HPP

namespace torsten {
  namespace dsolve {

    struct braid_data {
      // // Integrator settings
      // realtype rtol;        // relative tolerance
      // realtype atol;        // absolute tolerance
      // int      order;       // ARKode method order
      // bool     linear;      // enable/disable linearly implicit option
      // bool     diagnostics; // output diagnostics

      // XBraid settings
      double   x_tol;           // Xbraid stopping tolerance
      int      x_nt;            // number of fine grid time points
      int      x_skip;          // skip all work on first down cycle
      int      x_max_levels;    // max number of levels
      int      x_min_coarse;    // min possible coarse gird size
      int      x_nrelax;        // number of CF relaxation sweeps on all levels
      int      x_nrelax0;       // number of CF relaxation sweeps on level 0
      int      x_tnorm;         // temporal stopping norm
      int      x_cfactor;       // coarsening factor
      int      x_cfactor0;      // coarsening factor on level 0
      int      x_max_iter;      // max number of interations
      int      x_storage;       // Full storage on levels >= storage
      int      x_print_level;   // xbraid output level
      int      x_access_level;  // access level
      int      x_rfactor_limit; // refinement factor limit
      int      x_rfactor_fail;  // refinement factor on solver failure
      int      x_max_refine;    // max number of refinements
      bool     x_fmg;           // true = FMG cycle, false = V cycle
      bool     x_refine;        // enable refinement with XBraid
      bool     x_initseq;       // initialize with sequential solution
      bool     x_reltol;        // use relative tolerance
      bool     x_init_u0;       // initialize solution to initial condition

      braid_data() {
        x_tol           = 1.0e-6;
        x_nt            = 300;
        x_skip          = 1;
        x_max_levels    = 15;
        x_min_coarse    = 3;
        x_nrelax        = 1;
        x_nrelax0       = -1;
        x_tnorm         = 2;
        x_cfactor       = 10;
        x_cfactor0      = -1;
        x_max_iter      = 100;
        x_storage       = -1;
        x_print_level   = 2;
        x_access_level  = 1;
        x_rfactor_limit = 10;
        x_rfactor_fail  = 4;
        x_max_refine    = 8;
        x_fmg           = false;
        x_refine        = false;
        x_initseq       = false;
        x_reltol        = false;
        x_init_u0       = false;        
      }
    };
  }
}

#endif
