#ifndef STAN_MATH_TORSTEN_NONMEM_EVENT_ARGS
#define STAN_MATH_TORSTEN_NONMEM_EVENT_ARGS

/**
 * simpily pmx function declare
 * 
 */
#define TORSTEN_PMX_FUNC_EVENTS_ARGS const std::vector<T0>& time,\
    const std::vector<T1>& amt,                                  \
    const std::vector<T2>& rate,                                 \
    const std::vector<T3>& ii,                                   \
    const std::vector<int>& evid,                                \
    const std::vector<int>& cmt,                                 \
    const std::vector<int>& addl,                                \
    const std::vector<int>& ss

#endif
