#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class ChiralBondRestraint : public Potential {

private:
    int *d_idxs_;
    int *d_signs_;

    const int R_;

public:
    ChiralBondRestraint(
        const std::vector<int> &idxs, // [R, 4]
        const std::vector<int> &signs // [R]
    );

    ~ChiralBondRestraint();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        unsigned long long *d_du_dx, // buffered
        unsigned long long *d_du_dp,
        unsigned long long *d_u, // buffered
        int *d_u_overflow_count,
        cudaStream_t stream) override;
};

} // namespace timemachine
