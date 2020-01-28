#pragma once

#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType, int D>
class HarmonicAngle : public Gradient<RealType, D> {

private:

    int *d_angle_idxs_;
    int *d_param_idxs_;

    const int A_;

public:

    HarmonicAngle(
        const std::vector<int> &angle_idxs, // [n, 3]
        const std::vector<int> &param_idxs);

    ~HarmonicAngle();

    /*
    Execute the force computation, the semantics are:

    1. If d_coords_tangents == null, then out_coords != null, out_coords_tangent == null, out_params_tangents == null
    2. If d_coords_tangents != null, then out_coords == null, out_coords_tangent != null, out_params_tangents != null

    */
    virtual void execute_device(
        const int N,
        const int P,
        const RealType *d_coords,
        const RealType *d_coords_tangents,
        const RealType *d_params,
        unsigned long long *out_coords,
        RealType *out_coords_tangents,
        RealType *out_params_tangents
    ) override;


};


}