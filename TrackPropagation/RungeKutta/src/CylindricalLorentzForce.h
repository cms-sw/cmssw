#ifndef CylindricalLorentzForce_H
#define CylindricalLorentzForce_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "RKDerivative.h"

class RKLocalFieldProvider;

template <typename T, int N>
class dso_internal CylindricalLorentzForce final : public RKDerivative<T,N> {
public:

    typedef RKDerivative<T,N>                   Base;
    typedef typename Base::Scalar               Scalar;
    typedef typename Base::Vector               Vector;

    CylindricalLorentzForce( const RKLocalFieldProvider& field) : theField(field) {}

    Vector operator()( Scalar r, const Vector& state) const override;

private:

    const RKLocalFieldProvider& theField;

};

#include "TrackPropagation/RungeKutta/src/CylindricalLorentzForce.icc"

#endif
