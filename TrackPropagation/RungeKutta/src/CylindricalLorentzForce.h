#ifndef CylindricalLorentzForce_H
#define CylindricalLorentzForce_H

#include "RKDerivative.h"

class RKLocalFieldProvider;

template <typename T, int N>
class CylindricalLorentzForce : public RKDerivative<T,N> {
public:

    typedef RKDerivative<T,N>                   Base;
    typedef typename Base::Scalar               Scalar;
    typedef typename Base::Vector               Vector;

    CylindricalLorentzForce( const RKLocalFieldProvider& field) : theField(field) {}

    virtual Vector operator()( Scalar r, const Vector& state) const;

private:

    const RKLocalFieldProvider& theField;

};

#include "TrackPropagation/RungeKutta/src/CylindricalLorentzForce.icc"

#endif
