#ifndef CurvilinearLorentzForce_H
#define CurvilinearLorentzForce_H

#include "RKDerivative.h"

class RKLocalFieldProvider;

template <typename T, int N>
class CurvilinearLorentzForce : public RKDerivative<T,N> {
public:

    typedef RKDerivative<T,N>                   Base;
    typedef typename Base::Scalar               Scalar;
    typedef typename Base::Vector               Vector;

    CurvilinearLorentzForce( const RKLocalFieldProvider& field) : theField(field) {}

    virtual Vector operator()( Scalar z, const Vector& state) const;

private:

    const RKLocalFieldProvider& theField;

};

#include "TrackPropagation/RungeKutta/src/CurvilinearLorentzForce.icc"

#endif
