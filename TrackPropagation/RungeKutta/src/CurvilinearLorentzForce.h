#ifndef CurvilinearLorentzForce_H
#define CurvilinearLorentzForce_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "RKDerivative.h"

class RKLocalFieldProvider;

template <typename T, int N>
class dso_internal CurvilinearLorentzForce final : public RKDerivative<T,N> {
public:

    typedef RKDerivative<T,N>                   Base;
    typedef typename Base::Scalar               Scalar;
    typedef typename Base::Vector               Vector;

    CurvilinearLorentzForce( const RKLocalFieldProvider& field) : theField(field) {}

    Vector operator()( Scalar z, const Vector& state) const override;

private:

    const RKLocalFieldProvider& theField;

};

#include "TrackPropagation/RungeKutta/src/CurvilinearLorentzForce.icc"

#endif
