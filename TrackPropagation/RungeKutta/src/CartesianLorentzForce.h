#ifndef CartesianLorentzForce_H
#define CartesianLorentzForce_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "RKDerivative.h"
#include "RKLocalFieldProvider.h"

/// Derivative calculation for the 6D cartesian case.

class dso_internal CartesianLorentzForce final : public RKDerivative<double,6> {
public:

    typedef RKDerivative< double,6>             Base;
    typedef Base::Scalar                        Scalar;
    typedef Base::Vector                        Vector;

    CartesianLorentzForce( const RKLocalFieldProvider& field, float ch) : 
	theField(field), theCharge(ch) {}

    Vector operator()( Scalar z, const Vector& state) const override;

private:

    const RKLocalFieldProvider& theField;
    float theCharge;

};


#include "CartesianStateAdaptor.h"
inline
CartesianLorentzForce::Vector
CartesianLorentzForce::operator()( Scalar z, const Vector& state) const
{
    // derivatives in case S is the free parameter
    CartesianStateAdaptor start(state);
    auto bfield = theField.inTesla( RKLocalFieldProvider::LocalPoint(start.position()));
    constexpr float k = 2.99792458e-3; // conversion to [cm]

    /// Derivative d(pos)/ds is simply normalized momentum
    auto dpos = start.momentum().unit();

    /// Lorentz force in absence of electric field
    auto dmom = (k*theCharge) * dpos.cross( bfield);

    return CartesianStateAdaptor::rkstate( dpos, dmom);
}

#endif
