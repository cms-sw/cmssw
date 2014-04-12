#ifndef CartesianLorentzForce_H
#define CartesianLorentzForce_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "RKDerivative.h"
#include "RKLocalFieldProvider.h"

/// Derivative calculation for the 6D cartesian case.

class dso_internal CartesianLorentzForce GCC11_FINAL : public RKDerivative<double,6> {
public:

    typedef RKDerivative< double,6>             Base;
    typedef Base::Scalar                        Scalar;
    typedef Base::Vector                        Vector;

    CartesianLorentzForce( const RKLocalFieldProvider& field, double ch) : 
	theField(field), theCharge(ch) {}

    virtual Vector operator()( Scalar z, const Vector& state) const;

private:

    const RKLocalFieldProvider& theField;
    double theCharge;

};


#include "CartesianStateAdaptor.h"
inline
CartesianLorentzForce::Vector
CartesianLorentzForce::operator()( Scalar z, const Vector& state) const
{
    // derivatives in case S is the free parameter
    CartesianStateAdaptor start(state);
    RKLocalFieldProvider::Vector bfield = theField.inTesla( RKLocalFieldProvider::LocalPoint(start.position()));
    double k = 2.99792458e-3; // conversion to [cm]

    /// Derivative d(pos)/ds is simply normalized momentum
    CartesianStateAdaptor::Vector3D dpos = start.momentum().unit();

    /// Lorentz force in absence of electric field
    CartesianStateAdaptor::Vector3D dmom = (k*theCharge) * dpos.cross( bfield);

    return CartesianStateAdaptor::rkstate( dpos, dmom);
}

#endif
