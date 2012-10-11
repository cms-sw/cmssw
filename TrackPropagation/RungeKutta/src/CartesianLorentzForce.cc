
#include "CartesianLorentzForce.h"
#include "CartesianStateAdaptor.h"

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
    CartesianStateAdaptor::Vector3D dmom = k*theCharge * dpos.cross( bfield);

    return CartesianStateAdaptor::rkstate( dpos, dmom);
}
