
#include "TrackPropagation/RungeKutta/interface/CartesianLorentzForce.h"
#include "TrackPropagation/RungeKutta/interface/CartesianStateAdaptor.h"

CartesianLorentzForce::Vector
CartesianLorentzForce::operator()( Scalar z, const Vector& state) const
{
    // derivatives in case S is the free parameter
    CartesianStateAdaptor start(state);
    RKLocalFieldProvider::Vector bfield = theField.inTesla( RKLocalFieldProvider::LocalPoint(start.position()));
    double k = 2.99792458e-3; // conversion to [cm]

    CartesianStateAdaptor::Vector3D dpos = start.momentum().unit();
    CartesianStateAdaptor::Vector3D dmom = k*theCharge * dpos.cross( bfield);

    return CartesianStateAdaptor::rkstate( dpos, dmom);
}
