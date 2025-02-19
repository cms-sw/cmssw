#ifndef CartesianLorentzForce_H
#define CartesianLorentzForce_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "RKDerivative.h"
#include "RKLocalFieldProvider.h"

/// Derivative calculation for the 6D cartesian case.

class dso_internal CartesianLorentzForce : public RKDerivative<double,6> {
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

#endif
