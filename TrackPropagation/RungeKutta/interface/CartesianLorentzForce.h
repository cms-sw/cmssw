#ifndef CartesianLorentzForce_H
#define CartesianLorentzForce_H

#include "TrackPropagation/RungeKutta/interface/RKDerivative.h"
#include "TrackPropagation/RungeKutta/interface/RKLocalFieldProvider.h"

class CartesianLorentzForce : public RKDerivative<double,6> {
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
