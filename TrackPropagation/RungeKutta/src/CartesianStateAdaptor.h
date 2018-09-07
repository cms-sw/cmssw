#ifndef CartesianStateAdaptor_H
#define CartesianStateAdaptor_H

#include "RKSmallVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

class dso_internal CartesianStateAdaptor {
public:

    typedef float                             Scalar;
    typedef Basic3DVector<Scalar>             Vector3D;

    CartesianStateAdaptor( const RKSmallVector<double,6>& rk) :
      pos_(rk[0], rk[1], rk[2]), mom_(rk[3], rk[4], rk[5]) {}

    const Vector3D& position() const { return pos_;}
    const Vector3D& momentum() const { return mom_;}

    static Vector3D position(const RKSmallVector<double,6>& rk) {
      return Vector3D(rk[0], rk[1], rk[2]);
    }

    static Vector3D momentum(const RKSmallVector<double,6>& rk) {
      return Vector3D(rk[3], rk[4], rk[5]);
    }

    static RKSmallVector<double,6> rkstate( const Vector3D& pos, const Vector3D& mom) {
	RKSmallVector<double,6> res;
	res[0] = pos.x();
	res[1] = pos.y();
	res[2] = pos.z();
	res[3] = mom.x();
	res[4] = mom.y();
	res[5] = mom.z();
	return res;
    }

private:

    Vector3D pos_;
    Vector3D mom_;

};

#endif
