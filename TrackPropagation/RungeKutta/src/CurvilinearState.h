#ifndef CurvilinearState_H
#define CurvilinearState_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "VectorDoublet.h"

/**
State for solving the equation of motion with Z as free variable.
The dependent variables are
  x      - x coordinate
  y      - y coordinate
  dx/dz  - derivative of x versus z
  dy/dz  - derivative of y versus z
  q/p    - charge over momentum magnitude

The coordinate system is externally defined
*/

class dso_internal CurvilinearState {
public:

  typedef double                            Scalar;
  typedef Basic2DVector<Scalar>             Vector2D;
  typedef Basic3DVector<Scalar>             Vector3D;
  typedef VectorDoublet<Vector2D,Vector3D>  Vector;

  CurvilinearState() {}

  CurvilinearState( const Vector& v, Scalar z, Scalar pzsign) : 
    par_(v), z_(z), pzSign_(pzsign) {}

  CurvilinearState( const Vector3D& pos, const Vector3D& p, Scalar ch) : 
    par_(Vector2D(pos.x(),pos.y()), Vector3D( p.x()/p.z(), p.y()/p.z(), ch/p.mag())), 
    z_(pos.z()), pzSign_(p.z()>0. ? 1.:-1.) {}
       
  const Vector3D position() const { 
    return Vector3D(par_.first().x(),par_.first().y(),z_);
  }

  const Vector3D momentum() const {
    Scalar p = 1./fabs(par_.second().z());
    if ( p>1.e9 )  p = 1.e9;
    Scalar dxdz = par_.second().x();
    Scalar dydz = par_.second().y();
    Scalar dz = pzSign_/sqrt(1. + dxdz*dxdz + dydz*dydz);
    Scalar dx = dz*dxdz;
    Scalar dy = dz*dydz;
    return Vector3D(dx*p, dy*p, dz*p);
  }

  const Vector& parameters() const { return par_;}

  Scalar charge() const { return par_.second().z()>0 ? 1 : -1;}

  Scalar z() const {return z_;}

  double pzSign() const {return pzSign_;}

private:

  Vector par_;
  Scalar z_;
  Scalar pzSign_; ///< sign of local pz

};

inline CurvilinearState
operator+( const CurvilinearState& a, const CurvilinearState& b) {
  return CurvilinearState(a.parameters()+b.parameters(), a.z()+b.z(), a.pzSign());
}

inline CurvilinearState
operator-( const CurvilinearState& a, const CurvilinearState& b) {
  return CurvilinearState(a.parameters()-b.parameters(), a.z()-b.z(), a.pzSign());
}

inline CurvilinearState operator*( const CurvilinearState& v, 
				   const CurvilinearState::Scalar& s) {
  return CurvilinearState( v.parameters()*s, v.z()*s, v.pzSign());
}
inline CurvilinearState operator*( const CurvilinearState::Scalar& s,
				   const CurvilinearState& v) {
  return CurvilinearState( v.parameters()*s, v.z()*s, v.pzSign());
}

inline CurvilinearState operator/( const CurvilinearState& v, 
				   const CurvilinearState::Scalar& s) {
  return CurvilinearState( v.parameters()/s, v.z()/s, v.pzSign());
}


#endif
