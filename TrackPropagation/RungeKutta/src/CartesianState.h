#ifndef CartesianState_H
#define CartesianState_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "VectorDoublet.h"

class dso_internal CartesianState {
public:

  typedef double                            Scalar;
  typedef Basic3DVector<Scalar>             Vector3D;
  typedef VectorDoublet<Vector3D,Vector3D>  Vector;

  CartesianState() {}
  CartesianState( const Vector& v, Scalar s) : par_(v), charge_(s) {}
  CartesianState( const Vector3D& pos, const Vector3D& mom, Scalar s) : 
    par_(pos,mom), charge_(s) {}

  const Vector3D& position() const { return par_.first();}
  const Vector3D& momentum() const { return par_.second();}

  const Vector& parameters() const { return par_;}

  Scalar charge() const { return charge_;}

private:

  Vector par_;
  Scalar charge_;

};

inline CartesianState
operator+( const CartesianState& a, const CartesianState& b) {
  return CartesianState(a.parameters()+b.parameters(), a.charge());
}

inline CartesianState
operator-( const CartesianState& a, const CartesianState& b) {
  return CartesianState(a.parameters()-b.parameters(), a.charge());
}

inline CartesianState operator*( const CartesianState& v, const CartesianState::Scalar& s) {
  return CartesianState( v.parameters()*s, v.charge());
}
inline CartesianState operator*( const CartesianState::Scalar& s, const CartesianState& v) {
  return CartesianState( v.parameters()*s, v.charge());
}

inline CartesianState operator/( const CartesianState& v, const CartesianState::Scalar& s) {
  return CartesianState( v.parameters()/s, v.charge());
}


#endif
