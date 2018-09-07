#ifndef CylindricalState_H
#define CylindricalState_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "FWCore/Utilities/interface/Visibility.h"
#include "RKSmallVector.h"

#include <iostream>

/**
State for solving the equation of motion with radius (in cylindrical coordinates) as free variable.
The dependent variables are
  phi     - azimuthal angle
  z       - z coordinate
  dphi/dr - derivative of phi versus r
  dz/dr   - derivative of z versus r
  q/p     - charge over momentum magnitude

The coordinate system is externally defined
*/

class dso_internal CylindricalState {
public:

  typedef double                                   Scalar;
  typedef RKSmallVector<Scalar,5>                  Vector;

  CylindricalState() {}

  CylindricalState( const LocalPoint& pos, const LocalVector& mom, Scalar ch) {
      rho_ = pos.perp();
      Scalar cosphi = pos.x() / rho_;
      Scalar sinphi = pos.y() / rho_;
      Scalar p_rho   =  mom.x() * cosphi + mom.y() * sinphi;
      Scalar p_phi   = -mom.x() * sinphi + mom.y() * cosphi;

      par_(0) = pos.phi();
      par_(1) = pos.z();
      par_(2) = p_phi / (p_rho * rho_);
      par_(3) = mom.z() / p_rho;
      par_(4) = ch / mom.mag();

      prSign_ = p_rho > 0 ? 1.0 : -1.0;

      std::cout << "CylindricalState built from pos " << pos << " mom " << mom << " charge " << ch << std::endl;
      std::cout << "p_rho " << p_rho << " p_phi " << p_phi << " dphi_drho " << par_(2) << std::endl;
      std::cout << "Which results in                " << position() << " mom " << momentum() 
	   << " charge " << charge() << std::endl;
  }
  
  CylindricalState( Scalar rho, const Vector& par, Scalar prSign) :
      par_(par), rho_(rho), prSign_(prSign) {}


  const LocalPoint position() const { 
      return LocalPoint( LocalPoint::Cylindrical( rho_, par_(0), par_(1)));
  }

  const LocalVector momentum() const {
      Scalar cosphi = cos( par_(0));
      Scalar sinphi = sin( par_(0));
      Scalar Q = sqrt(1 + rho_*rho_ * par_(2)*par_(2) + par_(3)*par_(3));
      Scalar P = std::abs(1./par_(4));
      Scalar p_rho = prSign_*P/Q;
      Scalar p_phi = rho_*par_(2)*p_rho;
      Scalar p_z   = par_(3)*p_rho;
      LocalVector result( p_rho*cosphi - p_phi*sinphi,
			  p_rho*sinphi + p_phi*cosphi,
			  p_z);
      return result;
  }

  const Vector& parameters() const { return par_;}

  Scalar charge() const { return par_(4) > 0 ? 1 : -1;}

  Scalar rho() const {return rho_;}

  double prSign() const {return prSign_;}

private:

  Vector par_;
  Scalar rho_;
  Scalar prSign_; ///< sign of local p_r

};

#endif
