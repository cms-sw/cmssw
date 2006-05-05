#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"

PerigeeTrajectoryError::PerigeeTrajectoryError(const reco::perigee::Covariance & perigeeCov) :
  weightIsAvailable(false)
{
  thePerigeeError = AlgebraicSymMatrix(5,0);
  thePerigeeError(1,1) = perigeeCov(reco::perigee::i_tcurv,reco::perigee::i_tcurv);
  thePerigeeError(2,2) = perigeeCov(reco::perigee::i_theta,reco::perigee::i_theta);
  thePerigeeError(3,3) = perigeeCov(reco::perigee::i_phi0,reco::perigee::i_phi0);
  thePerigeeError(4,4) = perigeeCov(reco::perigee::i_d0,reco::perigee::i_d0);
  thePerigeeError(5,5) = perigeeCov(reco::perigee::i_dz,reco::perigee::i_dz);

  thePerigeeError(1,2) = perigeeCov(reco::perigee::i_tcurv,reco::perigee::i_theta);
  thePerigeeError(1,3) = perigeeCov(reco::perigee::i_tcurv,reco::perigee::i_phi0);
  thePerigeeError(1,4) = perigeeCov(reco::perigee::i_tcurv,reco::perigee::i_d0);
  thePerigeeError(1,5) = perigeeCov(reco::perigee::i_tcurv,reco::perigee::i_dz);

  thePerigeeError(2,3) = perigeeCov(reco::perigee::i_theta,reco::perigee::i_phi0);
  thePerigeeError(2,4) = perigeeCov(reco::perigee::i_theta,reco::perigee::i_d0);
  thePerigeeError(2,5) = perigeeCov(reco::perigee::i_theta,reco::perigee::i_dz);

  thePerigeeError(3,4) = perigeeCov(reco::perigee::i_phi0,reco::perigee::i_d0);
  thePerigeeError(3,5) = perigeeCov(reco::perigee::i_phi0,reco::perigee::i_dz);

  thePerigeeError(4,5) = perigeeCov(reco::perigee::i_d0,reco::perigee::i_dz);
    
}

PerigeeTrajectoryError::operator reco::perigee::Covariance() const
{
  reco::perigee::Covariance perigeeCov;
  perigeeCov(reco::perigee::i_tcurv,reco::perigee::i_tcurv) = thePerigeeError(1,1);
  perigeeCov(reco::perigee::i_theta,reco::perigee::i_theta) = thePerigeeError(2,2);
  perigeeCov(reco::perigee::i_phi0,reco::perigee::i_phi0)   = thePerigeeError(3,3);
  perigeeCov(reco::perigee::i_d0,reco::perigee::i_d0)	    = thePerigeeError(4,4);
  perigeeCov(reco::perigee::i_dz,reco::perigee::i_dz)	    = thePerigeeError(5,5);

  perigeeCov(reco::perigee::i_tcurv,reco::perigee::i_theta) = thePerigeeError(1,2);
  perigeeCov(reco::perigee::i_tcurv,reco::perigee::i_phi0)  = thePerigeeError(1,3);
  perigeeCov(reco::perigee::i_tcurv,reco::perigee::i_d0)    = thePerigeeError(1,4);
  perigeeCov(reco::perigee::i_tcurv,reco::perigee::i_dz)    = thePerigeeError(1,5);

  perigeeCov(reco::perigee::i_theta,reco::perigee::i_phi0)  = thePerigeeError(2,3);
  perigeeCov(reco::perigee::i_theta,reco::perigee::i_d0)    = thePerigeeError(2,4);
  perigeeCov(reco::perigee::i_theta,reco::perigee::i_dz)    = thePerigeeError(2,5);

  perigeeCov(reco::perigee::i_phi0,reco::perigee::i_d0)     = thePerigeeError(3,4);
  perigeeCov(reco::perigee::i_phi0,reco::perigee::i_dz)     = thePerigeeError(3,5);

  perigeeCov(reco::perigee::i_d0,reco::perigee::i_dz)	    = thePerigeeError(4,5);
  return perigeeCov;
}
