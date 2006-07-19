#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

PerigeeTrajectoryError::PerigeeTrajectoryError(const CovarianceMatrix & perigeeCov) :
  weightIsAvailable(false)
{
  thePerigeeError = AlgebraicSymMatrix(5,0);
  thePerigeeError(1,1) = perigeeCov(reco::TrackBase::i_transverseCurvature,reco::TrackBase::i_transverseCurvature);
  thePerigeeError(2,2) = perigeeCov(reco::TrackBase::i_theta,reco::TrackBase::i_theta);
  thePerigeeError(3,3) = perigeeCov(reco::TrackBase::i_phi0,reco::TrackBase::i_phi0);
  thePerigeeError(4,4) = perigeeCov(reco::TrackBase::i_d0,reco::TrackBase::i_d0);
  thePerigeeError(5,5) = perigeeCov(reco::TrackBase::i_dz,reco::TrackBase::i_dz);

  thePerigeeError(1,2) = perigeeCov(reco::TrackBase::i_transverseCurvature,reco::TrackBase::i_theta);
  thePerigeeError(1,3) = perigeeCov(reco::TrackBase::i_transverseCurvature,reco::TrackBase::i_phi0);
  thePerigeeError(1,4) = perigeeCov(reco::TrackBase::i_transverseCurvature,reco::TrackBase::i_d0);
  thePerigeeError(1,5) = perigeeCov(reco::TrackBase::i_transverseCurvature,reco::TrackBase::i_dz);

  thePerigeeError(2,3) = perigeeCov(reco::TrackBase::i_theta,reco::TrackBase::i_phi0);
  thePerigeeError(2,4) = perigeeCov(reco::TrackBase::i_theta,reco::TrackBase::i_d0);
  thePerigeeError(2,5) = perigeeCov(reco::TrackBase::i_theta,reco::TrackBase::i_dz);

  thePerigeeError(3,4) = perigeeCov(reco::TrackBase::i_phi0,reco::TrackBase::i_d0);
  thePerigeeError(3,5) = perigeeCov(reco::TrackBase::i_phi0,reco::TrackBase::i_dz);

  thePerigeeError(4,5) = perigeeCov(reco::TrackBase::i_d0,reco::TrackBase::i_dz);
    
}

PerigeeTrajectoryError::operator CovarianceMatrix() const
{
  CovarianceMatrix perigeeCov;
  perigeeCov(reco::TrackBase::i_transverseCurvature,reco::TrackBase::i_transverseCurvature) = thePerigeeError(1,1);
  perigeeCov(reco::TrackBase::i_theta,reco::TrackBase::i_theta) = thePerigeeError(2,2);
  perigeeCov(reco::TrackBase::i_phi0,reco::TrackBase::i_phi0)   = thePerigeeError(3,3);
  perigeeCov(reco::TrackBase::i_d0,reco::TrackBase::i_d0)	    = thePerigeeError(4,4);
  perigeeCov(reco::TrackBase::i_dz,reco::TrackBase::i_dz)	    = thePerigeeError(5,5);

  perigeeCov(reco::TrackBase::i_transverseCurvature,reco::TrackBase::i_theta) = thePerigeeError(1,2);
  perigeeCov(reco::TrackBase::i_transverseCurvature,reco::TrackBase::i_phi0)  = thePerigeeError(1,3);
  perigeeCov(reco::TrackBase::i_transverseCurvature,reco::TrackBase::i_d0)    = thePerigeeError(1,4);
  perigeeCov(reco::TrackBase::i_transverseCurvature,reco::TrackBase::i_dz)    = thePerigeeError(1,5);

  perigeeCov(reco::TrackBase::i_theta,reco::TrackBase::i_phi0)  = thePerigeeError(2,3);
  perigeeCov(reco::TrackBase::i_theta,reco::TrackBase::i_d0)    = thePerigeeError(2,4);
  perigeeCov(reco::TrackBase::i_theta,reco::TrackBase::i_dz)    = thePerigeeError(2,5);

  perigeeCov(reco::TrackBase::i_phi0,reco::TrackBase::i_d0)     = thePerigeeError(3,4);
  perigeeCov(reco::TrackBase::i_phi0,reco::TrackBase::i_dz)     = thePerigeeError(3,5);

  perigeeCov(reco::TrackBase::i_d0,reco::TrackBase::i_dz)	    = thePerigeeError(4,5);
  return perigeeCov;
}
