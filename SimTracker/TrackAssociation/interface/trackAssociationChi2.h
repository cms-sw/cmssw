#ifndef SimTracker_TrackAssociation_trackAssociationChi2_h
#define SimTracker_TrackAssociation_trackAssociationChi2_h

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

namespace track_associator {
  /// basic method where chi2 is computed
  double trackAssociationChi2(const reco::TrackBase::ParameterVector &rParameters,
                              const reco::TrackBase::CovarianceMatrix &recoTrackCovMatrix,
                              const Basic3DVector<double> &momAtVtx,
                              const Basic3DVector<double> &vert,
                              int charge,
                              const MagneticField &magfield,
                              const reco::BeamSpot &bs);

  double trackAssociationChi2(const reco::TrackBase::ParameterVector &rParameters,
                              const reco::TrackBase::CovarianceMatrix &recoTrackCovMatrix,
                              const TrackingParticle &trackingParticle,
                              const MagneticField &magfield,
                              const reco::BeamSpot &bs);

  double trackAssociationChi2(const reco::TrackBase &track,
                              const TrackingParticle &trackingParticle,
                              const MagneticField &magfield,
                              const reco::BeamSpot &bs);
}  // namespace track_associator

#endif
