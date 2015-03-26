#ifndef getChi2_h
#define getChi2_h

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

namespace track_associator {
  /// basic method where chi2 is computed
  double getChi2(const reco::TrackBase::ParameterVector& rParameters,
		 const reco::TrackBase::CovarianceMatrix& recoTrackCovMatrix,
		 const Basic3DVector<double>& momAtVtx,
		 const Basic3DVector<double>& vert,
		 int charge,
                 const MagneticField&,
		 const reco::BeamSpot&) ;
}

#endif
