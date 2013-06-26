#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerMultiRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/KalmanUpdators/interface/MRHChi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"

std::pair<bool,double>
MRHChi2MeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				   const TransientTrackingRecHit& aRecHit) const {
  if (!aRecHit.isValid()) {
	throw cms::Exception("MRHChi2MeasurementEstimator") << "Invalid RecHit passed to the MRHChi2MeasurementEstimator ";
  }

  typedef AlgebraicROOTObject<2>::Vector Vec;
  typedef AlgebraicROOTObject<2>::SymMatrix Mat;

  //better be a multihit...
  TSiTrackerMultiRecHit const & mHit = dynamic_cast<TSiTrackerMultiRecHit const &>(aRecHit);  
  MeasurementExtractor me(tsos);
  double est=0;
  double annealing = mHit.getAnnealingFactor();
  LogDebug("MRHChi2MeasurementEstimator") << "Current annealing factor is " << annealing; 		
  TransientTrackingRecHit::ConstRecHitContainer components = aRecHit.transientHits();
  LogDebug("MRHChi2MeasurementEstimator") << "this hit has " << components.size() << " components";	
  for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator iter = components.begin(); iter != components.end(); iter++){		
  	Vec r = asSVector<2>((*iter)->parameters()) - me.measuredParameters<2>(**iter);
  	Mat R = asSMatrix<2>((*iter)->parametersError())*annealing + me.measuredError<2>(**iter);
  	//int ierr = ! R.Invert(); // if (ierr != 0) throw exception; // 
	invertPosDefMatrix(R);
	LogDebug("MRHChi2MeasurementEstimator") << "Hit with weight " << (*iter)->weight(); 
  	est += ROOT::Math::Similarity(r, R)*((*iter)->weight());
  }	
  return returnIt(est);
}
