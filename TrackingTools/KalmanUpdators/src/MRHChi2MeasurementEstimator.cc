#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/KalmanUpdators/interface/MRHChi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "DataFormats/Math/interface/ProjectMatrix.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"

std::pair<bool, double> MRHChi2MeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
                                   const TrackingRecHit& aRecHit) const {

  switch (aRecHit.dimension()) {
    case 1:       return estimate<1>(tsos,aRecHit);
    case 2:       return estimate<2>(tsos,aRecHit);
    //avoid the not-(1D or 2D)  hit due to the final sum  
    //supposing all the hits inside of a MRH have the same dimension
    case 3:
    case 4:
    case 5:{
      LogDebug("MRHChi2MeasurementEstimator") << "WARNING:The hit is not 1D either 2D:" 
					      << " does not count in the MRH Chi2 estimation." ;
      double est = 0.0; 
      return  HitReturnType(false, est);
      }
    }
    throw cms::Exception("Rec hit of invalid dimension (not 1,2,3,4,5)") <<
      "The value was " << aRecHit.dimension() <<
      ", type is " << typeid(aRecHit).name() << "\n";
}

//---------------------------------------------------------------------------------------------------------------
template <unsigned int N>
std::pair<bool, double> MRHChi2MeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
                                                const TrackingRecHit& aRecHit) const {
 
  LogDebug("MRHChi2MeasurementEstimator") << "Calling MRHChi2MeasurementEstimator" ;               
  SiTrackerMultiRecHit const & mHit = dynamic_cast<SiTrackerMultiRecHit const &>(aRecHit);  
  double est=0;

  double annealing = mHit.getAnnealingFactor();
  LogDebug("MRHChi2MeasurementEstimator") << "Current annealing factor is " << annealing;               

  std::vector<const TrackingRecHit*> components = mHit.recHits();
  LogDebug("MRHChi2MeasurementEstimator") << "this hit has " << components.size() << " components";     

  int iComp = 0;
  for(std::vector<const TrackingRecHit*>::const_iterator iter = components.begin(); iter != components.end(); iter++, iComp++){

    // define variables that will be used to setup the KfComponentsHolder
    ProjectMatrix<double,5,N>  pf;
    typename AlgebraicROOTObject<N>::Vector r, rMeas;
    typename AlgebraicROOTObject<N,N>::SymMatrix V, VMeas;
    AlgebraicVector5 x = tsos.localParameters().vector();
    const AlgebraicSymMatrix55 &C = (tsos.localError().matrix());

    // setup the holder with the correct dimensions and get the values
    KfComponentsHolder holder;
    holder.template setup<N>(&r, &V, &pf, &rMeas, &VMeas, x, C);
    (**iter).getKfComponents(holder);

    r -= rMeas;
    V = V*annealing + VMeas;
    bool ierr = invertPosDefMatrix(V);
    if( !ierr ) {
      edm::LogError("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::ComputeParameters2dim: W not valid!"<<std::endl;
    }

    LogDebug("MRHChi2MeasurementEstimator") << "Hit with weight " << mHit.weight(iComp); 
    est += ROOT::Math::Similarity(r, V)*mHit.weight(iComp);

  }     

  return returnIt(est);

}

