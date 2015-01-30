#include "RecoTracker/SpecialSeedGenerators/interface/CosmicTrackingRegion.h"
#include "TrackingTools/KalmanUpdators/interface/EtaPhiMeasurementEstimator.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include "FWCore/Framework/interface/ESHandle.h"

namespace {
  template <class T> T sqr( T t) {return t*t;}
}

using namespace std;
using namespace ctfseeding; 


TrackingRegion::ctfHits CosmicTrackingRegion::hits(const edm::Event& ev,
						const edm::EventSetup& es,
						const  ctfseeding::SeedingLayer* layer) const
{
  TrackingRegion::ctfHits result;
  TrackingRegion::Hits tmp;
  hits_(ev, es, *layer, tmp);
  result.reserve(tmp.size());
  for ( auto h : tmp) result.emplace_back(*h); // not owned
  return result;
}

TrackingRegion::Hits CosmicTrackingRegion::hits(const edm::Event& ev,
						const edm::EventSetup& es,
						const SeedingLayerSetsHits::SeedingLayer& layer) const
{
  TrackingRegion::Hits result;
  hits_(ev, es, layer, result);
  return result;
}

template <typename T>
void CosmicTrackingRegion::hits_(
				 const edm::Event& ev,
				 const edm::EventSetup& es,
				 const T& layer, TrackingRegion::Hits  & result) const
{

  //get and name collections
  //++++++++++++++++++++++++

 
  //detector layer
  const DetLayer * detLayer = layer.detLayer();
  LogDebug("CosmicTrackingRegion") << "Looking at hits on subdet/layer " << layer.name();
  EtaPhiMeasurementEstimator est(0.3,0.3);

  //magnetic field
  edm::ESHandle<MagneticField> field;
  es.get<IdealMagneticFieldRecord>().get(field);
  const MagneticField * magField = field.product();

  //region
  const GlobalPoint vtx = origin();
  GlobalVector dir = direction();
  LogDebug("CosmicTrackingRegion") <<"The initial region characteristics are:" << "\n"
				   <<" Origin    = " << origin() << "\n"
				   <<" Direction = " << direction() << "\n" 
				   <<" Eta = " << origin().eta()  << "\n" 
				   <<" Phi = " << origin().phi();
     
  //trajectory state on surface
  float phi = dir.phi();
  Surface::RotationType rot( sin(phi), -cos(phi),           0,
                             0,                0,          -1,
                             cos(phi),  sin(phi),           0);

  Plane::PlanePointer surface = Plane::build(vtx, rot);
  FreeTrajectoryState fts( GlobalTrajectoryParameters(vtx, dir, 1, magField) );
  TrajectoryStateOnSurface tsos(fts, *surface);
  LogDebug("CosmicTrackingRegion") 
    << "The state used to find measurement with the measurement tracker is:\n" << tsos;

  //propagator
  AnalyticalPropagator prop( magField, alongMomentum);

  //propagation verification (debug)
  //++++++++++++++++++++++++++++++++

  //creation of the state
  TrajectoryStateOnSurface stateOnLayer = prop.propagate( *tsos.freeState(),
							  detLayer->surface());
  
  //verification of the state
  if (stateOnLayer.isValid()){
    LogDebug("CosmicTrackingRegion") << "The initial state propagates to the layer surface: \n" << stateOnLayer
				     << "R   = " << stateOnLayer.globalPosition().perp() << "\n"
				     << "Eta = " << stateOnLayer.globalPosition().eta() << "\n"
				     << "Phi = " << stateOnLayer.globalPosition().phi();

  }
  else{
    LogDebug("CosmicTrackingRegion") << "The initial state does not propagate to the layer surface.";
  }

  //number of compatible dets
  typedef DetLayer::DetWithState DetWithState;
  vector<DetWithState> compatDets = detLayer->compatibleDets(tsos, prop, est);
  LogDebug("CosmicTrackingRegion") << "Compatible dets = " << compatDets.size();
  

  //get hits
  //++++++++

  //measurement tracker (find hits)
  LayerMeasurements lm(theMeasurementTracker_->measurementTracker(), *theMeasurementTracker_);
  vector<TrajectoryMeasurement> meas = lm.measurements(*detLayer, tsos, prop, est);
  LogDebug("CosmicTrackingRegion") << "Number of Trajectory measurements = " << meas.size()
				   <<" but the last one is always an invalid hit, by construction.";

  //trajectory measurement

  // std::cout <<"CRegion b " << cache.size() << std::endl;

  // waiting for a migration at LayerMeasurements level and at seed builder level
  for (auto const & im : meas) {
    if(!im.recHit()->isValid()) continue;
    assert(!trackerHitRTTI::isUndef(*im.recHit()->hit()));
    auto ptrHit = (BaseTrackerRecHit *)(im.recHit()->hit()->clone());
    cache.emplace_back(ptrHit);
    result.emplace_back(ptrHit);
  }

  // std::cout <<"CRegion a " << cache.size() << std::endl;

}


