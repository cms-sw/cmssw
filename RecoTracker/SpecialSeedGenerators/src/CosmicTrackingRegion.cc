#include "RecoTracker/SpecialSeedGenerators/interface/CosmicTrackingRegion.h"
#include "TrackingTools/KalmanUpdators/interface/EtaPhiMeasurementEstimator.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

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


TrackingRegion::Hits CosmicTrackingRegion::hits(const edm::Event& ev,
						const edm::EventSetup& es,
						const  ctfseeding::SeedingLayer* layer) const
{
  return hits_(ev, es, *layer);
}

TrackingRegion::Hits CosmicTrackingRegion::hits(const edm::Event& ev,
						const edm::EventSetup& es,
						const SeedingLayerSetsHits::SeedingLayer& layer) const
{
  return hits_(ev, es, layer);
}

template <typename T>
TrackingRegion::Hits CosmicTrackingRegion::hits_(const edm::Event& ev,
						const edm::EventSetup& es,
						const T& layer) const
{

  //get and name collections
  //++++++++++++++++++++++++

  //tracking region
  TrackingRegion::Hits result;

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
  edm::ESHandle<MeasurementTracker> measurementTrackerESH;
  es.get<CkfComponentsRecord>().get(measurementTrackerName_,measurementTrackerESH);
  const MeasurementTracker * measurementTracker = measurementTrackerESH.product(); 
  edm::Handle<MeasurementTrackerEvent> mte;
  ev.getByLabel(edm::InputTag("MeasurementTrackerEvent"), mte);
  LayerMeasurements lm(*measurementTracker, *mte);
  vector<TrajectoryMeasurement> meas = lm.measurements(*detLayer, tsos, prop, est);
  LogDebug("CosmicTrackingRegion") << "Number of Trajectory measurements = " << meas.size()
				   <<" but the last one is always an invalid hit, by construction.";

  //trajectory measurement
  typedef vector<TrajectoryMeasurement>::const_iterator IM;

  for (IM im = meas.begin(); im != meas.end(); im++) {//loop on measurement tracker
    TrajectoryMeasurement::ConstRecHitPointer ptrHit = im->recHit();

    if (ptrHit->isValid()) { 
      LogDebug("CosmicTrackingRegion") << "Hit found in the region at position: "<<ptrHit->globalPosition();
      result.push_back(  ptrHit );
    }//end if isValid()

    else LogDebug("CosmicTrackingRegion") << "No valid hit";
  }//end loop on measurement tracker

  
  //result
  //++++++

  return result;
}


