//
// Package:         RecoTracker/RoadSearchTrackCandidateMaker
// Class:           RoadSearchTrackCandidateMakerAlgorithm
// 
// Description:     Converts cleaned clouds into
//                  TrackCandidates using the 
//                  TrajectoryBuilder framework
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 15 13:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/04/10 22:36:15 $
// $Revision: 1.5 $
//

#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/RoadSearchTrackCandidateMaker/interface/RoadSearchTrackCandidateMakerAlgorithm.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
//nclude "RecoTracker/CkfPattern/interface/CombinatorialTrajectoryBuilder.h"

RoadSearchTrackCandidateMakerAlgorithm::RoadSearchTrackCandidateMakerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

RoadSearchTrackCandidateMakerAlgorithm::~RoadSearchTrackCandidateMakerAlgorithm() {
}

void RoadSearchTrackCandidateMakerAlgorithm::run(const RoadSearchCloudCollection* input,
			      const edm::EventSetup& es,
			      TrackCandidateCollection &output)
{

//
// right now, track candidates are just filled from cleaned
// clouds. The trajectory of the seed is taken as the initial
// trajectory for the final fit
//

  edm::LogInfo("RoadSearch") << "Clean Clouds input size: " << input->size();

  for ( RoadSearchCloudCollection::const_iterator cloud = input->begin(); cloud != input->end(); ++cloud ) {

    // fill rechits from cloud into new OwnVector
    edm::OwnVector<TrackingRecHit> recHits;
    edm::OwnVector<TrackingRecHit> goodHits;

    RoadSearchCloud::RecHitOwnVector hits = cloud->recHits();
    for ( RoadSearchCloud::RecHitOwnVector::const_iterator rechit = hits.begin(); rechit != hits.end(); ++rechit) {      
      recHits.push_back(rechit->clone());
    }

    // take the first seed to fill in trackcandidate
    RoadSearchCloud::SeedRefs seeds = cloud->seeds();
    RoadSearchCloud::SeedRef ref = *(seeds.begin());

    // sort recHits, done after getting seed because propagationDirection is needed
    // get tracker geometry
    edm::ESHandle<TrackerGeometry> tracker;
    es.get<TrackerDigiGeometryRecord>().get(tracker);

    //const MeasurementTracker*  theMeasurementTracker = new MeasurementTracker(es); // will need this later


    recHits.sort(TrackingRecHitLessFromGlobalPosition(((TrackingGeometry*)(&(*tracker))),ref->direction()));

    // clone 
    TrajectorySeed seed = *((*ref).clone());
    PTrajectoryStateOnDet state = *((*ref).startingState().clone());
    TrajectoryStateOnSurface firstState;
      
    // check if Trajectory from seed is on first hit of the cloud, if not, propagate
    // exclude if first state on first hit is not valid
    edm::ESHandle<MagneticField> magField_;
    es.get<IdealMagneticFieldRecord>().get(magField_);

    const TrackerGeometry * geom = tracker.product();
    const MagneticField * magField = magField_.product();

    bool valid = true;
    if (recHits.begin()->geographicalId().rawId() != state.detId()) {
      AnalyticalPropagator prop(magField,anyDirection);
      const GeomDetUnit* det = geom->idToDetUnit(recHits.begin()->geographicalId());
      const GeomDetUnit* detState = geom->idToDetUnit(DetId(state.detId())  );
      
      TrajectoryStateTransform transformer;
      TrajectoryStateOnSurface before(transformer.transientState(state,  &(detState->surface()), magField));
      firstState = prop.propagate(before, det->surface());
      
      if (firstState.isValid() == false){
	valid=false;
      }
      state = *(transformer.persistentState(firstState,recHits.begin()->geographicalId().rawId()));
    }
    else{
      const GeomDetUnit* det = geom->idToDetUnit(recHits.begin()->geographicalId());
      const GeomDetUnit* detState = geom->idToDetUnit(DetId(state.detId())  );
      TrajectoryStateTransform transformer;
      TrajectoryStateOnSurface start(transformer.transientState(state,  &(detState->surface()), magField));
      firstState = start;
    }
    //if (valid == true) output.push_back(TrackCandidate(recHits,seed,state));
    if (!valid) continue;

    //convert the TrackingRecHit vector to a TransientTrackingRecHit vector
    edm::OwnVector<TransientTrackingRecHit> transHits;
    TransientTrackingRecHitBuilder * builder;
    builder = new TkTransientTrackingRecHitBuilder((TrackingGeometry*)(&(*tracker)));


    //Loop over RecHits and propagate trajectory to each hit

    double chi2cut = 30;
    Chi2MeasurementEstimator theEstimator(chi2cut);
    KFUpdator theUpdator;

    Trajectory traj(seed,ref->direction());
    edm::LogInfo("RoadSearch") << "Loop over hits to check measurements...";
    for (edm::OwnVector<TrackingRecHit>::const_iterator rhit=recHits.begin(); rhit!=recHits.end(); rhit++){

      TransientTrackingRecHit* ihit = builder->build(&(*rhit));

      //PropagatorWithMaterial thePropagator;
      AnalyticalPropagator thePropagator(magField,anyDirection);

      const GeomDetUnit* det = geom->idToDetUnit(rhit->geographicalId());

      TrajectoryStateOnSurface predTsos;
      TrajectoryStateOnSurface currTsos;

      if (traj.measurements().empty()) {
	predTsos = thePropagator.propagate(firstState, det->surface());
      } else {
	currTsos = traj.measurements().back().updatedState();
	predTsos = thePropagator.propagate(currTsos, det->surface());
      }
      if (!predTsos.isValid()) continue;
      TrajectoryMeasurement tm;

      GlobalPoint p = det->surface().toGlobal(ihit->localPosition());
      MeasurementEstimator::HitReturnType est = theEstimator.estimate(predTsos, *ihit);
      //std::cout << "Hit "<<nhit<<" of "<<transHits.size()<<" chi2 = " << est.second 
      //	<<"\t at r:z = "<<p.perp()<<":"<<p.z()<<std::endl;
      //std::cout<<predTsos;

      if (!est.first) continue;
      currTsos = theUpdator.update(predTsos, *ihit);
      tm = TrajectoryMeasurement((TrajectoryStateOnSurface)predTsos,(TrajectoryStateOnSurface)currTsos,&(*ihit));
      traj.push(tm);
      goodHits.push_back(rhit->clone());
      
    }
    if (goodHits.size()>2) output.push_back(TrackCandidate(goodHits,seed,state));

    delete builder;
  }

  edm::LogInfo("RoadSearch") << "Found " << output.size() << " track candidates.";

};


