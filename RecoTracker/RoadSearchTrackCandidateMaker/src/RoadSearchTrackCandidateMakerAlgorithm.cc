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
// $Author: burkett $
// $Date: 2006/08/28 23:05:45 $
// $Revision: 1.19 $
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
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
//nclude "RecoTracker/CkfPattern/interface/CombinatorialTrajectoryBuilder.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"


RoadSearchTrackCandidateMakerAlgorithm::RoadSearchTrackCandidateMakerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 

  theNumHitCut = (unsigned int)conf_.getParameter<int>("NumHitCut");
  theChi2Cut   = conf_.getParameter<double>("HitChi2Cut");

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

  //
  // get the transient builder
  //
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  es.get<TransientRecHitRecord>().get(builderName,theBuilder);

  // Create the trajectory cleaner 
  TrajectoryCleanerBySharedHits theTrajectoryCleaner;
  vector<Trajectory> FinalTrajectories;

  
  // need this to sort recHits, sorting done after getting seed because propagationDirection is needed
  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  
  edm::ESHandle<MagneticField> magField_;
  es.get<IdealMagneticFieldRecord>().get(magField_);
  
  const TrackerGeometry * geom = tracker.product();
  const MagneticField * magField = magField_.product();
  
  PropagatorWithMaterial thePropagator(alongMomentum,.1057,magField); 
  AnalyticalPropagator prop(magField,anyDirection);
  TrajectoryStateTransform transformer;

  Chi2MeasurementEstimator theEstimator(theChi2Cut);
  KFUpdator theUpdator;


  LogDebug("RoadSearch") << "Clean Clouds input size: " << input->size();

  int i_c = 0;
  for ( RoadSearchCloudCollection::const_iterator cloud = input->begin(); cloud != input->end(); ++cloud ) {

    LogDebug("RoadSearch") << "Cloud #"<<i_c ; ++i_c;
    // fill rechits from cloud into new OwnVector
    edm::OwnVector<TrackingRecHit> recHits;
    RoadSearchCloud::RecHitOwnVector hits = cloud->recHits();
    for ( RoadSearchCloud::RecHitOwnVector::const_iterator rechit = hits.begin(); rechit != hits.end(); ++rechit) {
      recHits.push_back(rechit->clone());
    }

    vector<Trajectory> CloudTrajectories;

    RoadSearchCloud::SeedRefs theSeeds = cloud->seeds();
    RoadSearchCloud::SeedRefs::const_iterator iseed;
    recHits.sort(TrackingRecHitLessFromGlobalPosition(((TrackingGeometry*)(&(*tracker))),alongMomentum));
      
    for ( iseed = theSeeds.begin(); iseed != theSeeds.end(); ++iseed ) {
      RoadSearchCloud::SeedRef ref = *iseed;
      
      // clone 
      TrajectorySeed seed = (*ref);
      PTrajectoryStateOnDet state = (*ref).startingState();
      TrajectoryStateOnSurface firstState;
      
      // check if Trajectory from seed is on first hit of the cloud, if not, propagate
      // exclude if first state on first hit is not valid
      
      bool valid = true;
      if (recHits.begin()->geographicalId().rawId() != state.detId()) {
	const GeomDet* det = geom->idToDet(recHits.begin()->geographicalId());
	const GeomDet* detState = geom->idToDet(DetId(state.detId())  );
	
	TrajectoryStateOnSurface before(transformer.transientState(state,  &(detState->surface()), magField));
	firstState = prop.propagate(before, det->surface());
	
	if (firstState.isValid() == false){
	  valid=false;
	}
	else {
	  state = *(transformer.persistentState(firstState,recHits.begin()->geographicalId().rawId()));
	}
      }
      else{
	//const GeomDet* det = geom->idToDet(recHits.begin()->geographicalId());
	const GeomDet* detState = geom->idToDet(DetId(state.detId())  );
	TrajectoryStateOnSurface start(transformer.transientState(state,  &(detState->surface()), magField));
	firstState = start;
      }
      //if (valid == true) output.push_back(TrackCandidate(recHits,seed,state));
      if (!valid) continue;
      
      //Loop over RecHits and propagate trajectory to each hit
            
      Trajectory traj(seed,ref->direction());
      //Trajectory traj( *((*ref).clone()),ref->direction());
      //edm::LogInfo("RoadSearch") << "Loop over hits to check measurements...";
      float my_chi = 0.0;
      for (edm::OwnVector<TrackingRecHit>::const_iterator rhit=recHits.begin(); rhit!=recHits.end(); rhit++){

	//TransientTrackingRecHit* ihit = theBuilder.product()->build(&(*rhit));
	TransientTrackingRecHit::RecHitPointer ihit = theBuilder.product()->build(&(*rhit));	
	
	const GeomDet* det = geom->idToDet(rhit->geographicalId());
	
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
	
	MeasurementEstimator::HitReturnType est = theEstimator.estimate(predTsos, *ihit);
	
	if (!est.first) continue;
	my_chi+=est.second;
	currTsos = theUpdator.update(predTsos, *ihit);
	tm = TrajectoryMeasurement((TrajectoryStateOnSurface)predTsos,
				   (TrajectoryStateOnSurface)currTsos,
				   &(*ihit));
	traj.push(tm,est.second);
	
      }
      //std::cout<<"This trajectory has chi2 = "<<my_chi<<std::endl;
      if (traj.recHits().size()>theNumHitCut) CloudTrajectories.push_back(traj);
            
    }

    theTrajectoryCleaner.clean(CloudTrajectories);
    for (vector<Trajectory>::const_iterator it = CloudTrajectories.begin();
	 it != CloudTrajectories.end(); it++) {
      if (it->isValid()) FinalTrajectories.push_back(*it);
    }

  } // End loop over Cloud Collection
    theTrajectoryCleaner.clean(FinalTrajectories);
    for (vector<Trajectory>::const_iterator it = FinalTrajectories.begin();
	 it != FinalTrajectories.end(); it++) {
      LogDebug("RoadSearch") << "Trajectory has "<<it->recHits().size()<<" hits with chi2="
			     <<it->chiSquared()<<" and is valid? "<<it->isValid();
      if (it->isValid()){

	edm::OwnVector<TrackingRecHit> goodHits;
	//edm::OwnVector<const TransientTrackingRecHit> ttHits = it->recHits();	
	//for (edm::OwnVector<const TransientTrackingRecHit>::const_iterator rhit=ttHits.begin(); 
	TransientTrackingRecHit::ConstRecHitContainer ttHits = it->recHits();		
	for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator rhit=ttHits.begin(); 
	     rhit!=ttHits.end(); ++rhit)
	  goodHits.push_back((*rhit)->hit()->clone());

	// clone 
	//TrajectorySeed seed = *((*ref).clone());
	PTrajectoryStateOnDet state = it->seed().startingState();
	TrajectoryStateOnSurface firstState;

	// check if Trajectory from seed is on first hit of the cloud, if not, propagate
	// exclude if first state on first hit is not valid
      
	bool valid = true;
	DetId FirstHitId = (*(it->recHits().begin()))->geographicalId();
	if (FirstHitId.rawId() != state.detId()) {
	  //if (it->recHits().begin()->geographicalId().rawId() != state.detId()) {
	  const GeomDet* det = geom->idToDet(FirstHitId);
	  const GeomDet* detState = geom->idToDet(DetId(state.detId())  );
	  
	  TrajectoryStateOnSurface before(transformer.transientState(state,  &(detState->surface()), magField));
	  firstState = prop.propagate(before, det->surface());
	  
	  if (firstState.isValid() == false){
	    valid=false;
	  }
	  state = *(transformer.persistentState(firstState,FirstHitId.rawId()));
	}
	else{
	  const GeomDet* detState = geom->idToDet(DetId(state.detId())  );
	  TrajectoryStateOnSurface start(transformer.transientState(state,  &(detState->surface()), magField));
	  firstState = start;
	}
	
	if (!valid){
	  continue;
	}
	output.push_back(TrackCandidate(goodHits,it->seed(),state));
      }
    }


  LogDebug("RoadSearch") << "Found " << output.size() << " track candidates.";

};


