//
// Package:         RecoTracker/RoadSearchTrackCandidateMaker
// Class:           RoadSearchTrackCandidateMaker
// 
// Description:     Calls RoadSeachTrackCandidateMakerAlgorithm
//                  to convert cleaned clouds into
//                  TrackCandidates using the 
//                  TrajectoryBuilder framework
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 15 13:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/07/08 20:32:40 $
// $Revision: 1.5 $
//

#include <memory>
#include <string>

#include "RoadSearchTrackCandidateMaker.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

RoadSearchTrackCandidateMaker::RoadSearchTrackCandidateMaker(edm::ParameterSet const& conf) : 
  roadSearchTrackCandidateMakerAlgorithm_(conf) ,
  conf_(conf)
{
  produces<TrackCandidateCollection>();

  cloudProducer_ = conf_.getParameter<edm::InputTag>("CloudProducer");

}


// Virtual destructor needed.
RoadSearchTrackCandidateMaker::~RoadSearchTrackCandidateMaker() { }  

// Functions that gets called by framework every event
void RoadSearchTrackCandidateMaker::produce(edm::Event& e, const edm::EventSetup& es)
{
  // Step A: Get Inputs 


  // retrieve producer name of raw CloudCollection
  edm::Handle<RoadSearchCloudCollection> cloudHandle;
  e.getByLabel(cloudProducer_, cloudHandle);
  const RoadSearchCloudCollection *clouds = cloudHandle.product();

  // Step B: create empty output collection
  std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);

  // Step C: Invoke the cloud cleaning algorithm
  roadSearchTrackCandidateMakerAlgorithm_.run(clouds,e,es,*output);

  // Step D: write output to file
  e.put(output);

}
