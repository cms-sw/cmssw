#ifndef RoadSearchTrackCandidateMaker_h
#define RoadSearchTrackCandidateMaker_h

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
// $Date: 2007/03/07 21:46:51 $
// $Revision: 1.3 $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "RecoTracker/RoadSearchTrackCandidateMaker/interface/RoadSearchTrackCandidateMakerAlgorithm.h"

class RoadSearchTrackCandidateMaker : public edm::EDProducer
{
public:

  explicit RoadSearchTrackCandidateMaker(const edm::ParameterSet& conf);

  virtual ~RoadSearchTrackCandidateMaker();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:
  RoadSearchTrackCandidateMakerAlgorithm roadSearchTrackCandidateMakerAlgorithm_;
  edm::ParameterSet conf_;
  edm::InputTag     cloudProducer_;

};

#endif
