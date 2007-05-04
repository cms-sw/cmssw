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
// $Date: 2007/03/01 08:18:19 $
// $Revision: 1.2 $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "RecoTracker/RoadSearchTrackCandidateMaker/interface/RoadSearchTrackCandidateMakerAlgorithm.h"

namespace cms
{
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
}

#endif
