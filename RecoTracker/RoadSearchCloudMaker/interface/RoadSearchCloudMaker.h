#ifndef RoadSearchCloudMaker_h
#define RoadSearchCloudMaker_h

//
// Package:         RecoTracker/RoadSearchCloudMaker
// Class:           RoadSearchCloudMaker
// 
// Description:     Calls RoadSeachCloudMakerAlgorithm
//                  to find RoadSearchClouds.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/01/14 22:00:00 $
// $Revision: 1.1 $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/EDProduct/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/RoadSearchCloudMaker/interface/RoadSearchCloudMakerAlgorithm.h"

namespace cms
{
  class RoadSearchCloudMaker : public edm::EDProducer
  {
  public:

    explicit RoadSearchCloudMaker(const edm::ParameterSet& conf);

    virtual ~RoadSearchCloudMaker();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    RoadSearchCloudMakerAlgorithm roadSearchCloudMakerAlgorithm_;
    edm::ParameterSet conf_;

  };
}


#endif
