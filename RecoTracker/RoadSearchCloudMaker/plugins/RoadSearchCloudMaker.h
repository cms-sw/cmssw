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
// $Author: wmtan $
// $Date: 2011/05/20 17:17:32 $
// $Revision: 1.8 $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RoadSearchCloudMakerAlgorithm.h"

class RoadSearchCloudMaker : public edm::EDProducer
{
public:

  explicit RoadSearchCloudMaker(const edm::ParameterSet& conf);

  virtual ~RoadSearchCloudMaker();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:
  RoadSearchCloudMakerAlgorithm roadSearchCloudMakerAlgorithm_;
  edm::ParameterSet conf_;
  edm::InputTag     seedProducer_;
  edm::InputTag     matchedStripRecHitsInputTag_;
  edm::InputTag     rphiStripRecHitsInputTag_;
  edm::InputTag     stereoStripRecHitsInputTag_;
  edm::InputTag     pixelRecHitsInputTag_;

};

#endif
