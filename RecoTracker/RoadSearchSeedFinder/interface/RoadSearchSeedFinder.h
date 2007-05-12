#ifndef RoadSearchSeedFinder_h
#define RoadSearchSeedFinder_h

//
// Package:         RecoTracker/RoadSearchSeedFinder
// Class:           RoadSearchSeedFinder
// 
// Description:     Calls RoadSeachSeedFinderAlgorithm
//                  to find TrackingSeeds.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/03/28 22:54:06 $
// $Revision: 1.3 $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "RecoTracker/RoadSearchSeedFinder/interface/RoadSearchSeedFinderAlgorithm.h"

class RoadSearchSeedFinder : public edm::EDProducer
{
 public:

  explicit RoadSearchSeedFinder(const edm::ParameterSet& conf);

  virtual ~RoadSearchSeedFinder();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:
  RoadSearchSeedFinderAlgorithm roadSearchSeedFinderAlgorithm_;
  edm::ParameterSet conf_;

};

#endif
