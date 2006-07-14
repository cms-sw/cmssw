#ifndef RoadSearchTrackListCleaner_h
#define RoadSearchTrackListCleaner_h

//
// Package:         RecoTracker/RoadSearchTrackListCleaner
// Class:           RoadSearchHelixMaker
// 
// Description:     Hit Dumper
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: burkett $
// $Date: 2006/03/15 16:07:16 $
// $Revision: 1.2 $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace cms
{
  class RoadSearchTrackListCleaner : public edm::EDProducer
  {
  public:

    explicit RoadSearchTrackListCleaner(const edm::ParameterSet& conf);

    virtual ~RoadSearchTrackListCleaner();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ParameterSet conf_;

  };
}


#endif
