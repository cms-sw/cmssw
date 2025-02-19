#ifndef TrackListMerger_h
#define TrackListMerger_h

//
// Package:         RecoTracker/TrackListMerger
// Class:           RoadSearchHelixMaker
// 
// Description:     Hit Dumper
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: wmtan $
// $Date: 2011/05/20 17:17:32 $
// $Revision: 1.3 $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace cms
{
  class TrackListMerger : public edm::EDProducer
  {
  public:

    explicit TrackListMerger(const edm::ParameterSet& conf);

    virtual ~TrackListMerger();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ParameterSet conf_;

  };
}


#endif
