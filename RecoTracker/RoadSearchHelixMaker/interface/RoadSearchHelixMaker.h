#ifndef RoadSearchHelixMaker_h
#define RoadSearchHelixMaker_h

//
// Package:         RecoTracker/HelixMaker
// Class:           RoadSearchHelixMaker
// 
// Description:     Calls RoadSeachHelixMakerAlgorithm
//                  to find RoadSearchClouds.
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Feb 19 22:00:00 UTC 2006
//
// $Author: stevew $
// $Date: 2006/03/22 22:47:37 $
// $Revision: 1.2 $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/RoadSearchHelixMaker/interface/RoadSearchHelixMakerAlgorithm.h"

namespace cms
{
  class RoadSearchHelixMaker : public edm::EDProducer
  {
  public:

    explicit RoadSearchHelixMaker(const edm::ParameterSet& conf);

    virtual ~RoadSearchHelixMaker();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    RoadSearchHelixMakerAlgorithm roadSearchHelixMakerAlgorithm_;
    edm::ParameterSet conf_;

  };
}


#endif
