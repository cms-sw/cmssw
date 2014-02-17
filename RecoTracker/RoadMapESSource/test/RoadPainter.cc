//
// Package:         RecoTracker/RingESSource/test
// Class:           RoadPainter
// 
// Description:     calls RoadPainterAlgorithm to
//                  paint rings
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Dec  7 08:52:54 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/03/01 07:46:30 $
// $Revision: 1.2 $
//

#include <memory>
#include <string>
#include <iostream>

#include "RecoTracker/RoadMapESSource/test/RoadPainter.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/RingRecord/interface/Rings.h"
#include "RecoTracker/RingRecord/interface/RingRecord.h"
#include "RecoTracker/RoadMapRecord/interface/Roads.h"
#include "RecoTracker/RoadMapRecord/interface/RoadMapRecord.h"

namespace cms
{

  RoadPainter::RoadPainter(edm::ParameterSet const& conf) : 
    roadPainterAlgorithm_(conf) ,
    conf_(conf)
  {

    ringLabel_ = conf.getUntrackedParameter<std::string>("RingLabel");
    roadLabel_ = conf.getUntrackedParameter<std::string>("RoadLabel");

  }

  // Virtual destructor needed.
  RoadPainter::~RoadPainter() { }  

  // Functions that gets called by framework every event
  void RoadPainter::analyze(const edm::Event& e, const edm::EventSetup& es)
  {

    edm::ESHandle<Rings> rings;
    es.get<RingRecord>().get(ringLabel_,rings);

    edm::ESHandle<Roads> roads;
    es.get<RoadMapRecord>().get(roadLabel_,roads);

    roadPainterAlgorithm_.run(rings.product(),roads.product());

  }

}
