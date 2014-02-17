//
// Package:         RecoTracker/RingESSource/test
// Class:           RingPainter
// 
// Description:     calls RingPainterAlgorithm to
//                  paint rings
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Dec  7 08:52:54 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/03/01 07:40:16 $
// $Revision: 1.3 $
//

#include <memory>
#include <string>
#include <iostream>

#include "RecoTracker/RingESSource/test/RingPainter.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/RingRecord/interface/RingRecord.h"
#include "RecoTracker/RingRecord/interface/Rings.h"

namespace cms
{

  RingPainter::RingPainter(edm::ParameterSet const& conf) : 
    ringPainterAlgorithm_(conf) ,
    conf_(conf)
  {

    ringLabel_ = conf.getUntrackedParameter<std::string>("RingLabel");

  }

  // Virtual destructor needed.
  RingPainter::~RingPainter() { }  

  // Functions that gets called by framework every event
  void RingPainter::analyze(const edm::Event& e, const edm::EventSetup& es)
  {

    edm::ESHandle<Rings> rings;
    es.get<RingRecord>().get(ringLabel_,rings);

    ringPainterAlgorithm_.run(rings.product());

  }

}
