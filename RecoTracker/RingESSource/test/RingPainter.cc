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
// $Date: 2006/06/20 09:09:19 $
// $Revision: 1.1 $
//

#include <memory>
#include <string>
#include <iostream>

#include "RecoTracker/RingESSource/test/RingPainter.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/RingRecord/interface/Rings.h"

namespace cms
{

  RingPainter::RingPainter(edm::ParameterSet const& conf) : 
    ringPainterAlgorithm_(conf) ,
    conf_(conf)
  {
  }

  // Virtual destructor needed.
  RingPainter::~RingPainter() { }  

  // Functions that gets called by framework every event
  void RingPainter::analyze(const edm::Event& e, const edm::EventSetup& es)
  {

    edm::ESHandle<Rings> rings;
    es.get<TrackerDigiGeometryRecord>().get(rings);

    ringPainterAlgorithm_.run(rings.product());

  }

}
