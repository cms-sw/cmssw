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
// $Date: 2006/06/20 09:09:19 $
// $Revision: 1.1 $
//

#ifndef RoadPainter_h
#define RoadPainter_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/RoadMapESSource/test/RoadPainterAlgorithm.h"

namespace cms
{
  class RoadPainter : public edm::EDAnalyzer
  {
  public:

    explicit RoadPainter(const edm::ParameterSet& conf);

    virtual ~RoadPainter();

    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    RoadPainterAlgorithm roadPainterAlgorithm_;
    edm::ParameterSet    conf_;

  };
}


#endif
