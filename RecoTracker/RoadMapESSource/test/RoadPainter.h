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

#ifndef RoadPainter_h
#define RoadPainter_h

#include <string>

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
    std::string          roadLabel_;
    std::string          ringLabel_;

  };
}


#endif
