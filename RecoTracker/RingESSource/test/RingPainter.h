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
// $Revision: 1.2 $
//

#ifndef RingPainter_h
#define RingPainter_h

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/RingESSource/test/RingPainterAlgorithm.h"

namespace cms
{
  class RingPainter : public edm::EDAnalyzer
  {
  public:

    explicit RingPainter(const edm::ParameterSet& conf);

    virtual ~RingPainter();

    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    RingPainterAlgorithm ringPainterAlgorithm_;
    edm::ParameterSet    conf_;
    std::string ringLabel_;

  };
}


#endif
