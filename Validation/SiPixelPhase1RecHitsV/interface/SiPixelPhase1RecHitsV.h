#ifndef SiPixelPhase1RecHitsV_h 
#define SiPixelPhase1RecHitsV_h 
// -*- C++ -*-
// 
// Package:     SiPixelPhase1RecHitsV
// Class  :     SiPixelPhase1RecHitsV
//

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

#include "Validation/SiPixelPhase1CommonV/interface/SiPixelPhase1BaseV.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

class SiPixelPhase1RecHitsV : public SiPixelPhase1BaseV {
  enum {
    IN_TIME_BUNCH,
    OUT_TIME_BUNCH,
    NSIMHITS,
    RECHIT_X,
    RECHIT_Y,
    RES_X,
    RES_Y,
    ERROR_X,
    ERROR_Y,
    PULL_X,
    PULL_Y,
  };

  public:
  explicit SiPixelPhase1RecHitsV(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&);

  private:
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  edm::EDGetTokenT<SiPixelRecHitCollection> srcToken_;
};

#endif
