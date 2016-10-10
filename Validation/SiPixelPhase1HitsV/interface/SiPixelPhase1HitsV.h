#ifndef SiPixelPhase1HitsV_h 
#define SiPixelPhase1HitsV_h 
// -*- C++ -*-
// 
// Package:     SiPixelPhase1HitsV
// Class  :     SiPixelPhase1HitsV
//

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

class SiPixelPhase1HitsV : public SiPixelPhase1Base {
  enum {
    ELOSS,
    ENTRY_EXIT_X,
    ENTRY_EXIT_Y,
    ENTRY_EXIT_Z,
    LOCAL_X,
    LOCAL_Y,
  };

  public:
  explicit SiPixelPhase1HitsV(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&);

  private:
  edm::EDGetTokenT<edm::PSimHitContainer> pixelBarrelLowToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> pixelBarrelHighToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> pixelForwardLowToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> pixelForwardHighToken_;
};

#endif
