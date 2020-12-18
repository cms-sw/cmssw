#ifndef SiPixelPhase1TrackClustersV_h
#define SiPixelPhase1TrackClustersV_h
// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackClustersV
// Class  :     SiPixelPhase1TrackClustersV
//

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

class SiPixelPhase1TrackClustersV : public SiPixelPhase1Base {
  enum {
    CHARGE,
    SIZE_X,
    SIZE_Y,
  };

public:
  explicit SiPixelPhase1TrackClustersV(const edm::ParameterSet &conf);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> clustersToken_;
};

#endif
