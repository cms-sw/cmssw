// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackClustersV
// Class:       SiPixelPhase1TrackClustersV
//

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/SiPixelPhase1TrackClustersV/interface/SiPixelPhase1TrackClustersV.h"

SiPixelPhase1TrackClustersV::SiPixelPhase1TrackClustersV(const edm::ParameterSet &iConfig)
    : SiPixelPhase1Base(iConfig) {
  clustersToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(iConfig.getParameter<edm::InputTag>("clusters"));
}

void SiPixelPhase1TrackClustersV::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get clusters
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> clusterColl;
  iEvent.getByToken(clustersToken_, clusterColl);

  edmNew::DetSetVector<SiPixelCluster>::const_iterator it;
  for (it = clusterColl->begin(); it != clusterColl->end(); ++it) {
    auto id = DetId(it->detId());

    for (auto subit = it->begin(); subit != it->end(); ++subit) {
      SiPixelCluster const &cluster = *subit;

      histo[CHARGE].fill(double(cluster.charge()), id, &iEvent);
      histo[SIZE_X].fill(double(cluster.sizeX()), id, &iEvent);
      histo[SIZE_Y].fill(double(cluster.sizeY()), id, &iEvent);
    }
  }
}

DEFINE_FWK_MODULE(SiPixelPhase1TrackClustersV);
