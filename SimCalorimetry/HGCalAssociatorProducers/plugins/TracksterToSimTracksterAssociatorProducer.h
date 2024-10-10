// Author: Felice Pantaleo, felice.pantaleo@cern.ch 06/2024

#ifndef TracksterToSimTracksterAssociatorProducer_h
#define TracksterToSimTracksterAssociatorProducer_h

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

class TracksterToSimTracksterAssociatorProducer : public edm::global::EDProducer<> {
public:
  explicit TracksterToSimTracksterAssociatorProducer(const edm::ParameterSet&);
  ~TracksterToSimTracksterAssociatorProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<std::vector<ticl::Trackster>> recoTracksterCollectionToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> simTracksterCollectionToken_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersCollectionToken_;
  edm::EDGetTokenT<
      ticl::AssociationMap<ticl::mapWithFraction, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>
      LayerClusterToTracksterMapToken_;
  edm::EDGetTokenT<
      ticl::AssociationMap<ticl::mapWithFraction, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>
      LayerClusterToSimTracksterMapToken_;
};

#endif
