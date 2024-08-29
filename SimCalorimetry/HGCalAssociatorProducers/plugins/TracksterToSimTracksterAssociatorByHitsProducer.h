// Author: Felice Pantaleo, felice.pantaleo@cern.ch 06/2024

#ifndef TracksterToSimTracksterAssociatorByHitsProducer_h
#define TracksterToSimTracksterAssociatorByHitsProducer_h

#include <vector>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

class TracksterToSimTracksterAssociatorByHitsProducer : public edm::global::EDProducer<> {
public:
  explicit TracksterToSimTracksterAssociatorByHitsProducer(const edm::ParameterSet&);
  ~TracksterToSimTracksterAssociatorByHitsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<std::vector<ticl::Trackster>> recoTracksterCollectionToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> simTracksterCollectionToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> simTracksterFromCPCollectionToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToTracksterMapToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToSimTracksterMapToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToSimTracksterFromCPMapToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToSimClusterMapToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> hitToCaloParticleMapToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> tracksterToHitMapToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> simTracksterToHitMapToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithFraction>> simTracksterFromCPToHitMapToken_;
  edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hitsTokens_;
};

#endif
