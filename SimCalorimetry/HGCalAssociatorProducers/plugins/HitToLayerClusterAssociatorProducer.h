#ifndef SimCalorimetry_HGCalAssociatorProducers_HitToLayerClusterAssociatorProducerT_h
#define SimCalorimetry_HGCalAssociatorProducers_HitToLayerClusterAssociatorProducerT_h
// Author: Felice Pantaleo, felice.pantaleo@cern.ch 06/2024

// system include files
#include <memory>
#include <string>
#include <unordered_map>

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "DataFormats/Common/interface/MultiSpan.h"

template <typename HIT>
class HitToLayerClusterAssociatorProducerT : public edm::global::EDProducer<> {
public:
  explicit HitToLayerClusterAssociatorProducerT(const edm::ParameterSet &);
  ~HitToLayerClusterAssociatorProducerT() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<std::vector<reco::CaloCluster>> LCCollectionToken_;
  edm::EDGetTokenT<std::unordered_map<DetId, unsigned int>> hitMapToken_;
  std::vector<edm::EDGetTokenT<std::vector<HIT>>> hitsTokens_;
};

template class HitToLayerClusterAssociatorProducerT<HGCRecHit>;
template class HitToLayerClusterAssociatorProducerT<reco::PFRecHit>;

using HitToLayerClusterAssociatorProducer = HitToLayerClusterAssociatorProducerT<HGCRecHit>;
DEFINE_FWK_MODULE(HitToLayerClusterAssociatorProducer);
using HitToBarrelLayerClusterAssociatorProducer = HitToLayerClusterAssociatorProducerT<reco::PFRecHit>;
DEFINE_FWK_MODULE(HitToBarrelLayerClusterAssociatorProducer);
#endif
