#ifndef SimCalorimetry_HGCalAssociatorProducers_HitToLayerClusterAssociatorProducer_h
#define SimCalorimetry_HGCalAssociatorProducers_HitToLayerClusterAssociatorProducer_h
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
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"

class HitToLayerClusterAssociatorProducer : public edm::global::EDProducer<> {
public:
  explicit HitToLayerClusterAssociatorProducer(const edm::ParameterSet &);
  ~HitToLayerClusterAssociatorProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<std::vector<reco::CaloCluster>> LCCollectionToken_;
  edm::EDGetTokenT<std::unordered_map<DetId, unsigned int>> hitMapToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hitsTokens_;
};

#endif
