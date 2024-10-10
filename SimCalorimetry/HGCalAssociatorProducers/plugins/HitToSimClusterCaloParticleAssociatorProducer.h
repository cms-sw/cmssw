#ifndef SimCalorimetry_HGCalAssociatorProducers_HitToSimClusterCaloParticleAssociatorProducer_h
#define SimCalorimetry_HGCalAssociatorProducers_HitToSimClusterCaloParticleAssociatorProducer_h

// Author: Felice Pantaleo, felice.pantaleo@cern.ch 06/2024

// user include files
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
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

class HitToSimClusterCaloParticleAssociatorProducer : public edm::global::EDProducer<> {
public:
  explicit HitToSimClusterCaloParticleAssociatorProducer(const edm::ParameterSet &);
  ~HitToSimClusterCaloParticleAssociatorProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<std::vector<SimCluster>> simClusterToken_;
  edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;

  edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMapToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hitsTokens_;
};

#endif
