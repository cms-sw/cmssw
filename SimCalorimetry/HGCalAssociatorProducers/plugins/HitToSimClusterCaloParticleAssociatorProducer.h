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
#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/Common/interface/MultiSpan.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

class HitToSimClusterCaloParticleAssociatorProducer : public edm::global::EDProducer<> {
public:
  explicit HitToSimClusterCaloParticleAssociatorProducer(const edm::ParameterSet &);
  ~HitToSimClusterCaloParticleAssociatorProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  const edm::EDGetTokenT<std::vector<SimCluster>> simClusterToken_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;

  const edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMapToken_;
  edm::EDGetTokenT<edm::RefProdVector<HGCRecHitCollection>> hitsToken_;
};

#endif
