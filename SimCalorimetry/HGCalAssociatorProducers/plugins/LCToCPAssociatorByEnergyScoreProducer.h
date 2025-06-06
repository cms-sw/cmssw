#ifndef SimCalorimetry_HGCalAssociatorProducers_LCToCPAssociatorByEnergyScoreProducer_H
#define SimCalorimetry_HGCalAssociatorProducers_LCToCPAssociatorByEnergyScoreProducer_H

// Original author: Marco Rovere

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "LCToCPAssociatorByEnergyScoreImpl.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

template <typename HIT, typename CLUSTER>
class LCToCPAssociatorByEnergyScoreProducerT : public edm::global::EDProducer<> {
public:
  using multiCollectionT = std::vector<edm::RefProd<std::vector<HIT>>>;

  explicit LCToCPAssociatorByEnergyScoreProducerT(const edm::ParameterSet &);
  ~LCToCPAssociatorByEnergyScoreProducerT() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMap_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometry_;
  const bool hardScatterOnly_;
  std::shared_ptr<hgcal::RecHitTools> rhtools_;
  edm::EDGetTokenT<multiCollectionT> hits_token_;
};

template class LCToCPAssociatorByEnergyScoreProducerT<HGCRecHit, reco::CaloClusterCollection>;
template class LCToCPAssociatorByEnergyScoreProducerT<reco::PFRecHit, reco::CaloClusterCollection>;
template class LCToCPAssociatorByEnergyScoreProducerT<HGCRecHit, reco::PFClusterCollection>;
template class LCToCPAssociatorByEnergyScoreProducerT<reco::PFRecHit, reco::PFClusterCollection>;

using HGCalLCToCPAssociatorByEnergyScoreProducer =
    LCToCPAssociatorByEnergyScoreProducerT<HGCRecHit, reco::CaloClusterCollection>;
DEFINE_FWK_MODULE(HGCalLCToCPAssociatorByEnergyScoreProducer);
using BarrelLCToCPAssociatorByEnergyScoreProducer =
    LCToCPAssociatorByEnergyScoreProducerT<reco::PFRecHit, reco::CaloClusterCollection>;
DEFINE_FWK_MODULE(BarrelLCToCPAssociatorByEnergyScoreProducer);
using HGCalPCToCPAssociatorByEnergyScoreProducer =
    LCToCPAssociatorByEnergyScoreProducerT<HGCRecHit, reco::PFClusterCollection>;
DEFINE_FWK_MODULE(HGCalPCToCPAssociatorByEnergyScoreProducer);
using BarrelPCToCPAssociatorByEnergyScoreProducer =
    LCToCPAssociatorByEnergyScoreProducerT<reco::PFRecHit, reco::PFClusterCollection>;
DEFINE_FWK_MODULE(BarrelPCToCPAssociatorByEnergyScoreProducer);

#endif
