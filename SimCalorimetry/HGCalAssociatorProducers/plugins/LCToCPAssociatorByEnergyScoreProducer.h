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

template <typename HIT>
class LCToCPAssociatorByEnergyScoreProducer : public edm::global::EDProducer<> {
public:
  explicit LCToCPAssociatorByEnergyScoreProducer(const edm::ParameterSet &);
  ~LCToCPAssociatorByEnergyScoreProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMap_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometry_;
  const bool hardScatterOnly_;
  std::shared_ptr<hgcal::RecHitTools> rhtools_;
  std::vector<edm::InputTag> hits_label_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcal_hits_token_;
  std::vector<edm::EDGetTokenT<std::vector<HIT>>> hits_token_;
};

template class LCToCPAssociatorByEnergyScoreProducer<HGCRecHit>;
template class LCToCPAssociatorByEnergyScoreProducer<reco::PFRecHit>;

using HGCalLCToCPAssociatorByEnergyScoreProducer = LCToCPAssociatorByEnergyScoreProducer<HGCRecHit>;
DEFINE_FWK_MODULE(HGCalLCToCPAssociatorByEnergyScoreProducer);
using BarrelLCToCPAssociatorByEnergyScoreProducer = LCToCPAssociatorByEnergyScoreProducer<reco::PFRecHit>;
DEFINE_FWK_MODULE(BarrelLCToCPAssociatorByEnergyScoreProducer);

#endif
