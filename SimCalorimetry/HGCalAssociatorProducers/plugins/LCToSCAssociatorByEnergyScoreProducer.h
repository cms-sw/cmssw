#ifndef SimCalorimetry_HGCalAssociatorProducers_LCToSCAssociatorByEnergyScoreProducer_H
#define SimCalorimetry_HGCalAssociatorProducers_LCToSCAssociatorByEnergyScoreProducer_H

// Original author: Leonardo Cristella

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/defaultModuleLabel.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"
#include "LCToSCAssociatorByEnergyScoreImpl.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

template <typename HIT>
class LCToSCAssociatorByEnergyScoreProducer : public edm::global::EDProducer<> {
public:
  explicit LCToSCAssociatorByEnergyScoreProducer(const edm::ParameterSet &);
  ~LCToSCAssociatorByEnergyScoreProducer() override;

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

template class LCToSCAssociatorByEnergyScoreProducer<HGCRecHit>;
template class LCToSCAssociatorByEnergyScoreProducer<reco::PFRecHit>;

using HGCalLCToSCAssociatorByEnergyScoreProducer = LCToSCAssociatorByEnergyScoreProducer<HGCRecHit>;
DEFINE_FWK_MODULE(HGCalLCToSCAssociatorByEnergyScoreProducer);
using BarrelLCToSCAssociatorByEnergyScoreProducer = LCToSCAssociatorByEnergyScoreProducer<reco::PFRecHit>;
DEFINE_FWK_MODULE(BarrelLCToSCAssociatorByEnergyScoreProducer);

#endif
