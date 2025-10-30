#ifndef SimCalorimetry_HGCalAssociatorProducers_LCToSCAssociatorByEnergyScoreProducerT_H
#define SimCalorimetry_HGCalAssociatorProducers_LCToSCAssociatorByEnergyScoreProducerT_H

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

#include "DataFormats/Common/interface/MultiCollection.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

template <typename HIT, typename CLUSTER>
class LCToSCAssociatorByEnergyScoreProducerT : public edm::global::EDProducer<> {
public:
  using multiCollectionT = edm::MultiCollection<std::vector<HIT>>;

  explicit LCToSCAssociatorByEnergyScoreProducerT(const edm::ParameterSet &);
  ~LCToSCAssociatorByEnergyScoreProducerT() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMap_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometry_;
  const bool hardScatterOnly_;
  std::shared_ptr<hgcal::RecHitTools> rhtools_;
  edm::EDGetTokenT<multiCollectionT> hits_token_;
};

template class LCToSCAssociatorByEnergyScoreProducerT<HGCRecHit, reco::CaloClusterCollection>;
template class LCToSCAssociatorByEnergyScoreProducerT<reco::PFRecHit, reco::CaloClusterCollection>;
template class LCToSCAssociatorByEnergyScoreProducerT<HGCRecHit, reco::PFClusterCollection>;
template class LCToSCAssociatorByEnergyScoreProducerT<reco::PFRecHit, reco::PFClusterCollection>;

using HGCalLCToSCAssociatorByEnergyScoreProducer =
    LCToSCAssociatorByEnergyScoreProducerT<HGCRecHit, reco::CaloClusterCollection>;
DEFINE_FWK_MODULE(HGCalLCToSCAssociatorByEnergyScoreProducer);
using BarrelLCToSCAssociatorByEnergyScoreProducer =
    LCToSCAssociatorByEnergyScoreProducerT<reco::PFRecHit, reco::CaloClusterCollection>;
DEFINE_FWK_MODULE(BarrelLCToSCAssociatorByEnergyScoreProducer);
using HGCalPCToSCAssociatorByEnergyScoreProducer =
    LCToSCAssociatorByEnergyScoreProducerT<HGCRecHit, reco::PFClusterCollection>;
DEFINE_FWK_MODULE(HGCalPCToSCAssociatorByEnergyScoreProducer);
using BarrelPCToSCAssociatorByEnergyScoreProducer =
    LCToSCAssociatorByEnergyScoreProducerT<reco::PFRecHit, reco::PFClusterCollection>;
DEFINE_FWK_MODULE(BarrelPCToSCAssociatorByEnergyScoreProducer);

#endif
