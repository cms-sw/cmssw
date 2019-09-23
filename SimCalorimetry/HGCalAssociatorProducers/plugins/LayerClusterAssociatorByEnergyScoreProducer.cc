// Original author: Marco Rovere

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "LayerClusterAssociatorByEnergyScoreImpl.h"

class LayerClusterAssociatorByEnergyScoreProducer : public edm::global::EDProducer<> {
public:
  explicit LayerClusterAssociatorByEnergyScoreProducer(const edm::ParameterSet &);
  ~LayerClusterAssociatorByEnergyScoreProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  // ----------member data ---------------------------
  float qualitySimToReco_;
  float qualityRecoRoSim_;
};

LayerClusterAssociatorByEnergyScoreProducer::LayerClusterAssociatorByEnergyScoreProducer(const edm::ParameterSet &) {
  // Register the product
  produces<hgcal::LayerClusterToCaloParticleAssociator>();
}

LayerClusterAssociatorByEnergyScoreProducer::~LayerClusterAssociatorByEnergyScoreProducer() {}

void LayerClusterAssociatorByEnergyScoreProducer::fillDescriptions(edm::ConfigurationDescriptions &desc) {}

void LayerClusterAssociatorByEnergyScoreProducer::produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const {}

//define this as a plug-in
DEFINE_FWK_MODULE(LayerClusterAssociatorByEnergyScoreProducer);
