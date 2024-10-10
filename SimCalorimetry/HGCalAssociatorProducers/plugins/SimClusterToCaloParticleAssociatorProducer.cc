// Author: Felice Pantaleo, felice.pantaleo@cern.ch 07/2024

#include "SimClusterToCaloParticleAssociatorProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

SimClusterToCaloParticleAssociatorProducer::SimClusterToCaloParticleAssociatorProducer(const edm::ParameterSet &pset)
    : simClusterToken_(consumes<std::vector<SimCluster>>(pset.getParameter<edm::InputTag>("simClusters"))),
      caloParticleToken_(consumes<std::vector<CaloParticle>>(pset.getParameter<edm::InputTag>("caloParticles"))) {
  produces<ticl::AssociationMap<ticl::oneToOneMapWithFraction, std::vector<SimCluster>, std::vector<CaloParticle>>>(
      "simClusterToCaloParticleMap");
}

SimClusterToCaloParticleAssociatorProducer::~SimClusterToCaloParticleAssociatorProducer() {}

void SimClusterToCaloParticleAssociatorProducer::produce(edm::StreamID,
                                                         edm::Event &iEvent,
                                                         const edm::EventSetup &iSetup) const {
  using namespace edm;

  Handle<std::vector<CaloParticle>> caloParticlesHandle;
  iEvent.getByToken(caloParticleToken_, caloParticlesHandle);
  const auto &caloParticles = *caloParticlesHandle;

  Handle<std::vector<SimCluster>> simClustersHandle;
  iEvent.getByToken(simClusterToken_, simClustersHandle);

  // Create association map
  auto simClusterToCaloParticleMap = std::make_unique<
      ticl::AssociationMap<ticl::oneToOneMapWithFraction, std::vector<SimCluster>, std::vector<CaloParticle>>>(
      simClustersHandle, caloParticlesHandle, iEvent);

  // Loop over caloParticles
  for (unsigned int cpId = 0; cpId < caloParticles.size(); ++cpId) {
    const auto &caloParticle = caloParticles[cpId];
    // Loop over simClusters in caloParticle
    for (const auto &simClusterRef : caloParticle.simClusters()) {
      const auto &simCluster = *simClusterRef;
      unsigned int scId = simClusterRef.key();
      // Calculate the fraction of the simCluster energy to the caloParticle energy
      float fraction = simCluster.energy() / caloParticle.energy();
      // Insert association
      simClusterToCaloParticleMap->insert(scId, cpId, fraction);
    }
  }
  iEvent.put(std::move(simClusterToCaloParticleMap), "simClusterToCaloParticleMap");
}

void SimClusterToCaloParticleAssociatorProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("simClusters", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  descriptions.add("SimClusterToCaloParticleAssociatorProducer", desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(SimClusterToCaloParticleAssociatorProducer);
