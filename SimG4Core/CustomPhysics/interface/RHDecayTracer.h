#ifndef SimG4Core_CustomPhysics_RHDecayTracer_H
#define SimG4Core_CustomPhysics_RHDecayTracer_H

#include "SimG4Core/CustomPhysics/interface/RHadronPythiaDecayDataManager.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

class RHDecayTracer : public edm::stream::EDProducer<> {
public:
  RHDecayTracer(edm::ParameterSet const& p);
  ~RHDecayTracer() override = default;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  void addGenRhadronHistoryToCollection(edm::Event& iEvent,
                                        reco::GenParticleCollection& genParticles,
                                        const edm::RefProd<reco::GenParticleCollection>& refProd);
  void addSimRhadronHistoryToCollection(edm::Event& iEvent,
                                        reco::GenParticleCollection& genParticles,
                                        const edm::RefProd<reco::GenParticleCollection>& refProd);
  void addRhadronDecayToCollection(reco::GenParticleCollection& genParticles,
                                   const edm::RefProd<reco::GenParticleCollection>& refProd);
  void linkSimTrackHistory(reco::GenParticleCollection& genParticles,
                           const edm::RefProd<reco::GenParticleCollection>& refProd,
                           std::vector<size_t>& refIndexVector,
                           std::vector<int>& parentIdVector,
                           std::vector<unsigned int>& trackIdVector);
  size_t getDecayParentIndex(const RHadronPythiaDecayDataManager::TrackData& parentData,
                             std::vector<size_t>& refIndexVector,
                             std::vector<unsigned int>& trackIdVector);
  void addDecayDaughtersToCollection(const std::vector<RHadronPythiaDecayDataManager::TrackData>& daughtersData,
                                     reco::GenParticleCollection& genParticles);

  edm::EDGetTokenT<edm::SimTrackContainer> edmSimTrackContainerToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> edmSimVertexContainerToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken_;
  edm::EDPutTokenT<reco::GenParticleCollection> genParticleRHadronDecayToken_;

  std::vector<size_t> refIndexVector;
  std::vector<int> parentIdVector;
  std::vector<unsigned int> trackIdVector;
};

#endif