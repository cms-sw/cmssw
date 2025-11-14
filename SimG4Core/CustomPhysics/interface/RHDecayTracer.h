#ifndef SimG4Core_CustomPhysics_RHDecayTracer_H
#define SimG4Core_CustomPhysics_RHDecayTracer_H

#include "SimG4Core/CustomPhysics/interface/RHadronPythiaDecayDataManager.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

class SimTrack;

namespace HepMC {
  class GenVertex;
}

namespace edm {
  class HepMCProduct;
}

class RHDecayTracer : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  RHDecayTracer(edm::ParameterSet const& p);
  ~RHDecayTracer() override = default;
  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  const SimTrack* findSimTrack(int trackID, const edm::SimTrackContainer& simTracks);
  void addSecondariesToGenVertex(std::map<int, std::vector<RHadronPythiaDecayDataManager::TrackData>> decayDaughters, const int decayID, HepMC::GenVertex* decayVertex);

  edm::EDGetTokenT<edm::HepMCProduct> genToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken_;
  edm::Handle<edm::HepMCProduct> genHandle_;
  edm::Handle<edm::SimTrackContainer> simTrackHandle_;
};

#endif