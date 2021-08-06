#ifndef SimG4CMSForwardSimG4FluxProducer_h
#define SimG4CMSForwardSimG4FluxProducer_h

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"

// to retreive hits
#include "SimDataFormats/CaloTest/interface/ParticleFlux.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Step.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PhysicalVolumeStore.hh"

#include <vector>
#include <string>
#include <map>

class SimG4FluxProducer : public SimProducer,
                          public Observer<const BeginOfRun *>,
                          public Observer<const BeginOfEvent *>,
                          public Observer<const G4Step *> {
public:
  SimG4FluxProducer(const edm::ParameterSet &p);
  SimG4FluxProducer(const SimG4FluxProducer &) = delete;  // stop default
  const SimG4FluxProducer &operator=(const SimG4FluxProducer &) = delete;
  ~SimG4FluxProducer() override;

  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  // observer classes
  void update(const BeginOfRun *run) override;
  void update(const BeginOfEvent *evt) override;
  void update(const G4Step *step) override;

  void endOfEvent(ParticleFlux &pflx, unsigned int k);
  G4VPhysicalVolume *getTopPV();
  std::map<G4LogicalVolume *, std::pair<unsigned int, std::string>>::iterator findLV(G4LogicalVolume *plv);

private:
  std::vector<std::string> LVNames_;
  std::vector<int> LVTypes_;
  G4VPhysicalVolume *topPV_;
  std::map<G4LogicalVolume *, std::pair<unsigned int, std::string>> mapLV_;

  // some private members for ananlysis
  unsigned int count_;
  bool init_;
  std::map<std::pair<G4LogicalVolume *, unsigned int>, ParticleFlux> store_;
};

#endif
