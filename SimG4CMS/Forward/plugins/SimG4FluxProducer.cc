#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CaloTest/interface/ParticleFlux.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"

#include "G4Step.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4Track.hh"
#include "G4TouchableHistory.hh"
#include "G4TransportationManager.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//#define EDM_ML_DEBUG

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

  std::vector<std::string> LVNames_;
  std::vector<int> LVTypes_;
  G4VPhysicalVolume *topPV_;
  std::map<G4LogicalVolume *, std::pair<unsigned int, std::string>> mapLV_;

  // some private members for ananlysis
  unsigned int count_;
  bool init_;
  std::map<std::pair<G4LogicalVolume *, unsigned int>, ParticleFlux> store_;
};

SimG4FluxProducer::SimG4FluxProducer(const edm::ParameterSet &p) : count_(0), init_(false) {
  edm::ParameterSet m_FP = p.getParameter<edm::ParameterSet>("SimG4FluxProducer");
  LVNames_ = m_FP.getUntrackedParameter<std::vector<std::string>>("LVNames");
  LVTypes_ = m_FP.getUntrackedParameter<std::vector<int>>("LVTypes");

  for (unsigned int k = 0; k < LVNames_.size(); ++k) {
    produces<ParticleFlux>(Form("%sParticleFlux", LVNames_[k].c_str()));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("SimG4FluxProducer")
        << "SimG4FluxProducer::Collection name[" << k << "] ParticleFlux" << LVNames_[k] << " and type " << LVTypes_[k];
#endif
  }
}

SimG4FluxProducer::~SimG4FluxProducer() {}

void SimG4FluxProducer::produce(edm::Event &e, const edm::EventSetup &) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("SimG4FluxProducer") << "SimG4FluxProducer: enters produce with " << LVNames_.size() << " LV's";
#endif
  for (unsigned int k = 0; k < LVNames_.size(); ++k) {
    std::unique_ptr<ParticleFlux> pflux(new ParticleFlux);
    endOfEvent(*pflux, k);
    std::string name = LVNames_[k] + "ParticleFlux";
    e.put(std::move(pflux), name);
  }
}

void SimG4FluxProducer::update(const BeginOfRun *run) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("SimG4FluxProducer") << "SimG4FluxProducer: enters BeginOfRun";
#endif
  topPV_ = getTopPV();
  if (topPV_ == nullptr) {
    edm::LogWarning("SimG4FluxProducer") << "Cannot find top level volume\n";
  } else {
    init_ = true;
    const G4LogicalVolumeStore *lvs = G4LogicalVolumeStore::GetInstance();
    for (auto lvcite : *lvs) {
      findLV(lvcite);
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("SimG4FluxProducer") << "SimG4FluxProducer::Finds " << mapLV_.size() << " logical volumes";
    unsigned int k(0);
    for (const auto &lvs : mapLV_) {
      edm::LogVerbatim("SimG4FluxProducer")
          << "Entry[" << k << "] " << lvs.first << ": (" << (lvs.second).first << ", " << (lvs.second).second << ")";
      ++k;
    }
#endif
  }
}

void SimG4FluxProducer::update(const BeginOfEvent *evt) {
  int iev = (*evt)()->GetEventID();
  edm::LogVerbatim("SimG4FluxProducer") << "SimG4FluxProducer: =====> Begin event = " << iev;
  ++count_;
  store_.clear();
}

void SimG4FluxProducer::update(const G4Step *aStep) {
  if (aStep != nullptr) {
    G4TouchableHistory *touchable = (G4TouchableHistory *)aStep->GetPreStepPoint()->GetTouchable();
    G4LogicalVolume *plv = (G4LogicalVolume *)touchable->GetVolume()->GetLogicalVolume();
    auto it = (init_) ? mapLV_.find(plv) : findLV(plv);
    //  edm::LogVerbatim("SimG4FluxProducer") << plv->GetName() << " Flag " << (it != mapLV_.end()) << " step " << aStep->IsFirstStepInVolume() << ":"  << aStep->IsLastStepInVolume();
    if (it != mapLV_.end()) {
      int type = LVTypes_[(it->second).first];
      if ((type == 0 && aStep->IsFirstStepInVolume()) || (type == 1 && aStep->IsLastStepInVolume())) {
        unsigned int copy = (unsigned int)(touchable->GetReplicaNumber(0));
        std::pair<G4LogicalVolume *, unsigned int> key(plv, copy);
        auto itr = store_.find(key);
        if (itr == store_.end()) {
          store_[key] = ParticleFlux((it->second).second, copy);
          itr = store_.find(key);
        }
        G4Track *track = aStep->GetTrack();
        int pdgid = track->GetDefinition()->GetPDGEncoding();
        int vxtyp = (track->GetCreatorProcess() == nullptr) ? 0 : 1;
        double time = (aStep->GetPostStepPoint()->GetGlobalTime());
        const double mmTocm(0.1), MeVToGeV(0.001);
        ParticleFlux::flux flx(pdgid, vxtyp, time);
        flx.vertex = math::GlobalPoint(mmTocm * track->GetVertexPosition().x(),
                                       mmTocm * track->GetVertexPosition().y(),
                                       mmTocm * track->GetVertexPosition().z());
        flx.hitPoint = math::GlobalPoint(
            mmTocm * track->GetPosition().x(), mmTocm * track->GetPosition().y(), mmTocm * track->GetPosition().z());
        flx.momentum = math::GlobalVector(MeVToGeV * track->GetMomentum().x(),
                                          MeVToGeV * track->GetMomentum().y(),
                                          MeVToGeV * track->GetMomentum().z());
        (itr->second).addFlux(flx);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("SimG4FluxProducer")
            << "SimG4FluxProducer: Element " << (it->second).first << ":" << (it->second).second << ":" << copy
            << " ID " << pdgid << " VxType " << vxtyp << " TOF " << time << " Hit Point " << flx.hitPoint << " p "
            << flx.momentum << " Vertex " << flx.vertex;
#endif
      }  //if(Step ok)
    }    //if( it != map.end() )
  }      //if (aStep != NULL)
}

void SimG4FluxProducer::endOfEvent(ParticleFlux &flux, unsigned int k) {
  bool done(false);
  for (const auto &element : store_) {
    G4LogicalVolume *lv = (element.first).first;
    auto it = mapLV_.find(lv);
    if (it != mapLV_.end()) {
      if ((it->second).first == k) {
        flux = element.second;
        done = true;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("SimG4FluxProducer") << "SimG4FluxProducer[" << k << "] Flux " << flux.getName() << ":"
                                              << flux.getId() << " with " << flux.getComponents() << " elements";
        std::vector<ParticleFlux::flux> fluxes = flux.getFlux();
        unsigned int k(0);
        for (auto element : fluxes) {
          edm::LogVerbatim("SimG4FluxProducer")
              << "Flux[" << k << "] PDGId " << element.pdgId << " VT " << element.vxType << " ToF " << element.tof
              << " Vertex " << element.vertex << " Hit " << element.hitPoint << " p " << element.momentum;
          ++k;
        }
#endif
      }
    }
    if (done)
      break;
  }
}

G4VPhysicalVolume *SimG4FluxProducer::getTopPV() {
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}

std::map<G4LogicalVolume *, std::pair<unsigned int, std::string>>::iterator SimG4FluxProducer::findLV(
    G4LogicalVolume *plv) {
  auto itr = mapLV_.find(plv);
  if (itr == mapLV_.end()) {
    std::string name = plv->GetName();
    for (unsigned int k = 0; k < LVNames_.size(); ++k) {
      if (name.find(LVNames_[k]) != std::string::npos) {
        mapLV_[plv] = std::pair<unsigned int, std::string>(k, name);
        itr = mapLV_.find(plv);
        break;
      }
    }
  }
  return itr;
}
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(SimG4FluxProducer);
