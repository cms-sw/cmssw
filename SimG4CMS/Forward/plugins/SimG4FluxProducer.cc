#include "SimG4FluxProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"
#include "G4TouchableHistory.hh"
#include "G4TransportationManager.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <utility>

//#define EDM_ML_DEBUG

SimG4FluxProducer::SimG4FluxProducer(const edm::ParameterSet& p) : count_(0), init_(false) {
  edm::ParameterSet m_FP = p.getParameter<edm::ParameterSet>("SimG4FluxProducer");
  LVNames_ = m_FP.getUntrackedParameter<std::vector<std::string>>("LVNames");
  LVTypes_ = m_FP.getUntrackedParameter<std::vector<int>>("LVTypes");

  for (unsigned int k = 0; k < LVNames_.size(); ++k) {
    produces<ParticleFlux>(Form("%sParticleFlux", LVNames_[k].c_str()));
#ifdef EDM_ML_DEBUG
    std::cout << "Collection name[" << k << "] ParticleFlux" << LVNames_[k] << " and type " << LVTypes_[k] << std::endl;
#endif
  }
}

SimG4FluxProducer::~SimG4FluxProducer() {}

void SimG4FluxProducer::produce(edm::Event& e, const edm::EventSetup&) {
  for (unsigned int k = 0; k < LVNames_.size(); ++k) {
    std::unique_ptr<ParticleFlux> pflux(new ParticleFlux);
    endOfEvent(*pflux, k);
    std::string name = LVNames_[k] + "ParticleFlux";
    e.put(std::move(pflux), name);
  }
}

void SimG4FluxProducer::update(const BeginOfRun* run) {
  topPV_ = getTopPV();
  if (topPV_ == nullptr) {
    edm::LogWarning("SimG4FluxProducer") << "Cannot find top level volume\n";
  } else {
    init_ = true;
    const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
    for (auto lvcite : *lvs) {
      findLV(lvcite);
    }

#ifdef EDM_ML_DEBUG
    std::cout << "SimG4FluxProducer::Finds " << mapLV_.size() << " logical volumes\n";
    unsigned int k(0);
    for (const auto& lvs : mapLV_) {
      std::cout << "Entry[" << k << "] " << lvs.first << ": (" << (lvs.second).first << ", " << (lvs.second).second
                << ")\n";
      ++k;
    }
#endif
  }
}

void SimG4FluxProducer::update(const BeginOfEvent* evt) {
  int iev = (*evt)()->GetEventID();
  edm::LogInfo("ValidHGCal") << "SimG4FluxProducer: =====> Begin event = " << iev << std::endl;
  ++count_;
  store_.clear();
}

void SimG4FluxProducer::update(const G4Step* aStep) {
  if (aStep != nullptr) {
    G4TouchableHistory* touchable = (G4TouchableHistory*)aStep->GetPreStepPoint()->GetTouchable();
    G4LogicalVolume* plv = (G4LogicalVolume*)touchable->GetVolume()->GetLogicalVolume();
    auto it = (init_) ? mapLV_.find(plv) : findLV(plv);
    //  std::cout << plv->GetName() << " Flag " << (it != mapLV_.end()) << " step " << aStep->IsFirstStepInVolume() << ":"  << aStep->IsLastStepInVolume() << std::endl;
    if (it != mapLV_.end()) {
      int type = LVTypes_[(it->second).first];
      if ((type == 0 && aStep->IsFirstStepInVolume()) || (type == 1 && aStep->IsLastStepInVolume())) {
        unsigned int copy = (unsigned int)(touchable->GetReplicaNumber(0));
        std::pair<G4LogicalVolume*, unsigned int> key(plv, copy);
        auto itr = store_.find(key);
        if (itr == store_.end()) {
          store_[key] = ParticleFlux((it->second).second, copy);
          itr = store_.find(key);
        }
        G4Track* track = aStep->GetTrack();
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
        std::cout << "SimG4FluxProducer: Element " << (it->second).first << ":" << (it->second).second << ":" << copy
                  << " ID " << pdgid << " VxType " << vxtyp << " TOF " << time << " Hit Point " << flx.hitPoint << " p "
                  << flx.momentum << " Vertex " << flx.vertex << std::endl;
#endif
      }  //if(Step ok)
    }    //if( it != map.end() )
  }      //if (aStep != NULL)
}

void SimG4FluxProducer::endOfEvent(ParticleFlux& flux, unsigned int k) {
  bool done(false);
  for (const auto& element : store_) {
    G4LogicalVolume* lv = (element.first).first;
    auto it = mapLV_.find(lv);
    if (it != mapLV_.end()) {
      if ((it->second).first == k) {
        flux = element.second;
        done = true;
#ifdef EDM_ML_DEBUG
        std::cout << "SimG4FluxProducer[" << k << "] Flux " << flux.getName() << ":" << flux.getId() << " with "
                  << flux.getComponents() << " elements" << std::endl;
        std::vector<ParticleFlux::flux> fluxes = flux.getFlux();
        unsigned int k(0);
        for (auto element : fluxes) {
          std::cout << "Flux[" << k << "] PDGId " << element.pdgId << " VT " << element.vxType << " ToF " << element.tof
                    << " Vertex " << element.vertex << " Hit " << element.hitPoint << " p " << element.momentum
                    << std::endl;
          ++k;
        }
#endif
      }
    }
    if (done)
      break;
  }
}

G4VPhysicalVolume* SimG4FluxProducer::getTopPV() {
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}

std::map<G4LogicalVolume*, std::pair<unsigned int, std::string>>::iterator SimG4FluxProducer::findLV(
    G4LogicalVolume* plv) {
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
