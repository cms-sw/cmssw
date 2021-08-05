///////////////////////////////////////////////////////////////////////////////
// File: HGCPassive.cc
// copied from SimG4HGCalValidation
// Description: Main analysis class for HGCal Validation of G4 Hits
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/HGCalTestBeam/interface/HGCPassive.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "G4TransportationManager.hh"
#include "DD4hep/Filter.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

//#define EDM_ML_DEBUG

HGCPassive::HGCPassive(const edm::ParameterSet& p) : topPV_(nullptr), topLV_(nullptr), count_(0), init_(false) {
  edm::ParameterSet m_Passive = p.getParameter<edm::ParameterSet>("HGCPassive");
  LVNames_ = m_Passive.getParameter<std::vector<std::string> >("LVNames");
  motherName_ = m_Passive.getParameter<std::string>("MotherName");
  bool dd4hep = m_Passive.getParameter<bool>("IfDD4Hep");
  addlevel_ = dd4hep ? 1 : 0;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Name of the mother volume " << motherName_ << " AddLevel " << addlevel_;
  unsigned k(0);
#endif
  for (const auto& name : LVNames_) {
    produces<edm::PassiveHitContainer>(Form("%sPassiveHits", name.c_str()));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "Collection name[" << k << "] " << name;
    ++k;
#endif
  }
}

HGCPassive::~HGCPassive() {}

void HGCPassive::produce(edm::Event& e, const edm::EventSetup&) {
  for (unsigned int k = 0; k < LVNames_.size(); ++k) {
    std::unique_ptr<edm::PassiveHitContainer> hgcPH(new edm::PassiveHitContainer);
    endOfEvent(*hgcPH, k);
    e.put(std::move(hgcPH), Form("%sPassiveHits", LVNames_[k].c_str()));
  }
}

void HGCPassive::update(const BeginOfRun* run) {
  topPV_ = getTopPV();
  if (topPV_ == nullptr) {
    edm::LogWarning("HGCSim") << "Cannot find top level volume\n";
  } else {
    init_ = true;
    const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
    for (auto lvcite : *lvs) {
      findLV(lvcite);
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCPassive::Finds " << mapLV_.size() << " logical volumes";
    unsigned int k(0);
    for (const auto& lvs : mapLV_) {
      edm::LogVerbatim("HGCSim") << "Entry[" << k << "] " << lvs.first << ": (" << (lvs.second).first << ", "
                                 << (lvs.second).second << ")";
      ++k;
    }
#endif
  }
}

//=================================================================== per EVENT
void HGCPassive::update(const BeginOfEvent* evt) {
  int iev = (*evt)()->GetEventID();
  edm::LogVerbatim("HGCSim") << "HGCPassive: =====> Begin event = " << iev << std::endl;

  ++count_;
  store_.clear();
}

// //=================================================================== each
// STEP
void HGCPassive::update(const G4Step* aStep) {
  if (aStep != nullptr) {
    G4VSensitiveDetector* curSD = aStep->GetPreStepPoint()->GetSensitiveDetector();
    const G4VTouchable* touchable = aStep->GetPreStepPoint()->GetTouchable();

    int level = (touchable->GetHistoryDepth());
    if (curSD == nullptr) {
      G4LogicalVolume* plv = touchable->GetVolume()->GetLogicalVolume();
      auto it = (init_) ? mapLV_.find(plv) : findLV(plv);
      double time = aStep->GetTrack()->GetGlobalTime();
      double energy = (aStep->GetTotalEnergyDeposit()) / CLHEP::GeV;

      unsigned int copy(0);
      if (((aStep->GetPostStepPoint() == nullptr) || (aStep->GetTrack()->GetNextVolume() == nullptr)) &&
          (aStep->IsLastStepInVolume())) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << static_cast<std::string>(dd4hep::dd::noNamespace(plv->GetName())) << " F|L Step "
                                   << aStep->IsFirstStepInVolume() << ":" << aStep->IsLastStepInVolume() << " Position"
                                   << aStep->GetPreStepPoint()->GetPosition() << " Track "
                                   << aStep->GetTrack()->GetDefinition()->GetParticleName() << " at"
                                   << aStep->GetTrack()->GetPosition() << " Volume " << aStep->GetTrack()->GetVolume()
                                   << ":" << aStep->GetTrack()->GetNextVolume() << " Status "
                                   << aStep->GetTrack()->GetTrackStatus() << " KE "
                                   << aStep->GetTrack()->GetKineticEnergy() << " Deposit "
                                   << aStep->GetTotalEnergyDeposit() << " Map " << (it != mapLV_.end());
#endif
        energy += (aStep->GetPreStepPoint()->GetKineticEnergy() / CLHEP::GeV);
      } else {
        time = (aStep->GetPostStepPoint()->GetGlobalTime());
        copy = (level < 2)
                   ? 0
                   : static_cast<unsigned int>(touchable->GetReplicaNumber(0) + 1000 * touchable->GetReplicaNumber(1));
      }
      if (it != mapLV_.end()) {
        storeInfo(it, plv, copy, time, energy, true);
      } else if (topLV_ != nullptr) {
        auto itr = findLV(topLV_);
        if (itr != mapLV_.end()) {
          storeInfo(itr, topLV_, copy, time, energy, true);
        }
      }
    }  // if (curSD==NULL)

    // Now for the mother volumes
    if (level > 0) {
      double energy = (aStep->GetTotalEnergyDeposit()) / CLHEP::GeV;
      double time = (aStep->GetTrack()->GetGlobalTime());

      for (int i = level; i > 0; --i) {
        G4LogicalVolume* plv = touchable->GetVolume(i)->GetLogicalVolume();
        auto it = (init_) ? mapLV_.find(plv) : findLV(plv);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "Level: " << level << ":" << i << " "
                                   << static_cast<std::string>(dd4hep::dd::noNamespace(plv->GetName()))
                                   << " flag in the List " << (it != mapLV_.end());
#endif
        if (it != mapLV_.end()) {
          unsigned int copy =
              (i == level) ? 0
                           : (unsigned int)(touchable->GetReplicaNumber(i) + 1000 * touchable->GetReplicaNumber(i + 1));
          storeInfo(it, plv, copy, time, energy, false);
        }
      }
    }
  }  // if (aStep != NULL)

}  // end update aStep

//================================================================ End of EVENT

void HGCPassive::endOfEvent(edm::PassiveHitContainer& hgcPH, unsigned int k) {
#ifdef EDM_ML_DEBUG
  unsigned int kount(0);
#endif
  for (const auto& element : store_) {
    G4LogicalVolume* lv = (element.first).first;
    auto it = mapLV_.find(lv);
    if (it != mapLV_.end()) {
      if ((it->second).first == k) {
        PassiveHit hit(
            (it->second).second, (element.first).second, (element.second)[1], (element.second)[2], (element.second)[0]);
        hgcPH.push_back(hit);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "HGCPassive[" << k << "] Hit[" << kount << "] " << hit;
        ++kount;
#endif
      }
    }
  }
}

G4VPhysicalVolume* HGCPassive::getTopPV() {
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}

HGCPassive::volumeIterator HGCPassive::findLV(G4LogicalVolume* plv) {
  auto itr = mapLV_.find(plv);
  if (itr == mapLV_.end()) {
    std::string name = static_cast<std::string>(dd4hep::dd::noNamespace(plv->GetName()));
    for (unsigned int k = 0; k < LVNames_.size(); ++k) {
      if (name.find(LVNames_[k]) != std::string::npos) {
        mapLV_[plv] = std::pair<unsigned int, std::string>(k, name);
        itr = mapLV_.find(plv);
        break;
      }
    }
  }
  if (topLV_ == nullptr) {
    if (static_cast<std::string>(dd4hep::dd::noNamespace(plv->GetName())) == motherName_)
      topLV_ = plv;
  }
  return itr;
}

void HGCPassive::storeInfo(const HGCPassive::volumeIterator it,
                           G4LogicalVolume* plv,
                           unsigned int copy,
                           double time,
                           double energy,
                           bool flag) {
  std::pair<G4LogicalVolume*, unsigned int> key(plv, copy);
  auto itr = store_.find(key);
  double ee = (flag) ? energy : 0;
  if (itr == store_.end()) {
    store_[key] = {{time, energy, energy}};
  } else {
    (itr->second)[1] += ee;
    (itr->second)[2] += energy;
  }
#ifdef EDM_ML_DEBUG
  itr = store_.find(key);
  edm::LogVerbatim("HGCSim") << "HGCPassive: Element " << (it->second).first << ":" << (it->second).second << ":"
                             << copy << " T " << (itr->second)[0] << " E " << (itr->second)[1] << ":"
                             << (itr->second)[2];
#endif
}

DEFINE_SIMWATCHER(HGCPassive);
