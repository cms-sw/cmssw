#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CaloHit/interface/MaterialInformation.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"

#include "G4LogicalVolumeStore.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4TransportationManager.hh"
#include "DD4hep/Filter.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

//#define EDM_ML_DEBUG

class MaterialBudgetVolume : public SimProducer,
                             public Observer<const BeginOfRun*>,
                             public Observer<const BeginOfEvent*>,
                             public Observer<const BeginOfTrack*>,
                             public Observer<const G4Step*>,
                             public Observer<const EndOfTrack*> {
public:
  MaterialBudgetVolume(const edm::ParameterSet& p);
  ~MaterialBudgetVolume() override {}

  void produce(edm::Event&, const edm::EventSetup&) override;

  struct MatInfo {
    double stepL, radL, intL;
    MatInfo(double s = 0, double r = 0, double l = 0) : stepL(s), radL(r), intL(l) {}
  };

private:
  MaterialBudgetVolume(const MaterialBudgetVolume&) = delete;  // stop default
  const MaterialBudgetVolume& operator=(const MaterialBudgetVolume&) = delete;

  // observer classes
  void update(const BeginOfRun* run) override;
  void update(const BeginOfEvent* evt) override;
  void update(const BeginOfTrack*) override;
  void update(const EndOfTrack*) override;
  void update(const G4Step* step) override;

  void endOfEvent(edm::MaterialInformationContainer& matbg);
  bool loadLV();
  int findLV(const G4VTouchable*);

private:
  std::vector<std::string> lvNames_;
  std::vector<int> lvLevel_;
  int iaddLevel_;
  bool init_;
  std::map<int, std::pair<G4LogicalVolume*, int> > mapLV_;
  std::vector<MatInfo> lengths_;
  std::vector<MaterialInformation> store_;
};

MaterialBudgetVolume::MaterialBudgetVolume(const edm::ParameterSet& p) : init_(false) {
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("MaterialBudgetVolume");

  lvNames_ = m_p.getParameter<std::vector<std::string> >("lvNames");
  lvLevel_ = m_p.getParameter<std::vector<int> >("lvLevels");
  iaddLevel_ = (m_p.getParameter<bool>("useDD4Hep")) ? 1 : 0;

  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolume: Studies Material budget for " << lvNames_.size()
                                     << " volumes with addLevel " << iaddLevel_;
  std::ostringstream st1;
  for (unsigned int k = 0; k < lvNames_.size(); ++k)
    st1 << " [" << k << "] " << lvNames_[k] << " at " << lvLevel_[k];
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolume: Volumes" << st1.str();

  produces<edm::MaterialInformationContainer>("MaterialInformation");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolume: will produce MaterialInformationContainer";
#endif
}

void MaterialBudgetVolume::produce(edm::Event& e, const edm::EventSetup&) {
  std::unique_ptr<edm::MaterialInformationContainer> matbg(new edm::MaterialInformationContainer);
  endOfEvent(*matbg);
  e.put(std::move(matbg), "MaterialInformation");
}

void MaterialBudgetVolume::update(const BeginOfRun* run) {
  init_ = loadLV();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolume::Finds " << mapLV_.size()
                                     << " logical volumes with return flag " << init_;
#endif
}

void MaterialBudgetVolume::update(const BeginOfEvent* evt) {
#ifdef EDM_ML_DEBUG
  int iev = (*evt)()->GetEventID();
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolume: =====> Begin event = " << iev << std::endl;
#endif
  if (!init_)
    init_ = loadLV();

  store_.clear();
}

void MaterialBudgetVolume::update(const BeginOfTrack* trk) {
  lengths_ = std::vector<MatInfo>(mapLV_.size());

#ifdef EDM_ML_DEBUG
  const G4Track* aTrack = (*trk)();
  const G4ThreeVector& mom = aTrack->GetMomentum();
  double theEnergy = aTrack->GetTotalEnergy();
  int theID = static_cast<int>(aTrack->GetDefinition()->GetPDGEncoding());
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolumme: Track " << aTrack->GetTrackID() << " Code " << theID
                                     << " Energy " << theEnergy / CLHEP::GeV << " GeV; Momentum " << mom / CLHEP::GeV
                                     << " GeV";
#endif
}

void MaterialBudgetVolume::update(const G4Step* aStep) {
  G4Material* material = aStep->GetPreStepPoint()->GetMaterial();
  double step = aStep->GetStepLength();
  double radl = material->GetRadlen();
  double intl = material->GetNuclearInterLength();
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int index = findLV(touch);
  if (index >= 0) {
    lengths_[index].stepL += step;
    lengths_[index].radL += (step / radl);
    lengths_[index].intL += (step / intl);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolume::Step in "
                                     << touch->GetVolume(0)->GetLogicalVolume()->GetName() << " Index " << index
                                     << " Step " << step << " RadL " << step / radl << " IntL " << step / intl;
#endif
}

void MaterialBudgetVolume::update(const EndOfTrack* trk) {
  const G4Track* aTrack = (*trk)();
  int id = aTrack->GetTrackID();
  double eta = aTrack->GetMomentumDirection().eta();
  double phi = aTrack->GetMomentumDirection().phi();
  for (unsigned int k = 0; k < lengths_.size(); ++k) {
    MaterialInformation info(lvNames_[k], id, eta, phi, lengths_[k].stepL, lengths_[k].radL, lengths_[k].intL);
    store_.emplace_back(info);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolume::Volume[" << k << "]: " << info;
#endif
  }
}

void MaterialBudgetVolume::endOfEvent(edm::MaterialInformationContainer& matbg) {
#ifdef EDM_ML_DEBUG
  unsigned int kount(0);
#endif
  for (const auto& element : store_) {
    MaterialInformation info(element.vname(),
                             element.id(),
                             element.trackEta(),
                             element.trackPhi(),
                             element.stepLength(),
                             element.radiationLength(),
                             element.interactionLength());
    matbg.push_back(info);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolume:: Info[" << kount << "] " << info;
    ++kount;
#endif
  }
}

bool MaterialBudgetVolume::loadLV() {
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  bool flag(false);
  if (lvs != nullptr) {
    std::vector<G4LogicalVolume*>::const_iterator lvcite;
    for (unsigned int i = 0; i < lvNames_.size(); i++) {
      G4LogicalVolume* lv = nullptr;
      std::string name(lvNames_[i]);
      for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
        std::string namx(dd4hep::dd::noNamespace((*lvcite)->GetName()));
        if (namx == name) {
          lv = (*lvcite);
          break;
        }
      }
      if (lv != nullptr)
        mapLV_[i] = std::make_pair(lv, (lvLevel_[i] + iaddLevel_));
    }
    flag = true;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolume::Finds " << mapLV_.size() << " logical volumes";
    unsigned int k(0);
    for (const auto& lvs : mapLV_) {
      edm::LogVerbatim("MaterialBudget") << "Entry[" << k << "] " << lvs.first << ": (" << (lvs.second).first << ", "
                                         << (lvs.second).second << ") : " << lvNames_[lvs.first];
      ++k;
    }
#endif
  }
  return flag;
}

int MaterialBudgetVolume::findLV(const G4VTouchable* touch) {
  int level(-1);
  int levels = ((touch->GetHistoryDepth()) + 1);
  for (const auto& lvs : mapLV_) {
    if ((lvs.second).second <= levels) {
      int ii = levels - (lvs.second).second;
      if ((touch->GetVolume(ii)->GetLogicalVolume()) == (lvs.second).first) {
        level = lvs.first;
        break;
      }
    }
  }
#ifdef EDM_ML_DEBUG
  if (level < 0) {
    edm::LogVerbatim("MaterialBudget") << "findLV: Gets " << level << " from " << levels << " levels in touchables";
    for (int i = 0; i < levels; ++i)
      edm::LogVerbatim("MaterialBudget") << "[" << (levels - i) << "] " << touch->GetVolume(i)->GetLogicalVolume()
                                         << " : " << touch->GetVolume(i)->GetLogicalVolume()->GetName();
  }
#endif
  return level;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

DEFINE_SIMWATCHER(MaterialBudgetVolume);
