#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"

#include "SimG4CMS/Calo/interface/CaloTrkProcessing.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "G4EventManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4SystemOfUnits.hh"
#include "DD4hep/Filter.h"

#include <sstream>
//#define EDM_ML_DEBUG

CaloTrkProcessing::CaloTrkProcessing(const std::string& name,
                                     const CaloSimulationParameters& csps,
                                     const SensitiveDetectorCatalog& clg,
                                     bool testBeam,
                                     double eMin,
                                     bool putHistory,
                                     bool doFineCalo,
                                     double eMinFine,
                                     int addlevel,
                                     const std::vector<std::string>& fineNames,
                                     const std::vector<int>& fineLevels,
                                     const std::vector<int>& useFines,
                                     const SimTrackManager*)
    : SensitiveCaloDetector(name, clg),
      testBeam_(testBeam),
      eMin_(eMin),
      putHistory_(putHistory),
      doFineCalo_(doFineCalo),
      eMinFine_(eMinFine),
      addlevel_(addlevel),
      lastTrackID_(-1) {
  //Initialise the parameter set

  edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: Initialised with TestBeam = " << testBeam_ << " Emin = " << eMin_
                              << " Flags " << putHistory_ << " (History), " << doFineCalo_ << " (Special Calorimeter)";
  edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: Have a possibility of " << fineNames.size()
                              << " fine calorimeters of which " << useFines.size() << " are selected";
  for (unsigned int k = 0; k < fineNames.size(); ++k)
    edm::LogVerbatim("CaloSim") << "[" << k << "] " << fineNames[k] << " at " << fineLevels[k];
  std::ostringstream st1;
  for (unsigned int k = 0; k < useFines.size(); ++k)
    st1 << " [" << k << "] " << useFines[k] << ":" << fineNames[useFines[k]];
  edm::LogVerbatim("CaloSim") << "CaloTrkProcessing used calorimeters" << st1.str();

  // Debug prints
  edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: " << csps.caloNames_.size() << " entries for caloNames:";
  for (unsigned int i = 0; i < csps.caloNames_.size(); i++)
    edm::LogVerbatim("CaloSim") << " (" << i << ") " << csps.caloNames_[i];
  edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: " << csps.levels_.size() << " entries for levels:";
  for (unsigned int i = 0; i < csps.levels_.size(); i++)
    edm::LogVerbatim("CaloSim") << " (" << i << ") " << (csps.levels_[i] + addlevel_);
  edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: " << csps.neighbours_.size() << " entries for neighbours:";
  for (unsigned int i = 0; i < csps.neighbours_.size(); i++)
    edm::LogVerbatim("CaloSim") << " (" << i << ") " << csps.neighbours_[i];
  edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: " << csps.insideNames_.size() << " entries for insideNames:";
  for (unsigned int i = 0; i < csps.insideNames_.size(); i++)
    edm::LogVerbatim("CaloSim") << " (" << i << ") " << csps.insideNames_[i];
  edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: " << csps.insideLevel_.size() << " entries for insideLevel:";
  for (unsigned int i = 0; i < csps.insideLevel_.size(); i++)
    edm::LogVerbatim("CaloSim") << " (" << i << ") " << (csps.insideLevel_[i] + addlevel_);

  if (csps.caloNames_.size() < csps.neighbours_.size()) {
    edm::LogError("CaloSim") << "CaloTrkProcessing: # of Calorimeter bins " << csps.caloNames_.size()
                             << " does not match with " << csps.neighbours_.size() << " ==> illegal ";
    throw cms::Exception("Unknown", "CaloTrkProcessing")
        << "Calorimeter array size does not match with size of neighbours\n";
  }

  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume*>::const_iterator lvcite;
  int istart = 0;
  for (unsigned int i = 0; i < csps.caloNames_.size(); i++) {
    G4LogicalVolume* lv = nullptr;
    G4String name(csps.caloNames_[i]);
    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
      G4String namx(static_cast<std::string>(dd4hep::dd::noNamespace((*lvcite)->GetName())));
      if (namx == name) {
        lv = (*lvcite);
        break;
      }
    }
    if (lv != nullptr) {
      CaloTrkProcessing::Detector detector;
      detector.name = name;
      detector.lv = lv;
      detector.level = (csps.levels_[i] + addlevel_);
      if (istart + csps.neighbours_[i] > static_cast<int>(csps.insideNames_.size())) {
        edm::LogError("CaloSim") << "CaloTrkProcessing: # of InsideNames bins " << csps.insideNames_.size()
                                 << " too few compaerd to " << istart + csps.neighbours_[i]
                                 << " requested ==> illegal ";
        throw cms::Exception("Unknown", "CaloTrkProcessing")
            << "InsideNames array size does not match with list of neighbours\n";
      }
      std::vector<std::string> inside;
      std::vector<G4LogicalVolume*> insideLV;
      std::vector<int> insideLevels;
      for (int k = 0; k < csps.neighbours_[i]; k++) {
        lv = nullptr;
        name = static_cast<G4String>(csps.insideNames_[istart + k]);
        for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
          G4String namx(static_cast<std::string>(dd4hep::dd::noNamespace((*lvcite)->GetName())));
          if (namx == name) {
            lv = (*lvcite);
            break;
          }
        }
        inside.push_back(name);
        insideLV.push_back(lv);
        insideLevels.push_back(csps.insideLevel_[istart + k] + addlevel_);
      }
      detector.fromDets = inside;
      detector.fromDetL = insideLV;
      detector.fromLevels = insideLevels;
      detectors_.emplace_back(detector);
    }
    istart += csps.neighbours_[i];
  }

  for (unsigned int i = 0; i < useFines.size(); i++) {
    G4LogicalVolume* lv = nullptr;
    G4String name = static_cast<G4String>(fineNames[useFines[i]]);
    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
      G4String namx(static_cast<std::string>(dd4hep::dd::noNamespace((*lvcite)->GetName())));
      if (namx == name) {
        lv = (*lvcite);
        break;
      }
    }
    if (lv != nullptr) {
      CaloTrkProcessing::Detector detector;
      detector.name = name;
      detector.lv = lv;
      detector.level = fineLevels[useFines[i]];
      detector.fromDets.clear();
      detector.fromDetL.clear();
      detector.fromLevels.clear();
      fineDetectors_.emplace_back(detector);
    }
  }

  edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: with " << detectors_.size() << " calorimetric volumes";
  for (unsigned int i = 0; i < detectors_.size(); i++) {
    edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: Calorimeter volume " << i << " " << detectors_[i].name << " LV "
                                << detectors_[i].lv << " at level " << detectors_[i].level << " with "
                                << detectors_[i].fromDets.size() << " neighbours";
    for (unsigned int k = 0; k < detectors_[i].fromDets.size(); k++)
      edm::LogVerbatim("CaloSim") << "                   Element " << k << " " << detectors_[i].fromDets[k] << " LV "
                                  << detectors_[i].fromDetL[k] << " at level " << detectors_[i].fromLevels[k];
  }

  doFineCalo_ = doFineCalo_ && !(fineDetectors_.empty());
  edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: with " << fineDetectors_.size() << " special calorimetric volumes";
  for (unsigned int i = 0; i < detectors_.size(); i++)
    edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: Calorimeter volume " << i << " " << detectors_[i].name << " LV "
                                << detectors_[i].lv << " at level " << detectors_[i].level;
}

CaloTrkProcessing::~CaloTrkProcessing() {}

void CaloTrkProcessing::update(const BeginOfEvent* evt) { lastTrackID_ = -1; }

void CaloTrkProcessing::update(const G4Step* aStep) {
  // define if you are at the surface of CALO

  G4Track* theTrack = aStep->GetTrack();
  int id = theTrack->GetTrackID();

  TrackInformation* trkInfo = dynamic_cast<TrackInformation*>(theTrack->GetUserInformation());

  if (trkInfo == nullptr) {
    edm::LogError("CaloSim") << "CaloTrkProcessing: No trk info !!!! abort ";
    throw cms::Exception("Unknown", "CaloTrkProcessing") << "cannot get trkInfo for Track " << id << "\n";
  }

  if (doFineCalo_) {
    int prestepLV = isItCalo(aStep->GetPreStepPoint()->GetTouchable(), fineDetectors_);
    int poststepLV = isItCalo(aStep->GetPostStepPoint()->GetTouchable(), fineDetectors_);

    // Once per track, determine whether track started in fine volume
    if (!trkInfo->startedInFineVolumeIsSet())
      trkInfo->setStartedInFineVolume(prestepLV >= 0);

    // Boundary-crossing logic
    if (prestepLV < 0 && poststepLV >= 0) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DoFineCalo") << "Track " << id << " entered a fine volume:"
                                     << " pdgid=" << theTrack->GetDefinition()->GetPDGEncoding()
                                     << " theTrack->GetCurrentStepNumber()=" << theTrack->GetCurrentStepNumber()
                                     << " prestepLV=" << prestepLV << " poststepLV=" << poststepLV
                                     << " GetKineticEnergy[GeV]=" << theTrack->GetKineticEnergy() / CLHEP::GeV
                                     << " prestepPosition[cm]=("
                                     << theTrack->GetStep()->GetPreStepPoint()->GetPosition().x() / CLHEP::cm << ","
                                     << theTrack->GetStep()->GetPreStepPoint()->GetPosition().y() / CLHEP::cm << ","
                                     << theTrack->GetStep()->GetPreStepPoint()->GetPosition().z() / CLHEP::cm << ")"
                                     << " poststepPosition[cm]=("
                                     << theTrack->GetStep()->GetPostStepPoint()->GetPosition().x() / CLHEP::cm << ","
                                     << theTrack->GetStep()->GetPostStepPoint()->GetPosition().y() / CLHEP::cm << ","
                                     << theTrack->GetStep()->GetPostStepPoint()->GetPosition().z() / CLHEP::cm << ")"
                                     << " position[cm]=(" << theTrack->GetPosition().x() / CLHEP::cm << ","
                                     << theTrack->GetPosition().y() / CLHEP::cm << ","
                                     << theTrack->GetPosition().z() / CLHEP::cm << ")"
                                     << " vertex_position[cm]=(" << theTrack->GetVertexPosition().x() / CLHEP::cm << ","
                                     << theTrack->GetVertexPosition().y() / CLHEP::cm << ","
                                     << theTrack->GetVertexPosition().z() / CLHEP::cm << ")"
                                     << " GetVertexKineticEnergy[GeV]="
                                     << theTrack->GetVertexKineticEnergy() / CLHEP::GeV;
#endif
      if (!trkInfo->startedInFineVolume() && !trkInfo->crossedBoundary()) {
        trkInfo->setCrossedBoundary(theTrack);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("DoFineCalo") << "Track " << id << " marked as boundary-crossing; sanity check:"
                                       << " theTrack->GetTrackID()=" << theTrack->GetTrackID()
                                       << " trkInfo->crossedBoundary()=" << trkInfo->crossedBoundary();
#endif
      }
#ifdef EDM_ML_DEBUG
      else {
        edm::LogVerbatim("DoFineCalo") << "Track " << id << " REENTERED a fine volume;"
                                       << " not counting this boundary crossing!";
      }
#endif

    }
#ifdef EDM_ML_DEBUG
    else if (prestepLV >= 0 && poststepLV < 0) {
      edm::LogVerbatim("DoFineCalo") << "Track " << id << " exited a fine volume:"
                                     << " theTrack->GetCurrentStepNumber()=" << theTrack->GetCurrentStepNumber()
                                     << " prestepLV=" << prestepLV << " poststepLV=" << poststepLV
                                     << " GetKineticEnergy[GeV]=" << theTrack->GetKineticEnergy() / CLHEP::GeV
                                     << " prestepPosition[cm]=("
                                     << theTrack->GetStep()->GetPreStepPoint()->GetPosition().x() / CLHEP::cm << ","
                                     << theTrack->GetStep()->GetPreStepPoint()->GetPosition().y() / CLHEP::cm << ","
                                     << theTrack->GetStep()->GetPreStepPoint()->GetPosition().z() / CLHEP::cm << ")"
                                     << " poststepPosition[cm]=("
                                     << theTrack->GetStep()->GetPostStepPoint()->GetPosition().x() / CLHEP::cm << ","
                                     << theTrack->GetStep()->GetPostStepPoint()->GetPosition().y() / CLHEP::cm << ","
                                     << theTrack->GetStep()->GetPostStepPoint()->GetPosition().z() / CLHEP::cm << ")";
    }
#endif
  }

  if (testBeam_) {
    if (trkInfo->getIDonCaloSurface() == 0) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("CaloSim") << "CaloTrkProcessing set IDonCaloSurface to " << id << " at step Number "
                                  << theTrack->GetCurrentStepNumber();
#endif
      trkInfo->setIDonCaloSurface(id, 0, 0, theTrack->GetDefinition()->GetPDGEncoding(), theTrack->GetMomentum().mag());
      lastTrackID_ = id;
      if (theTrack->GetKineticEnergy() / CLHEP::MeV > eMin_)
        trkInfo->putInHistory();
    }
  } else {
    if (putHistory_) {
      trkInfo->putInHistory();
      //      trkInfo->setAncestor();
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloTrkProcessing Entered for " << id << " at stepNumber "
                                << theTrack->GetCurrentStepNumber() << " IDonCaloSur.. "
                                << trkInfo->getIDonCaloSurface() << " CaloCheck " << trkInfo->caloIDChecked();
#endif
    if (trkInfo->getIDonCaloSurface() != 0) {
      if (trkInfo->caloIDChecked() == false) {
        G4StepPoint* postStepPoint = aStep->GetPostStepPoint();
        const G4VTouchable* post_touch = postStepPoint->GetTouchable();

        if (isItInside(post_touch, trkInfo->getIDCaloVolume(), trkInfo->getIDLastVolume()) > 0) {
          trkInfo->setIDonCaloSurface(0, -1, -1, 0, 0);
        } else {
          trkInfo->setCaloIDChecked(true);
        }
      }
    } else {
      G4StepPoint* postStepPoint = aStep->GetPostStepPoint();
      const G4VTouchable* post_touch = postStepPoint->GetTouchable();
      int ical = isItCalo(post_touch, detectors_);
      if (ical >= 0) {
        G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
        const G4VTouchable* pre_touch = preStepPoint->GetTouchable();
        int inside = isItInside(pre_touch, ical, -1);
        if (inside >= 0 || (theTrack->GetCurrentStepNumber() == 1)) {
          trkInfo->setIDonCaloSurface(
              id, ical, inside, theTrack->GetDefinition()->GetPDGEncoding(), theTrack->GetMomentum().mag());
          trkInfo->setCaloIDChecked(true);
          if (!doFineCalo_)
            trkInfo->setCrossedBoundary(theTrack);
          lastTrackID_ = id;
          if (theTrack->GetKineticEnergy() / CLHEP::MeV > eMin_)
            trkInfo->putInHistory();
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: set ID on Calo " << ical << " surface (Inside " << inside
                                      << ") to " << id << " of a Track with Kinetic Energy "
                                      << theTrack->GetKineticEnergy() / CLHEP::MeV << " MeV";
#endif
        }
      }
    }
  }
}

int CaloTrkProcessing::isItCalo(const G4VTouchable* touch, const std::vector<Detector>& detectors) {
  int lastLevel = -1;
  G4LogicalVolume* lv = nullptr;
  for (unsigned int it = 0; it < detectors.size(); it++) {
    if (lastLevel != detectors[it].level) {
      lastLevel = detectors[it].level;
      lv = detLV(touch, lastLevel);
#ifdef EDM_ML_DEBUG
      std::string name1 = "Unknown";
      if (lv != 0)
        name1 = lv->GetName();
      edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: volume " << name1 << " at Level " << lastLevel;
      int levels = detLevels(touch);
      if (levels > 0) {
        G4String name2[20];
        int copyno2[20];
        detectorLevel(touch, levels, copyno2, name2);
        for (int i2 = 0; i2 < levels; i2++)
          edm::LogVerbatim("CaloSim") << " " << i2 << " " << name2[i2] << " " << copyno2[i2];
      }
#endif
    }
    bool ok = (lv == detectors[it].lv);
    if (ok)
      return it;
  }
  return -1;
}

int CaloTrkProcessing::isItInside(const G4VTouchable* touch, int idcal, int idin) {
  int lastLevel = -1;
  G4LogicalVolume* lv = nullptr;
  int id1, id2;
  if (idcal < 0) {
    id1 = 0;
    id2 = static_cast<int>(detectors_.size());
  } else {
    id1 = idcal;
    id2 = id1 + 1;
  }
  for (int it1 = id1; it1 < id2; it1++) {
    if (idin < 0) {
      for (unsigned int it2 = 0; it2 < detectors_[it1].fromDets.size(); it2++) {
        if (lastLevel != detectors_[it1].fromLevels[it2]) {
          lastLevel = detectors_[it1].fromLevels[it2];
          lv = detLV(touch, lastLevel);
#ifdef EDM_ML_DEBUG
          std::string name1 = "Unknown";
          if (lv != 0)
            name1 = lv->GetName();
          edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: volume " << name1 << " at Level " << lastLevel;
          int levels = detLevels(touch);
          if (levels > 0) {
            G4String name2[20];
            int copyno2[20];
            detectorLevel(touch, levels, copyno2, name2);
            for (int i2 = 0; i2 < levels; i2++)
              edm::LogVerbatim("CaloSim") << " " << i2 << " " << name2[i2] << " " << copyno2[i2];
          }
#endif
        }
        bool ok = (lv == detectors_[it1].fromDetL[it2]);
        if (ok)
          return it2;
      }
    } else {
      lastLevel = detectors_[it1].fromLevels[idin];
      lv = detLV(touch, lastLevel);
#ifdef EDM_ML_DEBUG
      std::string name1 = "Unknown";
      if (lv != 0)
        name1 = lv->GetName();
      edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: volume " << name1 << " at Level " << lastLevel;
      int levels = detLevels(touch);
      if (levels > 0) {
        G4String name2[20];
        int copyno2[20];
        detectorLevel(touch, levels, copyno2, name2);
        for (int i2 = 0; i2 < levels; i2++)
          edm::LogVerbatim("CaloSim") << " " << i2 << " " << name2[i2] << " " << copyno2[i2];
      }
#endif
      bool ok = (lv == detectors_[it1].fromDetL[idin]);
      if (ok)
        return idin;
    }
  }
  return -1;
}

int CaloTrkProcessing::detLevels(const G4VTouchable* touch) const {
  //Return number of levels
  if (touch)
    return ((touch->GetHistoryDepth()) + 1);
  else
    return 0;
}

G4LogicalVolume* CaloTrkProcessing::detLV(const G4VTouchable* touch, int currentlevel) const {
  G4LogicalVolume* lv = nullptr;
  if (touch) {
    int level = ((touch->GetHistoryDepth()) + 1);
    if (level > 0 && level >= currentlevel) {
      int ii = level - currentlevel;
      lv = touch->GetVolume(ii)->GetLogicalVolume();
      return lv;
    }
  }
  return lv;
}

void CaloTrkProcessing::detectorLevel(const G4VTouchable* touch, int& level, int* copyno, G4String* name) const {
  static const std::string unknown("Unknown");
  //Get name and copy numbers
  if (level > 0) {
    for (int ii = 0; ii < level; ii++) {
      int i = level - ii - 1;
      G4VPhysicalVolume* pv = touch->GetVolume(i);
      if (pv != nullptr)
        name[ii] = pv->GetName();
      else
        name[ii] = unknown;
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloSimX") << "CaloTrkProcessing::detectorLevel "
                               << " with " << level << ":" << detLevels(touch) << " levels";
  for (int ii = 0; ii < level; ii++)
    edm::LogVerbatim("CaloSimX") << "[" << ii << "] " << name[ii] << ":" << copyno[ii];
#endif
}
