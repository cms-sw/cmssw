#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"

#include "SimG4CMS/Calo/interface/CaloTrkProcessing.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include "CondFormats/GeometryObjects/interface/CaloSimulationParameters.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
                                     const edm::EventSetup& es,
                                     const SensitiveDetectorCatalog& clg,
                                     edm::ParameterSet const& p,
                                     const SimTrackManager*)
    : SensitiveCaloDetector(name, clg), lastTrackID_(-1) {
  //Initialise the parameter set
  bool dd4hep = p.getParameter<bool>("g4GeometryDD4hepSource");
  int addlevel = dd4hep ? 1 : 0;
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("CaloTrkProcessing");
  testBeam_ = m_p.getParameter<bool>("TestBeam");
  eMin_ = m_p.getParameter<double>("EminTrack") * CLHEP::MeV;
  putHistory_ = m_p.getParameter<bool>("PutHistory");
  doFineCalo_ = m_p.getParameter<bool>("DoFineCalo");
  eMinFine_ = m_p.getParameter<double>("EminFineTrack") * CLHEP::MeV;
  std::vector<std::string> fineNames = m_p.getParameter<std::vector<std::string> >("FineCaloNames");
  std::vector<int> fineLevels = m_p.getParameter<std::vector<int> >("FineCaloLevels");
  std::vector<int> useFines = m_p.getParameter<std::vector<int> >("UseFineCalo");
  for (auto& level : fineLevels)
    level += addlevel;

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

  // Get pointer to CaloSimulationParameters
  edm::ESHandle<CaloSimulationParameters> csps;
  es.get<HcalParametersRcd>().get(csps);
  if (csps.isValid()) {
    const CaloSimulationParameters* csp = csps.product();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: " << csp->caloNames_.size() << " entries for caloNames:";
    for (unsigned int i = 0; i < csp->caloNames_.size(); i++)
      edm::LogVerbatim("CaloSim") << " (" << i << ") " << csp->caloNames_[i];
    edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: " << csp->levels_.size() << " entries for levels:";
    for (unsigned int i = 0; i < csp->levels_.size(); i++)
      edm::LogVerbatim("CaloSim") << " (" << i << ") " << (csp->levels_[i] + addlevel);
    edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: " << csp->neighbours_.size() << " entries for neighbours:";
    for (unsigned int i = 0; i < csp->neighbours_.size(); i++)
      edm::LogVerbatim("CaloSim") << " (" << i << ") " << csp->neighbours_[i];
    edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: " << csp->insideNames_.size() << " entries for insideNames:";
    for (unsigned int i = 0; i < csp->insideNames_.size(); i++)
      edm::LogVerbatim("CaloSim") << " (" << i << ") " << csp->insideNames_[i];
    edm::LogVerbatim("CaloSim") << "CaloTrkProcessing: " << csp->insideLevel_.size() << " entries for insideLevel:";
    for (unsigned int i = 0; i < csp->insideLevel_.size(); i++)
      edm::LogVerbatim("CaloSim") << " (" << i << ") " << (csp->insideLevel_[i] + addlevel);
#endif

    if (csp->caloNames_.size() < csp->neighbours_.size()) {
      edm::LogError("CaloSim") << "CaloTrkProcessing: # of Calorimeter bins " << csp->caloNames_.size()
                               << " does not match with " << csp->neighbours_.size() << " ==> illegal ";
      throw cms::Exception("Unknown", "CaloTrkProcessing")
          << "Calorimeter array size does not match with size of neighbours\n";
    }

    const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
    std::vector<G4LogicalVolume*>::const_iterator lvcite;
    int istart = 0;
    for (unsigned int i = 0; i < csp->caloNames_.size(); i++) {
      G4LogicalVolume* lv = nullptr;
      G4String name(csp->caloNames_[i]);
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
        detector.level = (csp->levels_[i] + addlevel);
        if (istart + csp->neighbours_[i] > static_cast<int>(csp->insideNames_.size())) {
          edm::LogError("CaloSim") << "CaloTrkProcessing: # of InsideNames bins " << csp->insideNames_.size()
                                   << " too few compaerd to " << istart + csp->neighbours_[i]
                                   << " requested ==> illegal ";
          throw cms::Exception("Unknown", "CaloTrkProcessing")
              << "InsideNames array size does not match with list of neighbours\n";
        }
        std::vector<std::string> inside;
        std::vector<G4LogicalVolume*> insideLV;
        std::vector<int> insideLevels;
        for (int k = 0; k < csp->neighbours_[i]; k++) {
          lv = nullptr;
          name = static_cast<G4String>(csp->insideNames_[istart + k]);
          for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
            G4String namx(static_cast<std::string>(dd4hep::dd::noNamespace((*lvcite)->GetName())));
            if (namx == name) {
              lv = (*lvcite);
              break;
            }
          }
          inside.push_back(name);
          insideLV.push_back(lv);
          insideLevels.push_back(csp->insideLevel_[istart + k] + addlevel);
        }
        detector.fromDets = inside;
        detector.fromDetL = insideLV;
        detector.fromLevels = insideLevels;
        detectors_.emplace_back(detector);
      }
      istart += csp->neighbours_[i];
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
  } else {
    edm::LogError("CaloSim") << "CaloTrkProcessing: Cannot find CaloSimulationParameters";
    throw cms::Exception("Unknown", "CaloTrkProcessing") << "Cannot find CaloSimulationParameters\n";
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
    // Boundary-crossing logic
    int prestepLV = isItCalo(aStep->GetPreStepPoint()->GetTouchable(), fineDetectors_);
    int poststepLV = isItCalo(aStep->GetPostStepPoint()->GetTouchable(), fineDetectors_);
    if (prestepLV < 0 && poststepLV >= 0
        // Allow back-scattering and filter it out later; ensure consistency during the SIM step
        // && std::abs(theTrack->GetStep()->GetPreStepPoint()->GetPosition().z()) < std::abs(theTrack->GetPosition().z())
    ) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DoFineCalo") << "Entered fine volume " << poststepLV << ":"
                                     << " Track " << id << " pdgid=" << theTrack->GetDefinition()->GetPDGEncoding()
                                     << " prestepLV=" << prestepLV << " poststepLV=" << poststepLV
                                     << " GetKineticEnergy[GeV]=" << theTrack->GetKineticEnergy() / CLHEP::GeV
                                     << " GetVertexKineticEnergy[GeV]="
                                     << theTrack->GetVertexKineticEnergy() / CLHEP::GeV << " prestepPosition[cm]=("
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
                                     << theTrack->GetVertexPosition().z() / CLHEP::cm << ")";
#endif
      trkInfo->setCrossedBoundary(theTrack);
    }
#ifdef EDM_ML_DEBUG
    else if (prestepLV >= 0 && poststepLV < 0) {
      edm::LogVerbatim("DoFineCalo") << "Exited fine volume " << prestepLV << ":"
                                     << " Track " << id
                                     << " GetKineticEnergy[GeV]=" << theTrack->GetKineticEnergy() / CLHEP::GeV
                                     << " GetVertexKineticEnergy[GeV]="
                                     << theTrack->GetVertexKineticEnergy() / CLHEP::GeV << " prestepPosition[cm]=("
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
