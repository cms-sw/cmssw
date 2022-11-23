#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"

#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4StepPoint.hh"
#include "G4Transform3D.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4TouchableHistory.hh"
#include "G4VUserTrackInformation.hh"

#include <sstream>

SensitiveDetector::SensitiveDetector(const std::string& iname, const SensitiveDetectorCatalog& clg, bool calo)
    : G4VSensitiveDetector(iname), m_isCalo(calo) {
  // for CMS hits
  m_namesOfSD.push_back(iname);

  // Geant4 hit collection
  collectionName.insert(iname);

  // register sensitive detector
  G4SDManager* SDman = G4SDManager::GetSDMpointer();
  SDman->AddNewDetector(this);

  const std::vector<std::string_view>& lvNames = clg.logicalNames(iname);
  std::stringstream ss;
  for (auto& lvname : lvNames) {
    this->AssignSD({lvname.data(), lvname.size()});
    ss << " " << lvname;
  }
  edm::LogVerbatim("SensitiveDetector") << " <" << iname << "> : Assigns SD to LVs " << ss.str();
}

SensitiveDetector::~SensitiveDetector() {}

void SensitiveDetector::Initialize(G4HCofThisEvent* eventHC) {}

void SensitiveDetector::EndOfEvent(G4HCofThisEvent* eventHC) {}

void SensitiveDetector::AssignSD(const std::string& vname) {
  G4LogicalVolumeStore* theStore = G4LogicalVolumeStore::GetInstance();
  for (auto& lv : *theStore) {
    if (vname == lv->GetName()) {
      lv->SetSensitiveDetector(this);
    }
  }
}

Local3DPoint SensitiveDetector::InitialStepPosition(const G4Step* step, coordinates cd) const {
  const G4StepPoint* preStepPoint = step->GetPreStepPoint();
  const G4ThreeVector& globalCoordinates = preStepPoint->GetPosition();
  if (cd == WorldCoordinates) {
    return ConvertToLocal3DPoint(globalCoordinates);
  }
  const G4TouchableHistory* theTouchable = static_cast<const G4TouchableHistory*>(preStepPoint->GetTouchable());
  const G4ThreeVector localCoordinates =
      theTouchable->GetHistory()->GetTopTransform().TransformPoint(globalCoordinates);
  return ConvertToLocal3DPoint(localCoordinates);
}

Local3DPoint SensitiveDetector::FinalStepPosition(const G4Step* step, coordinates cd) const {
  const G4StepPoint* postStepPoint = step->GetPostStepPoint();
  const G4ThreeVector& globalCoordinates = postStepPoint->GetPosition();
  if (cd == WorldCoordinates) {
    return ConvertToLocal3DPoint(globalCoordinates);
  }
  const G4StepPoint* preStepPoint = step->GetPreStepPoint();
  const G4ThreeVector localCoordinates =
      preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(globalCoordinates);
  return ConvertToLocal3DPoint(localCoordinates);
}

Local3DPoint SensitiveDetector::LocalPreStepPosition(const G4Step* step) const {
  const G4StepPoint* preStepPoint = step->GetPreStepPoint();
  G4ThreeVector localCoordinates =
      preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(preStepPoint->GetPosition());
  return ConvertToLocal3DPoint(localCoordinates);
}

Local3DPoint SensitiveDetector::LocalPostStepPosition(const G4Step* step) const {
  const G4ThreeVector& globalCoordinates = step->GetPostStepPoint()->GetPosition();
  G4ThreeVector localCoordinates =
      step->GetPreStepPoint()->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(globalCoordinates);
  return ConvertToLocal3DPoint(localCoordinates);
}

TrackInformation* SensitiveDetector::cmsTrackInformation(const G4Track* aTrack) {
  TrackInformation* info = (TrackInformation*)(aTrack->GetUserInformation());
  if (nullptr == info) {
    edm::LogWarning("SensitiveDetector") << " no TrackInformation available for trackID= " << aTrack->GetTrackID()
                                         << " inside SD " << GetName();
    G4Exception(
        "SensitiveDetector::cmsTrackInformation()", "sd01", FatalException, "cannot handle hits without trackinfo");
  }
  return info;
}

void SensitiveDetector::setNames(const std::vector<std::string>& hnames) {
  m_namesOfSD.clear();
  m_namesOfSD = hnames;
}

void SensitiveDetector::NaNTrap(const G4Step* aStep) const {
  G4Track* currentTrk = aStep->GetTrack();
  double ekin = currentTrk->GetKineticEnergy();
  if (ekin < 0.0) {
    const G4VPhysicalVolume* pCurrentVol = aStep->GetPreStepPoint()->GetPhysicalVolume();
    edm::LogWarning("SensitiveDetector") << "Negative kinetic energy Ekin(MeV)=" << ekin / CLHEP::MeV << " of "
                                         << currentTrk->GetDefinition()->GetParticleName()
                                         << " trackID= " << currentTrk->GetTrackID() << " inside "
                                         << pCurrentVol->GetName();
    currentTrk->SetKineticEnergy(0.0);
  }
  const G4ThreeVector& currentPos = currentTrk->GetPosition();
  double xyz = currentPos.x() + currentPos.y() + currentPos.z();
  const G4ThreeVector& currentMom = currentTrk->GetMomentum();
  xyz += currentMom.x() + currentMom.y() + currentMom.z();

  if (edm::isNotFinite(xyz)) {
    const G4VPhysicalVolume* pCurrentVol = aStep->GetPreStepPoint()->GetPhysicalVolume();
    edm::LogWarning("SensitiveDetector") << "NaN detected for trackID= " << currentTrk->GetTrackID() << " inside "
                                         << pCurrentVol->GetName();
    G4Exception("SensitiveDetector::NaNTrap()", "sd01", FatalException, "corrupted event or step");
  }
}
