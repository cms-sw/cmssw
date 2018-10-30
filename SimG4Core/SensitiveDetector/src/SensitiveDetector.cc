#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"

#include "SimG4Core/Notification/interface/SimG4Exception.h"
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

SensitiveDetector::SensitiveDetector(const std::string & iname, 
				     const DDCompactView & cpv,
				     const SensitiveDetectorCatalog & clg,
				     edm::ParameterSet const & p) :
  G4VSensitiveDetector(iname) 
{
  // for CMS hits
  namesOfSD.push_back(iname);

  // Geant4 hit collection
  collectionName.insert(iname);

  // register sensitive detector
  G4SDManager * SDman = G4SDManager::GetSDMpointer();
  SDman->AddNewDetector(this);

  const std::vector<std::string>& lvNames = clg.logicalNames(iname);
  std::stringstream ss;
  for (auto & lvname : lvNames) {
    this->AssignSD(lvname);
    ss << " " << lvname;
  }
  edm::LogVerbatim("SensitiveDetector") 
    << " <" << iname <<"> : Assigns SD to LVs " << ss.str();
}

SensitiveDetector::~SensitiveDetector() {}

void SensitiveDetector::Initialize(G4HCofThisEvent * eventHC) {}

void SensitiveDetector::EndOfEvent(G4HCofThisEvent * eventHC) {}

void SensitiveDetector::AssignSD(const std::string & vname)
{
  G4LogicalVolumeStore * theStore = G4LogicalVolumeStore::GetInstance();
  for (auto & lv : *theStore)
    {
      if (vname == lv->GetName()) { 
	lv->SetSensitiveDetector(this); 
      }
    }
}

Local3DPoint SensitiveDetector::InitialStepPosition(const G4Step * step, coordinates cd) const
{
  const G4StepPoint * preStepPoint = step->GetPreStepPoint();
  const G4ThreeVector& globalCoordinates = preStepPoint->GetPosition();
  if (cd == WorldCoordinates) { return ConvertToLocal3DPoint(globalCoordinates); }
  const G4TouchableHistory * theTouchable=static_cast<const G4TouchableHistory *>(preStepPoint->GetTouchable());
  const G4ThreeVector localCoordinates = theTouchable->GetHistory()
                  ->GetTopTransform().TransformPoint(globalCoordinates);
  return ConvertToLocal3DPoint(localCoordinates); 
}

Local3DPoint SensitiveDetector::FinalStepPosition(const G4Step * step, coordinates cd) const
{
  const G4StepPoint * postStepPoint = step->GetPostStepPoint();
  const G4ThreeVector& globalCoordinates = postStepPoint->GetPosition();
  if (cd == WorldCoordinates) { return ConvertToLocal3DPoint(globalCoordinates); }
  const G4StepPoint * preStepPoint  = step->GetPreStepPoint();
  const G4ThreeVector localCoordinates = preStepPoint->GetTouchable()->GetHistory()
                  ->GetTopTransform().TransformPoint(globalCoordinates);
  return ConvertToLocal3DPoint(localCoordinates); 
}

Local3DPoint SensitiveDetector::LocalPreStepPosition(const G4Step * step) const
{
  const G4StepPoint * preStepPoint = step->GetPreStepPoint();
  G4ThreeVector localCoordinates = preStepPoint->GetTouchable()->GetHistory()
    ->GetTopTransform().TransformPoint(preStepPoint->GetPosition());
  return ConvertToLocal3DPoint(localCoordinates); 
}

Local3DPoint SensitiveDetector::LocalPostStepPosition(const G4Step * step) const
{
  const G4ThreeVector& globalCoordinates = step->GetPostStepPoint()->GetPosition();
  G4ThreeVector localCoordinates = step->GetPreStepPoint()->GetTouchable()->GetHistory()
    ->GetTopTransform().TransformPoint(globalCoordinates);
  return ConvertToLocal3DPoint(localCoordinates);
}

TrackInformation* SensitiveDetector::cmsTrackInformation(const G4Track* aTrack)
{
  TrackInformation* info = (TrackInformation*)(aTrack->GetUserInformation());
  if (!info) { 
    edm::LogWarning("SensitiveDetector") 
      << " no TrackInformation available for trackID= " 
      << aTrack->GetTrackID();
    throw SimG4Exception("SimG4CoreSensitiveDetector: cannot handle hits for "
			 + GetName());
  }
  return info;
}

void SensitiveDetector::setNames(const std::vector<std::string>& hnames)
{
  namesOfSD.clear();
  namesOfSD = hnames;
}

void SensitiveDetector::NaNTrap(const G4Step* aStep) const
{
  const G4Track* CurrentTrk = aStep->GetTrack();
  const G4ThreeVector& CurrentPos = CurrentTrk->GetPosition();
  double xyz = CurrentPos.x() + CurrentPos.y() + CurrentPos.z();
  const G4ThreeVector& CurrentMom = CurrentTrk->GetMomentum();
  xyz += CurrentMom.x() + CurrentMom.y() + CurrentMom.z();

  if( edm::isNotFinite(xyz) ) {

    const G4VPhysicalVolume* pCurrentVol = aStep->GetPreStepPoint()->GetPhysicalVolume();
    const G4String& NameOfVol = pCurrentVol->GetName();
    throw SimG4Exception("SimG4CoreSensitiveDetector: Corrupted Event - NaN detected in volume "
			  + NameOfVol);
  }
}
