#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Transform3D.hh"
#include "G4LogicalVolumeStore.hh"

#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "G4TouchableHistory.hh"

SensitiveDetector::SensitiveDetector(std::string & iname, 
                                     const DDCompactView & cpv,
                                     const SensitiveDetectorCatalog & clg,
                                     edm::ParameterSet const & p) :
  G4VSensitiveDetector((G4String)iname), name(iname) {}

SensitiveDetector::~SensitiveDetector() {}

void SensitiveDetector::Initialize(G4HCofThisEvent * eventHC) {}

void SensitiveDetector::Register()
{
  G4SDManager * SDman = G4SDManager::GetSDMpointer();
  SDman->AddNewDetector(this);
}

void SensitiveDetector::AssignSD(const std::string & vname)
{
  G4LogicalVolume* v = 
    G4LogicalVolumeStore::GetInstance()->GetVolume((G4String)vname, true);
  if (v) { v->SetSensitiveDetector(this); }
}

void SensitiveDetector::EndOfEvent(G4HCofThisEvent * eventHC) {}

Local3DPoint SensitiveDetector::InitialStepPosition(const G4Step * step, coordinates cc)
{
  G4StepPoint * preStepPoint = step->GetPreStepPoint();
  return (cc == WorldCoordinates) ? ConvertToLocal3DPoint(preStepPoint->GetPosition())
    : ConvertToLocal3DPoint(preStepPoint->GetTouchable()->GetHistory()
                            ->GetTopTransform().TransformPoint(preStepPoint->GetPosition()));
}

Local3DPoint SensitiveDetector::FinalStepPosition(const G4Step * step, coordinates cc)
{
  // transformation is defined pre-step
  G4StepPoint * preStepPoint = step->GetPreStepPoint();
  G4StepPoint * postStepPoint = step->GetPostStepPoint();
  return (cc == WorldCoordinates) ? ConvertToLocal3DPoint(postStepPoint->GetPosition())
    : ConvertToLocal3DPoint(preStepPoint->GetTouchable()->GetHistory()
                            ->GetTopTransform().TransformPoint(postStepPoint->GetPosition()));
}

void SensitiveDetector::NaNTrap( G4Step* aStep )
{
  if( aStep != nullptr ) {   
    G4Track* CurrentTrk = aStep->GetTrack();

    double xyz = CurrentTrk->GetPosition().x() + CurrentTrk->GetPosition().y() + CurrentTrk->GetPosition().z();
    if( edm::isNotFinite(xyz))
    {
      G4VPhysicalVolume* pCurrentVol = CurrentTrk->GetVolume() ;
      G4String NameOfVol = ( pCurrentVol != nullptr ) ? pCurrentVol->GetName() 
        : "CorruptedVolumeInfo";
      throw SimG4Exception("SimG4CoreSensitiveDetector: Corrupted Event - NaN detected (position) in volume " 
                           + NameOfVol);
    }

    xyz = CurrentTrk->GetMomentum().x() + CurrentTrk->GetMomentum().y() + CurrentTrk->GetMomentum().z();
    if( edm::isNotFinite(xyz))
    {
      G4VPhysicalVolume* pCurrentVol = CurrentTrk->GetVolume() ;
      G4String NameOfVol = ( pCurrentVol != nullptr ) ? pCurrentVol->GetName() 
        : "CorruptedVolumeInfo";
      throw SimG4Exception("SimG4CoreSensitiveDetector: Corrupted Event - NaN detected (3-momentum) in volume "
                           + NameOfVol);
    }
  }
  return;
}

std::vector<std::string> SensitiveDetector::getNames()
{
  std::vector<std::string> temp;
  temp.push_back(name);
  return std::move(temp);
}  
