#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Transform3D.hh"
#include "G4LogicalVolumeStore.hh"

#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"

using std::string;

SensitiveDetector::SensitiveDetector(std::string & iname, 
				     const DDCompactView & cpv,
				     const SensitiveDetectorCatalog & clg,
				     edm::ParameterSet const & p) :
  G4VSensitiveDetector(iname), name(iname) {}

SensitiveDetector::~SensitiveDetector() {}

void SensitiveDetector::Initialize(G4HCofThisEvent * eventHC) {}

void SensitiveDetector::Register()
{
  G4SDManager * SDman = G4SDManager::GetSDMpointer();
  SDman->AddNewDetector(this);
}

void SensitiveDetector::AssignSD(const std::string & vname)
{
  G4LogicalVolumeStore * theStore = G4LogicalVolumeStore::GetInstance();
  G4LogicalVolumeStore::const_iterator it;
  for (it = theStore->begin(); it != theStore->end(); it++)
    {
      G4LogicalVolume * v = *it;
      if (vname==v->GetName()) { v->SetSensitiveDetector(this); }
    }
}

void SensitiveDetector::EndOfEvent(G4HCofThisEvent * eventHC) {}

#include "G4TouchableHistory.hh"

Local3DPoint SensitiveDetector::InitialStepPosition(G4Step * s, coordinates c)
{
    currentStep = s;
    G4StepPoint * preStepPoint = currentStep->GetPreStepPoint();
    G4ThreeVector globalCoordinates = preStepPoint->GetPosition();
    if (c == WorldCoordinates) return ConvertToLocal3DPoint(globalCoordinates);
    G4TouchableHistory * theTouchable=(G4TouchableHistory *)
                                      (preStepPoint->GetTouchable());
    G4ThreeVector localCoordinates = theTouchable->GetHistory()
                  ->GetTopTransform().TransformPoint(globalCoordinates);
    return ConvertToLocal3DPoint(localCoordinates); 
}

Local3DPoint SensitiveDetector::FinalStepPosition(G4Step * s, coordinates c)
{
    currentStep = s;
    G4StepPoint * postStepPoint = currentStep->GetPostStepPoint();
    G4StepPoint * preStepPoint  = currentStep->GetPreStepPoint();
    G4ThreeVector globalCoordinates = postStepPoint->GetPosition();
    if (c == WorldCoordinates) return ConvertToLocal3DPoint(globalCoordinates);
    G4TouchableHistory * theTouchable = (G4TouchableHistory *)
                                        (preStepPoint->GetTouchable());
    G4ThreeVector localCoordinates = theTouchable->GetHistory()
                  ->GetTopTransform().TransformPoint(globalCoordinates);
    return ConvertToLocal3DPoint(localCoordinates); 
}

Local3DPoint SensitiveDetector::ConvertToLocal3DPoint(const G4ThreeVector& p)
{
    return Local3DPoint(p.x(),p.y(),p.z());
}

void SensitiveDetector::NaNTrap( G4Step* aStep )
{

    if ( aStep == nullptr ) return ;
    
    G4Track* CurrentTrk = aStep->GetTrack() ;
    G4ThreeVector CurrentPos = CurrentTrk->GetPosition() ;
    G4ThreeVector CurrentMom = CurrentTrk->GetMomentum() ;
    G4VPhysicalVolume* pCurrentVol = CurrentTrk->GetVolume() ;
    G4String NameOfVol ;
    if ( pCurrentVol != nullptr )
    {
       NameOfVol = pCurrentVol->GetName() ;
    }
    else
    {
       NameOfVol = "CorruptedVolumeInfo" ;
    }
    
    // for simplicity... maybe edm::isNotFinite() will work on the 
    // 3-vector directly...

    double xyz[3] ;
    xyz[0] = CurrentPos.x() ;
    xyz[1] = CurrentPos.y() ;
    xyz[2] = CurrentPos.z() ;
    
    //
    // this is another trick to check on a NaN, maybe it's even CPU-faster...
    // but ler's stick to system function edm::isNotFinite(...) for now
    //
    // if ( !(xyz[0]==xyz[0]) || !(xyz[1]==xyz[1]) || !(xyz[2]==xyz[2]) )
    if( edm::isNotFinite(xyz[0]+xyz[1]+xyz[2]) != 0 )
    {
      throw SimG4Exception( "SimG4CoreSensitiveDetector: Corrupted Event - NaN detected (position) in volume " + NameOfVol);
    }

    xyz[0] = CurrentMom.x() ;
    xyz[1] = CurrentMom.y() ;
    xyz[2] = CurrentMom.z() ;
    if ( !(xyz[0]==xyz[0]) || !(xyz[1]==xyz[1]) || !(xyz[2]==xyz[2]) ||
         edm::isNotFinite(xyz[0]) != 0 || edm::isNotFinite(xyz[1]) != 0 || 
	 edm::isNotFinite(xyz[2]) != 0 )
    {
      throw SimG4Exception( "SimG4CoreSensitiveDetector: Corrupted Event - NaN detected (3-momentum) in volume " + NameOfVol);
    }

   return;

}
