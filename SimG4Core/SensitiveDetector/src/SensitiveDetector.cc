#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Transform3D.hh"
#include "G4LogicalVolumeStore.hh"

using std::string;

SensitiveDetector::SensitiveDetector(string & iname) :
    G4VSensitiveDetector(iname), name(iname) {}

SensitiveDetector::~SensitiveDetector() {}

void SensitiveDetector::Initialize(G4HCofThisEvent * eventHC) {}

void SensitiveDetector::Register()
{
    G4SDManager * SDman = G4SDManager::GetSDMpointer();
    SDman->AddNewDetector(this);
}

void SensitiveDetector::AssignSD(string & vname)
{
    G4LogicalVolumeStore * theStore = G4LogicalVolumeStore::GetInstance();
    G4LogicalVolumeStore::const_iterator it;
    for (it = theStore->begin(); it != theStore->end(); it++)
    {
	G4LogicalVolume * v = *it;
	if (v->GetName() == vname.c_str()) v->SetSensitiveDetector(this);
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

Local3DPoint SensitiveDetector::ConvertToLocal3DPoint(G4ThreeVector p)
{
    return Local3DPoint(p.x(),p.y(),p.z());
}




