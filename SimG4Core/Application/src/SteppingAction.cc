#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/EventAction.h"

#include "G4Track.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4UnitsTable.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using std::cout;
using std::endl;

SteppingAction::SteppingAction(EventAction* e,const edm::ParameterSet & p) 
  : eventAction_(e),
    killBeamPipe(p.getParameter<bool>("KillBeamPipe")),
    theCriticalEnergyForVacuum(p.getParameter<double>("CriticalEnergyForVacuum")*MeV),
    theCriticalDensity(p.getParameter<double>("CriticalDensity")*g/cm3),
    verbose(p.getUntrackedParameter<int>("Verbosity",0)), 
    initialized(false), tracker(0), calo(0) {}

SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step * aStep) {
  if (!initialized) initialized = initPointer();
  m_g4StepSignal(aStep);
  if (killBeamPipe){
    catchLowEnergyInVacuumHere(aStep);
    catchLowEnergyInVacuumNext(aStep);
  }
  if (aStep->GetPostStepPoint()->GetPhysicalVolume() != 0) {
    bool ok = (isThisVolume(aStep->GetPreStepPoint()->GetTouchable(),tracker)&&
	       isThisVolume(aStep->GetPostStepPoint()->GetTouchable(),calo));
    if (ok) {

      math::XYZVectorD pos((aStep->GetPreStepPoint()->GetPosition()).x(),
			   (aStep->GetPreStepPoint()->GetPosition()).y(),
			   (aStep->GetPreStepPoint()->GetPosition()).z());
      
      math::XYZTLorentzVectorD mom((aStep->GetPreStepPoint()->GetMomentum()).x(),
				   (aStep->GetPreStepPoint()->GetMomentum()).y(),
				   (aStep->GetPreStepPoint()->GetMomentum()).z(),
				   aStep->GetPreStepPoint()->GetTotalEnergy());
      
      uint32_t id = aStep->GetTrack()->GetTrackID();
      
      std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> p(pos,mom);
      eventAction_->addTkCaloStateInfo(id,p);
    }
  }
}

void SteppingAction::catchLowEnergyInVacuumHere(const G4Step * aStep) {
  G4Track * theTrack = aStep->GetTrack();
  double theKenergy = theTrack->GetKineticEnergy();
  if (theTrack->GetVolume()!=0) {
    double density = theTrack->GetVolume()->GetLogicalVolume()->GetMaterial()->GetDensity();
    if (theKenergy <= theCriticalEnergyForVacuum && theKenergy > 0.0 &&
	density <= theCriticalDensity && theTrack->GetDefinition()->GetPDGCharge() != 0 &&
	theTrack->GetTrackStatus() != fStopAndKill) {
      if (verbose>1)
	edm::LogInfo("SimG4CoreApplication") 
	  <<   " SteppingAction: LoopCatchSteppingAction:catchLowEnergyInVacuumHere: "
	  << " Track from " << theTrack->GetDefinition()->GetParticleName()
	  << " of kinetic energy " << theKenergy/MeV << " MeV "
	  << " killed in " << theTrack->GetVolume()->GetLogicalVolume()->GetName()
	  << " of density " << density/(g/cm3) << " g/cm3" ;
      theTrack->SetTrackStatus(fStopAndKill);
    }
  }
}

void SteppingAction::catchLowEnergyInVacuumNext(const G4Step * aStep) {
  G4Track * theTrack = aStep->GetTrack();
  double theKenergy = theTrack->GetKineticEnergy();
  if (theTrack->GetNextVolume()) {
    double density = theTrack->GetNextVolume()->GetLogicalVolume()->GetMaterial()->GetDensity();
    if (theKenergy <=  theCriticalEnergyForVacuum && theKenergy > 0.0 &&
	density <= theCriticalDensity && theTrack->GetDefinition()->GetPDGCharge() != 0 &&
	theTrack->GetTrackStatus() != fStopAndKill) {
      if (verbose>1)
	edm::LogInfo("SimG4CoreApplication") 
	  << " SteppingAction: LoopCatchSteppingAction::catchLowEnergyInVacuumNext: "
	  << " Track from " << theTrack->GetDefinition()->GetParticleName()
	  << " of kinetic energy " << theKenergy/MeV << " MeV "
	  << " stopped in " << theTrack->GetVolume()->GetLogicalVolume()->GetName()
	  << " before going into "<< theTrack->GetNextVolume()->GetLogicalVolume()->GetName()
	  << " of density " << density/(g/cm3) << " g/cm3" ;
      theTrack->SetTrackStatus(fStopButAlive);
    }
  }
}

bool SteppingAction::initPointer() {

  const G4PhysicalVolumeStore * pvs = G4PhysicalVolumeStore::GetInstance();
  if (pvs) {
    std::vector<G4VPhysicalVolume*>::const_iterator pvcite;
    for (pvcite = pvs->begin(); pvcite != pvs->end(); pvcite++) {
      if ((*pvcite)->GetName() == "Tracker") tracker = (*pvcite);
      if ((*pvcite)->GetName() == "CALO")    calo    = (*pvcite);
      if (tracker && calo) break;
    }
    edm::LogInfo("SimG4CoreApplication") << "Pointer for Tracker " << tracker
					 << " and for Calo " << calo;
    if (tracker) LogDebug("SimG4CoreApplication") << "Tracker vol name "
						  << tracker->GetName();
    if (calo)    LogDebug("SimG4CoreApplication") << "Calorimeter vol name "
						  << calo->GetName();
    return true;
  }
  return false;
}

bool SteppingAction::isThisVolume(const G4VTouchable* touch, 
				  G4VPhysicalVolume* pv) {

  int level = ((touch->GetHistoryDepth())+1);
  if (level > 0 && level >= 3) {
    int ii = level - 3;
    return (touch->GetVolume(ii) == pv);
  }
  return false;
}
