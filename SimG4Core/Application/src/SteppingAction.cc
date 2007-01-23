#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/EventAction.h"

#include "G4Track.hh"
#include "G4UnitsTable.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using std::cout;
using std::endl;

SteppingAction::SteppingAction(EventAction* e,const edm::ParameterSet & p) 
  : eventAction_(e),
    killBeamPipe(p.getParameter<bool>("KillBeamPipe")),
    theCriticalEnergyForVacuum(p.getParameter<double>("CriticalEnergyForVacuum")*MeV),
    theCriticalDensity(p.getParameter<double>("CriticalDensity")*g/cm3),
    verbose(p.getUntrackedParameter<int>("Verbosity",0))
{}

SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step * aStep)
{
    m_g4StepSignal(aStep);
    if (killBeamPipe)
    {
        catchLowEnergyInVacuumHere(aStep);
        catchLowEnergyInVacuumNext(aStep);
    }
    if((aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName()=="Tracker"&& 
       aStep->GetPostStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName()=="CALO")||
       (aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName()=="Tracker"&&
	aStep->GetPostStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName()!="EREG")){
      storeTkCaloStateInfo(aStep);
    }
}

void SteppingAction::catchLowEnergyInVacuumHere(const G4Step * aStep)
{
    G4Track * theTrack = aStep->GetTrack();
    double theKenergy = theTrack->GetKineticEnergy();
    if (theTrack->GetVolume()!=0)
    {
        double density = theTrack->GetVolume()->GetLogicalVolume()->GetMaterial()->GetDensity();
        if (theKenergy <= theCriticalEnergyForVacuum && theKenergy > 0.0 &&
            density <= theCriticalDensity && theTrack->GetDefinition()->GetPDGCharge() != 0 &&
            theTrack->GetTrackStatus() != fStopAndKill)
        {
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

void SteppingAction::catchLowEnergyInVacuumNext(const G4Step * aStep)
{
    G4Track * theTrack = aStep->GetTrack();
    double theKenergy = theTrack->GetKineticEnergy();
    if (theTrack->GetNextVolume())
    {
        double density = theTrack->GetNextVolume()->GetLogicalVolume()->GetMaterial()->GetDensity();
        if (theKenergy <=  theCriticalEnergyForVacuum && theKenergy > 0.0 &&
            density <= theCriticalDensity && theTrack->GetDefinition()->GetPDGCharge() != 0 &&
            theTrack->GetTrackStatus() != fStopAndKill)
        {
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

void SteppingAction::storeTkCaloStateInfo(const G4Step * aStep)
{
  Hep3Vector pos = aStep->GetPreStepPoint()->GetPosition();
  HepLorentzVector mom = HepLorentzVector(aStep->GetPreStepPoint()->GetMomentum()/GeV,
					       aStep->GetPreStepPoint()->GetTotalEnergy()/GeV);
  uint32_t id = aStep->GetTrack()->GetTrackID();
  
  std::pair<Hep3Vector,HepLorentzVector> p = std::pair<Hep3Vector,HepLorentzVector>(pos,mom);
  eventAction_->addTkCaloStateInfo(id,p);
}
