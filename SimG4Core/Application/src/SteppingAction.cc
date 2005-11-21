#include "SimG4Core/Application/interface/SteppingAction.h"

#include "G4Track.hh"
#include "G4UnitsTable.hh"

using std::cout;
using std::endl;

SteppingAction::SteppingAction(const edm::ParameterSet & p) 
    : killBeamPipe(p.getParameter<bool>("KillBeamPipe")),
      theCriticalEnergyForVacuum(p.getParameter<double>("CriticalEnergyForVacuum")*MeV),
      theCriticalDensity(p.getParameter<double>("CriticalDensity")*g/cm3),
      verbose(p.getParameter<int>("Verbosity"))
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
            cout << " OSCAR ACTION: LoopCatchSteppingAction:catchLowEnergyInVacuumHere: "
                 << " Track from " << theTrack->GetDefinition()->GetParticleName()
                 << " of kinetic energy " << theKenergy/MeV << " MeV "
                 << " killed in " << theTrack->GetVolume()->GetLogicalVolume()->GetName()
                 << " of density " << density/(g/cm3) << " g/cm3" << endl;
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
            cout << " OSCAR ACTION: LoopCatchSteppingAction::catchLowEnergyInVacuumNext: "
                 << " Track from " << theTrack->GetDefinition()->GetParticleName()
                 << " of kinetic energy " << theKenergy/MeV << " MeV "
                 << " stopped in " << theTrack->GetVolume()->GetLogicalVolume()->GetName()
                 << " before going into "<< theTrack->GetNextVolume()->GetLogicalVolume()->GetName()
                 << " of density " << density/(g/cm3) << " g/cm3" << endl;
            theTrack->SetTrackStatus(fStopButAlive);
        }
    }
}
