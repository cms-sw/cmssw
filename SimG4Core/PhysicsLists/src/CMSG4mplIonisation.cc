//
// -------------------------------------------------------------------
//
// GEANT4 Class file
//
//
// File name:     G4mplIonisation
//
// Author:        Vladimir Ivanchenko
//
// Creation date: 25.08.2005
//
// Modifications:
//
//
// -------------------------------------------------------------------
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#include "CMSG4mplIonisation.hh"
#include "G4Electron.hh"
#include "G4mplIonisationModel.hh"
#include "G4BohrFluctuations.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

using namespace std;

CMSG4mplIonisation::CMSG4mplIonisation(G4double mCharge, const G4String& name)
  : G4VEnergyLossProcess(name),
    magneticCharge(mCharge),
    isInitialised(false)
{
  // By default classical magnetic charge is used
  if(magneticCharge == 0.0) magneticCharge = eplus*0.5/fine_structure_const;

  SetVerboseLevel(0);
  SetProcessSubType(fIonisation);
  SetStepFunction(0.2, 1*mm);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

CMSG4mplIonisation::~CMSG4mplIonisation()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4bool CMSG4mplIonisation::IsApplicable(const G4ParticleDefinition&)
{
  return true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void CMSG4mplIonisation::InitialiseEnergyLossProcess(const G4ParticleDefinition*,
						  const G4ParticleDefinition*)
{
  if(isInitialised) return;

  SetBaseParticle(0);
  SetSecondaryParticle(G4Electron::Electron());

  G4mplIonisationModel* ion  = new G4mplIonisationModel(magneticCharge,"PAI");
  ion->SetLowEnergyLimit(MinKinEnergy());
  ion->SetHighEnergyLimit(MaxKinEnergy());
  AddEmModel(0,ion,ion);

  isInitialised = true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void CMSG4mplIonisation::PrintInfo()
{
  G4cout << "      No delta-electron production, only dE/dx"
         << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
