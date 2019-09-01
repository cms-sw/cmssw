//
// -------------------------------------------------------------------
//
// GEANT4 Class file
//
//
// File name:     CMSmplIonisation
//
// Author:        Vladimir Ivanchenko copied from Geant4 10.5p01
//
// Creation date: 02.03.2019
//
//
// -------------------------------------------------------------------
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#include "SimG4Core/PhysicsLists/interface/CMSmplIonisation.h"
#include "SimG4Core/PhysicsLists/interface/CMSmplIonisationWithDeltaModel.h"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4Electron.hh"
#include "G4EmParameters.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

using namespace std;

CMSmplIonisation::CMSmplIonisation(G4double mCharge, const G4String& name)
    : G4VEnergyLossProcess(name), magneticCharge(mCharge), isInitialised(false) {
  // By default classical magnetic charge is used
  if (magneticCharge == 0.0) {
    magneticCharge = eplus * 0.5 / fine_structure_const;
  }

  SetVerboseLevel(0);
  SetProcessSubType(fIonisation);
  SetStepFunction(0.2, 1 * mm);
  SetSecondaryParticle(G4Electron::Electron());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

CMSmplIonisation::~CMSmplIonisation() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4bool CMSmplIonisation::IsApplicable(const G4ParticleDefinition&) { return true; }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double CMSmplIonisation::MinPrimaryEnergy(const G4ParticleDefinition* mpl, const G4Material*, G4double cut) {
  G4double x = 0.5 * cut / electron_mass_c2;
  G4double mass = mpl->GetPDGMass();
  G4double ratio = electron_mass_c2 / mass;
  G4double gam = x * ratio + std::sqrt((1. + x) * (1. + x * ratio * ratio));
  return mass * (gam - 1.0);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void CMSmplIonisation::InitialiseEnergyLossProcess(const G4ParticleDefinition* p, const G4ParticleDefinition*) {
  if (isInitialised) {
    return;
  }

  SetBaseParticle(nullptr);

  // monopole model is responsible both for energy loss and fluctuations
  CMSmplIonisationWithDeltaModel* ion = new CMSmplIonisationWithDeltaModel(magneticCharge, "PAI");
  ion->SetParticle(p);

  // define size of dedx and range tables
  G4EmParameters* param = G4EmParameters::Instance();
  G4double emin = std::min(param->MinKinEnergy(), ion->LowEnergyLimit());
  G4double emax = std::max(param->MaxKinEnergy(), ion->HighEnergyLimit());
  G4int bin = G4lrint(param->NumberOfBinsPerDecade() * std::log10(emax / emin));
  ion->SetLowEnergyLimit(emin);
  ion->SetHighEnergyLimit(emax);
  SetMinKinEnergy(emin);
  SetMaxKinEnergy(emax);
  SetDEDXBinning(bin);

  SetEmModel(ion);
  AddEmModel(1, ion, ion);

  isInitialised = true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void CMSmplIonisation::PrintInfo() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void CMSmplIonisation::ProcessDescription(std::ostream& out) const {
  out << "No description available." << G4endl;
  G4VEnergyLossProcess::ProcessDescription(out);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
