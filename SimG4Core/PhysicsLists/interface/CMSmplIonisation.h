//
// -------------------------------------------------------------------
//
// GEANT4 Class header file
//
//
// File name:     CMSmplIonisation
//
// Author:        Vladimir Ivanchenko copied from Geant4 10.5p01
//
// Creation date: 02.03.2019
//
// Class Description:
//
// This class manages the ionisation process for a magnetic monopole
// it inherites from G4VContinuousDiscreteProcess via G4VEnergyLossProcess.
// Magnetic charge of the monopole should be defined in the constructor of
// the process, unless it is assumed that it is classic Dirac monopole with
// the charge 67.5*eplus. The name of the particle should be "monopole".
//

// -------------------------------------------------------------------
//

#ifndef CMSmplIonisation_h
#define CMSmplIonisation_h 1

#include "G4VEnergyLossProcess.hh"
#include "globals.hh"
#include "G4VEmModel.hh"

class G4Material;
class G4VEmFluctuationModel;

class CMSmplIonisation : public G4VEnergyLossProcess {
public:
  explicit CMSmplIonisation(G4double mCharge = 0.0, const G4String& name = "mplIoni");

  ~CMSmplIonisation() override;

  G4bool IsApplicable(const G4ParticleDefinition& p) override;

  G4double MinPrimaryEnergy(const G4ParticleDefinition* p, const G4Material*, G4double cut) final;

  // Print out of the class parameters
  void PrintInfo() override;

  // print description in html
  void ProcessDescription(std::ostream&) const override;

protected:
  void InitialiseEnergyLossProcess(const G4ParticleDefinition*, const G4ParticleDefinition*) override;

private:
  G4double magneticCharge;
  G4bool isInitialised;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
