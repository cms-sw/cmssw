//
// -------------------------------------------------------------------
//
// GEANT4 Class header file
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

#ifndef CMSG4mplIonisation_h
#define CMSG4mplIonisation_h 1

#include "G4VEnergyLossProcess.hh"
#include "globals.hh"
#include "G4VEmModel.hh"

class G4Material;
class G4VEmFluctuationModel;

class CMSG4mplIonisation : public G4VEnergyLossProcess
{

public:

  CMSG4mplIonisation(G4double mCharge = 0.0, const G4String& name = "mplIoni");

  virtual ~CMSG4mplIonisation();

  virtual G4bool IsApplicable(const G4ParticleDefinition& p);

  // Print out of the class parameters
  virtual void PrintInfo();

protected:

  virtual void InitialiseEnergyLossProcess(const G4ParticleDefinition*,
					   const G4ParticleDefinition*);

private:

  // hide assignment operator
  CMSG4mplIonisation & operator=(const CMSG4mplIonisation &right);
  CMSG4mplIonisation(const CMSG4mplIonisation&);

  G4double    magneticCharge;
  G4bool      isInitialised;

};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
