#ifndef CMSDarkPairProductionProcess_h
#define CMSDarkPairProductionProcess_h 1

#include "SimG4Core/CustomPhysics/interface/CMSDarkPairProduction.hh"
#include "globals.hh"
#include "G4VEmProcess.hh"
#include "G4Gamma.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class G4ParticleDefinition;
class G4VEmModel;
class G4MaterialCutsCouple;
class G4DynamicParticle;

class CMSDarkPairProductionProcess : public G4VEmProcess

{
public:  // with description

  CMSDarkPairProductionProcess(G4double df = 1E0,
  		      const G4String& processName ="conv",
                      //const G4ParticleDefinition* p = 0,
                      //const G4double df = 1E0,
		      G4ProcessType type = fElectromagnetic);

  virtual ~CMSDarkPairProductionProcess();

  // true for Gamma only.
  virtual G4bool IsApplicable(const G4ParticleDefinition&);

  virtual G4double MinPrimaryEnergy(const G4ParticleDefinition*,
				    const G4Material*);

  // Print few lines of informations about the process: validity range,
  virtual void PrintInfo();

protected:

  virtual void InitialiseProcess(const G4ParticleDefinition*);
  //double darkFactor;

private:
  G4bool  isInitialised;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
  
#endif
 
