/**
 * @file G4muDarkBremsstrahlung.h
 * @brief Class providing the Dark Bremsstrahlung process class.
 * @author Michael Revering, University of Minnesota
 */

#ifndef G4muDarkBremsstrahlung_h
#define G4muDarkBremsstrahlung_h

// Geant
#include "G4VEmProcess.hh"

class G4Material;

class G4muDarkBremsstrahlung : public G4VEmProcess {
public:
  G4muDarkBremsstrahlung(const G4String& scalefile, const G4double biasFactor, const G4String& name = "muDBrem");

  ~G4muDarkBremsstrahlung() override;

  G4bool IsApplicable(const G4ParticleDefinition& p) override;

  void PrintInfo() override;

  void SetMethod(std::string method_in);

  G4bool IsEnabled();
  void SetEnable(bool active);
  G4muDarkBremsstrahlung& operator=(const G4muDarkBremsstrahlung& right) = delete;
  G4muDarkBremsstrahlung(const G4muDarkBremsstrahlung&) = delete;

protected:
  void InitialiseProcess(const G4ParticleDefinition*) override;
  G4bool isInitialised;
  const G4String& mgfile;
  const G4double cxBias;
  G4bool isEnabled;
};

#endif
