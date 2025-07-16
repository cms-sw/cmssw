/**
 * @file CMSmuDarkBremsstrahlung.h
 * @brief Class providing the Dark Bremsstrahlung process class.
 * @author Michael Revering, University of Minnesota
 */

#ifndef CMSmuDarkBremsstrahlung_h
#define CMSmuDarkBremsstrahlung_h

// Geant
#include "G4VEmProcess.hh"

class G4Material;

class CMSmuDarkBremsstrahlung : public G4VEmProcess {
public:
  CMSmuDarkBremsstrahlung(const G4String& scalefile, const G4double biasFactor, const G4String& name = "muDBrem");

  ~CMSmuDarkBremsstrahlung() override = default;

  G4bool IsApplicable(const G4ParticleDefinition& p) override;

  void SetMethod(std::string method_in);

  G4bool IsEnabled();
  void SetEnable(bool active);
  CMSmuDarkBremsstrahlung& operator=(const CMSmuDarkBremsstrahlung& right) = delete;
  CMSmuDarkBremsstrahlung(const CMSmuDarkBremsstrahlung&) = delete;

protected:
  void InitialiseProcess(const G4ParticleDefinition*) override;
  G4bool isInitialised;
  const G4String& mgfile;
  const G4double cxBias;
  G4bool isEnabled;
};

#endif
