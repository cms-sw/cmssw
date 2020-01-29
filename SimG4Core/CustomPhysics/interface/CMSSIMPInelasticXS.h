#ifndef SimG4Core_CustomPhysics_CMSSIMPInelasticXS_H
#define SimG4Core_CustomPhysics_CMSSIMPInelasticXS_H

#include "G4VCrossSectionDataSet.hh"
#include "globals.hh"

class G4NeutronInelasticXS;

class CMSSIMPInelasticXS : public G4VCrossSectionDataSet {
public:
  CMSSIMPInelasticXS();

  ~CMSSIMPInelasticXS() override;

  G4bool IsElementApplicable(const G4DynamicParticle*, G4int Z, const G4Material*) override;

  G4bool IsIsoApplicable(const G4DynamicParticle*, G4int Z, G4int A, const G4Element*, const G4Material*) override;

  G4double GetElementCrossSection(const G4DynamicParticle*, G4int Z, const G4Material* mat = nullptr) override;

  G4double GetIsoCrossSection(const G4DynamicParticle*,
                              G4int Z,
                              G4int A,
                              const G4Isotope* iso,
                              const G4Element* elm,
                              const G4Material* mat) override;

  void BuildPhysicsTable(const G4ParticleDefinition&) override;

private:
  void Initialise(G4int Z, G4DynamicParticle* dp = nullptr, const char* = nullptr);

  CMSSIMPInelasticXS& operator=(const CMSSIMPInelasticXS& right) = delete;
  CMSSIMPInelasticXS(const CMSSIMPInelasticXS&) = delete;

  G4NeutronInelasticXS* nXsection;
  const G4ParticleDefinition* neutron;
  G4bool isInitialized;
};

#endif
