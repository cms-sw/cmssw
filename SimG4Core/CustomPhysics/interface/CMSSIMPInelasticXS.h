#ifndef SimG4Core_CustomPhysics_CMSSIMPInelasticXS_H
#define SimG4Core_CustomPhysics_CMSSIMPInelasticXS_H

#include "G4VCrossSectionDataSet.hh"
#include "globals.hh"
#include "G4ElementData.hh"
#include <vector>

const G4int MAXZINEL = 93;

class G4DynamicParticle;
class G4ParticleDefinition;
class G4Element;
class G4PhysicsVector;
class G4ComponentGGHadronNucleusXsc;
class G4HadronNucleonXsc;

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

  G4Isotope* SelectIsotope(const G4Element*, G4double kinEnergy) override;

  void BuildPhysicsTable(const G4ParticleDefinition&) override;

  void CrossSectionDescription(std::ostream&) const override;

private:
  void Initialise(G4int Z, G4DynamicParticle* dp = nullptr, const char* = nullptr);

  G4PhysicsVector* RetrieveVector(std::ostringstream& in, G4bool warn);

  G4double IsoCrossSection(G4double ekin, G4int Z, G4int A);

  CMSSIMPInelasticXS& operator=(const CMSSIMPInelasticXS& right) = delete;
  CMSSIMPInelasticXS(const CMSSIMPInelasticXS&) = delete;

  G4ComponentGGHadronNucleusXsc* ggXsection;
  G4HadronNucleonXsc* fNucleon;

  const G4ParticleDefinition* proton;

  G4ElementData data;
  std::vector<G4PhysicsVector*> work;
  std::vector<G4double> temp;
  std::vector<G4double> coeff;

  G4bool isInitialized;

  static const G4int amin[MAXZINEL];
  static const G4int amax[MAXZINEL];
};

#endif
