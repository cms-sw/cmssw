
#ifndef G4SQInelasticCrossSection_h
#define G4SQInelasticCrossSection_h

#include "globals.hh"
#include "G4VCrossSectionDataSet.hh"

class G4NistManager;
class G4SQ;
class G4AntiSQ;

class G4SQInelasticCrossSection : public G4VCrossSectionDataSet {
public:
  G4SQInelasticCrossSection(double mass);

  ~G4SQInelasticCrossSection();

  virtual G4bool IsElementApplicable(const G4DynamicParticle* aPart, G4int Z, const G4Material*);

  virtual G4double GetElementCrossSection(const G4DynamicParticle*, G4int Z, const G4Material*);

  G4double GetSQCrossSection(G4double kineticEnergy, G4int Z);

private:
  G4NistManager* nist;
  G4SQ* theSQ;
  G4AntiSQ* theAntiSQ;
};

#endif
