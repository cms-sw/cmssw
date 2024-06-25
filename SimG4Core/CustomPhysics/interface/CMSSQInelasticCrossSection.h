
#ifndef CMSSQInelasticCrossSection_h
#define CMSSQInelasticCrossSection_h

#include "globals.hh"
#include "G4VCrossSectionDataSet.hh"

class G4NistManager;
class CMSSQ;
class CMSAntiSQ;

class CMSSQInelasticCrossSection : public G4VCrossSectionDataSet {
public:
  CMSSQInelasticCrossSection(double mass);

  ~CMSSQInelasticCrossSection();

  virtual G4bool IsElementApplicable(const G4DynamicParticle* aPart, G4int Z, const G4Material*);

  virtual G4double GetElementCrossSection(const G4DynamicParticle*, G4int Z, const G4Material*);

  G4double GetSQCrossSection(G4double kineticEnergy, G4int Z);

private:
  G4NistManager* nist;
  CMSSQ* theSQ;
  CMSAntiSQ* theAntiSQ;
};

#endif
