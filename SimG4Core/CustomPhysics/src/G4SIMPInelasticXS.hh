 
#ifndef G4SIMPInelasticXS_h
#define G4SIMPInelasticXS_h 1

#include "G4VCrossSectionDataSet.hh"
#include "globals.hh"
#include "G4ElementData.hh"
#include <vector>

const G4int MAXZINEL = 93;

class G4DynamicParticle;
class G4ParticleDefinition;
class G4Element;
class G4PhysicsVector;
class G4GlauberGribovCrossSection;
class G4HadronNucleonXsc;

class G4SIMPInelasticXS : public G4VCrossSectionDataSet
{
public: 

  G4SIMPInelasticXS();

  virtual ~G4SIMPInelasticXS();

  virtual
  G4bool IsElementApplicable(const G4DynamicParticle*, G4int Z,
			     const G4Material*);

  virtual
  G4bool IsIsoApplicable(const G4DynamicParticle*, G4int Z, G4int A,
			 const G4Element*, const G4Material*);

  virtual
  G4double GetElementCrossSection(const G4DynamicParticle*, 
				  G4int Z, const G4Material* mat=0);

  virtual
  G4double GetIsoCrossSection(const G4DynamicParticle*, G4int Z, G4int A,
                              const G4Isotope* iso,
                              const G4Element* elm,
                              const G4Material* mat);

  virtual G4Isotope* SelectIsotope(const G4Element*, G4double kinEnergy);

  virtual
  void BuildPhysicsTable(const G4ParticleDefinition&);

  virtual void CrossSectionDescription(std::ostream&) const;

private: 

  void Initialise(G4int Z, G4DynamicParticle* dp = 0, const char* = 0);

  G4PhysicsVector* RetrieveVector(std::ostringstream& in, G4bool warn);

  G4double IsoCrossSection(G4double ekin, G4int Z, G4int A);

  G4SIMPInelasticXS & operator=(const G4SIMPInelasticXS &right);
  G4SIMPInelasticXS(const G4SIMPInelasticXS&);
  
  G4GlauberGribovCrossSection* ggXsection;
  G4HadronNucleonXsc* fNucleon;

  const G4ParticleDefinition* proton;

  G4ElementData data;
  std::vector<G4PhysicsVector*> work;
  std::vector<G4double>         temp;
  std::vector<G4double>         coeff;

  G4bool  isInitialized;

  static const G4int amin[MAXZINEL];
  static const G4int amax[MAXZINEL];
};

#endif
