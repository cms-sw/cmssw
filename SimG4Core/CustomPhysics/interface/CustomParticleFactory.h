#ifndef CustomParticleFactory_h
#define CustomParticleFactory_h 1

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/CustomPhysics/interface/CustomParticle.h"
#include <set>
#include <string>

class G4DecayTable;
// ######################################################################
// ###                          CustomParticle                                ###
// ######################################################################

class CustomParticleFactory {

 public:
  static void loadCustomParticles(const std::string & filePath);
  static bool isCustomParticle(G4ParticleDefinition *particle);

 private:

  static void addCustomParticle(int pdgCode, double mass, const std::string & name );
  static void getMassTable(std::ifstream *configFile);
  static G4DecayTable* getDecayTable(std::ifstream *configFile, int pdgId);
  static G4DecayTable* getAntiDecayTable(int pdgId, G4DecayTable *theDecayTable);

  static bool loaded;
  static std::set<G4ParticleDefinition *> m_particles;
  
};

#endif
