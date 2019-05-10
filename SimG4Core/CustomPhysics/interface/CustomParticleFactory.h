#ifndef SimG4Core_CustomPhysics_CustomParticleFactory_h
#define SimG4Core_CustomPhysics_CustomParticleFactory_h 1

#include "G4Threading.hh"
#include <vector>
#include <string>
#include <fstream>

class G4DecayTable;
class G4ParticleDefinition;

class CustomParticleFactory {
public:
  explicit CustomParticleFactory();
  ~CustomParticleFactory();

  void loadCustomParticles(const std::string &filePath);
  const std::vector<G4ParticleDefinition *> &GetCustomParticles();

private:
  void addCustomParticle(int pdgCode, double mass, const std::string &name);
  void getMassTable(std::ifstream *configFile);
  G4DecayTable *getDecayTable(std::ifstream *configFile, int pdgId);
  G4DecayTable *getAntiDecayTable(int pdgId, G4DecayTable *theDecayTable);
  std::string ToLower(std::string str);

  static bool loaded;
  static std::vector<G4ParticleDefinition *> m_particles;
#ifdef G4MULTITHREADED
  static G4Mutex customParticleFactoryMutex;
#endif
};

#endif
