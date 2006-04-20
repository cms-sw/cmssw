#ifndef SimG4Core_CustomParticleFactory_H
#define CSimG4Core_ustomParticleFactory_H

#include "CustomParticle.h"

#include <set>

class CustomParticleFactory 
{
private:
    static bool loaded;
    static std::set<G4ParticleDefinition *> m_particles;
public:
   static void loadCustomParticles();
   static bool isCustomParticle(G4ParticleDefinition *particle);
};

#endif
