#ifndef SimG4Core_GenParticleInfoExtractor_H
#define SimG4Core_GenParticleInfoExtractor_H

#include "SimG4Core/Notification/interface/GenParticleInfo.h"

class G4PrimaryParticle;

class GenParticleInfoExtractor 
{
public:
    const GenParticleInfo& operator()(const G4PrimaryParticle * p) const;
};

#endif
