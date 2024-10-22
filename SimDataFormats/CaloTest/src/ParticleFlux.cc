#include "SimDataFormats/CaloTest/interface/ParticleFlux.h"

void ParticleFlux::addFlux(ParticleFlux::flux f) { fluxVector_.push_back(f); }

void ParticleFlux::clear() { fluxVector_.clear(); }
