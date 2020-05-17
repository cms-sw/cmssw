#include "SimDataFormats/CaloTest/interface/ParticleFlux.h"

void ParticleFlux::addFlux(const ParticleFlux::flux& f) { fluxVector_.push_back(f); }

void ParticleFlux::clear() { fluxVector_.clear(); }
