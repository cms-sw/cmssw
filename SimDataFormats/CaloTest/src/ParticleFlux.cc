#include "SimDataFormats/CaloTest/interface/ParticleFlux.h"

void ParticleFlux::addFlux(ParticleFlux::flux f) {
  fluxVector.push_back(f);
}

void ParticleFlux::clear() {
  fluxVector.clear();
}
