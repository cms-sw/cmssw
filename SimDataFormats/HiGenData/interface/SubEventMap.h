#ifndef SimDataFormats_SubEventMap_h
#define SimDataFormats_SubEventMap_h

#include <vector>
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace edm {
   typedef AssociationMap<OneToValue<reco::GenParticleCollection, int, int> > SubEventMap;
}

#endif
