#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include <CLHEP/HepMC/GenParticle.h>

TrackingParticle::TrackingParticle(  const HepMC::GenParticle * p ) :
  genParticle_( p ), pdgId_( p->pdg_id() ) {
}
