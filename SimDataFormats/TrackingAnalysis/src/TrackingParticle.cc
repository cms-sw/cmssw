#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

TrackingParticle::TrackingParticle(  const HepMC::GenParticle * p ) :
  genParticle_( p ) {
}
