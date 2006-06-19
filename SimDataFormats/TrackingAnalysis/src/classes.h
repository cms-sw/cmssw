#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    TrackingParticleCollection c1;
    std::vector<TrackingVertex> c2;
    edm::Wrapper<TrackingParticleCollection> w1; 
    edm::Wrapper<TrackingVertexCollection> w2; 
    TrackingParticleRef r1;
    TrackingParticleRefVector rv1;
    TrackingParticleRefProd rp1;
    std::vector<HepMC::GenVertex> c3;
    edm::RefVector< std::vector<HepMC::GenVertex> > rv2;
    edm::Ref< std::vector<HepMC::GenVertex> > r2;
    edm::RefProd<std::vector<HepMC::GenVertex> > rp2;
  }
}
