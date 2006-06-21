#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

typedef edm::RefVector< std::vector<TrackingParticle> > TrackingParticleContainer;
typedef edm::RefVector< std::vector<TrackingVertex> >   TrackingVertexContainer;

namespace {
  namespace {
    
    TrackingParticle dummy10;
    TrackingParticleContainer dummy11;
    TrackingParticleCollection c1;
    edm::Wrapper<TrackingParticle> dummy12;
    edm::Wrapper<TrackingParticleContainer> dummy13;
    edm::Wrapper<TrackingParticleCollection> w1; 
    TrackingParticleRef r1;
    TrackingParticleRefVector rv1;
    TrackingParticleRefProd rp1;

    TrackingVertex dummy0;
    TrackingVertexContainer dummy1;
    TrackingVertexCollection c2;
    edm::Wrapper<TrackingVertex> dummy2;
    edm::Wrapper<TrackingVertexCollection> w2; 
    edm::Wrapper<TrackingVertexContainer> dummy3;
//    TrackingVertexRef tvr;
//    TrackingVertexRefVector tvrv;
//    TrackingVertexRefProd tvrp;
     
//    std::vector<HepMC::GenVertex> c3;
//    edm::RefVector< std::vector<HepMC::GenVertex> > rv2;
//    edm::Ref< std::vector<HepMC::GenVertex> > r2;
//    edm::RefProd<std::vector<HepMC::GenVertex> > rp2;
  }
}
