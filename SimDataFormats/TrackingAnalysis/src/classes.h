#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    TrackingParticleCollection c1;
    std::vector<TrackingVertex> c2;
    edm::Wrapper<TrackingParticleCollection> w1; 
    edm::Wrapper<TrackingVertexContainer> w2; 
    TrackingParticleRef r1;
    TrackingParticleRefVector rv1;
    TrackingParticleRefProd rp1;
  }
}
