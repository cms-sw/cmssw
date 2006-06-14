#include "DataFormats/Common/interface/Wrapper.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimGeneral/TrackingAnalysis/interface/TrackingTruthProducer.h"
#include <vector>

typedef edm::RefVector< std::vector<TrackingParticle> > TrackingParticleContainer;

namespace { namespace {
  //say which template classes should have dictionaries
  TrackingVertex dummy0;
  TrackingVertexContainer dummy1;
  edm::Wrapper<TrackingVertex> dummy2;
  edm::Wrapper<TrackingVertexContainer> dummy3;
  TrackingParticle dummy10;
  TrackingParticleContainer dummy11;
  edm::Wrapper<TrackingParticle> dummy12;
  edm::Wrapper<TrackingParticleContainer> dummy13;
} }

