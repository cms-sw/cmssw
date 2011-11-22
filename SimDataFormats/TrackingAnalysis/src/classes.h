#include "SimDataFormats/TrackingAnalysis/interface/ParticleBase.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace
{
struct dictionary
{

    std::vector<ParticleBase> dummy100;
    edm::Wrapper<std::vector<ParticleBase> > dummy101;

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
    TrackingVertexCollection::iterator tvcIt1;
    TrackingVertexCollection::const_iterator tvcIt2;
    edm::Wrapper<TrackingVertex> dummy2;
    edm::Wrapper<TrackingVertexCollection> w2;
    edm::Wrapper<TrackingVertexContainer> dummy3;
    TrackingVertexRef       tv_r;
    TrackingVertexRefVector tv_rv;
    TrackingVertexRefProd   tv_rp;

    std::vector<PSimHit>::const_iterator hcIt1;
    std::vector<PSimHit>::iterator hcIt2;

    edm::Wrapper<TrackingParticleRefVector> wrv1;
};
}
