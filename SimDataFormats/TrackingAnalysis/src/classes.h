#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/AssociationMapHelpers.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace
{
struct dictionary
{
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

    edm::Wrapper<TrackingParticleRefVector> wrv1;

    edm::helpers::KeyVal<edm::RefProd<std::vector<TrackingParticle> >,edm::RefToBaseProd<reco::Track> > aa;
    edm::helpers::KeyVal<edm::RefToBaseProd<reco::Track>,edm::RefProd<std::vector<TrackingParticle> > > aaa;
    std::map<unsigned int,edm::helpers::KeyVal<edm::Ref<std::vector<TrackingParticle>,TrackingParticle,edm::refhelper::FindUsingAdvance<std::vector<TrackingParticle>,TrackingParticle> >,std::vector<std::pair<edm::RefToBase<reco::Track>,double> > > > aaaa;
    std::map<unsigned int,edm::helpers::KeyVal<edm::RefToBase<reco::Track>,std::vector<std::pair<edm::Ref<std::vector<TrackingParticle>,TrackingParticle,edm::refhelper::FindUsingAdvance<std::vector<TrackingParticle>,TrackingParticle> >,double> > > > aaaaa;
    edm::helpers::KeyVal<edm::Ref<std::vector<TrackingParticle>,TrackingParticle,edm::refhelper::FindUsingAdvance<std::vector<TrackingParticle>,TrackingParticle> >,std::vector<std::pair<edm::RefToBase<reco::Track>,double> > > aaaaaaa;
    edm::helpers::KeyVal<edm::RefToBase<reco::Track>,std::vector<std::pair<edm::Ref<std::vector<TrackingParticle>,TrackingParticle,edm::refhelper::FindUsingAdvance<std::vector<TrackingParticle>,TrackingParticle> >,double> > > aaaaaaaa;
    
    std::vector<std::pair<edm::Ref<std::vector<TrackingParticle>,TrackingParticle,edm::refhelper::FindUsingAdvance<std::vector<TrackingParticle>,TrackingParticle> >,double> > aaaaaaaaa;
    std::pair<edm::Ref<std::vector<TrackingParticle>,TrackingParticle,edm::refhelper::FindUsingAdvance<std::vector<TrackingParticle>,TrackingParticle> >,double> aaaaaaaaaaa;  
  };
}
