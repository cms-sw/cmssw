#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackVertexMap.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

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
    TrackingVertexRef       tv_r;
    TrackingVertexRefVector tv_rv;
    TrackingVertexRefProd   tv_rp;
    
    TrackVertexAssociation                         tva;
    TrackVertexAssociationRef                      tva_r;
    TrackVertexAssociationRefVector                tva_rv;
    TrackVertexAssociationRefProd                  tva_rp;
    TrackVertexAssociationCollection               tva_c;
    edm::Wrapper<TrackVertexAssociationCollection> tva_w;

    VertexTrackAssociation                         vta;
    VertexTrackAssociationRef                      vta_r;
    VertexTrackAssociationRefVector                vta_rv;
    VertexTrackAssociationRefProd                  vta_rp;
    VertexTrackAssociationCollection               vta_c;
    edm::Wrapper<VertexTrackAssociationCollection> vta_w;
  }
}
