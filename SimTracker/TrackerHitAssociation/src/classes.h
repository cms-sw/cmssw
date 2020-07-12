#include "DataFormats/Common/interface/Wrapper.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "CUDADataFormats/Common/interface/Product.h"
#include "SimTracker/TrackerHitAssociation/interface/trackerHitAssociationHeterogeneous.h"

#include "DataFormats/Common/interface/AssociationMap.h"
namespace SimTracker_TrackerHitAssociation {
  struct dictionary {
    edm::AssociationMap<edm::OneToMany<std::vector<SimTrack>, std::vector<OmniClusterRef>, unsigned int> > dummy01;
    edm::Wrapper<edm::AssociationMap<edm::OneToMany<std::vector<SimTrack>, std::vector<OmniClusterRef>, unsigned int> > >
        dummy02;
    edm::helpers::KeyVal<edm::RefProd<std::vector<SimTrack> >, edm::RefProd<std::vector<OmniClusterRef> > > dummy03;
    edm::Wrapper<edm::helpers::KeyVal<edm::RefProd<std::vector<SimTrack> >, edm::RefProd<std::vector<OmniClusterRef> > > >
        dummy04;
    std::vector<OmniClusterRef> dummy05;
    edm::Wrapper<std::vector<OmniClusterRef> > dummy06;
    std::pair<OmniClusterRef, TrackingParticleRef> dummy13;
    edm::Wrapper<std::pair<OmniClusterRef, TrackingParticleRef> > dummy14;
    ClusterTPAssociation dummy07;
    edm::Wrapper<ClusterTPAssociation> dummy08;
    std::map<TrackingParticleRef, std::vector<OmniClusterRef> > dummy09;
    edm::Wrapper<std::map<TrackingParticleRef, std::vector<OmniClusterRef> > > dummy10;
    std::map<OmniClusterRef, std::vector<TrackingParticleRef> > dummy11;
    edm::Wrapper<std::map<OmniClusterRef, std::vector<TrackingParticleRef> > > dummy12;
  };
}  // namespace SimTracker_TrackerHitAssociation
