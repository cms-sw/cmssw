#include "DataFormats/Common/interface/Wrapper.h"

//Add includes for your classes here
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociator.h"
#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"
#include "SimDataFormats/Associations/interface/VertexAssociation.h"

namespace SimDataFormats_Associations {
   struct SimDataFormats_Associations {
      //add 'dummy' Wrapper variable for each class type you put into the Event
      edm::Wrapper<reco::TrackToTrackingParticleAssociator> dummy1;
      edm::Wrapper<reco::TrackToGenParticleAssociator> dummy2;
      edm::Wrapper<reco::MuonToTrackingParticleAssociator> dummy3;

      edm::Wrapper<reco::VertexToTrackingVertexAssociator> dummy4;

     reco::VertexSimToRecoCollection vstrc;
     reco::VertexSimToRecoCollection::const_iterator vstrci;
     edm::Wrapper<reco::VertexSimToRecoCollection > wvstrc;

     reco::VertexRecoToSimCollection vrtsc;
     reco::VertexRecoToSimCollection::const_iterator  vrtsci;
     edm::Wrapper<reco::VertexRecoToSimCollection > wvrtsci;
   };
}
