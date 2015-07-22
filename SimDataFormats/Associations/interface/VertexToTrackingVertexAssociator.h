#ifndef SimDataFormats_Associations_VertexToTrackingVertexAssociator_h
#define SimDataFormats_Associations_VertexToTrackingVertexAssociator_h

#include "SimDataFormats/Associations/interface/VertexAssociation.h"

#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociatorBaseImpl.h"

namespace reco {
  class VertexToTrackingVertexAssociator {
  public:

#ifndef __GCCXML__
    VertexToTrackingVertexAssociator( std::unique_ptr<reco::VertexToTrackingVertexAssociatorBaseImpl>);
#endif
    VertexToTrackingVertexAssociator();
    ~VertexToTrackingVertexAssociator();

    // ---------- const member functions ---------------------
    /// compare reco to sim the handle of reco::Vertex and TrackingVertex collections
    reco::VertexRecoToSimCollection associateRecoToSim(const edm::Handle<edm::View<reco::Vertex> >& vCH, 
                                                       const edm::Handle<TrackingVertexCollection>& tVCH ) const {
      return m_impl->associateRecoToSim(vCH,tVCH);
    }

    /// compare reco to sim the handle of reco::Vertex and TrackingVertex collections
    reco::VertexSimToRecoCollection associateSimToReco(const edm::Handle<edm::View<reco::Vertex> >& vCH, 
                                                       const edm::Handle<TrackingVertexCollection>& tVCH ) const {
      return m_impl->associateSimToReco(vCH,tVCH);
    }

    void swap(VertexToTrackingVertexAssociator& iOther) {
      std::swap(m_impl, iOther.m_impl);
    }
  private:
    VertexToTrackingVertexAssociator(const VertexToTrackingVertexAssociator&); // stop default
    
    const VertexToTrackingVertexAssociator& operator=(const VertexToTrackingVertexAssociator&); // stop default
    
    // ---------- member data --------------------------------
    VertexToTrackingVertexAssociatorBaseImpl* m_impl;
  };
}

#endif
