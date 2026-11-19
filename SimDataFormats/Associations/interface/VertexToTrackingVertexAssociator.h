#ifndef SimDataFormats_Associations_VertexToTrackingVertexAssociator_h
#define SimDataFormats_Associations_VertexToTrackingVertexAssociator_h

#include "SimDataFormats/Associations/interface/VertexAssociation.h"

#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociatorBaseImpl.h"
#include "DataFormats/Common/interface/Uninitialized.h"

namespace reco {
  template <typename VertexCollection>
  class VertexToTrackingVertexAssociator {
  public:
    using VertexType = typename VertexCollection::value_type;
    using SimToRecoCollection = VertexToTrackingVertexAssociatorBaseImpl<VertexCollection>::SimToRecoCollection;
    using RecoToSimCollection = VertexToTrackingVertexAssociatorBaseImpl<VertexCollection>::RecoToSimCollection;

#ifndef __GCCXML__
    VertexToTrackingVertexAssociator(std::unique_ptr<reco::VertexToTrackingVertexAssociatorBaseImpl<VertexCollection>>);
#endif
    VertexToTrackingVertexAssociator() = delete;
    explicit VertexToTrackingVertexAssociator(edm::Uninitialized) noexcept {};
    VertexToTrackingVertexAssociator(VertexToTrackingVertexAssociator &&) = default;
    VertexToTrackingVertexAssociator &operator=(VertexToTrackingVertexAssociator &&) = default;
    VertexToTrackingVertexAssociator(const VertexToTrackingVertexAssociator &) = delete;  // stop default
    const VertexToTrackingVertexAssociator &operator=(const VertexToTrackingVertexAssociator &) =
        delete;  // stop default

    ~VertexToTrackingVertexAssociator() = default;

    // ---------- const member functions ---------------------
    /// compare reco to sim the handle of reco::Vertex and TrackingVertex
    /// collections
    RecoToSimCollection associateRecoToSim(const edm::Handle<edm::View<VertexType>> &vCH,
                                           const edm::Handle<TrackingVertexCollection> &tVCH) const {
      return m_impl->associateRecoToSim(vCH, tVCH);
    }

    /// compare reco to sim the handle of reco::Vertex and TrackingVertex
    /// collections
    SimToRecoCollection associateSimToReco(const edm::Handle<edm::View<VertexType>> &vCH,
                                           const edm::Handle<TrackingVertexCollection> &tVCH) const {
      return m_impl->associateSimToReco(vCH, tVCH);
    }

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<VertexToTrackingVertexAssociatorBaseImpl<VertexCollection>> m_impl;
  };
}  // namespace reco

#endif
