#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"

template <typename VertexCollection>
reco::VertexToTrackingVertexAssociator<VertexCollection>::VertexToTrackingVertexAssociator(
    std::unique_ptr<reco::VertexToTrackingVertexAssociatorBaseImpl<VertexCollection>> iImpl)
    : m_impl{std::move(iImpl)} {}

template class reco::VertexToTrackingVertexAssociator<std::vector<reco::Vertex>>;
template class reco::VertexToTrackingVertexAssociator<std::vector<reco::VertexCompositePtrCandidate>>;
