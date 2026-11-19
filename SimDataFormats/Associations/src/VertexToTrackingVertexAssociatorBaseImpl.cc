#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociatorBaseImpl.h"

template <typename VertexCollection>
reco::VertexToTrackingVertexAssociatorBaseImpl<VertexCollection>::VertexToTrackingVertexAssociatorBaseImpl() {}

template <typename VertexCollection>
reco::VertexToTrackingVertexAssociatorBaseImpl<VertexCollection>::~VertexToTrackingVertexAssociatorBaseImpl() {}

template class reco::VertexToTrackingVertexAssociatorBaseImpl<std::vector<reco::Vertex>>;
template class reco::VertexToTrackingVertexAssociatorBaseImpl<std::vector<reco::VertexCompositePtrCandidate>>;
