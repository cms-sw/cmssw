#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"

reco::VertexToTrackingVertexAssociator::VertexToTrackingVertexAssociator(
    std::unique_ptr<reco::VertexToTrackingVertexAssociatorBaseImpl> iImpl)
    : m_impl{std::move(iImpl)} {}
