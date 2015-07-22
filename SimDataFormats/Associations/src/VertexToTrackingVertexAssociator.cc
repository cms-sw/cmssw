#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"

reco::VertexToTrackingVertexAssociator::VertexToTrackingVertexAssociator():
  m_impl{nullptr}
{}

reco::VertexToTrackingVertexAssociator::VertexToTrackingVertexAssociator(std::unique_ptr<reco::VertexToTrackingVertexAssociatorBaseImpl> iImpl):
  m_impl{iImpl.release()}
{}

reco::VertexToTrackingVertexAssociator::~VertexToTrackingVertexAssociator() {
  delete m_impl;
}
