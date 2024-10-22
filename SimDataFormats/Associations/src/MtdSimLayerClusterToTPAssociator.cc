#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociator.h"

reco::MtdSimLayerClusterToTPAssociator::MtdSimLayerClusterToTPAssociator(
    std::unique_ptr<reco::MtdSimLayerClusterToTPAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
