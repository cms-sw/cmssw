#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociator.h"

reco::MtdRecoClusterToSimLayerClusterAssociator::MtdRecoClusterToSimLayerClusterAssociator(
    std::unique_ptr<reco::MtdRecoClusterToSimLayerClusterAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
