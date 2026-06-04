#include "SimDataFormats/Associations/interface/MtdRecoMergedClusterToSimMergedClusterAssociator.h"

reco::MtdRecoMergedClusterToSimMergedClusterAssociator::MtdRecoMergedClusterToSimMergedClusterAssociator(
    std::unique_ptr<reco::MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
