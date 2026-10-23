#include "SimDataFormats/Associations/interface/MtdSimMergedClusterToTPAssociator.h"

reco::MtdSimMergedClusterToTPAssociator::MtdSimMergedClusterToTPAssociator(
    std::unique_ptr<reco::MtdSimMergedClusterToTPAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
