// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociator.h"

ticl::LayerClusterToSimTracksterAssociator::LayerClusterToSimTracksterAssociator(
    std::unique_ptr<ticl::LayerClusterToSimTracksterAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
