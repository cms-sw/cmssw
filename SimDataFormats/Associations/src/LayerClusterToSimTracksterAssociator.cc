// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociator.h"

hgcal::LayerClusterToSimTracksterAssociator::LayerClusterToSimTracksterAssociator(
    std::unique_ptr<hgcal::LayerClusterToSimTracksterAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
