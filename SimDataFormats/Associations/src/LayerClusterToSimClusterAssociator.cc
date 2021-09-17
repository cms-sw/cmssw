// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"

hgcal::LayerClusterToSimClusterAssociator::LayerClusterToSimClusterAssociator(
    std::unique_ptr<hgcal::LayerClusterToSimClusterAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
