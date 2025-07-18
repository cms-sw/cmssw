// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"

ticl::LayerClusterToSimClusterAssociator::LayerClusterToSimClusterAssociator(
    std::unique_ptr<ticl::LayerClusterToSimClusterAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
