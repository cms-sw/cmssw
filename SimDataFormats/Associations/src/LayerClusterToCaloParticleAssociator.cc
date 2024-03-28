// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"

ticl::LayerClusterToCaloParticleAssociator::LayerClusterToCaloParticleAssociator(
    std::unique_ptr<ticl::LayerClusterToCaloParticleAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
