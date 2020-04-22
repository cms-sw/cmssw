// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"

hgcal::LayerClusterToCaloParticleAssociator::LayerClusterToCaloParticleAssociator(
    std::unique_ptr<hgcal::LayerClusterToCaloParticleAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
