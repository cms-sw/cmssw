// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/MultiClusterToCaloParticleAssociator.h"

hgcal::MultiClusterToCaloParticleAssociator::MultiClusterToCaloParticleAssociator(
    std::unique_ptr<hgcal::MultiClusterToCaloParticleAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
