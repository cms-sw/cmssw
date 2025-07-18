// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/TracksterToSimClusterAssociator.h"

ticl::TracksterToSimClusterAssociator::TracksterToSimClusterAssociator(
    std::unique_ptr<ticl::TracksterToSimClusterAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
