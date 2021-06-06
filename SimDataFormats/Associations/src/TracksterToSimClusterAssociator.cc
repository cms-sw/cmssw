// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/TracksterToSimClusterAssociator.h"

hgcal::TracksterToSimClusterAssociator::TracksterToSimClusterAssociator(
    std::unique_ptr<hgcal::TracksterToSimClusterAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
