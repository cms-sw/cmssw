// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/TracksterToSimTracksterAssociator.h"

ticl::TracksterToSimTracksterAssociator::TracksterToSimTracksterAssociator(
    std::unique_ptr<ticl::TracksterToSimTracksterAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
