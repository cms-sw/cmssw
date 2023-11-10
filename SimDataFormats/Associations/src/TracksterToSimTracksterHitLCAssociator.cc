#include "SimDataFormats/Associations/interface/TracksterToSimTracksterHitLCAssociator.h"

ticl::TracksterToSimTracksterHitLCAssociator::TracksterToSimTracksterHitLCAssociator(
    std::unique_ptr<ticl::TracksterToSimTracksterHitLCAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
