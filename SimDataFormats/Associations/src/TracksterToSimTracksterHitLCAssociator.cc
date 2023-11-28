#include "SimDataFormats/Associations/interface/TracksterToSimTracksterHitLCAssociator.h"

hgcal::TracksterToSimTracksterHitLCAssociator::TracksterToSimTracksterHitLCAssociator(
    std::unique_ptr<hgcal::TracksterToSimTracksterHitLCAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
