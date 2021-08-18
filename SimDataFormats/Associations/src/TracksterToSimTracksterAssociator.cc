// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/TracksterToSimTracksterAssociator.h"

hgcal::TracksterToSimTracksterAssociator::TracksterToSimTracksterAssociator(
    std::unique_ptr<hgcal::TracksterToSimTracksterAssociatorBaseImpl> ptr)
    : m_impl(std::move(ptr)) {}
