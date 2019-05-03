// -*- C++ -*-
//
// Package:     SimDataFormats/Associations
// Class  :     TrackToTrackingParticleAssociator
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 31 Dec 2014 14:47:11 GMT
//

// system include files

// user include files
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

reco::TrackToTrackingParticleAssociator::TrackToTrackingParticleAssociator(
    std::unique_ptr<reco::TrackToTrackingParticleAssociatorBaseImpl> iImpl)
    : m_impl{std::move(iImpl)} {}
