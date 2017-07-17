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


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
reco::TrackToTrackingParticleAssociator::TrackToTrackingParticleAssociator():
  m_impl{nullptr}
{
}

reco::TrackToTrackingParticleAssociator::TrackToTrackingParticleAssociator(std::unique_ptr<reco::TrackToTrackingParticleAssociatorBaseImpl> iImpl):
  m_impl{iImpl.release()}
{
}

// TrackToTrackingParticleAssociator::TrackToTrackingParticleAssociator(const TrackToTrackingParticleAssociator& rhs)
// {
//    // do actual copying here;
// }

reco::TrackToTrackingParticleAssociator::~TrackToTrackingParticleAssociator()
{
  delete m_impl;
}

//
// assignment operators
//
// const TrackToTrackingParticleAssociator& TrackToTrackingParticleAssociator::operator=(const TrackToTrackingParticleAssociator& rhs)
// {
//   //An exception safe implementation is
//   TrackToTrackingParticleAssociator temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
