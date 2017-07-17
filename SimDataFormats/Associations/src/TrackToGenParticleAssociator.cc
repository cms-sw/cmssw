// -*- C++ -*-
//
// Package:     SimDataFormats/Associations
// Class  :     TrackToGenParticleAssociator
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 05 Jan 2015 14:50:48 GMT
//

// system include files

// user include files
#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociator.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
reco::TrackToGenParticleAssociator::TrackToGenParticleAssociator()
{
}

reco::TrackToGenParticleAssociator::TrackToGenParticleAssociator(std::unique_ptr<TrackToGenParticleAssociatorBaseImpl> impl): m_impl(impl.release())
{
}

reco::TrackToGenParticleAssociator::~TrackToGenParticleAssociator()
{
  delete m_impl;
}

