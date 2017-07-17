// -*- C++ -*-
//
// Package:     SimDataFormats/Associations
// Class  :     MuonToTrackingParticleAssociator
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 07 Jan 2015 21:15:13 GMT
//

// system include files

// user include files
#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociator.h"


//
// constructors and destructor
//
reco::MuonToTrackingParticleAssociator::MuonToTrackingParticleAssociator():
  impl_(nullptr)
{
}

reco::MuonToTrackingParticleAssociator::MuonToTrackingParticleAssociator(std::unique_ptr<MuonToTrackingParticleAssociatorBaseImpl> iImpl):
  impl_(iImpl.release())
{
}

reco::MuonToTrackingParticleAssociator::~MuonToTrackingParticleAssociator()
{
  delete impl_;
}

