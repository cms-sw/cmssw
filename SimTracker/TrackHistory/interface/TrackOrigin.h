/*
 *  TrackOrigin.h
 *  CMSSW_1_3_1
 *
 *  Created by Victor Eduardo Bazterra on 5/29/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TrackOrigin_h
#define TrackOrigin_h

#include <set>

#include "SimTracker/TrackHistory/interface/TrackHistory.h"

//! This class implement trace the generated origins of a given track.
class TrackOrigin : public TrackHistory {

public:

  //! Constructor by pset.
  /* Creates a TrackHistory with association given by pset.
  
     /param[in] pset with the consfiguration values
  */
  TrackOrigin(const edm::ParameterSet & iConfig) : TrackHistory(iConfig) 
  {
    // Set the history depth upto status 1 particles
    depth(-1);
  }
  
  //! Returns a pointer to most primitive status 1 or 2 particle.
  const HepMC::GenParticle * particle() const
  {
    if ( genParticleTrail_.empty() ) return 0;
    return genParticleTrail_[genParticleTrail_.size()-1];
  }

  //! Verify if any of the given particles exist in the track history.
  /* The input is a set of ints with all the pdgid of the particle 
     to verified.
     
     /param[in]: set with all pdgid of the particle to verify.
  */ 
  bool hasParticles(std::set<int> const & list) const;
};

#endif
