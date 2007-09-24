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

//! This class implement trace the origin of a given track.
class TrackOrigin : public TrackHistory {

public:

  //! Void constructor
  TrackOrigin() : TrackHistory() {}

  //! Constructor by depth.
  /* Creates a TrackHistory with any given depth. Positive values
     constrain the number of TrackingVertex visit in the history. 
     Negatives values set the limit of the iteration over generated 
     information i.e. (-1 -> status 1 or -2 -> status 2 particles).
  
     /param[in] depth the history
  */
  TrackOrigin(int depth) : TrackHistory(depth) {}
  
  //! Returns a pointer to most primitive status 1 or 2 particle.
  const HepMC::GenParticle * particle() const
  {
    if ( genParticleTrail_.empty() ) return 0;
    return genParticleTrail_[genParticleTrail_.size()-1];
  }

  //! Verify if the track comes from a TrackingVertex (displaced track).
  bool isDisplaced() const
  {
    return !simVertexTrail_.empty();
  }

  //! Verify if any of the given particles exist in the track history.
  /* The input is a set of ints with all the pdgid of the particle 
     to verified.
     
     /param[in]: set with all pdgid of the particle to verify.
  */ 
  bool hasParticles(std::set<int> const & list) const;

  //! Verify if there were any photon conversion in the particle history.
  bool hasPhotonConversion() const;
};

#endif
