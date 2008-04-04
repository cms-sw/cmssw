/*
 *  TrackHistory.h
 *  CMSSW_1_3_1
 *
 *  Created by Victor Eduardo Bazterra on 7/13/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TrackHistory_h
#define TrackHistory_h

#include <vector>

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"

//! This class trace the simulted and generated history of a given track.
class TrackHistory {

public:

  //! GenParticle trail type.
  typedef std::vector<const HepMC::GenParticle *> GenParticleTrail;

  //! GenVertex trail type.
  typedef std::vector<const HepMC::GenVertex *> GenVertexTrail;
  
  //! Void constructor.
  TrackHistory() 
  {
    depth_ = -1;
  }

  //! Constructor by depth.
  /* Creates a TrackHistory with any given depth. Positive values
     constrain the number of TrackingVertex visit in the history. 
     Negatives values set the limit of the iteration over generated 
     information i.e. (-1 -> status 1 or -2 -> status 2 particles).
  
     /param[in] depth the history
  */
  TrackHistory(int depth) 
  {
    depth_ = depth;
  }

  //! Set the depth of the history.
  void depth(int d)
  {
    depth_ = d;
  }

  //! Evaluate track history using a TrackingParticleRef.  
  /* Return false when the history cannot be determined upto a given depth.
     If not depth is pass to the function no restriction are apply to it.
     
     /param[in] TrackingParticleRef of a simulated track
     /param[in] depth of the track history
     /param[out] boolean that is true when history can be determined
  */
  bool evaluate(TrackingParticleRef tpr) 
  {
    resetTrails(tpr);
    return traceSimHistory(tpr, depth_);
  }
  
  //! Evaluate reco::Track history using a given association.
  /* Return false when the track association is not possible (fake track).
  
     /param[in] TrackRef to a reco::track
     /param[in] associator to be used in the mapping between reco and simulated tracks
     /param[in] boolean that if it is set true chose higher match
     /param[in] depth of the track history
     /param[out] boolean that is false when a fake track is detected
  */
  bool evaluate (reco::TrackRef, reco::RecoToSimCollection const &, bool maxMatch = true);

  //! Return all the simulated vertexes in the history.
  TrackingVertexContainer simVertexTrail()
  {
    return simVertexTrail_;
  }

  //! Return all the simulated particle in the history.
  TrackingParticleRefVector simParticleTrail()
  {
    return simParticleTrail_;
  }

  //! Return all generated vertex in the history.
  GenVertexTrail genVertexTrail()
  {
    return genVertexTrail_;
  }
  
  //! Return all generated particle in the history.
  GenParticleTrail genParticleTrail()
  {
    return genParticleTrail_;
  }

protected:

  int depth_;
  
  GenVertexTrail genVertexTrail_;
  GenParticleTrail genParticleTrail_;
  TrackingVertexContainer simVertexTrail_; 
  TrackingParticleRefVector simParticleTrail_;
  
private:

  void resetTrails(TrackingParticleRef tpr)
  { 
    // save the initial particle in the trail
    simParticleTrail_.clear();
    simParticleTrail_.push_back(tpr);
    
    // clear the remaining trails
    simVertexTrail_.clear();
    genVertexTrail_.clear();
    genParticleTrail_.clear();  
  }

  void traceGenHistory (const HepMC::GenParticle *);
  bool traceSimHistory (TrackingParticleRef, int);
};

#endif
