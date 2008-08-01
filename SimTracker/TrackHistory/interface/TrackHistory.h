/*
 *  TrackHistory.h
 *
 *  Created by Victor Eduardo Bazterra on 7/13/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TrackHistory_h
#define TrackHistory_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Framework/interface/Event.h"#include "FWCore/Framework/interface/ESHandle.h"#include "FWCore/Framework/interface/EventSetup.h"#include "FWCore/ParameterSet/interface/ParameterSet.h"

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

  //! SimParticle trail type.
  typedef std::vector<TrackingParticleRef> SimParticleTrail;

  //! SimVertex trail type.
  typedef std::vector<TrackingVertexRef> SimVertexTrail;
  
  //! Constructor by pset.
  /* Creates a TrackHistory with association given by pset.
  
     /param[in] pset with the consfiguration values
  */
  TrackHistory(const edm::ParameterSet &);
  
  //! Pre-process event information (for accessing reconstraction information)
  void newEvent(const edm::Event &, const edm::EventSetup &);

  //! Set the depth of the history.
  /* Set TrackHistory to given depth. Positive values
     constrain the number of TrackingVertex visit in the history. 
     Negatives values set the limit of the iteration over generated 
     information i.e. (-1 -> status 1 or -2 -> status 2 particles).
  
     /param[in] depth the history
  */
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
     /param[out] boolean that is false when a fake track is detected
  */
  bool evaluate (edm::RefToBase<reco::Track>);

  //! Return the initial tracking particle from the history.
  const TrackingParticleRef & simParticle() const
  {
    return simParticleTrail_[0];
  }

  //! Return all the simulated vertexes in the history.
  const SimVertexTrail & simVertexTrail() const
  {
    return simVertexTrail_;
  }

  //! Return all the simulated particle in the history.
  const SimParticleTrail & simParticleTrail() const
  {
    return simParticleTrail_;
  }

  //! Return all generated vertex in the history.
  const GenVertexTrail & genVertexTrail() const
  {
    return genVertexTrail_;
  }
  
  //! Return all generated particle in the history.
  const GenParticleTrail & genParticleTrail() const
  {
    return genParticleTrail_;
  }

protected:

  int depth_;
  
  GenVertexTrail genVertexTrail_;
  GenParticleTrail genParticleTrail_;
  SimVertexTrail simVertexTrail_; 
  SimParticleTrail simParticleTrail_;
  
private:

 bool newEvent_;

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
  
  bool bestMatchByMaxValue_;  
 
  std::string associationModule_;

  std::string recoTrackModule_;
  
  std::string trackingParticleModule_;

  std::string trackingParticleInstance_;

  reco::RecoToSimCollection association_;
  
  TrackingParticleRef match ( edm::RefToBase<reco::Track> );  
};

#endif
