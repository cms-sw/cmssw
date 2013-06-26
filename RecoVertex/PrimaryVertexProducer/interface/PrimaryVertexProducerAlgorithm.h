/////////////////////////   OBSOLETE    ///////////////////
#ifndef PrimaryVertexProducerAlgorithm_H
#define PrimaryVertexProducerAlgorithm_H
// -*- C++ -*-
//
// Package:    PrimaryVertexProducerAlgorithm
// Class:      PrimaryVertexProducerAlgorithm
// 
/**\class PrimaryVertexProducerAlgorithm PrimaryVertexProducerAlgorithm.cc RecoVertex/PrimaryVertexProducerAlgorithm/src/PrimaryVertexProducerAlgorithm.cc

 Description: allow redoing the primary vertex reconstruction from a list of tracks, considered obsolete

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Pascal Vanlaer
//         Created:  Tue Feb 28 11:06:34 CET 2006
// $Id: PrimaryVertexProducerAlgorithm.h,v 1.17 2012/04/27 16:12:39 werdmann Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFindingBase.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"
#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZ_vect.h"

#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include "RecoVertex/PrimaryVertexProducer/interface/HITrackFilterForPVFinding.h"
#include "RecoVertex/PrimaryVertexProducer/interface/GapClusterizerInZ.h"
#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZ.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
//#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include <algorithm>
#include "RecoVertex/PrimaryVertexProducer/interface/VertexHigherPtSquared.h"
#include "RecoVertex/VertexTools/interface/VertexCompatibleWithBeam.h"

//
// class declaration
//

class PrimaryVertexProducerAlgorithm : public VertexReconstructor {
public:

  explicit PrimaryVertexProducerAlgorithm(const edm::ParameterSet&);
  ~PrimaryVertexProducerAlgorithm();
  
  // obsolete method
  virtual std::vector<TransientVertex> 
  vertices(const std::vector<reco::TransientTrack> & tracks) const;

  virtual std::vector<TransientVertex> 
  vertices(const std::vector<reco::TransientTrack> & tracks, 
	   const reco::BeamSpot & beamSpot,
	   const std::string& label=""
	   ) const;
  /** Clone method
   */ 
  virtual PrimaryVertexProducerAlgorithm * clone() const {
    return new PrimaryVertexProducerAlgorithm(*this);
  }
  

  // access to config
  edm::ParameterSet config() const { return theConfig; }
  edm::InputTag trackLabel;
  edm::InputTag beamSpotLabel;
private:
  // ----------member data ---------------------------
  TrackFilterForPVFindingBase* theTrackFilter; 
  TrackClusterizerInZ* theTrackClusterizer;

  // vtx fitting algorithms
  struct algo {
    VertexFitter<5> * fitter;
    VertexCompatibleWithBeam * vertexSelector;
    std::string  label;
    bool useBeamConstraint;
    double minNdof;
  };

  std::vector< algo > algorithms;

  edm::ParameterSet theConfig;
  bool fVerbose;

};
#endif

