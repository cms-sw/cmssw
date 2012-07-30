/////////////////////////   OBSOLETE    ///////////////////
#ifndef PrimaryVertexProducerAlgorithm_H
#define PrimaryVertexProducerAlgorithm_H
// -*- C++ -*-
//
// Package:    PrimaryVertexProducerAlgorithm
// Class:      PrimaryVertexProducerAlgorithm
// 
/**\class PrimaryVertexProducerAlgorithm PrimaryVertexProducerAlgorithm.cc RecoVertex/PrimaryVertexProducerAlgorithm/src/PrimaryVertexProducerAlgorithm.cc

 Description: finds primary vertices, compatible with the beam line

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Pascal Vanlaer
//         Created:  Tue Feb 28 11:06:34 CET 2006
// $Id: PrimaryVertexProducerAlgorithm.h,v 1.15 2010/08/18 13:20:24 werdmann Exp $
//
//

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFindingBase.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"
#include "RecoVertex/TrimmedKalmanVertexFinder/interface/KalmanTrimmedVertexFinder.h"
#include "RecoVertex/VertexTools/interface/VertexCompatibleWithBeam.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//
// class declaration
//

class PrimaryVertexProducerAlgorithm : public VertexReconstructor {
public:

  explicit PrimaryVertexProducerAlgorithm(const edm::ParameterSet&);
  ~PrimaryVertexProducerAlgorithm();
  

  /** Find primary vertices
   */
// obsolete method
  virtual std::vector<TransientVertex> 
  vertices(const std::vector<reco::TransientTrack> & tracks) const;

  virtual std::vector<TransientVertex> 
  vertices(const std::vector<reco::TransientTrack> & tracks, 
	   const reco::BeamSpot & beamSpot) const;

  /** Clone method
   */ 
  virtual PrimaryVertexProducerAlgorithm * clone() const {
    return new PrimaryVertexProducerAlgorithm(*this);
  }
  
private:
  // ----------member data ---------------------------
  // vtx finding algorithm components
  edm::ParameterSet theConfig;
  TrackFilterForPVFindingBase* theTrackFilter; 
  TrackClusterizerInZ* theTrackClusterizer;
  KalmanTrimmedVertexFinder theFinder;
  VertexCompatibleWithBeam theVertexSelector;

  bool fVerbose;
  bool fUseBeamConstraint;
  double fMinNdof;
  VertexFitter<5> *theFitter;
  bool fapply_finder;
  bool fFailsafe;

};
#endif

