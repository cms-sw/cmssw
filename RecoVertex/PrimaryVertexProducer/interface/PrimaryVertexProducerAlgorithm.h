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
// $Id: PrimaryVertexProducerAlgorithm.h,v 1.3 2006/05/22 12:32:08 vanlaer Exp $
//
//

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"
#include "RecoVertex/TrimmedKalmanVertexFinder/interface/KalmanTrimmedVertexFinder.h"
#include "RecoVertex/VertexTools/interface/VertexCompatibleWithBeam.h"

//
// class declaration
//

class PrimaryVertexProducerAlgorithm : public VertexReconstructor {
public:

  explicit PrimaryVertexProducerAlgorithm(const edm::ParameterSet&);
  ~PrimaryVertexProducerAlgorithm();
  
  /** Find primary vertices
   */
  virtual vector<TransientVertex> 
  vertices(const vector<reco::TransientTrack> & tracks) const;

  /** Clone method
   */ 
  virtual PrimaryVertexProducerAlgorithm * clone() const {
    return new PrimaryVertexProducerAlgorithm(*this);
  }
  
private:
  // ----------member data ---------------------------
  // vtx finding algorithm components
  edm::ParameterSet theConfig;
  TrackFilterForPVFinding theTrackFilter;
  TrackClusterizerInZ theTrackClusterizer;
  KalmanTrimmedVertexFinder theFinder;
  VertexCompatibleWithBeam theVertexSelector;

};
