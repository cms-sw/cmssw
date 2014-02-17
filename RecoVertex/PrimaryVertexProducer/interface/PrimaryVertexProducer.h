// -*- C++ -*-
//
// Package:    PrimaryVertexProducer
// Class:      PrimaryVertexProducer
// 
/**\class PrimaryVertexProducer PrimaryVertexProducer.cc RecoVertex/PrimaryVertexProducer/src/PrimaryVertexProducer.cc

 Description: steers tracker primary vertex reconstruction and storage

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Pascal Vanlaer
//         Created:  Tue Feb 28 11:06:34 CET 2006
// $Id: PrimaryVertexProducer.h,v 1.12 2011/11/09 17:23:36 dpiparo Exp $
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

//#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducerAlgorithm.h"
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

class PrimaryVertexProducer : public edm::EDProducer {
public:
  explicit PrimaryVertexProducer(const edm::ParameterSet&);
  ~PrimaryVertexProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

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
