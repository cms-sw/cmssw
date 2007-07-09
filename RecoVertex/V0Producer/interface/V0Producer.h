// -*- C++ -*-
//
// Package:    V0Producer
// Class:      V0Producer
// 
/**\class V0Producer V0Producer.h RecoVertex/V0Producer/interface/V0Producer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Drell
//         Created:  Fri May 18 22:57:40 CEST 2007
// $Id: V0Producer.h,v 1.1 2007/07/05 12:25:39 drell Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/V0Candidate/interface/V0Candidate.h"

#include "RecoVertex/V0Producer/interface/V0Fitter.h"

class V0Producer : public edm::EDProducer {
public:
  explicit V0Producer(const edm::ParameterSet&);
  ~V0Producer();

private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  edm::ParameterSet theParams;

  // Track reconstruction algorithm label, so we know which tracks
  //  to pull from the Event to find Vees with
  std::string trackRecoAlgo;

  // Options to use the KalmanVertexFitter to refit tracks to the vertex
  //  and whether or not to store the full refitted tracks in the V0Candidate
  int useSmoothedTrax;
  int storeSmoothedTrax;

  // Parameters to select which V0 particles to reconstruct
  int reconstructKshorts;
  int reconstructLambdas;

  // Parameters for post-vertex-fit cuts:

  // Vertex chi2 cut
  double chi2Cut;
  // Vertex radius cut
  double rVtxCut;
  // Vertex significance cut (r_vtx / sigma(r_vtx))
  double vtxSigCut;
  // Particle collinearity cut (for lambda0)
  double collinCut;
  // Kshort mass width (will cut above and below by this amount)
  double kShortMassCut;
  double lambdaMassCut;
      
};
