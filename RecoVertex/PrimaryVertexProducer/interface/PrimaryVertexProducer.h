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
// $Id: PrimaryVertexProducer.h,v 1.8 2007/12/20 23:44:15 yumiceva Exp $
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
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducerAlgorithm.h"

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
  // vtx finding algorithm
  PrimaryVertexProducerAlgorithm theAlgo;

  edm::ParameterSet theConfig;
  bool fVerbose;
};
