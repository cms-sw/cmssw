// -*- C++ -*-
//
// Package:    PrimaryVertexProducer
// Class:      PrimaryVertexProducer
// 
/**\class PrimaryVertexProducer PrimaryVertexProducer.cc RecoVertex/PrimaryVertexProducer/src/PrimaryVertexProducer.cc

 Description: finds and stores primary vertices, compatible with the beam line

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Pascal Vanlaer
//         Created:  Tue Feb 28 11:06:34 CET 2006
// $Id: PrimaryVertexProducer.h,v 1.2 2006/03/13 17:16:16 vanlaer Exp $
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

#include "RecoVertex/TrimmedKalmanVertexFinder/interface/KalmanTrimmedVertexFinder.h"

//
// class decleration
//

class PrimaryVertexProducer : public edm::EDProducer {
public:
  explicit PrimaryVertexProducer(const edm::ParameterSet&);
  ~PrimaryVertexProducer();
  
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  // ----------member data ---------------------------
  // vtx finding algorithm
  KalmanTrimmedVertexFinder theFinder;
  edm::ParameterSet theConfig;

};
