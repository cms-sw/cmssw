// -*- C++ -*-
//
// Package:    PrimaryVertexProducer
// Class:      DummyPrimaryVertexProducer
// 
/**\class DummyPrimaryVertexProducer DummyPrimaryVertexProducer.cc RecoVertex/DummyPrimaryVertexProducer/src/DummyPrimaryVertexProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Pascal Vanlaer
//         Created:  Tue Feb 28 11:06:34 CET 2006
// $Id: DummyPrimaryVertexProducer.h,v 1.1 2006/02/28 11:45:39 vanlaer Exp $
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

//
// class decleration
//

class DummyPrimaryVertexProducer : public edm::EDProducer {
   public:
      explicit DummyPrimaryVertexProducer(const edm::ParameterSet&);
      ~DummyPrimaryVertexProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
};

