// -*- C++ -*-
//
// Package:    PileupVertexAccumulator
// Class:      PileupVertexAccumulator
// 
/**\class PileupVertexAccumulator PileupVertexAccumulator.cc SimTracker/PileupVertexAccumulator/src/PileupVertexAccumulator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author: Mike Hildreth - Notre Dame
//         Created:  Wed Jan 21 05:14:48 CET 2015
//
//


// system include files
#include <memory>
#include <set>

// user include files
#include "PileupVertexAccumulator.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupVertexContent.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/Exception.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
//using namespace std;


namespace cms
{
  PileupVertexAccumulator::PileupVertexAccumulator(const edm::ParameterSet& iConfig, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC)
  {
    edm::LogInfo ("PixelDigitizer ") <<"Enter the Pixel Digitizer";
    
    const std::string alias ("PileupVertexAccum"); 
    
    mixMod.produces<PileupVertexContent>().setBranchAlias(alias);

    Mtag_=edm::InputTag("generator");
    iC.consumes<edm::HepMCProduct>(Mtag_);
  }
  
  PileupVertexAccumulator::~PileupVertexAccumulator(){  
  }


  //
  // member functions
  //
  
  void
  PileupVertexAccumulator::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {
    // Make sure that the first crossing processed starts indexing the minbias events from zero.

    pT_Hats_.clear();
    z_posns_.clear();
  }

  void
  PileupVertexAccumulator::accumulate(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
    // don't do anything for hard-scatter signal events
  }

  void
  PileupVertexAccumulator::accumulate(PileUpEventPrincipal const& iEvent, edm::EventSetup const& iSetup, edm::StreamID const& streamID) {

    edm::Handle<edm::HepMCProduct> MCevt;
    iEvent.getByLabel(Mtag_, MCevt);

    const HepMC::GenEvent *myGenEvent = MCevt->GetEvent();

    double pthat = myGenEvent->event_scale();
    float pt_hat = float(pthat);

    pT_Hats_.push_back(pt_hat);

    HepMC::GenEvent::vertex_const_iterator viter;
    HepMC::GenEvent::vertex_const_iterator vbegin = myGenEvent->vertices_begin();
    HepMC::GenEvent::vertex_const_iterator vend = myGenEvent->vertices_end();

    // for production point, pick first vertex
    viter=vbegin; 

    if(viter!=vend){
      // The origin vertex (turn it to cm's from GenEvent mm's)
      HepMC::GenVertex* v = *viter;   
      float zpos = v->position().z()*0.1;
 
      z_posns_.push_back(zpos);
    }

    //    delete myGenEvent;

  }

  // ------------ method called to produce write the data  ------------
  void
  PileupVertexAccumulator::finalizeEvent(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    std::auto_ptr<PileupVertexContent> PUVtxC(new PileupVertexContent(pT_Hats_, z_posns_));

    // write output to event
    iEvent.put(PUVtxC);
  }


}// end namespace cms::

