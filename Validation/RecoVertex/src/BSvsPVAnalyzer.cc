// -*- C++ -*-
//
// Package:    Validation/RecoVertex
// Class:      BSvsPVAnalyzer
// 
/**\class BSvsPVAnalyzer BSvsPVAnalyzer.cc Validation/RecoVertex/plugins/BSvsPVAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Mon Oct 27 17:37:53 CET 2008
// $Id: BSvsPVAnalyzer.cc,v 1.1 2011/03/08 17:11:26 venturia Exp $
//
//


// system include files
#include <memory>

// user include files

#include <vector>
#include <map>
#include <limits>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "Validation/RecoVertex/interface/BSvsPVHistogramMaker.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//
// class decleration
//

class BSvsPVAnalyzer : public edm::EDAnalyzer {
   public:
      explicit BSvsPVAnalyzer(const edm::ParameterSet&);
      ~BSvsPVAnalyzer();


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  virtual void endJob() ;

      // ----------member data ---------------------------

  BSvsPVHistogramMaker _bspvhm;
  edm::InputTag _pvcollection;
  edm::InputTag _bscollection;
  bool _firstOnly;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
BSvsPVAnalyzer::BSvsPVAnalyzer(const edm::ParameterSet& iConfig):
  _bspvhm(iConfig.getParameter<edm::ParameterSet>("bspvHistogramMakerPSet")),
  _pvcollection(iConfig.getParameter<edm::InputTag>("pvCollection")),
  _bscollection(iConfig.getParameter<edm::InputTag>("bsCollection")),
  _firstOnly(iConfig.getUntrackedParameter<bool>("firstOnly",false))
{
   //now do what ever initialization is needed

  //

  _bspvhm.book();

}


BSvsPVAnalyzer::~BSvsPVAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
BSvsPVAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  // get BS

  Handle<reco::BeamSpot> bs;
  iEvent.getByLabel(_bscollection,bs);

  // get PV

  Handle<reco::VertexCollection> pvcoll;
  iEvent.getByLabel(_pvcollection,pvcoll);

  if(_firstOnly) {
    reco::VertexCollection firstpv;
    if(pvcoll->size()) firstpv.push_back((*pvcoll)[0]);
    _bspvhm.fill(iEvent,firstpv,*bs);
  }
  else {
    _bspvhm.fill(iEvent,*pvcoll,*bs);
  }
}


// ------------ method called once each job just before starting event loop  ------------
void 
BSvsPVAnalyzer::beginJob()
{ }

void
BSvsPVAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {

  _bspvhm.beginRun(iRun.run());

}

void
BSvsPVAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {

}
// ------------ method called once each job just after ending the event loop  ------------
void 
BSvsPVAnalyzer::endJob() {
}


//define this as a plug-in
DEFINE_FWK_MODULE(BSvsPVAnalyzer);
