// -*- C++ -*-
//
// Package:    DumpL1TrackerHits
// Class:      DumpL1TrackerHits
// 
/**\class DumpL1TrackerHits DumpL1TrackerHits.cc L1TriggerOffline/L1Trigger/src/DumpL1TrackerHits.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Mon Nov 26 00:16:29 CET 2007
// $Id: TrackTriggerHitsPrinter.cc,v 1.2 2010/02/03 09:46:37 arose Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "SimDataFormats/SLHC/interface/TrackTriggerCollections.h"

//
// class decleration
//

class DumpL1TrackerHits : public edm::EDAnalyzer {
public:
  explicit DumpL1TrackerHits(const edm::ParameterSet&);
  ~DumpL1TrackerHits();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  edm::InputTag inputTag_;

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
DumpL1TrackerHits::DumpL1TrackerHits(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
   inputTag_ = iConfig.getParameter<edm::InputTag>("inputTag");

}


DumpL1TrackerHits::~DumpL1TrackerHits()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
DumpL1TrackerHits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<TrackTriggerHitCollection> in;
   iEvent.getByLabel(inputTag_,in);

   TrackTriggerHitCollection::const_iterator itr;

   for (itr=in->begin(); itr!=in->end(); ++itr) {
     LogDebug("TrackTriggerHits") << (*itr) << std::endl;
     std::cout << (*itr) << std::endl;
   }

}


// ------------ method called once each job just before starting event loop  ------------
void 
DumpL1TrackerHits::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DumpL1TrackerHits::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpL1TrackerHits);

