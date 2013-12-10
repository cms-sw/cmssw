// -*- C++ -*-
//
// Package:    MuonGEMDigis
// Class:      MuonGEMDigis
// 
/**\class MuonGEMDigis MuonGEMDigis.cc Validation/MuonGEMDigis/plugins/MuonGEMDigis.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Geonmo RYU
//         Created:  Mon, 07 Oct 2013 12:45:56 GMT
//       Based on :  /GEMCode/GEMValidation/plugin/GEMDigiAnalyzer.cc
// $Id$
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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TTree.h"
#include "TFile.h"
#include "TGraphAsymmErrors.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Validation/MuonGEMDigis/interface/MuonGEMDigis.h"

///Data Format
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

///Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DQMServices/Core/interface/DQMStore.h"

///Log messages
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Validation/MuonGEMDigis/interface/SimTrackDigiMatchManager.h"
#include <vector>



//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonGEMDigis::MuonGEMDigis(const edm::ParameterSet& ps)
//  , simInputLabel_(ps.getUntrackedParameter<std::string>("simInputLabel", "g4SimHits"))
//  , verbose_(ps.getUntrackedParameter<int>("verbose", 0))
{
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder("MuonGEMDigisV/GEMDigiTask");
  outputFile_ =  ps.getParameter<std::string>("outputFile");

   //now do what ever initialization is needed
  

  theGEMStripDigiValidation  = new  GEMStripDigiValidation(dbe_, ps.getParameter<edm::InputTag>("stripLabel"));
  theGEMCSCPadDigiValidation = new GEMCSCPadDigiValidation(dbe_, ps.getParameter<edm::InputTag>("cscPadLabel"));
  theGEMCSCCoPadDigiValidation = new GEMCSCCoPadDigiValidation(dbe_, ps.getParameter<edm::InputTag>("cscCopadLabel"));
  theGEMTrackMatch = new GEMTrackMatch(dbe_, ps.getUntrackedParameter<std::string>("simInputLabel", "g4SimHits"), 
                                       ps.getParameterSet("simTrackMatching") );
  


}



MuonGEMDigis::~MuonGEMDigis()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)


  delete theGEMStripDigiValidation;
  delete theGEMCSCPadDigiValidation;
  delete theGEMCSCCoPadDigiValidation;
  delete theGEMTrackMatch;


}





//
// member functions
//

// ------------ method called for each event  ------------
void
MuonGEMDigis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  theGEMStripDigiValidation->analyze(iEvent,iSetup );  
  theGEMCSCPadDigiValidation->analyze(iEvent,iSetup );  
  theGEMCSCCoPadDigiValidation->analyze(iEvent,iSetup );  
  theGEMTrackMatch->analyze(iEvent,iSetup) ;
  

}


// ------------ method called once each job just before starting event loop  ------------

void 
MuonGEMDigis::beginJob()
{


}

// ------------ method called once each job just after ending the event loop  ------------

void 
MuonGEMDigis::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------

void 
MuonGEMDigis::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{

  iSetup.get<MuonGeometryRecord>().get(gem_geo_);
  gem_geometry_ = &*gem_geo_;

  theGEMStripDigiValidation->setGeometry(gem_geometry_);
  theGEMCSCPadDigiValidation->setGeometry(gem_geometry_);
  theGEMCSCCoPadDigiValidation->setGeometry(gem_geometry_);


  theGEMTrackMatch->setGeometry(gem_geometry_);




}


// ------------ method called when ending the processing of a run  ------------
void 
MuonGEMDigis::endRun(edm::Run const&, edm::EventSetup const&)
{

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
MuonGEMDigis::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
MuonGEMDigis::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/






// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MuonGEMDigis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGEMDigis);
