// -*- C++ -*-
//
// Package:    MuonGEMHits
// Class:      MuonGEMHits
// 
/**\class MuonGEMHits MuonGEMHits.cc Validation/MuonGEMHits/plugins/MuonGEMHits.cc

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

#include "Validation/MuonGEMHits/interface/MuonGEMHits.h"

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

#include "Validation/MuonGEMHits/interface/SimTrackMatchManager.h"




//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonGEMHits::MuonGEMHits(const edm::ParameterSet& ps)
{
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder("MuonGEMHitsV/GEMHitsTask");
  outputFile_ =  ps.getParameter<std::string>("outputFile");

   //now do what ever initialization is needed
  
  std::string simInputLabel_ = ps.getUntrackedParameter<std::string>("simInputLabel","g4SimHits"); 
  theGEMHitsValidation = new GEMHitsValidation(dbe_, edm::InputTag(simInputLabel_,"MuonGEMHits"),ps.getParameterSet("gemSystemSetting") );
  theGEMSimTrackMatch  = new GEMSimTrackMatch(dbe_, simInputLabel_ , ps.getParameterSet("simTrackMatching") );
}



MuonGEMHits::~MuonGEMHits()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)


  delete theGEMHitsValidation;
  delete theGEMSimTrackMatch;


}





//
// member functions
//

// ------------ method called for each event  ------------
void
MuonGEMHits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  theGEMHitsValidation->analyze(iEvent,iSetup );  
  theGEMSimTrackMatch->analyze(iEvent,iSetup );  

 
  



}


// ------------ method called once each job just before starting event loop  ------------

void 
MuonGEMHits::beginJob()
{


}

// ------------ method called once each job just after ending the event loop  ------------

void 
MuonGEMHits::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------

void 
MuonGEMHits::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{

  iSetup.get<MuonGeometryRecord>().get(gem_geom);
  gem_geometry_ = &*gem_geom;



  theGEMHitsValidation->setGeometry(gem_geometry_);
  theGEMSimTrackMatch->setGeometry(gem_geometry_);




}


// ------------ method called when ending the processing of a run  ------------
void 
MuonGEMHits::endRun(edm::Run const&, edm::EventSetup const&)
{
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MuonGEMHits::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGEMHits);
