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

#include "Validation/MuonGEMDigis/interface/SimTrackMatchManager.h"




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
  minPt_= ps.getUntrackedParameter<double>("minPt", 5.);
  //cfg_= ps.getParameterSet("simTrackMatching");
  outputFile_ =  ps.getParameter<std::string>("outputFile");

   //now do what ever initialization is needed
  

  theGEMStripDigiValidation = new GEMStripDigiValidation(dbe_, ps.getParameter<edm::InputTag>("stripLabel"));
//  theCSCPadDigiValidation = new GEMCSCPadValidation(dbe_, ps.getParameter<edm::InputTag>("simMuonGEMCSCPadDigis"));
//  theCSCCoPadDigiValidation = new GEMCSCCoPadValidation(dbe_, ps.getParameter<edm::InputTag>("simMuonGEMCSCPadDigis","Coincidence"));
  


}



MuonGEMDigis::~MuonGEMDigis()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)


  delete theGEMStripDigiValidation;
//  delete theCSCPadDigiValidation;
//  delete theCSCCoPadDigiValidation;



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


  const auto top_chamber = static_cast<const GEMEtaPartition*>(gem_geometry_->idToDetUnit(GEMDetId(1,1,1,1,1,1)));
   // TODO: it's really bad to hardcode max partition number!
  const auto bottom_chamber = static_cast<const GEMEtaPartition*>(gem_geometry_->idToDetUnit(GEMDetId(1,1,1,1,1,6)));
  const float top_half_striplength = top_chamber->specs()->specificTopology().stripLength()/2.;
  const float bottom_half_striplength = bottom_chamber->specs()->specificTopology().stripLength()/2.;
  const LocalPoint lp_top(0., top_half_striplength, 0.);
  const LocalPoint lp_bottom(0., -bottom_half_striplength, 0.);
  const GlobalPoint gp_top = top_chamber->toGlobal(lp_top);
  const GlobalPoint gp_bottom = bottom_chamber->toGlobal(lp_bottom);

  radiusCenter_ = (gp_bottom.perp() + gp_top.perp())/2.;
  chamberHeight_ = gp_top.perp() - gp_bottom.perp();

  using namespace std;
  cout<<"half top "<<top_half_striplength<<" bot "<<lp_bottom<<endl;
  cout<<"r  top "<<gp_top.perp()<<" bot "<<gp_bottom.perp()<<endl;
  LocalPoint p0(0.,0.,0.);
  cout<<"r0 top "<<top_chamber->toGlobal(p0).perp()<<" bot "<< bottom_chamber->toGlobal(p0).perp()<<endl;
  cout<<"rch "<<radiusCenter_<<" hch "<<chamberHeight_<<endl;

  buildLUT();


}


// ------------ method called when ending the processing of a run  ------------
void 
MuonGEMDigis::endRun(edm::Run const&, edm::EventSetup const&)
{
//    if ( theDQM && ! outputFileName_.empty() ) theDQM->save(outputFileName_);
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

void MuonGEMDigis::buildLUT()
{
  std::vector<int> pos_ids;
  pos_ids.push_back(GEMDetId(1,1,1,1,36,1).rawId());

  std::vector<int> neg_ids;
  neg_ids.push_back(GEMDetId(-1,1,1,1,36,1).rawId());

  // VK: I would really suggest getting phis from GEMGeometry
  
  std::vector<float> phis;
  phis.push_back(0.);
  for(int i=1; i<37; ++i)
  {
    pos_ids.push_back(GEMDetId(1,1,1,1,i,1).rawId());
    neg_ids.push_back(GEMDetId(-1,1,1,1,i,1).rawId());
    phis.push_back(i*10.);
  }
  positiveLUT_ = std::make_pair(phis,pos_ids);
  negativeLUT_ = std::make_pair(phis,neg_ids);
}





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
