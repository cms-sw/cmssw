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
#include "Validation/MuonGEMDigis/interface/GEMDigiTrackMatch.h"
#include "Validation/MuonGEMDigis/plugins/MuonGEMDigis.h"
#include <vector>


MuonGEMDigis::MuonGEMDigis(const edm::ParameterSet& ps)
{
  hasGEMGeometry_ = false;

  stripLabel_ = ps.getParameter<edm::InputTag>("stripLabel");
  cscPadLabel_ = ps.getParameter<edm::InputTag>("cscPadLabel");
  cscCopadLabel_ = ps.getParameter<edm::InputTag>("cscCopadLabel");
  simInputLabel_ = ps.getUntrackedParameter<std::string>("simInputLabel", "g4SimHits");
  simTrackMatching_ = ps.getParameterSet("simTrackMatching");
  
  dbe_ = edm::Service<DQMStore>().operator->();
  outputFile_ =  ps.getParameter<std::string>("outputFile");

  theGEMStripDigiValidation  = new  GEMStripDigiValidation(dbe_, stripLabel_ );
  theGEMCSCPadDigiValidation = new GEMCSCPadDigiValidation(dbe_, cscPadLabel_ );
  theGEMCSCCoPadDigiValidation = new GEMCSCCoPadDigiValidation(dbe_, cscCopadLabel_ );
  theGEMDigiTrackMatch = new GEMDigiTrackMatch(dbe_, simInputLabel_ , simTrackMatching_ );
}


MuonGEMDigis::~MuonGEMDigis()
{
  delete theGEMStripDigiValidation;
  delete theGEMCSCPadDigiValidation;
  delete theGEMCSCCoPadDigiValidation;
  delete theGEMDigiTrackMatch;
}


void
MuonGEMDigis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  try{
    iSetup.get<MuonGeometryRecord>().get(gem_geo_);
    gem_geometry_ = &*gem_geo_;
    hasGEMGeometry_ = true;
  }
  catch(edm::eventsetup::NoProxyException<GEMGeometry>& e){
    edm::LogError("MuonGEMDigis") << "+++ Error : GEM geometry is unavailable. +++\n";
    return;
  }
  if ( hasGEMGeometry_) { 
    theGEMStripDigiValidation->setGeometry(gem_geometry_);
    theGEMStripDigiValidation->analyze(iEvent,iSetup ); 
 
    theGEMCSCPadDigiValidation->setGeometry(gem_geometry_);
    theGEMCSCPadDigiValidation->analyze(iEvent,iSetup );  

    theGEMCSCCoPadDigiValidation->setGeometry(gem_geometry_);
    theGEMCSCCoPadDigiValidation->analyze(iEvent,iSetup ); 

    theGEMDigiTrackMatch->setGeometry(gem_geometry_);
    theGEMDigiTrackMatch->analyze(iEvent,iSetup) ;
  }
}


void 
MuonGEMDigis::beginJob()
{
}


void 
MuonGEMDigis::endJob() 
{
}


void 
MuonGEMDigis::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
  dbe_->setCurrentFolder("MuonGEMDigisV/GEMDigiTask");
  try{
    iSetup.get<MuonGeometryRecord>().get(gem_geo_);
    gem_geometry_ = &*gem_geo_;
    hasGEMGeometry_ = true;
  }
  catch(edm::eventsetup::NoProxyException<GEMGeometry>& e){
    edm::LogError("MuonGEMDigis") << "+++ Error : GEM geometry is unavailable. +++\n";
    return;
  }

  if ( hasGEMGeometry_ ) {
    theGEMStripDigiValidation->bookHisto(gem_geometry_);
    theGEMCSCPadDigiValidation->bookHisto(gem_geometry_);
    theGEMCSCCoPadDigiValidation->bookHisto(gem_geometry_);
    theGEMDigiTrackMatch->bookHisto();
  }
}


void 
MuonGEMDigis::endRun(edm::Run const&, edm::EventSetup const&)
{
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


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
