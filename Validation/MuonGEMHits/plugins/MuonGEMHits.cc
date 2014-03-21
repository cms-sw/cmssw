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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

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

#include "Validation/MuonGEMHits/interface/SimTrackMatchManager.h"
#include "Validation/MuonGEMHits/plugins/MuonGEMHits.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DQMServices/Core/interface/DQMStore.h"

/// ROOT
#include "TTree.h"
#include "TFile.h"


MuonGEMHits::MuonGEMHits(const edm::ParameterSet& ps)
{
  hasGEMGeometry_ = false;

  simInputLabel_ = ps.getUntrackedParameter<std::string>("simInputLabel","g4SimHits"); 

  dbe_ = edm::Service<DQMStore>().operator->();
  outputFile_ =  ps.getParameter<std::string>("outputFile");

  theGEMHitsValidation = new GEMHitsValidation(dbe_, edm::InputTag(simInputLabel_,"MuonGEMHits") );
  theGEMSimTrackMatch  = new GEMSimTrackMatch(dbe_, simInputLabel_ , ps.getParameterSet("simTrackMatching") );
}


MuonGEMHits::~MuonGEMHits()
{
  delete theGEMHitsValidation;
  delete theGEMSimTrackMatch;
}


void
MuonGEMHits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  hasGEMGeometry_ = false;
  try{
    iSetup.get<MuonGeometryRecord>().get(gem_geom);
    gem_geometry_ = &*gem_geom;
    hasGEMGeometry_ = true;
  } 
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
      edm::LogError("MuonGEMHits") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
      return;
  }
  if ( hasGEMGeometry_ ) {
    theGEMHitsValidation->setGeometry(gem_geometry_);
    theGEMHitsValidation->analyze(iEvent,iSetup );  
    theGEMSimTrackMatch->setGeometry(gem_geometry_);
    theGEMSimTrackMatch->analyze(iEvent,iSetup );  
  }
}


void 
MuonGEMHits::beginJob()
{
}


void 
MuonGEMHits::endJob() 
{
}


void 
MuonGEMHits::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
  try{
    iSetup.get<MuonGeometryRecord>().get(gem_geom);
    gem_geometry_ = &*gem_geom;
    hasGEMGeometry_ = true;
  } 
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
      edm::LogError("MuonGEMHits") << "+++ Error : GEM geometry is unavailable. +++\n";
      return;
  }
  dbe_->setCurrentFolder("MuonGEMHitsV/GEMHitsTask");
  if ( hasGEMGeometry_) { 
    theGEMHitsValidation->bookHisto(gem_geometry_);
    theGEMSimTrackMatch->bookHisto();   // GEMSimTrackMatch needs not Geometry information for booking histogram.
  }
}


void 
MuonGEMHits::endRun(edm::Run const&, edm::EventSetup const&)
{
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


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
