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
#include "Validation/MuonGEMHits/plugins/MuonGEMHits_Harvesting.h"

using namespace std;
MuonGEMHits_Harvesting::MuonGEMHits_Harvesting(const edm::ParameterSet& ps)
{
  dbe_ = edm::Service<DQMStore>().operator->();
	dbe_path_ = std::string("MuonGEMHitsV/GEMHitsTask/");
}


MuonGEMHits_Harvesting::~MuonGEMHits_Harvesting()
{
}

void
MuonGEMHits_Harvesting::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


void 
MuonGEMHits_Harvesting::beginJob()
{
}


void 
MuonGEMHits_Harvesting::endJob() 
{
}


void 
MuonGEMHits_Harvesting::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
}


void 
MuonGEMHits_Harvesting::endRun(edm::Run const&, edm::EventSetup const&)
{
  dbe_->cd();
  dbe_->setCurrentFolder(dbe_path_.c_str());
	string l_suffix[4] = {"_l1","_l2","_l1or2","_l1and2"};
	string s_suffix[3] = {"_st1","_st2_short","_st2_long"};
	//string c_suffix[2] = {"_even","_odd"};
  TH1F* track_eta[3];
  TH1F* track_phi[3];
	//TH1F* sh_eta[4][3];
	//TH1F* sh_phi[4][3];
	//TH1F* gem_lx[3][2];
	//TH1F* gem_ly[3][2];

	for ( int i = 0 ; i< 3 ; i++) {
		track_eta[i]=nullptr;
		track_phi[i]=nullptr;
		for ( int j= 0 ; j<4 ; j++) {
			//sh_eta[j][i]=nullptr;
			//sh_phi[j][i]=nullptr;
		}
		for ( int j=0; j<2; j++) {
		  //gem_lx[i][j]=nullptr;
		  //gem_ly[i][j]=nullptr;
		}
	}

	for(int i=0 ; i<3 ; i++) {
		string suffix = s_suffix[i];
		string track_eta_name = dbe_path_+"track_eta"+suffix;
		if ( dbe_->get(track_eta_name.c_str()) != nullptr) track_eta[i] = (TH1F*)dbe_->get(track_eta_name.c_str())->getTH1F()->Clone();
		string track_phi_name = dbe_path_+"track_phi"+suffix;
		if ( dbe_->get(track_phi_name.c_str()) != nullptr) track_phi[i] = (TH1F*)dbe_->get(track_phi_name.c_str())->getTH1F()->Clone();
		for( int j=0 ; j<4; j++) {
			suffix = l_suffix[j]+s_suffix[i];
			ProcessBooking( "sh_eta"+suffix,track_eta[i]);
			ProcessBooking( "sh_phi"+suffix,track_phi[i]);
  	}
		/*
		for( int j=0 ; j<2 ; j++) {
			suffix = s_suffix[i]+c_suffix[j]; 
			string gem_lx_title = dir+"gem_lx"+suffix;
		  if ( dbe_->get(gem_lx_title.c_str())!=nullptr ) gem_lx[i][j] = (TH1F*)dbe_->get(gem_lx_title.c_str())->getTH1F()->Clone();
			
		} 
		*/
	}
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void
MuonGEMHits_Harvesting::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGEMHits_Harvesting);
