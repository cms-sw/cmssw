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
#include "Validation/MuonGEMDigis/plugins/MuonGEMDigis_Harvesting.h"


MuonGEMDigis_Harvesting::MuonGEMDigis_Harvesting(const edm::ParameterSet& ps)
{
  dbe_ = edm::Service<DQMStore>().operator->();
}


MuonGEMDigis_Harvesting::~MuonGEMDigis_Harvesting()
{
}
TProfile* MuonGEMDigis_Harvesting::ComputeEff(TH1F* num, TH1F* denum )
{
	std::string name  = "eff_"+std::string(num->GetName());
	std::string title = "Eff. "+std::string(num->GetTitle());
	TProfile * efficHist = new TProfile(name.c_str(), title.c_str(),num->GetXaxis()->GetNbins(), num->GetXaxis()->GetXmin(),num->GetXaxis()->GetXmax());
  for (int i=1; i <= num->GetNbinsX(); i++) {

	  const double nNum = num->GetBinContent(i);
    const double nDenum = denum->GetBinContent(i);

    if ( nDenum == 0 || nNum > nDenum ) continue;
    if ( nNum == 0 ) continue;
    const double effVal = nNum/nDenum;

    const double errLo = TEfficiency::ClopperPearson((int)nDenum,(int)nNum,0.683,false);
    const double errUp = TEfficiency::ClopperPearson((int)nDenum,(int)nNum,0.683,true);
    const double errVal = (effVal - errLo > errUp - effVal) ? effVal - errLo : errUp - effVal;
    efficHist->SetBinContent(i, effVal);
    efficHist->SetBinEntries(i, 1);
    efficHist->SetBinError(i, sqrt(effVal * effVal + errVal * errVal));
  }

	//TF1 *f1 = new TF1("eff_fit", "pol0", num->GetXaxis()->GetXmin(), num->GetXaxis()->GetXmax());
	//f1->SetParameter(0,98.);
  //efficHist->Fit("eff_fit","MES");	
	return efficHist;
}

void MuonGEMDigis_Harvesting::ProcessBooking( DQMStore* dbe_, const char* label, TString suffix, TH1F* track_hist, TH1F* sh_hist )
{
	const char* dbe_path = "MuonGEMDigisV/GEMDigisTask/";
	
	TString dbe_label = TString(dbe_path)+label+suffix;
	if( dbe_->get(dbe_label.Data()) != nullptr && sh_hist !=nullptr && track_hist !=nullptr ) {
		TH1F* hist =    (TH1F*)dbe_->get( dbe_label.Data() )->getTH1F()->Clone();
		TProfile* profile = ComputeEff( hist, track_hist);
		TProfile* profile_sh = ComputeEff( hist, sh_hist );
		profile_sh->SetName( (profile->GetName()+std::string("_sh")).c_str());
		TString x_axis_title = TString(hist->GetXaxis()->GetTitle());
		TString title  = TString::Format("Eff. for a SimTrack to have an associated GEM Strip in %s;%s;Eff.",suffix.Data(),x_axis_title.Data());
		TString title2 = TString::Format("Eff. for a SimTrack to have an associated GEM Strip in %s with a matched SimHit;%s;Eff.",suffix.Data(),x_axis_title.Data() );
		profile->SetTitle( title.Data());
		profile_sh->SetTitle( title2.Data() );
	  dbe_->bookProfile( profile->GetName(),profile); 
	  dbe_->bookProfile( profile_sh->GetName(),profile_sh); 
	}
	return;
}

void
MuonGEMDigis_Harvesting::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


void 
MuonGEMDigis_Harvesting::beginJob()
{
}


void 
MuonGEMDigis_Harvesting::endJob() 
{
}


void 
MuonGEMDigis_Harvesting::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
}


void 
MuonGEMDigis_Harvesting::endRun(edm::Run const&, edm::EventSetup const&)
{
	const char* dbe_path = "MuonGEMDigisV/GEMDigisTask/";
  dbe_->cd();
  dbe_->setCurrentFolder(dbe_path);
 
	const char* l_suffix[4] = {"_l1","_l2","_l1or2","_l1and2"};
	const char* s_suffix[3] = {"_st1","_st2_short","_st2_long"};	 

  TH1F* gem_trk_phi = nullptr;  
  TH1F* gem_trk_eta = nullptr;
  TH1F* sh_eta[3][4];
  TH1F* sh_phi[3][4];
	
	if ( dbe_->get("MuonGEMDigisV/GEMDigisTask/track_phi") != nullptr && dbe_->get("MuonGEMDigisV/GEMDigisTask/track_eta") !=nullptr ) {
    gem_trk_phi = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/track_phi")->getTH1F()->Clone();
    gem_trk_eta = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/track_eta")->getTH1F()->Clone();
    gem_trk_phi->Sumw2();
    gem_trk_eta->Sumw2();
  }
	for( int i = 0 ; i < 3 ; i++) {
		for( int j = 0; j < 4 ; j++) { 
 			TString suffix = TString( l_suffix[j])+TString( s_suffix[i] );
			TString eta_label = TString(dbe_path)+"dg_sh_eta"+suffix;
			TString phi_label = TString(dbe_path)+"dg_sh_phi"+suffix;
  		if( dbe_->get(eta_label.Data() ) !=nullptr && dbe_->get(phi_label.Data()) !=nullptr ) {
				TString label = "MuonGEMDigisV/GEMDigisTask/dg_sh_eta"+suffix;
			  sh_eta[i][j] = (TH1F*)dbe_->get(label.Data())->getTH1F()->Clone();
		    sh_eta[i][j]->Sumw2();
				label = "MuonGEMDigisV/GEMDigisTask/dg_sh_phi"+suffix;
		    sh_phi[i][j] = (TH1F*)dbe_->get(label.Data())->getTH1F()->Clone();
		    sh_phi[i][j]->Sumw2();
  		}
			ProcessBooking( dbe_, "dg_eta", suffix, gem_trk_eta, sh_eta[i][j]); 
			ProcessBooking( dbe_, "pad_eta", suffix, gem_trk_eta, sh_eta[i][j]); 
			ProcessBooking( dbe_, "dg_phi",suffix, gem_trk_phi, sh_phi[i][j]);
			ProcessBooking( dbe_, "pad_phi",suffix,gem_trk_phi, sh_phi[i][j]);
  	}
	}
  /*
  if( dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_lx_even") != nullptr) { 

    TH1F* gem_dg_lx_even[5];
    gem_dg_lx_even[0] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_lx_even")->getTH1F()->Clone(); 
    gem_dg_lx_even[1] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_lx_even_l1")->getTH1F()->Clone();
    gem_dg_lx_even[2] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_lx_even_l2")->getTH1F()->Clone();
    gem_dg_lx_even[3] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_lx_even_l1or2")->getTH1F()->Clone();
    gem_dg_lx_even[4] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_lx_even_l1and2")->getTH1F()->Clone();
 
    TH1F* gem_dg_ly_even[5];
    gem_dg_ly_even[0] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_ly_even")->getTH1F()->Clone(); 
    gem_dg_ly_even[1] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_ly_even_l1")->getTH1F()->Clone();
    gem_dg_ly_even[2] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_ly_even_l2")->getTH1F()->Clone();
    gem_dg_ly_even[3] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_ly_even_l1or2")->getTH1F()->Clone();
    gem_dg_ly_even[4] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_ly_even_l1and2")->getTH1F()->Clone();
 
    TH1F* gem_dg_lx_odd[5];
    gem_dg_lx_odd[0] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_lx_odd")->getTH1F()->Clone(); 
    gem_dg_lx_odd[1] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_lx_odd_l1")->getTH1F()->Clone();
    gem_dg_lx_odd[2] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_lx_odd_l2")->getTH1F()->Clone();
    gem_dg_lx_odd[3] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_lx_odd_l1or2")->getTH1F()->Clone();
    gem_dg_lx_odd[4] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_lx_odd_l1and2")->getTH1F()->Clone();
 
    TH1F* gem_dg_ly_odd[5];
    gem_dg_ly_odd[0] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_ly_odd")->getTH1F()->Clone(); 
    gem_dg_ly_odd[1] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_ly_odd_l1")->getTH1F()->Clone();
    gem_dg_ly_odd[2] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_ly_odd_l2")->getTH1F()->Clone();
    gem_dg_ly_odd[3] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_ly_odd_l1or2")->getTH1F()->Clone();
    gem_dg_ly_odd[4] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_ly_odd_l1and2")->getTH1F()->Clone();
 
    for ( int i= 0; i<5 ; i++) {
      gem_dg_lx_even[i]->Sumw2(); 
      gem_dg_ly_even[i]->Sumw2(); 
      gem_dg_lx_odd[i]->Sumw2(); 
      gem_dg_ly_odd[i]->Sumw2(); 
 
    }
    for( int i=1 ; i<5 ; i++) {
      //gem_dg_lx_even[i]->Divide( gem_dg_lx_even[i], gem_dg_lx_even[0],1.0,1.0,"B");
			TProfile* eff_dg_lx_even = ComputeEff(gem_dg_lx_even[i],gem_dg_lx_even[0]);
			TProfile* eff_dg_ly_even = ComputeEff(gem_dg_ly_even[i],gem_dg_ly_even[0]);
			TProfile* eff_dg_lx_odd  = ComputeEff(gem_dg_lx_odd[i] ,gem_dg_lx_odd[0]);
			TProfile* eff_dg_ly_odd  = ComputeEff(gem_dg_ly_odd[i] ,gem_dg_ly_odd[0]);
      
      dbe_->bookProfile( eff_dg_lx_even->GetName(), eff_dg_lx_even ); 
      dbe_->bookProfile( eff_dg_ly_even->GetName(), eff_dg_ly_even ); 
      dbe_->bookProfile( eff_dg_lx_odd->GetName(), eff_dg_lx_odd ); 
      dbe_->bookProfile( eff_dg_ly_odd->GetName(), eff_dg_ly_odd ); 

			delete eff_dg_lx_even;
			delete eff_dg_ly_even;
			delete eff_dg_lx_odd;
			delete eff_dg_ly_odd;
    }
  }
  */

}


void
MuonGEMDigis_Harvesting::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGEMDigis_Harvesting);
