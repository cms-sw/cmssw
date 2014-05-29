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
  dbe_->cd();
  dbe_->setCurrentFolder("MuonGEMDigisV/GEMDigisTask");
  
  std::vector< std::string > suffix;
  suffix.push_back("_l1");
  suffix.push_back("_l2");
  suffix.push_back("_l1or2");
  suffix.push_back("_l1and2");
  TH1F* gem_trk_phi = nullptr;  
  TH1F* gem_trk_eta = nullptr;
  TH1F* sh_eta[4]={nullptr};
  TH1F* sh_phi[4]={nullptr};
  
  if ( dbe_->get("MuonGEMDigisV/GEMDigisTask/track_phi") != nullptr && dbe_->get("MuonGEMDigisV/GEMDigisTask/track_eta") !=nullptr ) {
    gem_trk_phi = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/track_phi")->getTH1F()->Clone();
    gem_trk_eta = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/track_eta")->getTH1F()->Clone();
    gem_trk_phi->Sumw2();
    gem_trk_eta->Sumw2();
  }
  
  if( dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_sh_eta"+suffix[0]) !=nullptr && dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_sh_phi"+suffix[0]) !=nullptr ) {
    for ( int i =0 ; i< 4 ; i++) {
      sh_eta[i] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_sh_eta"+suffix[i])->getTH1F()->Clone();
      sh_eta[i]->Sumw2();
      sh_phi[i] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_sh_phi"+suffix[i])->getTH1F()->Clone();
      sh_phi[i]->Sumw2();
    }

  }
  if( dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_eta"+suffix[0]) !=nullptr && sh_eta[0] !=nullptr && gem_trk_eta !=nullptr ) {
    for ( int i =0 ; i< 4 ; i++) {
      TH1F* dg_eta =    (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_eta"+suffix[i])->getTH1F()->Clone();
			TProfile* p_dg_eta = ComputeEff( dg_eta, gem_trk_eta);
			TProfile* p_dg_eta_2 = ComputeEff( dg_eta, sh_eta[i] );
			p_dg_eta_2->SetName( (p_dg_eta_2->GetName()+std::string("_sh")).c_str());
			TString title  = TString::Format("Eff. for a SimTrack to have an associated GEM Strip in %s;|#eta|;Eff.",suffix[i].c_str());
			TString title2 = TString::Format("Eff. for a SimTrack to have an associated GEM Strip in %s with a matched SimHit;|#eta|;Eff.",suffix[i].c_str());
			p_dg_eta->SetTitle( title.Data());
			p_dg_eta_2->SetTitle( title2.Data() );
      dbe_->bookProfile( p_dg_eta->GetName(),p_dg_eta); 
      dbe_->bookProfile( p_dg_eta_2->GetName(),p_dg_eta_2); 
    }
  }
  if( dbe_->get("MuonGEMDigisV/GEMDigisTask/pad_eta"+suffix[0]) !=nullptr && sh_eta[0] !=nullptr && gem_trk_eta !=nullptr )  {
    for ( int i =0 ; i< 4 ; i++) {
      TH1F* pad_eta;
      if ( i <3) { 
        pad_eta = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/pad_eta"+suffix[i])->getTH1F()->Clone();
      }
      else       {
        pad_eta = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/copad_eta")->getTH1F()->Clone();
      }
      pad_eta->Sumw2();
      
      TProfile* p_pad_eta = ComputeEff( pad_eta, gem_trk_eta);
      TProfile* p_pad_eta_2 = ComputeEff( pad_eta, sh_eta[i]);
			p_pad_eta_2->SetName( (p_pad_eta_2->GetName()+std::string("_sh")).c_str());

			TString title  = TString::Format("Eff. for a SimTrack to have an associated GEM Pad in %s;|#eta|;Eff.",suffix[i].c_str());
			TString title2 = TString::Format("Eff. for a SimTrack to have an associated GEM Pad in %s with a matched SimHit;|#eta|;Eff.",suffix[i].c_str());
			p_pad_eta->SetTitle( title.Data() );
			p_pad_eta_2->SetTitle( title2.Data());
      dbe_->bookProfile( p_pad_eta->GetName(),p_pad_eta); 
      dbe_->bookProfile( p_pad_eta_2->GetName(),p_pad_eta_2); 
    }
  }

  if( dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_phi"+suffix[0]) !=nullptr && sh_eta[0] !=nullptr && gem_trk_phi !=nullptr) {
    for ( int i =0 ; i< 4 ; i++) {
      TH1F* dg_phi =    (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/dg_phi"+suffix[i])->getTH1F()->Clone();
      dg_phi->Sumw2();

			TProfile* p_dg_phi = ComputeEff( dg_phi, gem_trk_phi);
			TProfile* p_dg_phi_2 = ComputeEff( dg_phi, sh_phi[i]);

			p_dg_phi_2->SetName( (p_dg_phi_2->GetName()+std::string("_sh")).c_str());

			TString title  = TString::Format("Eff. for a SimTrack to have an associated GEM Strip in %s;|#phi|;Eff.",suffix[i].c_str());
			TString title2 = TString::Format("Eff. for a SimTrack to have an associated GEM Strip in %s with a matched SimHit;|#phi|;Eff.",suffix[i].c_str());
			p_dg_phi->SetTitle( title.Data() );
			p_dg_phi_2->SetTitle( title2.Data() );
      dbe_->bookProfile( p_dg_phi->GetName(), p_dg_phi) ; 
      dbe_->bookProfile( p_dg_phi_2->GetName(),p_dg_phi_2); 
    }
  }
  if( dbe_->get("MuonGEMDigisV/GEMDigisTask/pad_phi"+suffix[0]) !=nullptr && sh_phi[0] !=nullptr && gem_trk_phi !=nullptr)  {
    for ( int i =0 ; i< 4 ; i++) {
      TH1F* pad_phi;
      if ( i <3) { 
        pad_phi = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/pad_phi"+suffix[i])->getTH1F()->Clone();
      }
      else       {
        pad_phi = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigisTask/copad_phi")->getTH1F()->Clone();
      }
      pad_phi->Sumw2();
			TProfile* p_pad_phi   = ComputeEff( pad_phi, gem_trk_phi);
			TProfile* p_pad_phi_2 = ComputeEff( pad_phi, sh_phi[i]);
			p_pad_phi_2->SetName( (p_pad_phi_2->GetName()+std::string("_sh")).c_str());

			TString title  = TString::Format("Eff. for a SimTrack to have an associated GEM Pad in %s;|#phi|;Eff.",suffix[i].c_str());
			TString title2 = TString::Format("Eff. for a SimTrack to have an associated GEM Pad in %s with a matched SimHit;|#phi|;Eff.",suffix[i].c_str());
			p_pad_phi->SetTitle( title.Data());
			p_pad_phi_2->SetTitle( title2.Data()); 

      dbe_->bookProfile( p_pad_phi->GetName(),p_pad_phi); 
      dbe_->bookProfile( p_pad_phi_2->GetName(),p_pad_phi_2); 
    }
  }

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
