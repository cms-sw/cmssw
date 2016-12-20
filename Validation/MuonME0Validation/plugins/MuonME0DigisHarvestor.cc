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
#include <DataFormats/GEMDigi/interface/ME0DigiPreReco.h>
#include <DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h>
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

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DQMServices/Core/interface/DQMStore.h"

///Log messages
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Validation/MuonME0Validation/plugins/MuonME0DigisHarvestor.h"


MuonME0DigisHarvestor::MuonME0DigisHarvestor(const edm::ParameterSet& ps)
{
  dbe_path_ = std::string("MuonME0DigisV/ME0DigisTasks/");
  outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "myfile.root");
}


MuonME0DigisHarvestor::MuonME0DigisHarvestor()
{
}

TProfile* MuonME0DigisHarvestor::ComputeEff(TH1F* num, TH1F* denum )
{
  std::string name  = "eff_"+std::string(num->GetName());
  std::string title = "Eff. "+std::string(num->GetTitle());
  TProfile * efficHist = new TProfile(name.c_str(), title.c_str(),denum->GetXaxis()->GetNbins(), denum->GetXaxis()->GetXmin(),denum->GetXaxis()->GetXmax());

  for (int i=1; i <= denum->GetNbinsX(); i++) {

    double nNum = num->GetBinContent(i);
    double nDenum = denum->GetBinContent(i);
    if ( nDenum == 0 || nNum ==0  ) {
      continue;
    }
    if ( nNum > nDenum ) {
      double temp = nDenum;
      nDenum = nNum;
      nNum = temp;
      std::cout<<"Alert! specific bin's num is bigger than denum"<<std::endl;
    }
    const double effVal = nNum/nDenum;
    efficHist->SetBinContent(i, effVal);
    efficHist->SetBinEntries(i,1);
    efficHist->SetBinError(i,0);
    const double errLo = TEfficiency::ClopperPearson((int)nDenum,(int)nNum,0.683,false);
    const double errUp = TEfficiency::ClopperPearson((int)nDenum,(int)nNum,0.683,true);
    const double errVal = (effVal - errLo > errUp - effVal) ? effVal - errLo : errUp - effVal;
    efficHist->SetBinError(i, sqrt(effVal * effVal + errVal * errVal));
  }
  return efficHist;
}


void MuonME0DigisHarvestor::ProcessBooking( DQMStore::IBooker& ibooker, DQMStore::IGetter& ig, const char* label, TString suffix, TH1F* num, TH1F* den )
{
    
  if( num !=nullptr && den !=nullptr ) {
      
    TProfile* profile = ComputeEff(num, den);
    TProfile* profile_sh = ComputeEff( hist, sh_hist );

    TString x_axis_title = TString(num->GetXaxis()->GetTitle());
    TString title  = TString::Format("Digi Efficiency;%s;Eff.",x_axis_title.Data());

    profile->SetTitle( title.Data());
    ibooker.bookProfile( profile->GetName(),profile);
      
  }
  else {
      
    std::cout<<"Can not find histograms of "<<dbe_label<<std::endl;
    if ( num == nullptr) std::cout<<"num not found"<<std::endl;
    if ( den == nullptr) std::cout<<"den not found"<<std::endl;
      
  }
  return;
    
}


void 
MuonME0DigisHarvestor::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& ig)
{
  ig.setCurrentFolder(dbe_path_.c_str());
 
  const char* l_suffix[6] = {"_l1","_l2","_l3","_l4","_l5","_l6"};
  const char* r_suffix[2] = {"-1","1"};

  TH1F* num_vs_eta_tot[2];
  TH1F* num_vs_eta[2][6];
  TH1F* den_vs_eta_tot[2];
  TH1F* den_vs_eta[2][6];
  
  for( int i = 0 ; i < 2 ; i++) {
      
    TString eta_label_den_tot = TString(dbe_path_)+"me0_dg_strip_den_eta_tot";
    TString eta_label_num_tot = TString(dbe_path_)+"me0_dg_strip_num_eta_tot";
    if( ig.get(eta_label_num_tot.Data()) !=nullptr && ig.get(eta_label_den_tot.Data()) !=nullptr ) {
        
        num_vs_eta_tot[i] = (TH1F*)ig.get(eta_label_num_tot.Data())->getTH1F()->Clone();
        num_vs_eta_tot[i]->Sumw2();
        den_vs_eta_tot[i] = (TH1F*)ig.get(eta_label_den_tot.Data())->getTH1F()->Clone();
        den_vs_eta_tot[i]->Sumw2();
        
    }
    else std::cout<<"Can not find histograms: "<<eta_label_num<<" or "<<eta_label_den<<std::endl;
    ProcessBooking( ibooker, ig, "rh_eta", suffix, num_vs_eta_tot[i], den_vs_eta_tot[i]);
    
    if ( ig.get(eta_label_num.Data()) != nullptr && ig.get(eta_label_den.Data()) != nullptr ) {
        
      for( int j = 0; j < 6 ; j++) {
          
        TString eta_label_den = TString(dbe_path_)+"me0_dg_strip_den_eta"+r_suffix[i]+l_suffix[j];
        TString eta_label_num = TString(dbe_path_)+"me0_dg_strip_num_eta"+r_suffix[i]+l_suffix[j];
        std::cout<<eta_label_num<<" "<<eta_label_den<<std:endl;

        if( ig.get(eta_label_num.Data()) !=nullptr && ig.get(eta_label_den.Data()) !=nullptr) {
            
          num_vs_eta[i][j] = (TH1F*)ig.get(eta_label_num.Data())->getTH1F()->Clone();
          num_vs_eta[i][j]->Sumw2();
          den_vs_eta[i][j] = (TH1F*)ig.get(eta_label_den.Data())->getTH1F()->Clone();
          den_vs_eta[i][j]->Sumw2();
            
        }
        else std::cout<<"Can not find histograms: "<<eta_label_num<<" "<<eta_label_den<<std::endl;
        ProcessBooking( ibooker, ig, "rh_eta", suffix, num_vs_eta[i][j], den_vs_eta[i][j]);

      }
    }
    else std::cout<<"Can not find eta or phi of all track"<<std::endl;
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(MuonME0DigisHarvestor);
