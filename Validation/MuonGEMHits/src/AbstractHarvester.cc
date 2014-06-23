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
#include "Validation/MuonGEMHits/interface/AbstractHarvester.h"


TProfile* AbstractHarvester::ComputeEff(TH1F* num, TH1F* denum )
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

void AbstractHarvester::ProcessBooking( std::string label_suffix, TH1F* track_hist, TH1F* sh_hist )
{
  TString dbe_label = TString(dbe_path_)+label_suffix;
  if( dbe_->get(dbe_label.Data()) != nullptr && track_hist !=nullptr ) {
    TH1F* hist =    (TH1F*)dbe_->get( dbe_label.Data() )->getTH1F()->Clone();
    TProfile* profile = ComputeEff( hist, track_hist);
    TString x_axis_title = TString(hist->GetXaxis()->GetTitle());
    TString title  = TString::Format("Eff. for a SimTrack to have an associated GEM Strip in %s;%s;Eff.",label_suffix.c_str(),x_axis_title.Data());
    profile->SetTitle( title.Data());
    dbe_->bookProfile( profile->GetName(),profile);
		if ( sh_hist!=nullptr) {
	    TProfile* profile_sh = ComputeEff( hist, sh_hist );
	    profile_sh->SetName( (profile->GetName()+std::string("_sh")).c_str());
	    TString title2 = TString::Format("Eff. for a SimTrack to have an associated GEM Strip in %s with a matched SimHit;%s;Eff.",label_suffix.c_str(),x_axis_title.Data() );
	    profile_sh->SetTitle( title2.Data() );
	    dbe_->bookProfile( profile_sh->GetName(),profile_sh);
  	}
	}
  return;
}

void
AbstractHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

