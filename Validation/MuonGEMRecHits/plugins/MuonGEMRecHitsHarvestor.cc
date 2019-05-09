// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TFile.h"
#include "TGraphAsymmErrors.h"
#include "TTree.h"

/// Data Format
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

/// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "DQMServices/Core/interface/DQMStore.h"

/// Log messages
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Validation/MuonGEMHits/interface/GEMDetLabel.h"
#include "Validation/MuonGEMRecHits/plugins/MuonGEMRecHitsHarvestor.h"

MuonGEMRecHitsHarvestor::MuonGEMRecHitsHarvestor(const edm::ParameterSet &ps) {
  dbe_path_ = std::string("MuonGEMRecHitsV/GEMRecHitsTask/");
  outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "myfile.root");
}

MuonGEMRecHitsHarvestor::~MuonGEMRecHitsHarvestor() {}
TProfile *MuonGEMRecHitsHarvestor::ComputeEff(TH1F *num, TH1F *denum) {
  std::string name = "eff_" + std::string(num->GetName());
  std::string title = "Eff. " + std::string(num->GetTitle());
  TProfile *efficHist = new TProfile(name.c_str(),
                                     title.c_str(),
                                     denum->GetXaxis()->GetNbins(),
                                     denum->GetXaxis()->GetXmin(),
                                     denum->GetXaxis()->GetXmax());

  for (int i = 1; i <= denum->GetNbinsX(); i++) {
    double nNum = num->GetBinContent(i);
    double nDenum = denum->GetBinContent(i);
    if (nDenum == 0 || nNum == 0) {
      continue;
    }
    if (nNum > nDenum) {
      double temp = nDenum;
      nDenum = nNum;
      nNum = temp;
      std::cout << "Alert! specific bin's num is bigger than denum" << std::endl;
    }
    const double effVal = nNum / nDenum;
    efficHist->SetBinContent(i, effVal);
    efficHist->SetBinEntries(i, 1);
    efficHist->SetBinError(i, 0);
    const double errLo = TEfficiency::ClopperPearson((int)nDenum, (int)nNum, 0.683, false);
    const double errUp = TEfficiency::ClopperPearson((int)nDenum, (int)nNum, 0.683, true);
    const double errVal = (effVal - errLo > errUp - effVal) ? effVal - errLo : errUp - effVal;
    efficHist->SetBinError(i, sqrt(effVal * effVal + errVal * errVal));
  }
  return efficHist;
}

void MuonGEMRecHitsHarvestor::ProcessBooking(DQMStore::IBooker &ibooker,
                                             DQMStore::IGetter &ig,
                                             const char *label,
                                             TString suffix,
                                             TH1F *track_hist,
                                             TH1F *sh_hist) {
  TString dbe_label = TString(dbe_path_) + label + suffix;
  if (ig.get(dbe_label.Data()) != nullptr && sh_hist != nullptr && track_hist != nullptr) {
    TH1F *hist = (TH1F *)ig.get(dbe_label.Data())->getTH1F()->Clone();
    TProfile *profile = ComputeEff(hist, track_hist);
    TProfile *profile_sh = ComputeEff(hist, sh_hist);
    profile_sh->SetName((profile->GetName() + std::string("_sh")).c_str());
    TString x_axis_title = TString(hist->GetXaxis()->GetTitle());
    TString title = TString::Format(
        "Eff. for a SimTrack to have an associated GEM RecHits in %s;%s;Eff.", suffix.Data(), x_axis_title.Data());
    TString title2 = TString::Format(
        "Eff. for a SimTrack to have an associated GEM RecHits "
        "in %s with a matched SimHit;%s;Eff.",
        suffix.Data(),
        x_axis_title.Data());
    profile->SetTitle(title.Data());
    profile_sh->SetTitle(title2.Data());
    ibooker.bookProfile(profile->GetName(), profile);
    ibooker.bookProfile(profile_sh->GetName(), profile_sh);
  } else {
    std::cout << "Can not found histogram of " << dbe_label << std::endl;
    if (track_hist == nullptr)
      std::cout << "track not found" << std::endl;
    if (sh_hist == nullptr)
      std::cout << "sh_hist not found" << std::endl;
  }
  return;
}

void MuonGEMRecHitsHarvestor::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &ig) {
  ig.setCurrentFolder(dbe_path_);

  using namespace GEMDetLabel;

  TH1F *gem_trk_eta[s_suffix.size()];
  TH1F *gem_trk_phi[s_suffix.size()][c_suffix.size()];

  TH1F *sh_eta[s_suffix.size()][l_suffix.size()];
  TH1F *sh_phi[s_suffix.size()][l_suffix.size()][c_suffix.size()];

  for (unsigned int i = 0; i < s_suffix.size(); i++) {
    TString eta_label = TString(dbe_path_) + "track_eta" + s_suffix[i];
    TString phi_label;
    if (ig.get(eta_label.Data()) != nullptr) {
      gem_trk_eta[i] = (TH1F *)ig.get(eta_label.Data())->getTH1F()->Clone();
      gem_trk_eta[i]->Sumw2();
    } else
      std::cout << "Can not found track_eta" << std::endl;
    for (unsigned int k = 0; k < c_suffix.size(); k++) {
      phi_label = TString(dbe_path_.c_str()) + "track_phi" + s_suffix[i] + c_suffix[k];
      if (ig.get(phi_label.Data()) != nullptr) {
        gem_trk_phi[i][k] = (TH1F *)ig.get(phi_label.Data())->getTH1F()->Clone();
        gem_trk_phi[i][k]->Sumw2();
      } else
        std::cout << "Can not found track_phi" << std::endl;
    }

    if (ig.get(eta_label.Data()) != nullptr && ig.get(phi_label.Data()) != nullptr) {
      for (unsigned int j = 0; j < l_suffix.size(); j++) {
        TString suffix = TString(s_suffix[i]) + TString(l_suffix[j]);
        TString eta_label = TString(dbe_path_) + "rh_sh_eta" + suffix;
        if (ig.get(eta_label.Data()) != nullptr) {
          sh_eta[i][j] = (TH1F *)ig.get(eta_label.Data())->getTH1F()->Clone();
          sh_eta[i][j]->Sumw2();
        } else
          std::cout << "Can not found eta histogram : " << eta_label << std::endl;
        ProcessBooking(ibooker, ig, "rh_eta", suffix, gem_trk_eta[i], sh_eta[i][j]);
        for (unsigned int k = 0; k < c_suffix.size(); k++) {
          suffix = TString(s_suffix[i]) + TString(l_suffix[j]) + TString(c_suffix[k]);
          TString phi_label = TString(dbe_path_) + "rh_sh_phi" + suffix;
          if (ig.get(phi_label.Data()) != nullptr) {
            sh_phi[i][j][k] = (TH1F *)ig.get(phi_label.Data())->getTH1F()->Clone();
            sh_phi[i][j][k]->Sumw2();
          } else {
            std::cout << "Can not found phi plots : " << phi_label << std::endl;
            continue;
          }
          ProcessBooking(ibooker, ig, "rh_phi", suffix, gem_trk_phi[i][k], sh_phi[i][j][k]);
        }
      }
    } else
      std::cout << "Can not find eta or phi of all track" << std::endl;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(MuonGEMRecHitsHarvestor);
