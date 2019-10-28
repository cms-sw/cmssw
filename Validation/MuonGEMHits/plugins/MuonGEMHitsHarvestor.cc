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

#include "Validation/MuonGEMHits/plugins/MuonGEMHitsHarvestor.h"
#include "Validation/MuonGEMHits/interface/GEMDetLabel.h"

using namespace GEMDetLabel;
using namespace std;
MuonGEMHitsHarvestor::MuonGEMHitsHarvestor(const edm::ParameterSet& ps) {
  dbe_path_ = std::string("MuonGEMHitsV/GEMHitsTask/");
  outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "myfile.root");
}

MuonGEMHitsHarvestor::~MuonGEMHitsHarvestor() {}
TProfile* MuonGEMHitsHarvestor::ComputeEff(TH1F* num, TH1F* denum) {
  if (num == nullptr || denum == nullptr) {
    std::cout << "num or denum are missing" << std::endl;
    return nullptr;
  }
  if (num->GetNbinsX() != denum->GetNbinsX()) {
    std::cout << "Wrong Xbin. Please, check histogram's name" << std::endl;
    return nullptr;
  }
  std::string name = "eff_" + std::string(num->GetName());
  std::string title = "Eff. " + std::string(num->GetTitle());
  TProfile* efficHist = new TProfile(
      name.c_str(), title.c_str(), num->GetXaxis()->GetNbins(), num->GetXaxis()->GetXmin(), num->GetXaxis()->GetXmax());
  for (int i = 1; i <= num->GetNbinsX(); i++) {
    const double nNum = num->GetBinContent(i);
    const double nDenum = denum->GetBinContent(i);
    if (nDenum == 0 || nNum > nDenum)
      continue;
    if (nNum == 0)
      continue;
    const double effVal = nNum / nDenum;
    const double errLo = TEfficiency::ClopperPearson((int)nDenum, (int)nNum, 0.683, false);
    const double errUp = TEfficiency::ClopperPearson((int)nDenum, (int)nNum, 0.683, true);
    const double errVal = (effVal - errLo > errUp - effVal) ? effVal - errLo : errUp - effVal;
    efficHist->SetBinContent(i, effVal);
    efficHist->SetBinEntries(i, 1);
    efficHist->SetBinError(i, sqrt(effVal * effVal + errVal * errVal));
  }
  return efficHist;
}
void MuonGEMHitsHarvestor::ProcessBooking(
    DQMStore::IBooker& ibooker, DQMStore::IGetter& ig, std::string label_suffix, TH1F* track_hist, TH1F* sh_hist) {
  TString dbe_label = TString(dbe_path_) + label_suffix;
  //std::cout<<dbe_label<<"   "<<track_hist->GetName()<<std::endl;
  if (ig.get(dbe_label.Data()) != nullptr && track_hist != nullptr) {
    TH1F* hist = (TH1F*)ig.get(dbe_label.Data())->getTH1F()->Clone();
    TProfile* profile = ComputeEff(hist, track_hist);
    TString x_axis_title = TString(hist->GetXaxis()->GetTitle());
    TString title = TString::Format(
        "Eff. for a SimTrack to have an associated GEM Strip in %s;%s;Eff.", label_suffix.c_str(), x_axis_title.Data());
    profile->SetTitle(title.Data());
    ibooker.bookProfile(profile->GetName(), profile);
    if (sh_hist != nullptr) {
      TProfile* profile_sh = ComputeEff(hist, sh_hist);
      profile_sh->SetName((profile->GetName() + std::string("_sh")).c_str());
      TString title2 =
          TString::Format("Eff. for a SimTrack to have an associated GEM Strip in %s with a matched SimHit;%s;Eff.",
                          label_suffix.c_str(),
                          x_axis_title.Data());
      profile_sh->SetTitle(title2.Data());
      ibooker.bookProfile(profile_sh->GetName(), profile_sh);
    }
  }
  return;
}

void MuonGEMHitsHarvestor::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& ig) {
  ig.setCurrentFolder(dbe_path_);
  TH1F* track_eta[3];
  TH1F* track_phi[3][3];

  for (unsigned int i = 0; i < 3; i++) {
    track_eta[i] = nullptr;
    for (unsigned int j = 0; j < 3; j++) {
      track_phi[i][j] = nullptr;
    }
  }

  for (unsigned int i = 0; i < s_suffix.size(); i++) {
    string suffix = s_suffix[i];
    string track_eta_name = dbe_path_ + "track_eta" + suffix;
    if (ig.get(track_eta_name) != nullptr)
      track_eta[i] = (TH1F*)ig.get(track_eta_name)->getTH1F()->Clone();
    for (unsigned int j = 0; j < l_suffix.size(); j++) {
      suffix = s_suffix[i] + l_suffix[j];
      ProcessBooking(ibooker, ig, "sh_eta" + suffix, track_eta[i]);
    }
  }
  for (unsigned int i = 0; i < s_suffix.size(); i++) {
    for (unsigned int j = 0; j < c_suffix.size(); j++) {
      string suffix = s_suffix[i] + c_suffix[j];
      string track_phi_name = dbe_path_ + "track_phi" + suffix;
      if (ig.get(track_phi_name) != nullptr)
        track_phi[i][j] = (TH1F*)ig.get(track_phi_name)->getTH1F()->Clone();
      for (unsigned int k = 0; k < l_suffix.size(); k++) {
        suffix = s_suffix[i] + l_suffix[k] + c_suffix[j];
        ProcessBooking(ibooker, ig, "sh_phi" + suffix, track_phi[i][j]);
      }
    }
  }
}
