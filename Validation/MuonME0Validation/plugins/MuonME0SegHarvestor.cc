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
#include "TGraphAsymmErrors.h"

/// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include "DQMServices/Core/interface/DQMStore.h"

/// Log messages
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Validation/MuonME0Validation/plugins/MuonME0SegHarvestor.h"

MuonME0SegHarvestor::MuonME0SegHarvestor(const edm::ParameterSet &ps) {
  dbe_path_ = std::string("MuonME0RecHitsV/ME0SegmentsTask/");
}

MuonME0SegHarvestor::~MuonME0SegHarvestor() {}

TProfile *MuonME0SegHarvestor::ComputeEff(TH1F *num, TH1F *denum, std::string nameHist) {
  std::string name = "eff_" + nameHist;
  std::string title = "Segment Efficiency" + std::string(num->GetTitle());
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
      edm::LogWarning("MuonME0SegHarvestor")
          << "Alert! specific bin's num is bigger than denum " << i << " " << nNum << " " << nDenum;
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

void MuonME0SegHarvestor::ProcessBooking(
    DQMStore::IBooker &ibooker, DQMStore::IGetter &ig, std::string nameHist, TH1F *num, TH1F *den) {
  if (num != nullptr && den != nullptr) {
    TProfile *profile = ComputeEff(num, den, nameHist);

    TString x_axis_title = TString(num->GetXaxis()->GetTitle());
    TString title = TString::Format("Segment Efficiency;%s;Eff.", x_axis_title.Data());

    profile->SetTitle(title.Data());
    ibooker.bookProfile(profile->GetName(), profile);

    delete profile;

  } else {
    edm::LogWarning("MuonME0SegHarvestor") << "Can not find histograms";
    if (num == nullptr)
      edm::LogWarning("MuonME0SegHarvestor") << "num not found";
    if (den == nullptr)
      edm::LogWarning("MuonME0SegHarvestor") << "den not found";
  }
  return;
}

void MuonME0SegHarvestor::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &ig) {
  ig.setCurrentFolder(dbe_path_);

  TString eta_label_den = TString(dbe_path_) + "me0_simsegment_eta";
  TString eta_label_num = TString(dbe_path_) + "me0_matchedsimsegment_eta";
  TString pt_label_den = TString(dbe_path_) + "me0_simsegment_pt";
  TString pt_label_num = TString(dbe_path_) + "me0_matchedsimsegment_pt";
  TString phi_label_den = TString(dbe_path_) + "me0_simsegment_phi";
  TString phi_label_num = TString(dbe_path_) + "me0_matchedsimsegment_phi";

  if (ig.get(eta_label_num.Data()) != nullptr && ig.get(eta_label_den.Data()) != nullptr) {
    TH1F *num_vs_eta = (TH1F *)ig.get(eta_label_num.Data())->getTH1F()->Clone();
    num_vs_eta->Sumw2();
    TH1F *den_vs_eta = (TH1F *)ig.get(eta_label_den.Data())->getTH1F()->Clone();
    den_vs_eta->Sumw2();

    ProcessBooking(ibooker, ig, "me0segment_eff_vs_eta", num_vs_eta, den_vs_eta);

    delete num_vs_eta;
    delete den_vs_eta;

  } else
    edm::LogWarning("MuonME0SegHarvestor") << "Can not find histograms: " << eta_label_num << " or " << eta_label_den;

  if (ig.get(pt_label_num.Data()) != nullptr && ig.get(pt_label_den.Data()) != nullptr) {
    TH1F *num_vs_pt = (TH1F *)ig.get(pt_label_num.Data())->getTH1F()->Clone();
    num_vs_pt->Sumw2();
    TH1F *den_vs_pt = (TH1F *)ig.get(pt_label_den.Data())->getTH1F()->Clone();
    den_vs_pt->Sumw2();

    ProcessBooking(ibooker, ig, "me0segment_eff_vs_pt", num_vs_pt, den_vs_pt);

    delete num_vs_pt;
    delete den_vs_pt;

  } else
    edm::LogWarning("MuonME0SegHarvestor") << "Can not find histograms: " << pt_label_num << " or " << pt_label_den;

  if (ig.get(phi_label_num.Data()) != nullptr && ig.get(phi_label_den.Data()) != nullptr) {
    TH1F *num_vs_phi = (TH1F *)ig.get(phi_label_num.Data())->getTH1F()->Clone();
    num_vs_phi->Sumw2();
    TH1F *den_vs_phi = (TH1F *)ig.get(phi_label_den.Data())->getTH1F()->Clone();
    den_vs_phi->Sumw2();

    ProcessBooking(ibooker, ig, "me0segment_eff_vs_phi", num_vs_phi, den_vs_phi);

    delete num_vs_phi;
    delete den_vs_phi;

  } else
    edm::LogWarning("MuonME0SegHarvestor") << "Can not find histograms: " << phi_label_num << " or " << phi_label_den;
}

// define this as a plug-in
DEFINE_FWK_MODULE(MuonME0SegHarvestor);
