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

#include "Validation/MuonME0Validation/plugins/MuonME0DigisHarvestor.h"

MuonME0DigisHarvestor::MuonME0DigisHarvestor(const edm::ParameterSet &ps) {
  dbe_path_ = std::string("MuonME0DigisV/ME0DigisTask/");
}

MuonME0DigisHarvestor::~MuonME0DigisHarvestor() {}

TProfile *MuonME0DigisHarvestor::ComputeEff(TH1F *num, TH1F *denum, std::string nameHist) {
  std::string name = "eff_" + nameHist;
  std::string title = "Digi Efficiency" + std::string(num->GetTitle());
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
      edm::LogWarning("MuonME0DigisHarvestor")
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

void MuonME0DigisHarvestor::ProcessBooking(
    DQMStore::IBooker &ibooker, DQMStore::IGetter &ig, std::string nameHist, TH1F *num, TH1F *den) {
  if (num != nullptr && den != nullptr) {
    TProfile *profile = ComputeEff(num, den, nameHist);

    TString x_axis_title = TString(num->GetXaxis()->GetTitle());
    TString title = TString::Format("Digi Efficiency;%s;Eff.", x_axis_title.Data());

    profile->SetTitle(title.Data());
    ibooker.bookProfile(profile->GetName(), profile);

    delete profile;

  } else {
    edm::LogWarning("MuonME0DigisHarvestor") << "Can not find histograms";
    if (num == nullptr)
      edm::LogWarning("MuonME0DigisHarvestor") << "num not found";
    if (den == nullptr)
      edm::LogWarning("MuonME0DigisHarvestor") << "den not found";
  }
  return;
}

TH1F *MuonME0DigisHarvestor::ComputeBKG(TH1F *hist1, TH1F *hist2, std::string nameHist) {
  std::string name = "rate_" + nameHist;
  hist1->SetName(name.c_str());
  for (int bin = 1; bin <= hist1->GetNbinsX(); ++bin) {
    double R_min = hist1->GetBinCenter(bin) - 0.5 * hist1->GetBinWidth(bin);
    double R_max = hist1->GetBinCenter(bin) + 0.5 * hist1->GetBinWidth(bin);

    double Area = TMath::Pi() * (R_max * R_max - R_min * R_min);
    hist1->SetBinContent(bin, (hist1->GetBinContent(bin)) / Area);
    hist1->SetBinError(bin, (hist1->GetBinError(bin)) / Area);
  }

  int nEvts = hist2->GetEntries();
  float scale = 6 * 2 * nEvts * 3 * 25e-9;  // New redigitizer saves hits only in the BX range: [-1,+1], so the
                                            // number of background hits has to be divided by 3
  hist1->Scale(1.0 / scale);
  return hist1;
}

void MuonME0DigisHarvestor::ProcessBookingBKG(
    DQMStore::IBooker &ibooker, DQMStore::IGetter &ig, std::string nameHist, TH1F *hist1, TH1F *hist2) {
  if (hist1 != nullptr && hist2 != nullptr) {
    TH1F *rate = ComputeBKG(hist1, hist2, nameHist);

    TString x_axis_title = TString(hist1->GetXaxis()->GetTitle());
    TString origTitle = TString(hist1->GetTitle());
    TString title = TString::Format((origTitle + ";%s;Rate [Hz/cm^{2}]").Data(), x_axis_title.Data());

    rate->SetTitle(title.Data());
    ibooker.book1D(rate->GetName(), rate);

  } else {
    edm::LogWarning("MuonME0DigisHarvestor") << "Can not find histograms";
    if (hist1 == nullptr)
      edm::LogWarning("MuonME0DigisHarvestor") << "num not found";
    if (hist2 == nullptr)
      edm::LogWarning("MuonME0DigisHarvestor") << "den not found";
  }
  return;
}

void MuonME0DigisHarvestor::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &ig) {
  ig.setCurrentFolder(dbe_path_);

  const char *l_suffix[6] = {"_l1", "_l2", "_l3", "_l4", "_l5", "_l6"};
  const char *r_suffix[2] = {"-1", "1"};

  TString eta_label_den_tot = TString(dbe_path_) + "me0_strip_dg_den_eta_tot";
  TString eta_label_num_tot = TString(dbe_path_) + "me0_strip_dg_num_eta_tot";
  if (ig.get(eta_label_num_tot.Data()) != nullptr && ig.get(eta_label_den_tot.Data()) != nullptr) {
    TH1F *num_vs_eta_tot = (TH1F *)ig.get(eta_label_num_tot.Data())->getTH1F()->Clone();
    num_vs_eta_tot->Sumw2();
    TH1F *den_vs_eta_tot = (TH1F *)ig.get(eta_label_den_tot.Data())->getTH1F()->Clone();
    den_vs_eta_tot->Sumw2();

    ProcessBooking(ibooker, ig, "me0_strip_dg_eta_tot", num_vs_eta_tot, den_vs_eta_tot);

    delete num_vs_eta_tot;
    delete den_vs_eta_tot;

  } else
    edm::LogWarning("MuonME0DigisHarvestor")
        << "Can not find histograms: " << eta_label_num_tot << " or " << eta_label_den_tot;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 6; j++) {
      TString eta_label_den = TString(dbe_path_) + "me0_strip_dg_den_eta" + r_suffix[i] + l_suffix[j];
      TString eta_label_num = TString(dbe_path_) + "me0_strip_dg_num_eta" + r_suffix[i] + l_suffix[j];

      if (ig.get(eta_label_num.Data()) != nullptr && ig.get(eta_label_den.Data()) != nullptr) {
        TH1F *num_vs_eta = (TH1F *)ig.get(eta_label_num.Data())->getTH1F()->Clone();
        num_vs_eta->Sumw2();
        TH1F *den_vs_eta = (TH1F *)ig.get(eta_label_den.Data())->getTH1F()->Clone();
        den_vs_eta->Sumw2();

        std::string r_s = r_suffix[i];
        std::string l_s = l_suffix[j];
        std::string name = "me0_strip_dg_eta" + r_s + l_s;
        ProcessBooking(ibooker, ig, name, num_vs_eta, den_vs_eta);

        delete num_vs_eta;
        delete den_vs_eta;

      } else
        edm::LogWarning("MuonME0DigisHarvestor")
            << "Can not find histograms: " << eta_label_num << " " << eta_label_den;
    }
  }

  TString label_eleBkg = TString(dbe_path_) + "me0_strip_dg_bkgElePos_radius";
  TString label_neuBkg = TString(dbe_path_) + "me0_strip_dg_bkgNeutral_radius";
  TString label_totBkg = TString(dbe_path_) + "me0_strip_dg_bkg_radius_tot";
  TString label_evts = TString(dbe_path_) + "num_evts";

  if (ig.get(label_evts.Data()) != nullptr) {
    TH1F *numEvts = (TH1F *)ig.get(label_evts.Data())->getTH1F()->Clone();

    if (ig.get(label_eleBkg.Data()) != nullptr) {
      TH1F *eleBkg = (TH1F *)ig.get(label_eleBkg.Data())->getTH1F()->Clone();
      eleBkg->Sumw2();
      ProcessBookingBKG(ibooker, ig, "me0_strip_dg_elePosBkg_rad", eleBkg, numEvts);

      delete eleBkg;
    }
    if (ig.get(label_neuBkg.Data()) != nullptr) {
      TH1F *neuBkg = (TH1F *)ig.get(label_neuBkg.Data())->getTH1F()->Clone();
      neuBkg->Sumw2();
      ProcessBookingBKG(ibooker, ig, "me0_strip_dg_neuBkg_rad", neuBkg, numEvts);

      delete neuBkg;
    }
    if (ig.get(label_totBkg.Data()) != nullptr) {
      TH1F *totBkg = (TH1F *)ig.get(label_totBkg.Data())->getTH1F()->Clone();
      totBkg->Sumw2();
      ProcessBookingBKG(ibooker, ig, "me0_strip_dg_totBkg_rad", totBkg, numEvts);

      delete totBkg;
    }

    delete numEvts;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(MuonME0DigisHarvestor);
