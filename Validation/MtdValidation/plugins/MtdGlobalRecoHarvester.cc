#include <string>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

class MtdGlobalRecoHarvester : public DQMEDHarvester {
public:
  explicit MtdGlobalRecoHarvester(const edm::ParameterSet& iConfig);
  ~MtdGlobalRecoHarvester() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  const std::string folder_;

  // --- Histograms
  MonitorElement* meBtlEtaEff_;
  MonitorElement* meBtlPhiEff_;
  MonitorElement* meBtlPtEff_;
  MonitorElement* meEtlEtaEff_[2];
  MonitorElement* meEtlPhiEff_[2];
  MonitorElement* meEtlPtEff_[2];
};

// ------------ constructor and destructor --------------
MtdGlobalRecoHarvester::MtdGlobalRecoHarvester(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {}

MtdGlobalRecoHarvester::~MtdGlobalRecoHarvester() {}

// ------------ endjob tasks ----------------------------
void MtdGlobalRecoHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& igetter) {
  // --- Get the monitoring histograms
  MonitorElement* meBTLTrackEffEtaTot = igetter.get(folder_ + "TrackBTLEffEtaTot");
  MonitorElement* meBTLTrackEffPhiTot = igetter.get(folder_ + "TrackBTLEffPhiTot");
  MonitorElement* meBTLTrackEffPtTot = igetter.get(folder_ + "TrackBTLEffPtTot");
  MonitorElement* meBTLTrackEffEtaMtd = igetter.get(folder_ + "TrackBTLEffEtaMtd");
  MonitorElement* meBTLTrackEffPhiMtd = igetter.get(folder_ + "TrackBTLEffPhiMtd");
  MonitorElement* meBTLTrackEffPtMtd = igetter.get(folder_ + "TrackBTLEffPtMtd");
  MonitorElement* meETLTrackEffEtaTotZneg = igetter.get(folder_ + "TrackETLEffEtaTotZneg");
  MonitorElement* meETLTrackEffPhiTotZneg = igetter.get(folder_ + "TrackETLEffPhiTotZneg");
  MonitorElement* meETLTrackEffPtTotZneg = igetter.get(folder_ + "TrackETLEffPtTotZneg");
  MonitorElement* meETLTrackEffEtaMtdZneg = igetter.get(folder_ + "TrackETLEffEtaMtdZneg");
  MonitorElement* meETLTrackEffPhiMtdZneg = igetter.get(folder_ + "TrackETLEffPhiMtdZneg");
  MonitorElement* meETLTrackEffPtMtdZneg = igetter.get(folder_ + "TrackETLEffPtMtdZneg");
  MonitorElement* meETLTrackEffEtaTotZpos = igetter.get(folder_ + "TrackETLEffEtaTotZpos");
  MonitorElement* meETLTrackEffPhiTotZpos = igetter.get(folder_ + "TrackETLEffPhiTotZpos");
  MonitorElement* meETLTrackEffPtTotZpos = igetter.get(folder_ + "TrackETLEffPtTotZpos");
  MonitorElement* meETLTrackEffEtaMtdZpos = igetter.get(folder_ + "TrackETLEffEtaMtdZpos");
  MonitorElement* meETLTrackEffPhiMtdZpos = igetter.get(folder_ + "TrackETLEffPhiMtdZpos");
  MonitorElement* meETLTrackEffPtMtdZpos = igetter.get(folder_ + "TrackETLEffPtMtdZpos");

  if (!meBTLTrackEffEtaTot || !meBTLTrackEffPhiTot || !meBTLTrackEffPtTot || !meBTLTrackEffEtaMtd ||
      !meBTLTrackEffPhiMtd || !meBTLTrackEffPtMtd || !meETLTrackEffEtaTotZneg || !meETLTrackEffPhiTotZneg ||
      !meETLTrackEffPtTotZneg || !meETLTrackEffEtaMtdZneg || !meETLTrackEffPhiMtdZneg || !meETLTrackEffPtMtdZneg ||
      !meETLTrackEffEtaTotZpos || !meETLTrackEffPhiTotZpos || !meETLTrackEffPtTotZpos || !meETLTrackEffEtaMtdZpos ||
      !meETLTrackEffPhiMtdZpos || !meETLTrackEffPtMtdZpos) {
    edm::LogError("MtdGlobalRecoHarvester") << "Monitoring histograms not found!" << std::endl;
    return;
  }

  // --- Book  histograms
  ibook.cd(folder_);
  meBtlEtaEff_ = ibook.book1D("BtlEtaEff",
                              " Track Efficiency VS Eta;#eta;Efficiency",
                              meBTLTrackEffEtaTot->getNbinsX(),
                              meBTLTrackEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                              meBTLTrackEffEtaTot->getTH1()->GetXaxis()->GetXmax());
  meBtlPhiEff_ = ibook.book1D("BtlPhiEff",
                              "Track Efficiency VS Phi;#phi [rad];Efficiency",
                              meBTLTrackEffPhiTot->getNbinsX(),
                              meBTLTrackEffPhiTot->getTH1()->GetXaxis()->GetXmin(),
                              meBTLTrackEffPhiTot->getTH1()->GetXaxis()->GetXmax());
  meBtlPtEff_ = ibook.book1D("BtlPtEff",
                             "Track Efficiency VS Pt;Pt [GeV];Efficiency",
                             meBTLTrackEffPtTot->getNbinsX(),
                             meBTLTrackEffPtTot->getTH1()->GetXaxis()->GetXmin(),
                             meBTLTrackEffPtTot->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff_[0] = ibook.book1D("EtlEtaEffZneg",
                                 " Track Efficiency VS Eta (-Z);#eta;Efficiency",
                                 meETLTrackEffEtaTotZneg->getNbinsX(),
                                 meETLTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff_[0] = ibook.book1D("EtlPhiEffZneg",
                                 "Track Efficiency VS Phi (-Z);#phi [rad];Efficiency",
                                 meETLTrackEffPhiTotZneg->getNbinsX(),
                                 meETLTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff_[0] = ibook.book1D("EtlPtEffZneg",
                                "Track Efficiency VS Pt (-Z);Pt [GeV];Efficiency",
                                meETLTrackEffPtTotZneg->getNbinsX(),
                                meETLTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                meETLTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff_[1] = ibook.book1D("EtlEtaEffZpos",
                                 " Track Efficiency VS Eta (+Z);#eta;Efficiency",
                                 meETLTrackEffEtaTotZpos->getNbinsX(),
                                 meETLTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff_[1] = ibook.book1D("EtlPhiEffZpos",
                                 "Track Efficiency VS Phi (+Z);#phi [rad];Efficiency",
                                 meETLTrackEffPhiTotZpos->getNbinsX(),
                                 meETLTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff_[1] = ibook.book1D("EtlPtEffZpos",
                                "Track Efficiency VS Pt (+Z);Pt [GeV];Efficiency",
                                meETLTrackEffPtTotZpos->getNbinsX(),
                                meETLTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                meETLTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmax());

  meBtlEtaEff_->getTH1()->SetMinimum(0.);
  meBtlPhiEff_->getTH1()->SetMinimum(0.);
  meBtlPtEff_->getTH1()->SetMinimum(0.);
  for (int i = 0; i < 2; i++) {
    meEtlEtaEff_[i]->getTH1()->SetMinimum(0.);
    meEtlPhiEff_[i]->getTH1()->SetMinimum(0.);
    meEtlPtEff_[i]->getTH1()->SetMinimum(0.);
  }

  // --- Calculate efficiency BTL
  for (int ibin = 1; ibin <= meBTLTrackEffEtaTot->getNbinsX(); ibin++) {
    double eff = meBTLTrackEffEtaMtd->getBinContent(ibin) / meBTLTrackEffEtaTot->getBinContent(ibin);
    double bin_err = sqrt((meBTLTrackEffEtaMtd->getBinContent(ibin) *
                           (meBTLTrackEffEtaTot->getBinContent(ibin) - meBTLTrackEffEtaMtd->getBinContent(ibin))) /
                          pow(meBTLTrackEffEtaTot->getBinContent(ibin), 3));
    if (meBTLTrackEffEtaTot->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meBtlEtaEff_->setBinContent(ibin, eff);
    meBtlEtaEff_->setBinError(ibin, bin_err);
  }
  for (int ibin = 1; ibin <= meBTLTrackEffPhiTot->getNbinsX(); ibin++) {
    double eff = meBTLTrackEffPhiMtd->getBinContent(ibin) / meBTLTrackEffPhiTot->getBinContent(ibin);
    double bin_err = sqrt((meBTLTrackEffPhiMtd->getBinContent(ibin) *
                           (meBTLTrackEffPhiTot->getBinContent(ibin) - meBTLTrackEffPhiMtd->getBinContent(ibin))) /
                          pow(meBTLTrackEffPhiTot->getBinContent(ibin), 3));
    if (meBTLTrackEffPhiTot->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meBtlPhiEff_->setBinContent(ibin, eff);
    meBtlPhiEff_->setBinError(ibin, bin_err);
  }
  for (int ibin = 1; ibin <= meBTLTrackEffPtTot->getNbinsX(); ibin++) {
    double eff = meBTLTrackEffPtMtd->getBinContent(ibin) / meBTLTrackEffPtTot->getBinContent(ibin);
    double bin_err = sqrt((meBTLTrackEffPtMtd->getBinContent(ibin) *
                           (meBTLTrackEffPtTot->getBinContent(ibin) - meBTLTrackEffPtMtd->getBinContent(ibin))) /
                          pow(meBTLTrackEffPtTot->getBinContent(ibin), 3));
    if (meBTLTrackEffPtTot->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meBtlPtEff_->setBinContent(ibin, eff);
    meBtlPtEff_->setBinError(ibin, bin_err);
  }
  // --- Calculate efficiency ETL
  for (int ibin = 1; ibin <= meETLTrackEffEtaTotZneg->getNbinsX(); ibin++) {
    double eff = meETLTrackEffEtaMtdZneg->getBinContent(ibin) / meETLTrackEffEtaTotZneg->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffEtaMtdZneg->getBinContent(ibin) *
              (meETLTrackEffEtaTotZneg->getBinContent(ibin) - meETLTrackEffEtaMtdZneg->getBinContent(ibin))) /
             pow(meETLTrackEffEtaTotZneg->getBinContent(ibin), 3));
    if (meETLTrackEffEtaTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlEtaEff_[0]->setBinContent(ibin, eff);
    meEtlEtaEff_[0]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffEtaTotZpos->getNbinsX(); ibin++) {
    double eff = meETLTrackEffEtaMtdZpos->getBinContent(ibin) / meETLTrackEffEtaTotZpos->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffEtaMtdZpos->getBinContent(ibin) *
              (meETLTrackEffEtaTotZpos->getBinContent(ibin) - meETLTrackEffEtaMtdZpos->getBinContent(ibin))) /
             pow(meETLTrackEffEtaTotZpos->getBinContent(ibin), 3));
    if (meETLTrackEffEtaTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlEtaEff_[1]->setBinContent(ibin, eff);
    meEtlEtaEff_[1]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPhiTotZneg->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPhiMtdZneg->getBinContent(ibin) / meETLTrackEffPhiTotZneg->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPhiMtdZneg->getBinContent(ibin) *
              (meETLTrackEffPhiTotZneg->getBinContent(ibin) - meETLTrackEffPhiMtdZneg->getBinContent(ibin))) /
             pow(meETLTrackEffPhiTotZneg->getBinContent(ibin), 3));
    if (meETLTrackEffPhiTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPhiEff_[0]->setBinContent(ibin, eff);
    meEtlPhiEff_[0]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPhiTotZpos->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPhiMtdZpos->getBinContent(ibin) / meETLTrackEffPhiTotZpos->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPhiMtdZpos->getBinContent(ibin) *
              (meETLTrackEffPhiTotZpos->getBinContent(ibin) - meETLTrackEffPhiMtdZpos->getBinContent(ibin))) /
             pow(meETLTrackEffPhiTotZpos->getBinContent(ibin), 3));
    if (meETLTrackEffPhiTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPhiEff_[1]->setBinContent(ibin, eff);
    meEtlPhiEff_[1]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPtTotZneg->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPtMtdZneg->getBinContent(ibin) / meETLTrackEffPtTotZneg->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPtMtdZneg->getBinContent(ibin) *
              (meETLTrackEffPtTotZneg->getBinContent(ibin) - meETLTrackEffPtMtdZneg->getBinContent(ibin))) /
             pow(meETLTrackEffPtTotZneg->getBinContent(ibin), 3));
    if (meETLTrackEffPtTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPtEff_[0]->setBinContent(ibin, eff);
    meEtlPtEff_[0]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPtTotZpos->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPtMtdZpos->getBinContent(ibin) / meETLTrackEffPtTotZpos->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPtMtdZpos->getBinContent(ibin) *
              (meETLTrackEffPtTotZpos->getBinContent(ibin) - meETLTrackEffPtMtdZpos->getBinContent(ibin))) /
             pow(meETLTrackEffPtTotZpos->getBinContent(ibin), 3));
    if (meETLTrackEffPtTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPtEff_[1]->setBinContent(ibin, eff);
    meEtlPtEff_[1]->setBinError(ibin, bin_err);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ----------
void MtdGlobalRecoHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/GlobalReco/");

  descriptions.add("MtdGlobalRecoPostProcessor", desc);
}

DEFINE_FWK_MODULE(MtdGlobalRecoHarvester);
