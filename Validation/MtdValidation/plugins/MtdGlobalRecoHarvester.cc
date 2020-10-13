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
  MonitorElement* meEtlEtaEff_[4];
  MonitorElement* meEtlPhiEff_[4];
  MonitorElement* meEtlPtEff_[4];
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
  MonitorElement* meETLTrackEffEtaMtdZnegD1 = igetter.get(folder_ + "TrackETLEffEtaMtdZnegD1");
  MonitorElement* meETLTrackEffEtaMtdZnegD2 = igetter.get(folder_ + "TrackETLEffEtaMtdZnegD2");
  MonitorElement* meETLTrackEffPhiMtdZnegD1 = igetter.get(folder_ + "TrackETLEffPhiMtdZnegD1");
  MonitorElement* meETLTrackEffPhiMtdZnegD2 = igetter.get(folder_ + "TrackETLEffPhiMtdZnegD2");
  MonitorElement* meETLTrackEffPtMtdZnegD1 = igetter.get(folder_ + "TrackETLEffPtMtdZnegD1");
  MonitorElement* meETLTrackEffPtMtdZnegD2 = igetter.get(folder_ + "TrackETLEffPtMtdZnegD2");
  MonitorElement* meETLTrackEffEtaTotZpos = igetter.get(folder_ + "TrackETLEffEtaTotZpos");
  MonitorElement* meETLTrackEffPhiTotZpos = igetter.get(folder_ + "TrackETLEffPhiTotZpos");
  MonitorElement* meETLTrackEffPtTotZpos = igetter.get(folder_ + "TrackETLEffPtTotZpos");
  MonitorElement* meETLTrackEffEtaMtdZposD1 = igetter.get(folder_ + "TrackETLEffEtaMtdZposD1");
  MonitorElement* meETLTrackEffEtaMtdZposD2 = igetter.get(folder_ + "TrackETLEffEtaMtdZposD2");
  MonitorElement* meETLTrackEffPhiMtdZposD1 = igetter.get(folder_ + "TrackETLEffPhiMtdZposD1");
  MonitorElement* meETLTrackEffPhiMtdZposD2 = igetter.get(folder_ + "TrackETLEffPhiMtdZposD2");
  MonitorElement* meETLTrackEffPtMtdZposD1 = igetter.get(folder_ + "TrackETLEffPtMtdZposD1");
  MonitorElement* meETLTrackEffPtMtdZposD2 = igetter.get(folder_ + "TrackETLEffPtMtdZposD2");

  if (!meBTLTrackEffEtaTot || !meBTLTrackEffPhiTot || !meBTLTrackEffPtTot || !meBTLTrackEffEtaMtd ||
      !meBTLTrackEffPhiMtd || !meBTLTrackEffPtMtd || !meETLTrackEffEtaTotZneg || !meETLTrackEffPhiTotZneg ||
      !meETLTrackEffPtTotZneg || !meETLTrackEffEtaMtdZnegD1 || !meETLTrackEffPhiMtdZnegD1 ||
      !meETLTrackEffPtMtdZnegD1 || !meETLTrackEffEtaTotZpos || !meETLTrackEffPhiTotZpos || !meETLTrackEffPtTotZpos ||
      !meETLTrackEffEtaMtdZposD1 || !meETLTrackEffPhiMtdZposD1 || !meETLTrackEffPtMtdZposD1 ||
      !meETLTrackEffEtaMtdZnegD2 || !meETLTrackEffPhiMtdZnegD2 || !meETLTrackEffPtMtdZnegD2 ||
      !meETLTrackEffEtaMtdZposD2 || !meETLTrackEffPhiMtdZposD2 || !meETLTrackEffPtMtdZposD2) {
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
  meEtlEtaEff_[0] = ibook.book1D("EtlEtaEffZnegD1",
                                 " Track Efficiency VS Eta (-Z, Single(topo1D)/First(topo2D) Disk);#eta;Efficiency",
                                 meETLTrackEffEtaTotZneg->getNbinsX(),
                                 meETLTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff_[1] = ibook.book1D("EtlEtaEffZnegD2",
                                 " Track Efficiency VS Eta (-Z, Second Disk);#eta;Efficiency",
                                 meETLTrackEffEtaTotZneg->getNbinsX(),
                                 meETLTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff_[0] =
      ibook.book1D("EtlPhiEffZnegD1",
                   "Track Efficiency VS Phi (-Z, Single(topo1D)/First(topo2D) Disk);#phi [rad];Efficiency",
                   meETLTrackEffPhiTotZneg->getNbinsX(),
                   meETLTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmin(),
                   meETLTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff_[1] = ibook.book1D("EtlPhiEffZnegD2",
                                 "Track Efficiency VS Phi (-Z, Second Disk);#phi [rad];Efficiency",
                                 meETLTrackEffPhiTotZneg->getNbinsX(),
                                 meETLTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff_[0] = ibook.book1D("EtlPtEffZnegD1",
                                "Track Efficiency VS Pt (-Z, Single(topo1D)/First(topo2D) Disk);Pt [GeV];Efficiency",
                                meETLTrackEffPtTotZneg->getNbinsX(),
                                meETLTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                meETLTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff_[1] = ibook.book1D("EtlPtEffZnegD2",
                                "Track Efficiency VS Pt (-Z, Second Disk);Pt [GeV];Efficiency",
                                meETLTrackEffPtTotZneg->getNbinsX(),
                                meETLTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                meETLTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff_[2] = ibook.book1D("EtlEtaEffZposD1",
                                 " Track Efficiency VS Eta (+Z, Single(topo1D)/First(topo2D) Disk);#eta;Efficiency",
                                 meETLTrackEffEtaTotZpos->getNbinsX(),
                                 meETLTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff_[3] = ibook.book1D("EtlEtaEffZposD2",
                                 " Track Efficiency VS Eta (+Z, Second Disk);#eta;Efficiency",
                                 meETLTrackEffEtaTotZpos->getNbinsX(),
                                 meETLTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff_[2] =
      ibook.book1D("EtlPhiEffZposD1",
                   "Track Efficiency VS Phi (+Z, Single(topo1D)/First(topo2D) Disk);#phi [rad];Efficiency",
                   meETLTrackEffPhiTotZpos->getNbinsX(),
                   meETLTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmin(),
                   meETLTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff_[3] = ibook.book1D("EtlPhiEffZposD2",
                                 "Track Efficiency VS Phi (+Z, Second Disk);#phi [rad];Efficiency",
                                 meETLTrackEffPhiTotZpos->getNbinsX(),
                                 meETLTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff_[2] = ibook.book1D("EtlPtEffZposD1",
                                "Track Efficiency VS Pt (+Z, Single(topo1D)/First(topo2D) Disk);Pt [GeV];Efficiency",
                                meETLTrackEffPtTotZpos->getNbinsX(),
                                meETLTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                meETLTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff_[3] = ibook.book1D("EtlPtEffZposD2",
                                "Track Efficiency VS Pt (+Z, Second Disk);Pt [GeV];Efficiency",
                                meETLTrackEffPtTotZpos->getNbinsX(),
                                meETLTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                meETLTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmax());
  meBtlEtaEff_->getTH1()->SetMinimum(0.);
  meBtlPhiEff_->getTH1()->SetMinimum(0.);
  meBtlPtEff_->getTH1()->SetMinimum(0.);
  for (int i = 0; i < 4; i++) {
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
    double eff = meETLTrackEffEtaMtdZnegD1->getBinContent(ibin) / meETLTrackEffEtaTotZneg->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffEtaMtdZnegD1->getBinContent(ibin) *
              (meETLTrackEffEtaTotZneg->getBinContent(ibin) - meETLTrackEffEtaMtdZnegD1->getBinContent(ibin))) /
             pow(meETLTrackEffEtaTotZneg->getBinContent(ibin), 3));
    if (meETLTrackEffEtaTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlEtaEff_[0]->setBinContent(ibin, eff);
    meEtlEtaEff_[0]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffEtaTotZneg->getNbinsX(); ibin++) {
    double eff = meETLTrackEffEtaMtdZnegD2->getBinContent(ibin) / meETLTrackEffEtaTotZneg->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffEtaMtdZnegD2->getBinContent(ibin) *
              (meETLTrackEffEtaTotZneg->getBinContent(ibin) - meETLTrackEffEtaMtdZnegD2->getBinContent(ibin))) /
             pow(meETLTrackEffEtaTotZneg->getBinContent(ibin), 3));
    if (meETLTrackEffEtaTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlEtaEff_[1]->setBinContent(ibin, eff);
    meEtlEtaEff_[1]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffEtaTotZpos->getNbinsX(); ibin++) {
    double eff = meETLTrackEffEtaMtdZposD1->getBinContent(ibin) / meETLTrackEffEtaTotZpos->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffEtaMtdZposD1->getBinContent(ibin) *
              (meETLTrackEffEtaTotZpos->getBinContent(ibin) - meETLTrackEffEtaMtdZposD1->getBinContent(ibin))) /
             pow(meETLTrackEffEtaTotZpos->getBinContent(ibin), 3));
    if (meETLTrackEffEtaTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlEtaEff_[2]->setBinContent(ibin, eff);
    meEtlEtaEff_[2]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffEtaTotZpos->getNbinsX(); ibin++) {
    double eff = meETLTrackEffEtaMtdZposD2->getBinContent(ibin) / meETLTrackEffEtaTotZpos->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffEtaMtdZposD2->getBinContent(ibin) *
              (meETLTrackEffEtaTotZpos->getBinContent(ibin) - meETLTrackEffEtaMtdZposD2->getBinContent(ibin))) /
             pow(meETLTrackEffEtaTotZpos->getBinContent(ibin), 3));
    if (meETLTrackEffEtaTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlEtaEff_[3]->setBinContent(ibin, eff);
    meEtlEtaEff_[3]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPhiTotZneg->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPhiMtdZnegD1->getBinContent(ibin) / meETLTrackEffPhiTotZneg->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPhiMtdZnegD1->getBinContent(ibin) *
              (meETLTrackEffPhiTotZneg->getBinContent(ibin) - meETLTrackEffPhiMtdZnegD1->getBinContent(ibin))) /
             pow(meETLTrackEffPhiTotZneg->getBinContent(ibin), 3));
    if (meETLTrackEffPhiTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPhiEff_[0]->setBinContent(ibin, eff);
    meEtlPhiEff_[0]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPhiTotZneg->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPhiMtdZnegD2->getBinContent(ibin) / meETLTrackEffPhiTotZneg->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPhiMtdZnegD2->getBinContent(ibin) *
              (meETLTrackEffPhiTotZneg->getBinContent(ibin) - meETLTrackEffPhiMtdZnegD2->getBinContent(ibin))) /
             pow(meETLTrackEffPhiTotZneg->getBinContent(ibin), 3));
    if (meETLTrackEffPhiTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPhiEff_[1]->setBinContent(ibin, eff);
    meEtlPhiEff_[1]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPhiTotZpos->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPhiMtdZposD1->getBinContent(ibin) / meETLTrackEffPhiTotZpos->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPhiMtdZposD1->getBinContent(ibin) *
              (meETLTrackEffPhiTotZpos->getBinContent(ibin) - meETLTrackEffPhiMtdZposD1->getBinContent(ibin))) /
             pow(meETLTrackEffPhiTotZpos->getBinContent(ibin), 3));
    if (meETLTrackEffPhiTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPhiEff_[2]->setBinContent(ibin, eff);
    meEtlPhiEff_[2]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPhiTotZpos->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPhiMtdZposD2->getBinContent(ibin) / meETLTrackEffPhiTotZpos->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPhiMtdZposD2->getBinContent(ibin) *
              (meETLTrackEffPhiTotZpos->getBinContent(ibin) - meETLTrackEffPhiMtdZposD2->getBinContent(ibin))) /
             pow(meETLTrackEffPhiTotZpos->getBinContent(ibin), 3));
    if (meETLTrackEffPhiTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPhiEff_[3]->setBinContent(ibin, eff);
    meEtlPhiEff_[3]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPtTotZneg->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPtMtdZnegD1->getBinContent(ibin) / meETLTrackEffPtTotZneg->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPtMtdZnegD1->getBinContent(ibin) *
              (meETLTrackEffPtTotZneg->getBinContent(ibin) - meETLTrackEffPtMtdZnegD1->getBinContent(ibin))) /
             pow(meETLTrackEffPtTotZneg->getBinContent(ibin), 3));
    if (meETLTrackEffPtTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPtEff_[0]->setBinContent(ibin, eff);
    meEtlPtEff_[0]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPtTotZneg->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPtMtdZnegD2->getBinContent(ibin) / meETLTrackEffPtTotZneg->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPtMtdZnegD2->getBinContent(ibin) *
              (meETLTrackEffPtTotZneg->getBinContent(ibin) - meETLTrackEffPtMtdZnegD2->getBinContent(ibin))) /
             pow(meETLTrackEffPtTotZneg->getBinContent(ibin), 3));
    if (meETLTrackEffPtTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPtEff_[1]->setBinContent(ibin, eff);
    meEtlPtEff_[1]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPtTotZpos->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPtMtdZposD1->getBinContent(ibin) / meETLTrackEffPtTotZpos->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPtMtdZposD1->getBinContent(ibin) *
              (meETLTrackEffPtTotZpos->getBinContent(ibin) - meETLTrackEffPtMtdZposD1->getBinContent(ibin))) /
             pow(meETLTrackEffPtTotZpos->getBinContent(ibin), 3));
    if (meETLTrackEffPtTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPtEff_[2]->setBinContent(ibin, eff);
    meEtlPtEff_[2]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meETLTrackEffPtTotZpos->getNbinsX(); ibin++) {
    double eff = meETLTrackEffPtMtdZposD2->getBinContent(ibin) / meETLTrackEffPtTotZpos->getBinContent(ibin);
    double bin_err =
        sqrt((meETLTrackEffPtMtdZposD2->getBinContent(ibin) *
              (meETLTrackEffPtTotZpos->getBinContent(ibin) - meETLTrackEffPtMtdZposD2->getBinContent(ibin))) /
             pow(meETLTrackEffPtTotZpos->getBinContent(ibin), 3));
    if (meETLTrackEffPtTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPtEff_[3]->setBinContent(ibin, eff);
    meEtlPtEff_[3]->setBinError(ibin, bin_err);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ----------
void MtdGlobalRecoHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/GlobalReco/");

  descriptions.add("MtdGlobalRecoPostProcessor", desc);
}

DEFINE_FWK_MODULE(MtdGlobalRecoHarvester);
