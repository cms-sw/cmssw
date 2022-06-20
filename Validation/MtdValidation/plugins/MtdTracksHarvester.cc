#include <string>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

class MtdTracksHarvester : public DQMEDHarvester {
public:
  explicit MtdTracksHarvester(const edm::ParameterSet& iConfig);
  ~MtdTracksHarvester() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  void computeEfficiency1D(MonitorElement* num, MonitorElement* den, MonitorElement* result);

  const std::string folder_;

  // --- Histograms
  MonitorElement* meBtlEtaEff_;
  MonitorElement* meBtlPhiEff_;
  MonitorElement* meBtlPtEff_;
  MonitorElement* meEtlEtaEff_[2];
  MonitorElement* meEtlPhiEff_[2];
  MonitorElement* meEtlPtEff_[2];
  MonitorElement* meEtlEtaEff2_[2];
  MonitorElement* meEtlPhiEff2_[2];
  MonitorElement* meEtlPtEff2_[2];
  MonitorElement* meMVAPtSelEff_;
  MonitorElement* meMVAEtaSelEff_;
  MonitorElement* meMVAPtMatchEff_;
  MonitorElement* meMVAEtaMatchEff_;
  MonitorElement* meTPPtSelEff_;
  MonitorElement* meTPEtaSelEff_;
  MonitorElement* meTPPtMatchEff_;
  MonitorElement* meTPEtaMatchEff_;
  MonitorElement* meTPPtMatchEtl2Eff_;
  MonitorElement* meTPEtaMatchEtl2Eff_;
  MonitorElement* meTPmtdPtSelEff_;
  MonitorElement* meTPmtdEtaSelEff_;
  MonitorElement* meTPmtdPtMatchEff_;
  MonitorElement* meTPmtdEtaMatchEff_;
  MonitorElement* meTPAssocEff_;
};

// ------------ constructor and destructor --------------
MtdTracksHarvester::MtdTracksHarvester(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {}

MtdTracksHarvester::~MtdTracksHarvester() {}

// auxiliary method to compute efficiency from the ratio of two 1D MonitorElement
void MtdTracksHarvester::computeEfficiency1D(MonitorElement* num, MonitorElement* den, MonitorElement* result) {
  for (int ibin = 1; ibin <= den->getNbinsX(); ibin++) {
    double eff = num->getBinContent(ibin) / den->getBinContent(ibin);
    double bin_err = sqrt((num->getBinContent(ibin) * (den->getBinContent(ibin) - num->getBinContent(ibin))) /
                          pow(den->getBinContent(ibin), 3));
    if (den->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    result->setBinContent(ibin, eff);
    result->setBinError(ibin, bin_err);
  }
}

// ------------ endjob tasks ----------------------------
void MtdTracksHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& igetter) {
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
  MonitorElement* meETLTrackEffEta2MtdZneg = igetter.get(folder_ + "TrackETLEffEta2MtdZneg");
  MonitorElement* meETLTrackEffPhi2MtdZneg = igetter.get(folder_ + "TrackETLEffPhi2MtdZneg");
  MonitorElement* meETLTrackEffPt2MtdZneg = igetter.get(folder_ + "TrackETLEffPt2MtdZneg");
  MonitorElement* meETLTrackEffEtaTotZpos = igetter.get(folder_ + "TrackETLEffEtaTotZpos");
  MonitorElement* meETLTrackEffPhiTotZpos = igetter.get(folder_ + "TrackETLEffPhiTotZpos");
  MonitorElement* meETLTrackEffPtTotZpos = igetter.get(folder_ + "TrackETLEffPtTotZpos");
  MonitorElement* meETLTrackEffEtaMtdZpos = igetter.get(folder_ + "TrackETLEffEtaMtdZpos");
  MonitorElement* meETLTrackEffPhiMtdZpos = igetter.get(folder_ + "TrackETLEffPhiMtdZpos");
  MonitorElement* meETLTrackEffPtMtdZpos = igetter.get(folder_ + "TrackETLEffPtMtdZpos");
  MonitorElement* meETLTrackEffEta2MtdZpos = igetter.get(folder_ + "TrackETLEffEta2MtdZpos");
  MonitorElement* meETLTrackEffPhi2MtdZpos = igetter.get(folder_ + "TrackETLEffPhi2MtdZpos");
  MonitorElement* meETLTrackEffPt2MtdZpos = igetter.get(folder_ + "TrackETLEffPt2MtdZpos");
  MonitorElement* meMVATrackEffPtTot = igetter.get(folder_ + "MVAEffPtTot");
  MonitorElement* meMVATrackMatchedEffPtTot = igetter.get(folder_ + "MVAMatchedEffPtTot");
  MonitorElement* meMVATrackMatchedEffPtMtd = igetter.get(folder_ + "MVAMatchedEffPtMtd");
  MonitorElement* meTrackMatchedTPEffPtTot = igetter.get(folder_ + "MatchedTPEffPtTot");
  MonitorElement* meTrackMatchedTPEffPtMtd = igetter.get(folder_ + "MatchedTPEffPtMtd");
  MonitorElement* meTrackMatchedTPEffPtEtl2Mtd = igetter.get(folder_ + "MatchedTPEffPtEtl2Mtd");
  MonitorElement* meTrackMatchedTPmtdEffPtTot = igetter.get(folder_ + "MatchedTPmtdEffPtTot");
  MonitorElement* meTrackMatchedTPmtdEffPtMtd = igetter.get(folder_ + "MatchedTPmtdEffPtMtd");
  MonitorElement* meMVATrackEffEtaTot = igetter.get(folder_ + "MVAEffEtaTot");
  MonitorElement* meMVATrackMatchedEffEtaTot = igetter.get(folder_ + "MVAMatchedEffEtaTot");
  MonitorElement* meMVATrackMatchedEffEtaMtd = igetter.get(folder_ + "MVAMatchedEffEtaMtd");
  MonitorElement* meTrackMatchedTPEffEtaTot = igetter.get(folder_ + "MatchedTPEffEtaTot");
  MonitorElement* meTrackMatchedTPEffEtaMtd = igetter.get(folder_ + "MatchedTPEffEtaMtd");
  MonitorElement* meTrackMatchedTPEffEtaEtl2Mtd = igetter.get(folder_ + "MatchedTPEffEtaEtl2Mtd");
  MonitorElement* meTrackMatchedTPmtdEffEtaTot = igetter.get(folder_ + "MatchedTPmtdEffEtaTot");
  MonitorElement* meTrackMatchedTPmtdEffEtaMtd = igetter.get(folder_ + "MatchedTPmtdEffEtaMtd");
  MonitorElement* meNTrackingParticles = igetter.get(folder_ + "NTrackingParticles");
  MonitorElement* meUnassDeposit = igetter.get(folder_ + "UnassDeposit");

  if (!meBTLTrackEffEtaTot || !meBTLTrackEffPhiTot || !meBTLTrackEffPtTot || !meBTLTrackEffEtaMtd ||
      !meBTLTrackEffPhiMtd || !meBTLTrackEffPtMtd || !meETLTrackEffEtaTotZneg || !meETLTrackEffPhiTotZneg ||
      !meETLTrackEffPtTotZneg || !meETLTrackEffEtaMtdZneg || !meETLTrackEffPhiMtdZneg || !meETLTrackEffPtMtdZneg ||
      !meETLTrackEffEta2MtdZneg || !meETLTrackEffPhi2MtdZneg || !meETLTrackEffPt2MtdZneg || !meETLTrackEffEtaTotZpos ||
      !meETLTrackEffPhiTotZpos || !meETLTrackEffPtTotZpos || !meETLTrackEffEtaMtdZpos || !meETLTrackEffPhiMtdZpos ||
      !meETLTrackEffPtMtdZpos || !meETLTrackEffEta2MtdZpos || !meETLTrackEffPhi2MtdZpos || !meETLTrackEffPt2MtdZpos ||
      !meMVATrackEffPtTot || !meMVATrackMatchedEffPtTot || !meMVATrackMatchedEffPtMtd || !meMVATrackEffEtaTot ||
      !meMVATrackMatchedEffEtaTot || !meMVATrackMatchedEffEtaMtd || !meTrackMatchedTPEffPtTot ||
      !meTrackMatchedTPEffPtMtd || !meTrackMatchedTPEffPtEtl2Mtd || !meTrackMatchedTPmtdEffPtTot ||
      !meTrackMatchedTPmtdEffPtMtd || !meTrackMatchedTPEffEtaTot || !meTrackMatchedTPEffEtaMtd ||
      !meTrackMatchedTPEffEtaEtl2Mtd || !meTrackMatchedTPmtdEffEtaTot || !meTrackMatchedTPmtdEffEtaMtd ||
      !meNTrackingParticles || !meUnassDeposit) {
    edm::LogError("MtdTracksHarvester") << "Monitoring histograms not found!" << std::endl;
    return;
  }

  // --- Book  histograms
  ibook.cd(folder_);
  meBtlEtaEff_ = ibook.book1D("BtlEtaEff",
                              " Track Efficiency VS Eta;#eta;Efficiency",
                              meBTLTrackEffEtaTot->getNbinsX(),
                              meBTLTrackEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                              meBTLTrackEffEtaTot->getTH1()->GetXaxis()->GetXmax());
  meBtlEtaEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackEffEtaMtd, meBTLTrackEffEtaTot, meBtlEtaEff_);

  meBtlPhiEff_ = ibook.book1D("BtlPhiEff",
                              "Track Efficiency VS Phi;#phi [rad];Efficiency",
                              meBTLTrackEffPhiTot->getNbinsX(),
                              meBTLTrackEffPhiTot->getTH1()->GetXaxis()->GetXmin(),
                              meBTLTrackEffPhiTot->getTH1()->GetXaxis()->GetXmax());
  meBtlPhiEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackEffPhiMtd, meBTLTrackEffPhiTot, meBtlPhiEff_);

  meBtlPtEff_ = ibook.book1D("BtlPtEff",
                             "Track Efficiency VS Pt;Pt [GeV];Efficiency",
                             meBTLTrackEffPtTot->getNbinsX(),
                             meBTLTrackEffPtTot->getTH1()->GetXaxis()->GetXmin(),
                             meBTLTrackEffPtTot->getTH1()->GetXaxis()->GetXmax());
  meBtlPtEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackEffPtMtd, meBTLTrackEffPtTot, meBtlPtEff_);

  meEtlEtaEff_[0] = ibook.book1D("EtlEtaEffZneg",
                                 " Track Efficiency VS Eta (-Z);#eta;Efficiency",
                                 meETLTrackEffEtaTotZneg->getNbinsX(),
                                 meETLTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff_[0]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffEtaMtdZneg, meETLTrackEffEtaTotZneg, meEtlEtaEff_[0]);

  meEtlPhiEff_[0] = ibook.book1D("EtlPhiEffZneg",
                                 "Track Efficiency VS Phi (-Z);#phi [rad];Efficiency",
                                 meETLTrackEffPhiTotZneg->getNbinsX(),
                                 meETLTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff_[0]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffPhiMtdZneg, meETLTrackEffPhiTotZneg, meEtlPhiEff_[0]);

  meEtlPtEff_[0] = ibook.book1D("EtlPtEffZneg",
                                "Track Efficiency VS Pt (-Z);Pt [GeV];Efficiency",
                                meETLTrackEffPtTotZneg->getNbinsX(),
                                meETLTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                meETLTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff_[0]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffPtMtdZneg, meETLTrackEffPtTotZneg, meEtlPtEff_[0]);

  meEtlEtaEff_[1] = ibook.book1D("EtlEtaEffZpos",
                                 " Track Efficiency VS Eta (+Z);#eta;Efficiency",
                                 meETLTrackEffEtaTotZpos->getNbinsX(),
                                 meETLTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff_[1]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffEtaMtdZpos, meETLTrackEffEtaTotZpos, meEtlEtaEff_[1]);

  meEtlPhiEff_[1] = ibook.book1D("EtlPhiEffZpos",
                                 "Track Efficiency VS Phi (+Z);#phi [rad];Efficiency",
                                 meETLTrackEffPhiTotZpos->getNbinsX(),
                                 meETLTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff_[1]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffPhiMtdZpos, meETLTrackEffPhiTotZpos, meEtlPhiEff_[1]);

  meEtlPtEff_[1] = ibook.book1D("EtlPtEffZpos",
                                "Track Efficiency VS Pt (+Z);Pt [GeV];Efficiency",
                                meETLTrackEffPtTotZpos->getNbinsX(),
                                meETLTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                meETLTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff_[1]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffPtMtdZpos, meETLTrackEffPtTotZpos, meEtlPtEff_[1]);

  meEtlEtaEff2_[0] = ibook.book1D("EtlEtaEff2Zneg",
                                  " Track Efficiency VS Eta (-Z, 2 hit);#eta;Efficiency",
                                  meETLTrackEffEtaTotZneg->getNbinsX(),
                                  meETLTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                  meETLTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff2_[0]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffEta2MtdZneg, meETLTrackEffEtaTotZneg, meEtlEtaEff2_[0]);

  meEtlPhiEff2_[0] = ibook.book1D("EtlPhiEff2Zneg",
                                  "Track Efficiency VS Phi (-Z, 2 hit);#phi [rad];Efficiency",
                                  meETLTrackEffPhiTotZneg->getNbinsX(),
                                  meETLTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                  meETLTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff2_[0]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffPhi2MtdZneg, meETLTrackEffPhiTotZneg, meEtlPhiEff2_[0]);

  meEtlPtEff2_[0] = ibook.book1D("EtlPtEff2Zneg",
                                 "Track Efficiency VS Pt (-Z, 2 hit);Pt [GeV];Efficiency",
                                 meETLTrackEffPtTotZneg->getNbinsX(),
                                 meETLTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff2_[0]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffPt2MtdZneg, meETLTrackEffPtTotZneg, meEtlPtEff2_[0]);

  meEtlEtaEff2_[1] = ibook.book1D("EtlEtaEff2Zpos",
                                  "Track Efficiency VS Eta (+Z, 2 hit);#eta;Efficiency",
                                  meETLTrackEffEtaTotZpos->getNbinsX(),
                                  meETLTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                  meETLTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff2_[1]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffEta2MtdZpos, meETLTrackEffEtaTotZpos, meEtlEtaEff2_[1]);

  meEtlPhiEff2_[1] = ibook.book1D("EtlPhiEff2Zpos",
                                  "Track Efficiency VS Phi (+Z, 2 hit);#phi [rad];Efficiency",
                                  meETLTrackEffPhiTotZpos->getNbinsX(),
                                  meETLTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                  meETLTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff2_[1]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffPhi2MtdZpos, meETLTrackEffPhiTotZpos, meEtlPhiEff2_[1]);

  meEtlPtEff2_[1] = ibook.book1D("EtlPtEff2Zpos",
                                 "Track Efficiency VS Pt (+Z, 2 hit);Pt [GeV];Efficiency",
                                 meETLTrackEffPtTotZpos->getNbinsX(),
                                 meETLTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                 meETLTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff2_[1]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEffPt2MtdZpos, meETLTrackEffPtTotZpos, meEtlPtEff2_[1]);

  meMVAPtSelEff_ = ibook.book1D("MVAPtSelEff",
                                "Track selected efficiency VS Pt;Pt [GeV];Efficiency",
                                meMVATrackEffPtTot->getNbinsX(),
                                meMVATrackEffPtTot->getTH1()->GetXaxis()->GetXmin(),
                                meMVATrackEffPtTot->getTH1()->GetXaxis()->GetXmax());
  meMVAPtSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meMVATrackMatchedEffPtTot, meMVATrackEffPtTot, meMVAPtSelEff_);

  meMVAEtaSelEff_ = ibook.book1D("MVAEtaSelEff",
                                 "Track selected efficiency VS Eta;Eta;Efficiency",
                                 meMVATrackEffEtaTot->getNbinsX(),
                                 meMVATrackEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                 meMVATrackEffEtaTot->getTH1()->GetXaxis()->GetXmax());
  meMVAEtaSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meMVATrackMatchedEffEtaTot, meMVATrackEffEtaTot, meMVAEtaSelEff_);

  meMVAPtMatchEff_ = ibook.book1D("MVAPtMatchEff",
                                  "Track matched to GEN efficiency VS Pt;Pt [GeV];Efficiency",
                                  meMVATrackMatchedEffPtTot->getNbinsX(),
                                  meMVATrackMatchedEffPtTot->getTH1()->GetXaxis()->GetXmin(),
                                  meMVATrackMatchedEffPtTot->getTH1()->GetXaxis()->GetXmax());
  meMVAPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meMVATrackMatchedEffPtMtd, meMVATrackMatchedEffPtTot, meMVAPtMatchEff_);

  meMVAEtaMatchEff_ = ibook.book1D("MVAEtaMatchEff",
                                   "Track matched to GEN efficiency VS Eta;Eta;Efficiency",
                                   meMVATrackMatchedEffEtaTot->getNbinsX(),
                                   meMVATrackMatchedEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                   meMVATrackMatchedEffEtaTot->getTH1()->GetXaxis()->GetXmax());
  meMVAEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meMVATrackMatchedEffEtaMtd, meMVATrackMatchedEffEtaTot, meMVAEtaMatchEff_);

  meTPPtSelEff_ = ibook.book1D("TPPtSelEff",
                               "Track selected efficiency TP VS Pt;Pt [GeV];Efficiency",
                               meMVATrackEffPtTot->getNbinsX(),
                               meMVATrackEffPtTot->getTH1()->GetXaxis()->GetXmin(),
                               meMVATrackEffPtTot->getTH1()->GetXaxis()->GetXmax());
  meTPPtSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meTrackMatchedTPEffPtTot, meMVATrackEffPtTot, meTPPtSelEff_);

  meTPEtaSelEff_ = ibook.book1D("TPEtaSelEff",
                                "Track selected efficiency TP VS Eta;Eta;Efficiency",
                                meMVATrackEffEtaTot->getNbinsX(),
                                meMVATrackEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                meMVATrackEffEtaTot->getTH1()->GetXaxis()->GetXmax());
  meTPEtaSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meTrackMatchedTPEffEtaTot, meMVATrackEffEtaTot, meTPEtaSelEff_);

  meTPPtMatchEff_ = ibook.book1D("TPPtMatchEff",
                                 "Track matched to TP efficiency VS Pt;Pt [GeV];Efficiency",
                                 meTrackMatchedTPEffPtTot->getNbinsX(),
                                 meTrackMatchedTPEffPtTot->getTH1()->GetXaxis()->GetXmin(),
                                 meTrackMatchedTPEffPtTot->getTH1()->GetXaxis()->GetXmax());
  meTPPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meTrackMatchedTPEffPtMtd, meTrackMatchedTPEffPtTot, meTPPtMatchEff_);

  meTPEtaMatchEff_ = ibook.book1D("TPEtaMatchEff",
                                  "Track matched to TP efficiency VS Eta;Eta;Efficiency",
                                  meTrackMatchedTPEffEtaTot->getNbinsX(),
                                  meTrackMatchedTPEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                  meTrackMatchedTPEffEtaTot->getTH1()->GetXaxis()->GetXmax());
  meTPEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meTrackMatchedTPEffEtaMtd, meTrackMatchedTPEffEtaTot, meTPEtaMatchEff_);

  meTPPtMatchEtl2Eff_ = ibook.book1D("TPPtMatchEtl2Eff",
                                     "Track matched to TP efficiency VS Pt, 2 ETL hits;Pt [GeV];Efficiency",
                                     meTrackMatchedTPEffPtTot->getNbinsX(),
                                     meTrackMatchedTPEffPtTot->getTH1()->GetXaxis()->GetXmin(),
                                     meTrackMatchedTPEffPtTot->getTH1()->GetXaxis()->GetXmax());
  meTPPtMatchEtl2Eff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meTrackMatchedTPEffPtEtl2Mtd, meTrackMatchedTPEffPtTot, meTPPtMatchEtl2Eff_);

  meTPEtaMatchEtl2Eff_ = ibook.book1D("TPEtaMatchEtl2Eff",
                                      "Track matched to TP efficiency VS Eta, 2 ETL hits;Eta;Efficiency",
                                      meTrackMatchedTPEffEtaTot->getNbinsX(),
                                      meTrackMatchedTPEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                      meTrackMatchedTPEffEtaTot->getTH1()->GetXaxis()->GetXmax());
  meTPEtaMatchEtl2Eff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meTrackMatchedTPEffEtaEtl2Mtd, meTrackMatchedTPEffEtaTot, meTPEtaMatchEtl2Eff_);

  meTPmtdPtSelEff_ = ibook.book1D("TPmtdPtSelEff",
                                  "Track selected efficiency TP-mtd hit VS Pt;Pt [GeV];Efficiency",
                                  meMVATrackEffPtTot->getNbinsX(),
                                  meMVATrackEffPtTot->getTH1()->GetXaxis()->GetXmin(),
                                  meMVATrackEffPtTot->getTH1()->GetXaxis()->GetXmax());
  meTPmtdPtSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meTrackMatchedTPmtdEffPtTot, meMVATrackEffPtTot, meTPmtdPtSelEff_);

  meTPmtdEtaSelEff_ = ibook.book1D("TPmtdEtaSelEff",
                                   "Track selected efficiency TPmtd hit VS Eta;Eta;Efficiency",
                                   meMVATrackEffEtaTot->getNbinsX(),
                                   meMVATrackEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                   meMVATrackEffEtaTot->getTH1()->GetXaxis()->GetXmax());
  meTPmtdEtaSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meTrackMatchedTPmtdEffEtaTot, meMVATrackEffEtaTot, meTPmtdEtaSelEff_);

  meTPmtdPtMatchEff_ = ibook.book1D("TPmtdPtMatchEff",
                                    "Track matched to TP-mtd hit efficiency VS Pt;Pt [GeV];Efficiency",
                                    meTrackMatchedTPmtdEffPtTot->getNbinsX(),
                                    meTrackMatchedTPmtdEffPtTot->getTH1()->GetXaxis()->GetXmin(),
                                    meTrackMatchedTPmtdEffPtTot->getTH1()->GetXaxis()->GetXmax());
  meTPmtdPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meTrackMatchedTPmtdEffPtMtd, meTrackMatchedTPmtdEffPtTot, meTPmtdPtMatchEff_);

  meTPmtdEtaMatchEff_ = ibook.book1D("TPmtdEtaMatchEff",
                                     "Track matched to TP-mtd hit efficiency VS Eta;Eta;Efficiency",
                                     meTrackMatchedTPmtdEffEtaTot->getNbinsX(),
                                     meTrackMatchedTPmtdEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                     meTrackMatchedTPmtdEffEtaTot->getTH1()->GetXaxis()->GetXmax());
  meTPmtdEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meTrackMatchedTPmtdEffEtaMtd, meTrackMatchedTPmtdEffEtaTot, meTPmtdEtaMatchEff_);

  meTPAssocEff_ =
      ibook.book1D("TPAssocEff",
                   "Tracking particles not associated to any MTD cell in events with at least one cell over threshold",
                   meNTrackingParticles->getNbinsX(),
                   meNTrackingParticles->getTH1()->GetXaxis()->GetXmin(),
                   meNTrackingParticles->getTH1()->GetXaxis()->GetXmax());
  meTPAssocEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meUnassDeposit, meNTrackingParticles, meTPAssocEff_);

  meBtlEtaEff_->getTH1()->SetMinimum(0.);
  meBtlPhiEff_->getTH1()->SetMinimum(0.);
  meBtlPtEff_->getTH1()->SetMinimum(0.);
  for (int i = 0; i < 2; i++) {
    meEtlEtaEff_[i]->getTH1()->SetMinimum(0.);
    meEtlPhiEff_[i]->getTH1()->SetMinimum(0.);
    meEtlPtEff_[i]->getTH1()->SetMinimum(0.);
    meEtlEtaEff2_[i]->getTH1()->SetMinimum(0.);
    meEtlPhiEff2_[i]->getTH1()->SetMinimum(0.);
    meEtlPtEff2_[i]->getTH1()->SetMinimum(0.);
  }
  meMVAPtSelEff_->getTH1()->SetMinimum(0.);
  meMVAEtaSelEff_->getTH1()->SetMinimum(0.);
  meMVAPtMatchEff_->getTH1()->SetMinimum(0.);
  meMVAEtaMatchEff_->getTH1()->SetMinimum(0.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ----------
void MtdTracksHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/Tracks/");

  descriptions.add("MtdTracksPostProcessor", desc);
}

DEFINE_FWK_MODULE(MtdTracksHarvester);
