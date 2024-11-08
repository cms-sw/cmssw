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
  void normalize(MonitorElement* h, double scale);

  const std::string folder_;

  // --- Histograms
  MonitorElement* meBtlEtaEff_;
  MonitorElement* meBtlPhiEff_;
  MonitorElement* meBtlPtEff_;
  MonitorElement* meEtlEtaEff_;
  MonitorElement* meEtlPhiEff_;
  MonitorElement* meEtlPtEff_;
  MonitorElement* meEtlEtaEff2_;
  MonitorElement* meEtlPhiEff2_;
  MonitorElement* meEtlPtEff2_;
  MonitorElement* meEtlEtaEffLowPt_[2];
  MonitorElement* meEtlEtaEff2LowPt_[2];

  MonitorElement* meBTLTPPtSelEff_;
  MonitorElement* meBTLTPEtaSelEff_;
  MonitorElement* meBTLTPPtMatchEff_;
  MonitorElement* meBTLTPEtaMatchEff_;
  MonitorElement* meETLTPPtSelEff_;
  MonitorElement* meETLTPEtaSelEff_;
  MonitorElement* meETLTPPtMatchEff_;
  MonitorElement* meETLTPEtaMatchEff_;
  MonitorElement* meETLTPPtMatchEff2_;
  MonitorElement* meETLTPEtaMatchEff2_;

  // - BTL track-mtd matching efficiencies
  MonitorElement* meBTLTPmtdDirectEtaSelEff_;
  MonitorElement* meBTLTPmtdDirectPtSelEff_;
  MonitorElement* meBTLTPmtdOtherEtaSelEff_;
  MonitorElement* meBTLTPmtdOtherPtSelEff_;
  MonitorElement* meBTLTPnomtdEtaSelEff_;
  MonitorElement* meBTLTPnomtdPtSelEff_;

  MonitorElement* meBTLTPmtdDirectCorrectAssocEtaMatchEff_;
  MonitorElement* meBTLTPmtdDirectCorrectAssocPtMatchEff_;
  MonitorElement* meBTLTPmtdDirectWrongAssocEtaMatchEff_;
  MonitorElement* meBTLTPmtdDirectWrongAssocPtMatchEff_;
  MonitorElement* meBTLTPmtdDirectNoAssocEtaMatchEff_;
  MonitorElement* meBTLTPmtdDirectNoAssocPtMatchEff_;

  MonitorElement* meBTLTPmtdOtherCorrectAssocEtaMatchEff_;
  MonitorElement* meBTLTPmtdOtherCorrectAssocPtMatchEff_;
  MonitorElement* meBTLTPmtdOtherWrongAssocEtaMatchEff_;
  MonitorElement* meBTLTPmtdOtherWrongAssocPtMatchEff_;
  MonitorElement* meBTLTPmtdOtherNoAssocEtaMatchEff_;
  MonitorElement* meBTLTPmtdOtherNoAssocPtMatchEff_;

  MonitorElement* meBTLTPnomtdEtaMatchEff_;
  MonitorElement* meBTLTPnomtdPtMatchEff_;

  // - ETL track-mtd matching efficiencies
  MonitorElement* meETLTPmtd1EtaSelEff_;
  MonitorElement* meETLTPmtd1PtSelEff_;
  MonitorElement* meETLTPmtd2EtaSelEff_;
  MonitorElement* meETLTPmtd2PtSelEff_;
  MonitorElement* meETLTPnomtdEtaSelEff_;
  MonitorElement* meETLTPnomtdPtSelEff_;

  MonitorElement* meETLTPmtd1CorrectAssocEtaMatchEff_;
  MonitorElement* meETLTPmtd1CorrectAssocPtMatchEff_;
  MonitorElement* meETLTPmtd1WrongAssocEtaMatchEff_;
  MonitorElement* meETLTPmtd1WrongAssocPtMatchEff_;
  MonitorElement* meETLTPmtd1NoAssocEtaMatchEff_;
  MonitorElement* meETLTPmtd1NoAssocPtMatchEff_;

  MonitorElement* meETLTPmtd2CorrectAssocEtaMatchEff_;
  MonitorElement* meETLTPmtd2CorrectAssocPtMatchEff_;
  MonitorElement* meETLTPmtd2WrongAssocEtaMatchEff_;
  MonitorElement* meETLTPmtd2WrongAssocPtMatchEff_;
  MonitorElement* meETLTPmtd2NoAssocEtaMatchEff_;
  MonitorElement* meETLTPmtd2NoAssocPtMatchEff_;

  MonitorElement* meETLTPnomtdEtaMatchEff_;
  MonitorElement* meETLTPnomtdPtMatchEff_;

  // -
  MonitorElement* meNoTimeFraction_;
  MonitorElement* meExtraPtEff_;
  MonitorElement* meExtraPtEtl2Eff_;
  MonitorElement* meExtraEtaEff_;
  MonitorElement* meExtraEtaEtl2Eff_;
  MonitorElement* meExtraPhiAtBTLEff_;
  MonitorElement* meExtraMTDfailExtenderEtaEff_;
  MonitorElement* meExtraMTDfailExtenderPtEff_;
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

void MtdTracksHarvester::normalize(MonitorElement* h, double scale) {
  double integral = h->getTH1F()->Integral();
  double norma = (integral > 0.) ? scale / integral : 0.;
  for (int ibin = 1; ibin <= h->getNbinsX(); ibin++) {
    double eff = h->getBinContent(ibin) * norma;
    double bin_err = h->getBinError(ibin) * norma;
    h->setBinContent(ibin, eff);
    h->setBinError(ibin, bin_err);
  }
}

// ------------ endjob tasks ----------------------------
void MtdTracksHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& igetter) {
  // --- Get the monitoring histograms
  MonitorElement* meBTLTrackEtaTot = igetter.get(folder_ + "TrackBTLEtaTot");
  MonitorElement* meBTLTrackPhiTot = igetter.get(folder_ + "TrackBTLPhiTot");
  MonitorElement* meBTLTrackPtTot = igetter.get(folder_ + "TrackBTLPtTot");
  MonitorElement* meBTLTrackEtaMtd = igetter.get(folder_ + "TrackBTLEtaMtd");
  MonitorElement* meBTLTrackPhiMtd = igetter.get(folder_ + "TrackBTLPhiMtd");
  MonitorElement* meBTLTrackPtMtd = igetter.get(folder_ + "TrackBTLPtMtd");

  MonitorElement* meETLTrackEtaTot = igetter.get(folder_ + "TrackETLEtaTot");
  MonitorElement* meETLTrackPhiTot = igetter.get(folder_ + "TrackETLPhiTot");
  MonitorElement* meETLTrackPtTot = igetter.get(folder_ + "TrackETLPtTot");
  MonitorElement* meETLTrackEtaMtd = igetter.get(folder_ + "TrackETLEtaMtd");
  MonitorElement* meETLTrackPhiMtd = igetter.get(folder_ + "TrackETLPhiMtd");
  MonitorElement* meETLTrackPtMtd = igetter.get(folder_ + "TrackETLPtMtd");
  MonitorElement* meETLTrackEta2Mtd = igetter.get(folder_ + "TrackETLEta2Mtd");
  MonitorElement* meETLTrackPhi2Mtd = igetter.get(folder_ + "TrackETLPhi2Mtd");
  MonitorElement* meETLTrackPt2Mtd = igetter.get(folder_ + "TrackETLPt2Mtd");

  MonitorElement* meETLTrackEtaTotLowPt0 = igetter.get(folder_ + "TrackETLEtaTotLowPt0");
  MonitorElement* meETLTrackEtaTotLowPt1 = igetter.get(folder_ + "TrackETLEtaTotLowPt1");
  MonitorElement* meETLTrackEtaMtdLowPt0 = igetter.get(folder_ + "TrackETLEtaMtdLowPt0");
  MonitorElement* meETLTrackEtaMtdLowPt1 = igetter.get(folder_ + "TrackETLEtaMtdLowPt1");
  MonitorElement* meETLTrackEta2MtdLowPt0 = igetter.get(folder_ + "TrackETLEta2MtdLowPt0");
  MonitorElement* meETLTrackEta2MtdLowPt1 = igetter.get(folder_ + "TrackETLEta2MtdLowPt1");

  MonitorElement* meExtraPtMtd = igetter.get(folder_ + "ExtraPtMtd");
  MonitorElement* meExtraPtEtl2Mtd = igetter.get(folder_ + "ExtraPtEtl2Mtd");
  MonitorElement* meTrackMatchedTPPtTotLV = igetter.get(folder_ + "MatchedTPPtTotLV");
  MonitorElement* meExtraEtaMtd = igetter.get(folder_ + "ExtraEtaMtd");
  MonitorElement* meExtraEtaEtl2Mtd = igetter.get(folder_ + "ExtraEtaEtl2Mtd");
  MonitorElement* meTrackMatchedTPEtaTotLV = igetter.get(folder_ + "MatchedTPEtaTotLV");

  MonitorElement* meBTLTrackMatchedTPPtTot = igetter.get(folder_ + "BTLTrackMatchedTPPtTot");
  MonitorElement* meBTLTrackMatchedTPPtMtd = igetter.get(folder_ + "BTLTrackMatchedTPPtMtd");
  MonitorElement* meBTLTrackMatchedTPEtaTot = igetter.get(folder_ + "BTLTrackMatchedTPEtaTot");
  MonitorElement* meBTLTrackMatchedTPEtaMtd = igetter.get(folder_ + "BTLTrackMatchedTPEtaMtd");
  MonitorElement* meETLTrackMatchedTPPtTot = igetter.get(folder_ + "ETLTrackMatchedTPPtTot");
  MonitorElement* meETLTrackMatchedTPPtMtd = igetter.get(folder_ + "ETLTrackMatchedTPPtMtd");
  MonitorElement* meETLTrackMatchedTPPt2Mtd = igetter.get(folder_ + "ETLTrackMatchedTPPt2Mtd");
  MonitorElement* meETLTrackMatchedTPEtaTot = igetter.get(folder_ + "ETLTrackMatchedTPEtaTot");
  MonitorElement* meETLTrackMatchedTPEtaMtd = igetter.get(folder_ + "ETLTrackMatchedTPEtaMtd");
  MonitorElement* meETLTrackMatchedTPEta2Mtd = igetter.get(folder_ + "ETLTrackMatchedTPEta2Mtd");

  //
  MonitorElement* meBTLTrackMatchedTPmtdDirectEta = igetter.get(folder_ + "BTLTrackMatchedTPmtdDirectEta");
  MonitorElement* meBTLTrackMatchedTPmtdDirectPt = igetter.get(folder_ + "BTLTrackMatchedTPmtdDirectPt");
  MonitorElement* meBTLTrackMatchedTPmtdOtherEta = igetter.get(folder_ + "BTLTrackMatchedTPmtdOtherEta");
  MonitorElement* meBTLTrackMatchedTPmtdOtherPt = igetter.get(folder_ + "BTLTrackMatchedTPmtdOtherPt");
  ;
  MonitorElement* meBTLTrackMatchedTPnomtdEta = igetter.get(folder_ + "BTLTrackMatchedTPnomtdEta");
  MonitorElement* meBTLTrackMatchedTPnomtdPt = igetter.get(folder_ + "BTLTrackMatchedTPnomtdPt");

  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocEta =
      igetter.get(folder_ + "BTLTrackMatchedTPmtdDirectCorrectAssocEta");
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocPt =
      igetter.get(folder_ + "BTLTrackMatchedTPmtdDirectCorrectAssocPt");
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocEta =
      igetter.get(folder_ + "BTLTrackMatchedTPmtdDirectWrongAssocEta");
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocPt =
      igetter.get(folder_ + "BTLTrackMatchedTPmtdDirectWrongAssocPt");
  MonitorElement* meBTLTrackMatchedTPmtdDirectNoAssocEta =
      igetter.get(folder_ + "BTLTrackMatchedTPmtdDirectNoAssocEta");
  MonitorElement* meBTLTrackMatchedTPmtdDirectNoAssocPt = igetter.get(folder_ + "BTLTrackMatchedTPmtdDirectNoAssocPt");

  MonitorElement* meBTLTrackMatchedTPmtdOtherCorrectAssocEta =
      igetter.get(folder_ + "BTLTrackMatchedTPmtdOtherCorrectAssocEta");
  MonitorElement* meBTLTrackMatchedTPmtdOtherCorrectAssocPt =
      igetter.get(folder_ + "BTLTrackMatchedTPmtdOtherCorrectAssocPt");
  MonitorElement* meBTLTrackMatchedTPmtdOtherWrongAssocEta =
      igetter.get(folder_ + "BTLTrackMatchedTPmtdOtherWrongAssocEta");
  MonitorElement* meBTLTrackMatchedTPmtdOtherWrongAssocPt =
      igetter.get(folder_ + "BTLTrackMatchedTPmtdOtherWrongAssocPt");
  MonitorElement* meBTLTrackMatchedTPmtdOtherNoAssocEta = igetter.get(folder_ + "BTLTrackMatchedTPmtdOtherNoAssocEta");
  MonitorElement* meBTLTrackMatchedTPmtdOtherNoAssocPt = igetter.get(folder_ + "BTLTrackMatchedTPmtdOtherNoAssocPt");

  MonitorElement* meBTLTrackMatchedTPnomtdAssocEta = igetter.get(folder_ + "BTLTrackMatchedTPnomtdAssocEta");
  MonitorElement* meBTLTrackMatchedTPnomtdAssocPt = igetter.get(folder_ + "BTLTrackMatchedTPnomtdAssocPt");

  MonitorElement* meETLTrackMatchedTPmtd1Eta = igetter.get(folder_ + "ETLTrackMatchedTPmtd1Eta");
  MonitorElement* meETLTrackMatchedTPmtd1Pt = igetter.get(folder_ + "ETLTrackMatchedTPmtd1Pt");
  MonitorElement* meETLTrackMatchedTPmtd2Eta = igetter.get(folder_ + "ETLTrackMatchedTPmtd2Eta");
  MonitorElement* meETLTrackMatchedTPmtd2Pt = igetter.get(folder_ + "ETLTrackMatchedTPmtd2Pt");
  ;
  MonitorElement* meETLTrackMatchedTPnomtdEta = igetter.get(folder_ + "ETLTrackMatchedTPnomtdEta");
  MonitorElement* meETLTrackMatchedTPnomtdPt = igetter.get(folder_ + "ETLTrackMatchedTPnomtdPt");

  MonitorElement* meETLTrackMatchedTPmtd1CorrectAssocEta =
      igetter.get(folder_ + "ETLTrackMatchedTPmtd1CorrectAssocEta");
  MonitorElement* meETLTrackMatchedTPmtd1CorrectAssocPt = igetter.get(folder_ + "ETLTrackMatchedTPmtd1CorrectAssocPt");
  MonitorElement* meETLTrackMatchedTPmtd1WrongAssocEta = igetter.get(folder_ + "ETLTrackMatchedTPmtd1WrongAssocEta");
  MonitorElement* meETLTrackMatchedTPmtd1WrongAssocPt = igetter.get(folder_ + "ETLTrackMatchedTPmtd1WrongAssocPt");
  MonitorElement* meETLTrackMatchedTPmtd1NoAssocEta = igetter.get(folder_ + "ETLTrackMatchedTPmtd1NoAssocEta");
  MonitorElement* meETLTrackMatchedTPmtd1NoAssocPt = igetter.get(folder_ + "ETLTrackMatchedTPmtd1NoAssocPt");

  MonitorElement* meETLTrackMatchedTPmtd2CorrectAssocEta =
      igetter.get(folder_ + "ETLTrackMatchedTPmtd2CorrectAssocEta");
  MonitorElement* meETLTrackMatchedTPmtd2CorrectAssocPt = igetter.get(folder_ + "ETLTrackMatchedTPmtd2CorrectAssocPt");
  MonitorElement* meETLTrackMatchedTPmtd2WrongAssocEta = igetter.get(folder_ + "ETLTrackMatchedTPmtd2WrongAssocEta");
  MonitorElement* meETLTrackMatchedTPmtd2WrongAssocPt = igetter.get(folder_ + "ETLTrackMatchedTPmtd2WrongAssocPt");
  MonitorElement* meETLTrackMatchedTPmtd2NoAssocEta = igetter.get(folder_ + "ETLTrackMatchedTPmtd2NoAssocEta");
  MonitorElement* meETLTrackMatchedTPmtd2NoAssocPt = igetter.get(folder_ + "ETLTrackMatchedTPmtd2NoAssocPt");

  MonitorElement* meETLTrackMatchedTPnomtdAssocEta = igetter.get(folder_ + "ETLTrackMatchedTPnomtdAssocEta");
  MonitorElement* meETLTrackMatchedTPnomtdAssocPt = igetter.get(folder_ + "ETLTrackMatchedTPnomtdAssocPt");

  //
  MonitorElement* meTrackNumHits = igetter.get(folder_ + "TrackNumHits");
  MonitorElement* meTrackNumHitsNT = igetter.get(folder_ + "TrackNumHitsNT");
  MonitorElement* meExtraPhiAtBTL = igetter.get(folder_ + "ExtraPhiAtBTL");
  MonitorElement* meExtraPhiAtBTLmatched = igetter.get(folder_ + "ExtraPhiAtBTLmatched");
  MonitorElement* meExtraBTLeneInCone = igetter.get(folder_ + "ExtraBTLeneInCone");
  MonitorElement* meExtraMTDfailExtenderEta = igetter.get(folder_ + "ExtraMTDfailExtenderEta");
  MonitorElement* meExtraMTDfailExtenderPt = igetter.get(folder_ + "ExtraMTDfailExtenderPt");

  if (!meBTLTrackEtaTot || !meBTLTrackPhiTot || !meBTLTrackPtTot || !meBTLTrackEtaMtd || !meBTLTrackPhiMtd ||
      !meBTLTrackPtMtd || !meETLTrackEtaTot || !meETLTrackPhiTot || !meETLTrackPtTot || !meETLTrackEtaMtd ||
      !meETLTrackPhiMtd || !meETLTrackPtMtd || !meETLTrackEta2Mtd || !meETLTrackPhi2Mtd || !meETLTrackPt2Mtd ||
      !meETLTrackEtaTotLowPt0 || !meETLTrackEtaTotLowPt1 || !meETLTrackEtaMtdLowPt0 || !meETLTrackEtaMtdLowPt1 ||
      !meETLTrackEta2MtdLowPt0 || !meETLTrackEta2MtdLowPt1 || !meTrackMatchedTPPtTotLV || !meTrackMatchedTPEtaTotLV ||

      !meBTLTrackMatchedTPPtTot || !meBTLTrackMatchedTPPtMtd || !meBTLTrackMatchedTPEtaTot ||
      !meBTLTrackMatchedTPEtaMtd || !meETLTrackMatchedTPPtTot || !meETLTrackMatchedTPPtMtd ||
      !meETLTrackMatchedTPPt2Mtd || !meETLTrackMatchedTPEtaTot || !meETLTrackMatchedTPEtaMtd ||
      !meETLTrackMatchedTPEta2Mtd ||

      !meBTLTrackMatchedTPmtdDirectEta || !meBTLTrackMatchedTPmtdDirectPt || !meBTLTrackMatchedTPmtdOtherEta ||
      !meBTLTrackMatchedTPmtdOtherPt || !meBTLTrackMatchedTPnomtdEta || !meBTLTrackMatchedTPnomtdPt ||
      !meBTLTrackMatchedTPmtdDirectCorrectAssocEta || !meBTLTrackMatchedTPmtdDirectCorrectAssocPt ||
      !meBTLTrackMatchedTPmtdDirectWrongAssocEta || !meBTLTrackMatchedTPmtdDirectWrongAssocPt ||
      !meBTLTrackMatchedTPmtdDirectNoAssocEta || !meBTLTrackMatchedTPmtdDirectNoAssocPt ||
      !meBTLTrackMatchedTPmtdOtherCorrectAssocEta || !meBTLTrackMatchedTPmtdOtherCorrectAssocPt ||
      !meBTLTrackMatchedTPmtdOtherWrongAssocEta || !meBTLTrackMatchedTPmtdOtherWrongAssocPt ||
      !meBTLTrackMatchedTPmtdOtherNoAssocEta || !meBTLTrackMatchedTPmtdOtherNoAssocPt ||
      !meBTLTrackMatchedTPnomtdAssocEta || !meBTLTrackMatchedTPnomtdAssocPt || !meETLTrackMatchedTPmtd1Eta ||
      !meETLTrackMatchedTPmtd1Pt || !meETLTrackMatchedTPmtd2Eta || !meETLTrackMatchedTPmtd2Pt ||
      !meETLTrackMatchedTPnomtdEta || !meETLTrackMatchedTPnomtdPt || !meETLTrackMatchedTPmtd1CorrectAssocEta ||
      !meETLTrackMatchedTPmtd1CorrectAssocPt || !meETLTrackMatchedTPmtd1WrongAssocEta ||
      !meETLTrackMatchedTPmtd1WrongAssocPt || !meETLTrackMatchedTPmtd1NoAssocEta || !meETLTrackMatchedTPmtd1NoAssocPt ||
      !meETLTrackMatchedTPmtd2CorrectAssocEta || !meETLTrackMatchedTPmtd2CorrectAssocPt ||
      !meETLTrackMatchedTPmtd2WrongAssocEta || !meETLTrackMatchedTPmtd2WrongAssocPt ||
      !meETLTrackMatchedTPmtd2NoAssocEta || !meETLTrackMatchedTPmtd2NoAssocPt || !meETLTrackMatchedTPnomtdAssocEta ||
      !meETLTrackMatchedTPnomtdAssocPt || !meTrackNumHits || !meTrackNumHitsNT || !meExtraPtMtd || !meExtraPtEtl2Mtd ||
      !meExtraEtaMtd || !meExtraEtaEtl2Mtd || !meExtraPhiAtBTL || !meExtraPhiAtBTLmatched || !meExtraBTLeneInCone ||
      !meExtraMTDfailExtenderEta || !meExtraMTDfailExtenderPt) {
    edm::LogError("MtdTracksHarvester") << "Monitoring histograms not found!" << std::endl;
    return;
  }

  // --- Book  histograms
  ibook.cd(folder_);
  meBtlEtaEff_ = ibook.book1D("BtlEtaEff",
                              " Track Efficiency VS Eta;#eta;Efficiency",
                              meBTLTrackEtaTot->getNbinsX(),
                              meBTLTrackEtaTot->getTH1()->GetXaxis()->GetXmin(),
                              meBTLTrackEtaTot->getTH1()->GetXaxis()->GetXmax());
  meBtlEtaEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackEtaMtd, meBTLTrackEtaTot, meBtlEtaEff_);

  meBtlPhiEff_ = ibook.book1D("BtlPhiEff",
                              "Track Efficiency VS Phi;#phi [rad];Efficiency",
                              meBTLTrackPhiTot->getNbinsX(),
                              meBTLTrackPhiTot->getTH1()->GetXaxis()->GetXmin(),
                              meBTLTrackPhiTot->getTH1()->GetXaxis()->GetXmax());
  meBtlPhiEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackPhiMtd, meBTLTrackPhiTot, meBtlPhiEff_);

  meBtlPtEff_ = ibook.book1D("BtlPtEff",
                             "Track Efficiency VS Pt;Pt [GeV];Efficiency",
                             meBTLTrackPtTot->getNbinsX(),
                             meBTLTrackPtTot->getTH1()->GetXaxis()->GetXmin(),
                             meBTLTrackPtTot->getTH1()->GetXaxis()->GetXmax());
  meBtlPtEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackPtMtd, meBTLTrackPtTot, meBtlPtEff_);

  meEtlEtaEff_ = ibook.book1D("EtlEtaEff",
                              " Track Efficiency VS Eta;#eta;Efficiency",
                              meETLTrackEtaTot->getNbinsX(),
                              meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmin(),
                              meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEtaMtd, meETLTrackEtaTot, meEtlEtaEff_);

  meEtlPhiEff_ = ibook.book1D("EtlPhiEff",
                              "Track Efficiency VS Phi;#phi [rad];Efficiency",
                              meETLTrackPhiTot->getNbinsX(),
                              meETLTrackPhiTot->getTH1()->GetXaxis()->GetXmin(),
                              meETLTrackPhiTot->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackPhiMtd, meETLTrackPhiTot, meEtlPhiEff_);

  meEtlPtEff_ = ibook.book1D("EtlPtEff",
                             "Track Efficiency VS Pt;Pt [GeV];Efficiency",
                             meETLTrackPtTot->getNbinsX(),
                             meETLTrackPtTot->getTH1()->GetXaxis()->GetXmin(),
                             meETLTrackPtTot->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackPtMtd, meETLTrackPtTot, meEtlPtEff_);

  meEtlEtaEff2_ = ibook.book1D("EtlEtaEff2",
                               " Track Efficiency VS Eta (2 hits);#eta;Efficiency",
                               meETLTrackEtaTot->getNbinsX(),
                               meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmin(),
                               meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff2_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEta2Mtd, meETLTrackEtaTot, meEtlEtaEff2_);

  meEtlPhiEff2_ = ibook.book1D("EtlPhiEff2",
                               "Track Efficiency VS Phi (2 hits);#phi [rad];Efficiency",
                               meETLTrackPhiTot->getNbinsX(),
                               meETLTrackPhiTot->getTH1()->GetXaxis()->GetXmin(),
                               meETLTrackPhiTot->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff2_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackPhi2Mtd, meETLTrackPhiTot, meEtlPhiEff2_);

  meEtlPtEff2_ = ibook.book1D("EtlPtEff2",
                              "Track Efficiency VS Pt (2 hits);Pt [GeV];Efficiency",
                              meETLTrackPtTot->getNbinsX(),
                              meETLTrackPtTot->getTH1()->GetXaxis()->GetXmin(),
                              meETLTrackPtTot->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff2_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackPt2Mtd, meETLTrackPtTot, meEtlPtEff2_);

  // low pT
  meEtlEtaEffLowPt_[0] = ibook.book1D("EtlEtaEffLowPt0",
                                      " Track Efficiency VS Eta, 0.2 < pt < 0.45;#eta;Efficiency",
                                      meETLTrackEtaTotLowPt0->getNbinsX(),
                                      meETLTrackEtaTotLowPt0->getTH1()->GetXaxis()->GetXmin(),
                                      meETLTrackEtaTotLowPt0->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEffLowPt_[0]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEtaMtdLowPt0, meETLTrackEtaTotLowPt0, meEtlEtaEffLowPt_[0]);

  meEtlEtaEffLowPt_[1] = ibook.book1D("EtlEtaEffLowPt1",
                                      " Track Efficiency VS Eta, 0.45 < pt < 0.7;#eta;Efficiency",
                                      meETLTrackEtaTotLowPt1->getNbinsX(),
                                      meETLTrackEtaTotLowPt1->getTH1()->GetXaxis()->GetXmin(),
                                      meETLTrackEtaTotLowPt1->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEffLowPt_[1]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEtaMtdLowPt1, meETLTrackEtaTotLowPt1, meEtlEtaEffLowPt_[1]);

  meEtlEtaEff2LowPt_[0] = ibook.book1D("EtlEtaEff2LowPt0",
                                       " Track Efficiency VS Eta (2 hits), 0.2 < pt < 0.45;#eta;Efficiency",
                                       meETLTrackEtaTotLowPt0->getNbinsX(),
                                       meETLTrackEtaTotLowPt0->getTH1()->GetXaxis()->GetXmin(),
                                       meETLTrackEtaTotLowPt0->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff2LowPt_[0]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEta2MtdLowPt0, meETLTrackEtaTotLowPt0, meEtlEtaEff2LowPt_[0]);

  meEtlEtaEff2LowPt_[1] = ibook.book1D("EtlEtaEff2LowPt1",
                                       " Track Efficiency VS Eta (2 hits), 0.45 < pt < 0.7;#eta;Efficiency",
                                       meETLTrackEtaTotLowPt1->getNbinsX(),
                                       meETLTrackEtaTotLowPt1->getTH1()->GetXaxis()->GetXmin(),
                                       meETLTrackEtaTotLowPt1->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff2LowPt_[1]->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackEta2MtdLowPt1, meETLTrackEtaTotLowPt1, meEtlEtaEff2LowPt_[1]);

  meExtraPtEff_ =
      ibook.book1D("ExtraPtEff",
                   "MTD matching efficiency wrt extrapolated track associated to LV VS Pt;Pt [GeV];Efficiency",
                   meTrackMatchedTPPtTotLV->getNbinsX(),
                   meTrackMatchedTPPtTotLV->getTH1()->GetXaxis()->GetXmin(),
                   meTrackMatchedTPPtTotLV->getTH1()->GetXaxis()->GetXmax());
  meExtraPtEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meExtraPtMtd, meTrackMatchedTPPtTotLV, meExtraPtEff_);

  meExtraPtEtl2Eff_ =
      ibook.book1D("ExtraPtEtl2Eff",
                   "MTD matching efficiency (2 ETL) wrt extrapolated track associated to LV VS Pt;Pt [GeV];Efficiency",
                   meTrackMatchedTPPtTotLV->getNbinsX(),
                   meTrackMatchedTPPtTotLV->getTH1()->GetXaxis()->GetXmin(),
                   meTrackMatchedTPPtTotLV->getTH1()->GetXaxis()->GetXmax());
  meExtraPtEtl2Eff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meExtraPtEtl2Mtd, meTrackMatchedTPPtTotLV, meExtraPtEtl2Eff_);

  meExtraEtaEff_ = ibook.book1D("ExtraEtaEff",
                                "MTD matching efficiency wrt extrapolated track associated to LV VS Eta;Eta;Efficiency",
                                meTrackMatchedTPEtaTotLV->getNbinsX(),
                                meTrackMatchedTPEtaTotLV->getTH1()->GetXaxis()->GetXmin(),
                                meTrackMatchedTPEtaTotLV->getTH1()->GetXaxis()->GetXmax());
  meExtraEtaEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meExtraEtaMtd, meTrackMatchedTPEtaTotLV, meExtraEtaEff_);

  meExtraEtaEtl2Eff_ =
      ibook.book1D("ExtraEtaEtl2Eff",
                   "MTD matching efficiency (2 ETL) wrt extrapolated track associated to LV VS Eta;Eta;Efficiency",
                   meTrackMatchedTPEtaTotLV->getNbinsX(),
                   meTrackMatchedTPEtaTotLV->getTH1()->GetXaxis()->GetXmin(),
                   meTrackMatchedTPEtaTotLV->getTH1()->GetXaxis()->GetXmax());
  meExtraEtaEtl2Eff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meExtraEtaEtl2Mtd, meTrackMatchedTPEtaTotLV, meExtraEtaEtl2Eff_);

  // Efficiency for TP matched tracks
  meBTLTPPtSelEff_ = ibook.book1D("BTLTPPtSelEff",
                                  "Track selected efficiency TP VS Pt;Pt [GeV];Efficiency",
                                  meBTLTrackPtTot->getNbinsX(),
                                  meBTLTrackPtTot->getTH1()->GetXaxis()->GetXmin(),
                                  meBTLTrackPtTot->getTH1()->GetXaxis()->GetXmax());
  meBTLTPPtSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPPtTot, meBTLTrackPtTot, meBTLTPPtSelEff_);

  meBTLTPEtaSelEff_ = ibook.book1D("BTLTPEtaSelEff",
                                   "Track selected efficiency TP VS Eta;Eta;Efficiency",
                                   meBTLTrackEtaTot->getNbinsX(),
                                   meBTLTrackEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                   meBTLTrackEtaTot->getTH1()->GetXaxis()->GetXmax());
  meBTLTPEtaSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPEtaTot, meBTLTrackEtaTot, meBTLTPEtaSelEff_);

  meBTLTPPtMatchEff_ = ibook.book1D("BTLTPPtMatchEff",
                                    "Track matched to TP efficiency VS Pt;Pt [GeV];Efficiency",
                                    meBTLTrackMatchedTPPtTot->getNbinsX(),
                                    meBTLTrackMatchedTPPtTot->getTH1()->GetXaxis()->GetXmin(),
                                    meBTLTrackMatchedTPPtTot->getTH1()->GetXaxis()->GetXmax());
  meBTLTPPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPPtMtd, meBTLTrackMatchedTPPtTot, meBTLTPPtMatchEff_);

  meBTLTPEtaMatchEff_ = ibook.book1D("BTLTPEtaMatchEff",
                                     "Track matched to TP efficiency VS Eta;Eta;Efficiency",
                                     meBTLTrackMatchedTPEtaTot->getNbinsX(),
                                     meBTLTrackMatchedTPEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                     meBTLTrackMatchedTPEtaTot->getTH1()->GetXaxis()->GetXmax());
  meBTLTPEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPEtaMtd, meBTLTrackMatchedTPEtaTot, meBTLTPEtaMatchEff_);

  meETLTPPtSelEff_ = ibook.book1D("ETLTPPtSelEff",
                                  "Track selected efficiency TP VS Pt;Pt [GeV];Efficiency",
                                  meETLTrackPtTot->getNbinsX(),
                                  meETLTrackPtTot->getTH1()->GetXaxis()->GetXmin(),
                                  meETLTrackPtTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPPtSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPPtTot, meETLTrackPtTot, meETLTPPtSelEff_);

  meETLTPEtaSelEff_ = ibook.book1D("ETLTPEtaSelEff",
                                   "Track selected efficiency TP VS Eta;Eta;Efficiency",
                                   meETLTrackEtaTot->getNbinsX(),
                                   meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                   meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPEtaSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPEtaTot, meETLTrackEtaTot, meETLTPEtaSelEff_);

  meETLTPPtMatchEff_ = ibook.book1D("ETLTPPtMatchEff",
                                    "Track matched to TP efficiency VS Pt;Pt [GeV];Efficiency",
                                    meETLTrackMatchedTPPtTot->getNbinsX(),
                                    meETLTrackMatchedTPPtTot->getTH1()->GetXaxis()->GetXmin(),
                                    meETLTrackMatchedTPPtTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPPtMtd, meETLTrackMatchedTPPtTot, meETLTPPtMatchEff_);

  meETLTPEtaMatchEff_ = ibook.book1D("ETLTPEtaMatchEff",
                                     "Track matched to TP efficiency VS Eta;Eta;Efficiency",
                                     meETLTrackMatchedTPEtaTot->getNbinsX(),
                                     meETLTrackMatchedTPEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                     meETLTrackMatchedTPEtaTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPEtaMtd, meETLTrackMatchedTPEtaTot, meETLTPEtaMatchEff_);

  meETLTPPtMatchEff2_ = ibook.book1D("ETLTPPtMatchEff2",
                                     "Track matched to TP efficiency VS Pt (2 ETL hits);Pt [GeV];Efficiency",
                                     meETLTrackMatchedTPPtTot->getNbinsX(),
                                     meETLTrackMatchedTPPtTot->getTH1()->GetXaxis()->GetXmin(),
                                     meETLTrackMatchedTPPtTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPPtMatchEff2_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPPt2Mtd, meETLTrackMatchedTPPtTot, meETLTPPtMatchEff2_);

  meETLTPEtaMatchEff2_ = ibook.book1D("ETLTPEtaMatchEff2",
                                      "Track matched to TP efficiency VS Eta (2 hits);Eta;Efficiency",
                                      meETLTrackMatchedTPEtaTot->getNbinsX(),
                                      meETLTrackMatchedTPEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                      meETLTrackMatchedTPEtaTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPEtaMatchEff2_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPEta2Mtd, meETLTrackMatchedTPEtaTot, meETLTPEtaMatchEff2_);

  // == Track-cluster matching efficiencies based on mc truth
  // -- BTL
  meBTLTPmtdDirectEtaSelEff_ = ibook.book1D("BTLTPmtdDirectEtaSelEff",
                                            "Track selected efficiency TP-mtd hit (direct) VS Eta",
                                            meBTLTrackEtaTot->getNbinsX(),
                                            meBTLTrackEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                            meBTLTrackEtaTot->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdDirectEtaSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPmtdDirectEta, meBTLTrackEtaTot, meBTLTPmtdDirectEtaSelEff_);

  meBTLTPmtdDirectPtSelEff_ = ibook.book1D("BTLTPmtdDirectPtSelEff",
                                           "Track selected efficiency TP-mtd hit (direct) VS Pt",
                                           meBTLTrackPtTot->getNbinsX(),
                                           meBTLTrackPtTot->getTH1()->GetXaxis()->GetXmin(),
                                           meBTLTrackPtTot->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdDirectPtSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPmtdDirectPt, meBTLTrackPtTot, meBTLTPmtdDirectPtSelEff_);

  meBTLTPmtdOtherEtaSelEff_ = ibook.book1D("BTLTPmtdOtherEtaSelEff",
                                           "Track selected efficiency TP-mtd hit (other) VS Eta",
                                           meBTLTrackEtaTot->getNbinsX(),
                                           meBTLTrackEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                           meBTLTrackEtaTot->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdOtherEtaSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPmtdOtherEta, meBTLTrackEtaTot, meBTLTPmtdOtherEtaSelEff_);

  meBTLTPmtdOtherPtSelEff_ = ibook.book1D("BTLTPmtdOtherPtSelEff",
                                          "Track selected efficiency TP-mtd hit (other) VS Pt",
                                          meBTLTrackPtTot->getNbinsX(),
                                          meBTLTrackPtTot->getTH1()->GetXaxis()->GetXmin(),
                                          meBTLTrackPtTot->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdOtherPtSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPmtdOtherPt, meBTLTrackPtTot, meBTLTPmtdOtherPtSelEff_);

  meBTLTPnomtdEtaSelEff_ = ibook.book1D("BTLTPnomtdEtaSelEff",
                                        "Track selected efficiency TP-no mtd hit VS Eta",
                                        meBTLTrackEtaTot->getNbinsX(),
                                        meBTLTrackEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                        meBTLTrackEtaTot->getTH1()->GetXaxis()->GetXmax());
  meBTLTPnomtdEtaSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPnomtdEta, meBTLTrackEtaTot, meBTLTPnomtdEtaSelEff_);

  meBTLTPnomtdPtSelEff_ = ibook.book1D("BTLTPnomtdPtSelEff",
                                       "Track selected efficiency TP-no mtd hit VS Pt",
                                       meBTLTrackPtTot->getNbinsX(),
                                       meBTLTrackPtTot->getTH1()->GetXaxis()->GetXmin(),
                                       meBTLTrackPtTot->getTH1()->GetXaxis()->GetXmax());
  meBTLTPnomtdPtSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPnomtdPt, meBTLTrackPtTot, meBTLTPnomtdPtSelEff_);

  meBTLTPmtdDirectCorrectAssocEtaMatchEff_ =
      ibook.book1D("BTLTPmtdDirectCorrectAssocEtaMatchEff",
                   "Track efficiency TP-mtd hit (direct), correct reco match VS Eta",
                   meBTLTrackMatchedTPmtdDirectEta->getNbinsX(),
                   meBTLTrackMatchedTPmtdDirectEta->getTH1()->GetXaxis()->GetXmin(),
                   meBTLTrackMatchedTPmtdDirectEta->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdDirectCorrectAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPmtdDirectCorrectAssocEta,
                      meBTLTrackMatchedTPmtdDirectEta,
                      meBTLTPmtdDirectCorrectAssocEtaMatchEff_);

  meBTLTPmtdDirectCorrectAssocPtMatchEff_ =
      ibook.book1D("BTLTPmtdDirectCorrectAssocPtMatchEff",
                   "Track efficiency TP-mtd hit (direct), correct reco match VS Pt",
                   meBTLTrackMatchedTPmtdDirectPt->getNbinsX(),
                   meBTLTrackMatchedTPmtdDirectPt->getTH1()->GetXaxis()->GetXmin(),
                   meBTLTrackMatchedTPmtdDirectPt->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdDirectCorrectAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPmtdDirectCorrectAssocPt,
                      meBTLTrackMatchedTPmtdDirectPt,
                      meBTLTPmtdDirectCorrectAssocPtMatchEff_);

  meBTLTPmtdDirectWrongAssocEtaMatchEff_ =
      ibook.book1D("BTLTPmtdDirectWrongAssocEtaMatchEff",
                   "Track efficiency TP-mtd hit (direct), incorrect reco match VS Eta",
                   meBTLTrackMatchedTPmtdDirectEta->getNbinsX(),
                   meBTLTrackMatchedTPmtdDirectEta->getTH1()->GetXaxis()->GetXmin(),
                   meBTLTrackMatchedTPmtdDirectEta->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdDirectWrongAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPmtdDirectWrongAssocEta,
                      meBTLTrackMatchedTPmtdDirectEta,
                      meBTLTPmtdDirectWrongAssocEtaMatchEff_);

  meBTLTPmtdDirectWrongAssocPtMatchEff_ =
      ibook.book1D("BTLTPmtdDirectWrongAssocPtMatchEff",
                   "Track efficiency TP-mtd hit (direct), incorrect reco match VS Pt",
                   meBTLTrackMatchedTPmtdDirectPt->getNbinsX(),
                   meBTLTrackMatchedTPmtdDirectPt->getTH1()->GetXaxis()->GetXmin(),
                   meBTLTrackMatchedTPmtdDirectPt->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdDirectWrongAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meBTLTrackMatchedTPmtdDirectWrongAssocPt, meBTLTrackMatchedTPmtdDirectPt, meBTLTPmtdDirectWrongAssocPtMatchEff_);

  meBTLTPmtdDirectNoAssocEtaMatchEff_ = ibook.book1D("BTLTPmtdDirectNoAssocEtaMatchEff",
                                                     "Track efficiency TP-mtd hit (direct), no reco match VS Eta",
                                                     meBTLTrackMatchedTPmtdDirectEta->getNbinsX(),
                                                     meBTLTrackMatchedTPmtdDirectEta->getTH1()->GetXaxis()->GetXmin(),
                                                     meBTLTrackMatchedTPmtdDirectEta->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdDirectNoAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meBTLTrackMatchedTPmtdDirectNoAssocEta, meBTLTrackMatchedTPmtdDirectEta, meBTLTPmtdDirectNoAssocEtaMatchEff_);

  meBTLTPmtdDirectNoAssocPtMatchEff_ = ibook.book1D("BTLTPmtdDirectNoAssocPtMatchEff",
                                                    "Track efficiency TP-mtd hit (direct), no reco match VS Pt",
                                                    meBTLTrackMatchedTPmtdDirectPt->getNbinsX(),
                                                    meBTLTrackMatchedTPmtdDirectPt->getTH1()->GetXaxis()->GetXmin(),
                                                    meBTLTrackMatchedTPmtdDirectPt->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdDirectNoAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meBTLTrackMatchedTPmtdDirectNoAssocPt, meBTLTrackMatchedTPmtdDirectPt, meBTLTPmtdDirectNoAssocPtMatchEff_);

  meBTLTPmtdOtherCorrectAssocEtaMatchEff_ =
      ibook.book1D("BTLTPmtdOtherCorrectAssocEtaMatchEff",
                   "Track efficiency TP-mtd hit (other), correct reco match VS Eta",
                   meBTLTrackMatchedTPmtdOtherEta->getNbinsX(),
                   meBTLTrackMatchedTPmtdOtherEta->getTH1()->GetXaxis()->GetXmin(),
                   meBTLTrackMatchedTPmtdOtherEta->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdOtherCorrectAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPmtdOtherCorrectAssocEta,
                      meBTLTrackMatchedTPmtdOtherEta,
                      meBTLTPmtdOtherCorrectAssocEtaMatchEff_);

  meBTLTPmtdOtherCorrectAssocPtMatchEff_ = ibook.book1D("BTLTPmtdOtherCorrectAssocPtMatchEff",
                                                        "Track efficiency TP-mtd hit (other), correct reco match VS Pt",
                                                        meBTLTrackMatchedTPmtdOtherPt->getNbinsX(),
                                                        meBTLTrackMatchedTPmtdOtherPt->getTH1()->GetXaxis()->GetXmin(),
                                                        meBTLTrackMatchedTPmtdOtherPt->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdOtherCorrectAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meBTLTrackMatchedTPmtdOtherCorrectAssocPt, meBTLTrackMatchedTPmtdOtherPt, meBTLTPmtdOtherCorrectAssocPtMatchEff_);

  meBTLTPmtdOtherWrongAssocEtaMatchEff_ =
      ibook.book1D("BTLTPmtdOtherWrongAssocEtaMatchEff",
                   "Track efficiency TP-mtd hit (other), incorrect reco match VS Eta",
                   meBTLTrackMatchedTPmtdOtherEta->getNbinsX(),
                   meBTLTrackMatchedTPmtdOtherEta->getTH1()->GetXaxis()->GetXmin(),
                   meBTLTrackMatchedTPmtdOtherEta->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdOtherWrongAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meBTLTrackMatchedTPmtdOtherWrongAssocEta, meBTLTrackMatchedTPmtdOtherEta, meBTLTPmtdOtherWrongAssocEtaMatchEff_);

  meBTLTPmtdOtherWrongAssocPtMatchEff_ = ibook.book1D("BTLTPmtdOtherWrongAssocPtMatchEff",
                                                      "Track efficiency TP-mtd hit (other), incorrect reco match VS Pt",
                                                      meBTLTrackMatchedTPmtdOtherPt->getNbinsX(),
                                                      meBTLTrackMatchedTPmtdOtherPt->getTH1()->GetXaxis()->GetXmin(),
                                                      meBTLTrackMatchedTPmtdOtherPt->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdOtherWrongAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meBTLTrackMatchedTPmtdOtherWrongAssocPt, meBTLTrackMatchedTPmtdOtherPt, meBTLTPmtdOtherWrongAssocPtMatchEff_);

  meBTLTPmtdOtherNoAssocEtaMatchEff_ = ibook.book1D("BTLTPmtdOtherNoAssocEtaMatchEff",
                                                    "Track efficiency TP-mtd hit (other), no reco match VS Eta",
                                                    meBTLTrackMatchedTPmtdOtherEta->getNbinsX(),
                                                    meBTLTrackMatchedTPmtdOtherEta->getTH1()->GetXaxis()->GetXmin(),
                                                    meBTLTrackMatchedTPmtdOtherEta->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdOtherNoAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meBTLTrackMatchedTPmtdOtherNoAssocEta, meBTLTrackMatchedTPmtdOtherEta, meBTLTPmtdOtherNoAssocEtaMatchEff_);

  meBTLTPmtdOtherNoAssocPtMatchEff_ = ibook.book1D("BTLTPmtdOtherNoAssocPtMatchEff",
                                                   "Track efficiency TP-mtd hit (other), no reco match VS Pt",
                                                   meBTLTrackMatchedTPmtdOtherPt->getNbinsX(),
                                                   meBTLTrackMatchedTPmtdOtherPt->getTH1()->GetXaxis()->GetXmin(),
                                                   meBTLTrackMatchedTPmtdOtherPt->getTH1()->GetXaxis()->GetXmax());
  meBTLTPmtdOtherNoAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meBTLTrackMatchedTPmtdOtherNoAssocPt, meBTLTrackMatchedTPmtdOtherPt, meBTLTPmtdOtherNoAssocPtMatchEff_);

  meBTLTPnomtdEtaMatchEff_ = ibook.book1D("BTLTPnomtdEtaMatchEff",
                                          "Track efficiency TP- no mtd hit, with reco match VS Eta",
                                          meBTLTrackMatchedTPnomtdEta->getNbinsX(),
                                          meBTLTrackMatchedTPnomtdEta->getTH1()->GetXaxis()->GetXmin(),
                                          meBTLTrackMatchedTPnomtdEta->getTH1()->GetXaxis()->GetXmax());
  meBTLTPnomtdEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPnomtdAssocEta, meBTLTrackMatchedTPnomtdEta, meBTLTPnomtdEtaMatchEff_);

  meBTLTPnomtdPtMatchEff_ = ibook.book1D("BTLTPnomtdPtMatchEff",
                                         "Track efficiency TP- no mtd hit, with reco match VS Pt",
                                         meBTLTrackMatchedTPnomtdPt->getNbinsX(),
                                         meBTLTrackMatchedTPnomtdPt->getTH1()->GetXaxis()->GetXmin(),
                                         meBTLTrackMatchedTPnomtdPt->getTH1()->GetXaxis()->GetXmax());
  meBTLTPnomtdPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBTLTrackMatchedTPnomtdAssocPt, meBTLTrackMatchedTPnomtdPt, meBTLTPnomtdPtMatchEff_);

  // -- ETL
  meETLTPmtd1EtaSelEff_ = ibook.book1D("ETLTPmtd1EtaSelEff",
                                       "Track selected efficiency TP-mtd hit (>=1 sim hit) VS Eta",
                                       meETLTrackEtaTot->getNbinsX(),
                                       meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                       meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd1EtaSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPmtd1Eta, meETLTrackEtaTot, meETLTPmtd1EtaSelEff_);

  meETLTPmtd1PtSelEff_ = ibook.book1D("ETLTPmtd1PtSelEff",
                                      "Track selected efficiency TP-mtd hit (>=1 sim hit) VS Pt",
                                      meETLTrackPtTot->getNbinsX(),
                                      meETLTrackPtTot->getTH1()->GetXaxis()->GetXmin(),
                                      meETLTrackPtTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd1PtSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPmtd1Pt, meETLTrackPtTot, meETLTPmtd1PtSelEff_);

  meETLTPmtd2EtaSelEff_ = ibook.book1D("ETLTPmtd2EtaSelEff",
                                       "Track selected efficiency TP-mtd hit (2 sim hits) VS Eta",
                                       meETLTrackEtaTot->getNbinsX(),
                                       meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                       meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd2EtaSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPmtd2Eta, meETLTrackEtaTot, meETLTPmtd2EtaSelEff_);

  meETLTPmtd2PtSelEff_ = ibook.book1D("ETLTPmtd2PtSelEff",
                                      "Track selected efficiency TP-mtd hit (2 sim hits) VS Pt",
                                      meETLTrackPtTot->getNbinsX(),
                                      meETLTrackPtTot->getTH1()->GetXaxis()->GetXmin(),
                                      meETLTrackPtTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd2PtSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPmtd2Pt, meETLTrackPtTot, meETLTPmtd2PtSelEff_);

  meETLTPnomtdEtaSelEff_ = ibook.book1D("ETLTPnomtdEtaSelEff",
                                        "Track selected efficiency TP-no mtd hit VS Eta",
                                        meETLTrackEtaTot->getNbinsX(),
                                        meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                        meETLTrackEtaTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPnomtdEtaSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPnomtdEta, meETLTrackEtaTot, meETLTPnomtdEtaSelEff_);

  meETLTPnomtdPtSelEff_ = ibook.book1D("ETLTPnomtdPtSelEff",
                                       "Track selected efficiency TP-no mtd hit VS Pt",
                                       meETLTrackPtTot->getNbinsX(),
                                       meETLTrackPtTot->getTH1()->GetXaxis()->GetXmin(),
                                       meETLTrackPtTot->getTH1()->GetXaxis()->GetXmax());
  meETLTPnomtdPtSelEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPnomtdPt, meETLTrackPtTot, meETLTPnomtdPtSelEff_);

  meETLTPmtd1CorrectAssocEtaMatchEff_ =
      ibook.book1D("ETLTPmtd1CorrectAssocEtaMatchEff",
                   "Track efficiency TP-mtd hit (>=1 sim hit), correct reco match VS Eta",
                   meETLTrackMatchedTPmtd1Eta->getNbinsX(),
                   meETLTrackMatchedTPmtd1Eta->getTH1()->GetXaxis()->GetXmin(),
                   meETLTrackMatchedTPmtd1Eta->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd1CorrectAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meETLTrackMatchedTPmtd1CorrectAssocEta, meETLTrackMatchedTPmtd1Eta, meETLTPmtd1CorrectAssocEtaMatchEff_);

  meETLTPmtd1CorrectAssocPtMatchEff_ =
      ibook.book1D("ETLTPmtd1CorrectAssocPtMatchEff",
                   "Track efficiency TP-mtd hit (>=1 sim hit), correct reco match VS Pt",
                   meETLTrackMatchedTPmtd1Pt->getNbinsX(),
                   meETLTrackMatchedTPmtd1Pt->getTH1()->GetXaxis()->GetXmin(),
                   meETLTrackMatchedTPmtd1Pt->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd1CorrectAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meETLTrackMatchedTPmtd1CorrectAssocPt, meETLTrackMatchedTPmtd1Pt, meETLTPmtd1CorrectAssocPtMatchEff_);

  meETLTPmtd1WrongAssocEtaMatchEff_ =
      ibook.book1D("ETLTPmtd1WrongAssocEtaMatchEff",
                   "Track efficiency TP-mtd hit (>=1 sim hit), incorrect reco match VS Eta",
                   meETLTrackMatchedTPmtd1Eta->getNbinsX(),
                   meETLTrackMatchedTPmtd1Eta->getTH1()->GetXaxis()->GetXmin(),
                   meETLTrackMatchedTPmtd1Eta->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd1WrongAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meETLTrackMatchedTPmtd1WrongAssocEta, meETLTrackMatchedTPmtd1Eta, meETLTPmtd1WrongAssocEtaMatchEff_);

  meETLTPmtd1WrongAssocPtMatchEff_ =
      ibook.book1D("ETLTPmtd1WrongAssocPtMatchEff",
                   "Track efficiency TP-mtd hit (>=1 sim hit), incorrect reco match VS Pt",
                   meETLTrackMatchedTPmtd1Pt->getNbinsX(),
                   meETLTrackMatchedTPmtd1Pt->getTH1()->GetXaxis()->GetXmin(),
                   meETLTrackMatchedTPmtd1Pt->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd1WrongAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPmtd1WrongAssocPt, meETLTrackMatchedTPmtd1Pt, meETLTPmtd1WrongAssocPtMatchEff_);

  meETLTPmtd1NoAssocEtaMatchEff_ = ibook.book1D("ETLTPmtd1NoAssocEtaMatchEff",
                                                "Track efficiency TP-mtd hit (>=1 sim hit), no reco match VS Eta",
                                                meETLTrackMatchedTPmtd1Eta->getNbinsX(),
                                                meETLTrackMatchedTPmtd1Eta->getTH1()->GetXaxis()->GetXmin(),
                                                meETLTrackMatchedTPmtd1Eta->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd1NoAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPmtd1NoAssocEta, meETLTrackMatchedTPmtd1Eta, meETLTPmtd1NoAssocEtaMatchEff_);

  meETLTPmtd1NoAssocPtMatchEff_ = ibook.book1D("ETLTPmtd1NoAssocPtMatchEff",
                                               "Track efficiency TP-mtd hit (>=1 sim hit), no reco match VS Pt",
                                               meETLTrackMatchedTPmtd1Pt->getNbinsX(),
                                               meETLTrackMatchedTPmtd1Pt->getTH1()->GetXaxis()->GetXmin(),
                                               meETLTrackMatchedTPmtd1Pt->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd1NoAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPmtd1NoAssocPt, meETLTrackMatchedTPmtd1Pt, meETLTPmtd1NoAssocPtMatchEff_);

  meETLTPmtd2CorrectAssocEtaMatchEff_ =
      ibook.book1D("ETLTPmtd2CorrectAssocEtaMatchEff",
                   "Track efficiency TP-mtd hit (2 sim hits), correct reco match VS Eta",
                   meETLTrackMatchedTPmtd2Eta->getNbinsX(),
                   meETLTrackMatchedTPmtd2Eta->getTH1()->GetXaxis()->GetXmin(),
                   meETLTrackMatchedTPmtd2Eta->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd2CorrectAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meETLTrackMatchedTPmtd2CorrectAssocEta, meETLTrackMatchedTPmtd2Eta, meETLTPmtd2CorrectAssocEtaMatchEff_);

  meETLTPmtd2CorrectAssocPtMatchEff_ =
      ibook.book1D("ETLTPmtd2CorrectAssocPtMatchEff",
                   "Track efficiency TP-mtd hit (2 sim hits), correct reco match VS Pt",
                   meETLTrackMatchedTPmtd2Pt->getNbinsX(),
                   meETLTrackMatchedTPmtd2Pt->getTH1()->GetXaxis()->GetXmin(),
                   meETLTrackMatchedTPmtd2Pt->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd2CorrectAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meETLTrackMatchedTPmtd2CorrectAssocPt, meETLTrackMatchedTPmtd2Pt, meETLTPmtd2CorrectAssocPtMatchEff_);

  meETLTPmtd2WrongAssocEtaMatchEff_ =
      ibook.book1D("ETLTPmtd2WrongAssocEtaMatchEff",
                   "Track efficiency TP-mtd hit (2 sim hits), incorrect reco match VS Eta",
                   meETLTrackMatchedTPmtd2Eta->getNbinsX(),
                   meETLTrackMatchedTPmtd2Eta->getTH1()->GetXaxis()->GetXmin(),
                   meETLTrackMatchedTPmtd2Eta->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd2WrongAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(
      meETLTrackMatchedTPmtd2WrongAssocEta, meETLTrackMatchedTPmtd2Eta, meETLTPmtd2WrongAssocEtaMatchEff_);

  meETLTPmtd2WrongAssocPtMatchEff_ =
      ibook.book1D("ETLTPmtd2WrongAssocPtMatchEff",
                   "Track efficiency TP-mtd hit (2 sim hits), incorrect reco match VS Pt",
                   meETLTrackMatchedTPmtd2Pt->getNbinsX(),
                   meETLTrackMatchedTPmtd2Pt->getTH1()->GetXaxis()->GetXmin(),
                   meETLTrackMatchedTPmtd2Pt->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd2WrongAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPmtd2WrongAssocPt, meETLTrackMatchedTPmtd2Pt, meETLTPmtd2WrongAssocPtMatchEff_);

  meETLTPmtd2NoAssocEtaMatchEff_ = ibook.book1D("ETLTPmtd2NoAssocEtaMatchEff",
                                                "Track efficiency TP-mtd hit (2 sim hits), no reco match VS Eta",
                                                meETLTrackMatchedTPmtd2Eta->getNbinsX(),
                                                meETLTrackMatchedTPmtd2Eta->getTH1()->GetXaxis()->GetXmin(),
                                                meETLTrackMatchedTPmtd2Eta->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd2NoAssocEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPmtd2NoAssocEta, meETLTrackMatchedTPmtd2Eta, meETLTPmtd2NoAssocEtaMatchEff_);

  meETLTPmtd2NoAssocPtMatchEff_ = ibook.book1D("ETLTPmtd2NoAssocPtMatchEff",
                                               "Track efficiency TP-mtd hit (2 sim hits), no reco match VS Pt",
                                               meETLTrackMatchedTPmtd2Pt->getNbinsX(),
                                               meETLTrackMatchedTPmtd2Pt->getTH1()->GetXaxis()->GetXmin(),
                                               meETLTrackMatchedTPmtd2Pt->getTH1()->GetXaxis()->GetXmax());
  meETLTPmtd2NoAssocPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPmtd2NoAssocPt, meETLTrackMatchedTPmtd2Pt, meETLTPmtd2NoAssocPtMatchEff_);

  meETLTPnomtdEtaMatchEff_ = ibook.book1D("ETLTPnomtdEtaMatchEff",
                                          "Track efficiency TP- no mtd hit, with reco match VS Eta",
                                          meETLTrackMatchedTPnomtdEta->getNbinsX(),
                                          meETLTrackMatchedTPnomtdEta->getTH1()->GetXaxis()->GetXmin(),
                                          meETLTrackMatchedTPnomtdEta->getTH1()->GetXaxis()->GetXmax());
  meETLTPnomtdEtaMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPnomtdAssocEta, meETLTrackMatchedTPnomtdEta, meETLTPnomtdEtaMatchEff_);

  meETLTPnomtdPtMatchEff_ = ibook.book1D("ETLTPnomtdPtMatchEff",
                                         "Track efficiency TP- no mtd hit, with reco match VS Pt",
                                         meETLTrackMatchedTPnomtdPt->getNbinsX(),
                                         meETLTrackMatchedTPnomtdPt->getTH1()->GetXaxis()->GetXmin(),
                                         meETLTrackMatchedTPnomtdPt->getTH1()->GetXaxis()->GetXmax());
  meETLTPnomtdPtMatchEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meETLTrackMatchedTPnomtdAssocPt, meETLTrackMatchedTPnomtdPt, meETLTPnomtdPtMatchEff_);

  meNoTimeFraction_ = ibook.book1D("NoTimeFraction",
                                   "Fraction of tracks with MTD hits and no time associated; Num. of hits",
                                   meTrackNumHits->getNbinsX(),
                                   meTrackNumHits->getTH1()->GetXaxis()->GetXmin(),
                                   meTrackNumHits->getTH1()->GetXaxis()->GetXmax());
  meNoTimeFraction_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meTrackNumHitsNT, meTrackNumHits, meNoTimeFraction_);

  meBtlEtaEff_->getTH1()->SetMinimum(0.);
  meBtlPhiEff_->getTH1()->SetMinimum(0.);
  meBtlPtEff_->getTH1()->SetMinimum(0.);
  meEtlEtaEff_->getTH1()->SetMinimum(0.);
  meEtlPhiEff_->getTH1()->SetMinimum(0.);
  meEtlPtEff_->getTH1()->SetMinimum(0.);
  meEtlEtaEff2_->getTH1()->SetMinimum(0.);
  meEtlPhiEff2_->getTH1()->SetMinimum(0.);
  meEtlPtEff2_->getTH1()->SetMinimum(0.);
  for (int i = 0; i < 2; i++) {
    meEtlEtaEffLowPt_[i]->getTH1()->SetMinimum(0.);
    meEtlEtaEff2LowPt_[i]->getTH1()->SetMinimum(0.);
  }

  meExtraPhiAtBTLEff_ = ibook.book1D("ExtraPhiAtBTLEff",
                                     "Efficiency to match hits at BTL surface of extrapolated tracks associated to LV",
                                     meExtraPhiAtBTL->getNbinsX(),
                                     meExtraPhiAtBTL->getTH1()->GetXaxis()->GetXmin(),
                                     meExtraPhiAtBTL->getTH1()->GetXaxis()->GetXmax());
  meExtraPhiAtBTLEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meExtraPhiAtBTLmatched, meExtraPhiAtBTL, meExtraPhiAtBTLEff_);

  normalize(meExtraBTLeneInCone, 1.);

  meExtraMTDfailExtenderEtaEff_ =
      ibook.book1D("ExtraMTDfailExtenderEtaEff",
                   "Track associated to LV extrapolated at MTD surface no extender efficiency VS Eta;Eta;Efficiency",
                   meTrackMatchedTPEtaTotLV->getNbinsX(),
                   meTrackMatchedTPEtaTotLV->getTH1()->GetXaxis()->GetXmin(),
                   meTrackMatchedTPEtaTotLV->getTH1()->GetXaxis()->GetXmax());
  meExtraMTDfailExtenderEtaEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meExtraMTDfailExtenderEta, meTrackMatchedTPEtaTotLV, meExtraMTDfailExtenderEtaEff_);

  meExtraMTDfailExtenderPtEff_ = ibook.book1D(
      "ExtraMTDfailExtenderPtEff",
      "Track associated to LV extrapolated at MTD surface no extender efficiency VS Pt;Pt [GeV];Efficiency",
      meTrackMatchedTPPtTotLV->getNbinsX(),
      meTrackMatchedTPPtTotLV->getTH1()->GetXaxis()->GetXmin(),
      meTrackMatchedTPPtTotLV->getTH1()->GetXaxis()->GetXmax());
  meExtraMTDfailExtenderPtEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meExtraMTDfailExtenderPt, meTrackMatchedTPPtTotLV, meExtraMTDfailExtenderPtEff_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ----------
void MtdTracksHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/Tracks/");

  descriptions.add("MtdTracksPostProcessor", desc);
}

DEFINE_FWK_MODULE(MtdTracksHarvester);
