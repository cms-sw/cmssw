#include <string>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

class Primary4DVertexHarvester : public DQMEDHarvester {
public:
  explicit Primary4DVertexHarvester(const edm::ParameterSet& iConfig);
  ~Primary4DVertexHarvester() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:

  void computeEfficiency1D(MonitorElement* num, MonitorElement* den, MonitorElement* result);

  void incrementME(MonitorElement* base, MonitorElement* toBeAdded);

  const std::string folder_;

  // --- Histograms
  MonitorElement* meMVAPtSelEff_;
  MonitorElement* meMVAEtaSelEff_;
  MonitorElement* meMVAPtMatchEff_;
  MonitorElement* meMVAEtaMatchEff_;

  MonitorElement* meBarrelTruePi_;
  MonitorElement* meBarrelTrueK_;
  MonitorElement* meBarrelTrueP_;

  MonitorElement* meEndcapTruePi_;
  MonitorElement* meEndcapTrueK_;
  MonitorElement* meEndcapTrueP_;

  MonitorElement* meBarrelAsPi_;
  MonitorElement* meBarrelAsK_;
  MonitorElement* meBarrelAsP_;
  MonitorElement* meBarrelNoPID_;

  MonitorElement* meEndcapAsPi_;
  MonitorElement* meEndcapAsK_;
  MonitorElement* meEndcapAsP_;
  MonitorElement* meEndcapNoPID_;

  MonitorElement* meBarrelPIDPiAsPiEff_;
  MonitorElement* meBarrelPIDPiAsKEff_;
  MonitorElement* meBarrelPIDPiAsPEff_;
  MonitorElement* meBarrelPIDPiNoPIDEff_;

  MonitorElement* meBarrelPIDKAsPiEff_;
  MonitorElement* meBarrelPIDKAsKEff_;
  MonitorElement* meBarrelPIDKAsPEff_;
  MonitorElement* meBarrelPIDKNoPIDEff_;

  MonitorElement* meBarrelPIDPAsPiEff_;
  MonitorElement* meBarrelPIDPAsKEff_;
  MonitorElement* meBarrelPIDPAsPEff_;
  MonitorElement* meBarrelPIDPNoPIDEff_;

  MonitorElement* meEndcapPIDPiAsPiEff_;
  MonitorElement* meEndcapPIDPiAsKEff_;
  MonitorElement* meEndcapPIDPiAsPEff_;
  MonitorElement* meEndcapPIDPiNoPIDEff_;

  MonitorElement* meEndcapPIDKAsPiEff_;
  MonitorElement* meEndcapPIDKAsKEff_;
  MonitorElement* meEndcapPIDKAsPEff_;
  MonitorElement* meEndcapPIDKNoPIDEff_;

  MonitorElement* meEndcapPIDPAsPiEff_;
  MonitorElement* meEndcapPIDPAsKEff_;
  MonitorElement* meEndcapPIDPAsPEff_;
  MonitorElement* meEndcapPIDPNoPIDEff_;
};

// ------------ constructor and destructor --------------
Primary4DVertexHarvester::Primary4DVertexHarvester(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {}

Primary4DVertexHarvester::~Primary4DVertexHarvester() {}

// auxiliary method to compute efficiency from the ratio of two 1D MonitorElement
void Primary4DVertexHarvester::computeEfficiency1D(MonitorElement* num, MonitorElement* den, MonitorElement* result) {

  for (int ibin = 1; ibin <= den->getNbinsX(); ibin++) {
    double eff = num->getBinContent(ibin) / den->getBinContent(ibin);
    double bin_err = sqrt((num->getBinContent(ibin) *
                           (den->getBinContent(ibin) - num->getBinContent(ibin))) /
                          pow(den->getBinContent(ibin), 3));
    if (den->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    result->setBinContent(ibin, eff);
    result->setBinError(ibin, bin_err);
  }

}

// auxiliary method to add 1D MonitorElement toBeAdded to a base ME
void Primary4DVertexHarvester::incrementME(MonitorElement* base, MonitorElement* toBeAdded) {

  for (int ibin = 1; ibin <= base->getNbinsX(); ibin++) {
     double newC = base->getBinContent(ibin) + toBeAdded->getBinContent(ibin);
     double newE = std::sqrt(newC);
     base->setBinContent(ibin, newC);
     base->setBinError(ibin, newE);
  }

}

// ------------ endjob tasks ----------------------------
void Primary4DVertexHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& igetter) {
  // --- Get the monitoring histograms
  MonitorElement* meMVATrackEffPtTot = igetter.get(folder_ + "MVAEffPtTot");
  MonitorElement* meMVATrackMatchedEffPtTot = igetter.get(folder_ + "MVAMatchedEffPtTot");
  MonitorElement* meMVATrackMatchedEffPtMtd = igetter.get(folder_ + "MVAMatchedEffPtMtd");
  MonitorElement* meMVATrackEffEtaTot = igetter.get(folder_ + "MVAEffEtaTot");
  MonitorElement* meMVATrackMatchedEffEtaTot = igetter.get(folder_ + "MVAMatchedEffEtaTot");
  MonitorElement* meMVATrackMatchedEffEtaMtd = igetter.get(folder_ + "MVAMatchedEffEtaMtd");
  MonitorElement* meRecoVtxVsLineDensity = igetter.get(folder_ + "RecoVtxVsLineDensity");
  MonitorElement* meRecVerNumber = igetter.get(folder_ + "RecVerNumber");

  if (!meMVATrackEffEtaTot || !meMVATrackMatchedEffEtaTot || !meMVATrackMatchedEffEtaMtd || !meMVATrackEffEtaTot ||
      !meMVATrackMatchedEffEtaTot || !meMVATrackMatchedEffEtaMtd || !meRecoVtxVsLineDensity || !meRecVerNumber) {
    edm::LogError("Primary4DVertexHarvester") << "Monitoring histograms not found!" << std::endl;
    return;
  }

  // Normalize line density plot
  double nEvt = meRecVerNumber->getEntries();
  if (nEvt > 0.) {
    nEvt = 1. / nEvt;
    double nEntries = meRecoVtxVsLineDensity->getEntries();
    for (int ibin = 1; ibin <= meRecoVtxVsLineDensity->getNbinsX(); ibin++) {
      double cont = meRecoVtxVsLineDensity->getBinContent(ibin) * nEvt;
      double bin_err = meRecoVtxVsLineDensity->getBinError(ibin) * nEvt;
      meRecoVtxVsLineDensity->setBinContent(ibin, cont);
      meRecoVtxVsLineDensity->setBinError(ibin, bin_err);
    }
    meRecoVtxVsLineDensity->setEntries(nEntries);
  }

  // --- Book  histograms
  ibook.cd(folder_);
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

  MonitorElement* meBarrelMatchedp = igetter.get(folder_ + "BarrelMatchedp");
  MonitorElement* meEndcapMatchedp = igetter.get(folder_ + "EndcapMatchedp");

  MonitorElement* meBarrelTruePiNoPID = igetter.get(folder_ + "BarrelTruePiNoPID");
  MonitorElement* meBarrelTrueKNoPID = igetter.get(folder_ + "BarrelTrueKNoPID");
  MonitorElement* meBarrelTruePNoPID = igetter.get(folder_ + "BarrelTruePNoPID");
  MonitorElement* meEndcapTruePiNoPID = igetter.get(folder_ + "EndcapTruePiNoPID");
  MonitorElement* meEndcapTrueKNoPID = igetter.get(folder_ + "EndcapTrueKNoPID");
  MonitorElement* meEndcapTruePNoPID = igetter.get(folder_ + "EndcapTruePNoPID");

  MonitorElement* meBarrelTruePiAsPi = igetter.get(folder_ + "BarrelTruePiAsPi");
  MonitorElement* meBarrelTrueKAsPi = igetter.get(folder_ + "BarrelTrueKAsPi");
  MonitorElement* meBarrelTruePAsPi = igetter.get(folder_ + "BarrelTruePAsPi");
  MonitorElement* meEndcapTruePiAsPi = igetter.get(folder_ + "EndcapTruePiAsPi");
  MonitorElement* meEndcapTrueKAsPi = igetter.get(folder_ + "EndcapTrueKAsPi");
  MonitorElement* meEndcapTruePAsPi = igetter.get(folder_ + "EndcapTruePAsPi");

  MonitorElement* meBarrelTruePiAsK = igetter.get(folder_ + "BarrelTruePiAsK");
  MonitorElement* meBarrelTrueKAsK = igetter.get(folder_ + "BarrelTrueKAsK");
  MonitorElement* meBarrelTruePAsK = igetter.get(folder_ + "BarrelTruePAsK");
  MonitorElement* meEndcapTruePiAsK = igetter.get(folder_ + "EndcapTruePiAsK");
  MonitorElement* meEndcapTrueKAsK = igetter.get(folder_ + "EndcapTrueKAsK");
  MonitorElement* meEndcapTruePAsK = igetter.get(folder_ + "EndcapTruePAsK");

  MonitorElement* meBarrelTruePiAsP = igetter.get(folder_ + "BarrelTruePiAsP");
  MonitorElement* meBarrelTrueKAsP = igetter.get(folder_ + "BarrelTrueKAsP");
  MonitorElement* meBarrelTruePAsP = igetter.get(folder_ + "BarrelTruePAsP");
  MonitorElement* meEndcapTruePiAsP = igetter.get(folder_ + "EndcapTruePiAsP");
  MonitorElement* meEndcapTrueKAsP = igetter.get(folder_ + "EndcapTrueKAsP");
  MonitorElement* meEndcapTruePAsP = igetter.get(folder_ + "EndcapTruePAsP");

  if ( !meBarrelMatchedp || !meEndcapMatchedp || !meBarrelTruePiNoPID || !meBarrelTrueKNoPID || !meBarrelTruePNoPID || !meEndcapTruePiNoPID || !meEndcapTrueKNoPID || !meEndcapTruePNoPID
  || !meBarrelTruePiAsPi || !meBarrelTrueKAsPi || !meBarrelTruePAsPi || !meEndcapTruePiAsPi || !meEndcapTrueKAsPi || !meEndcapTruePAsPi
  || !meBarrelTruePiAsK || !meBarrelTrueKAsK || !meBarrelTruePAsK || !meEndcapTruePiAsK || !meEndcapTrueKAsK || !meEndcapTruePAsK
  || !meBarrelTruePiAsP || !meBarrelTrueKAsP || !meBarrelTruePAsP || !meEndcapTruePiAsP || !meEndcapTrueKAsP || !meEndcapTruePAsP) {
    edm::LogError("Primary4DVertexHarvester") << "PID Monitoring histograms not found!" << std::endl;
    return;
  }

  meBarrelTruePi_ = meBarrelTruePiAsPi;
  incrementME(meBarrelTruePi_, meBarrelTruePiAsK);
  incrementME(meBarrelTruePi_, meBarrelTruePiAsP);
  incrementME(meBarrelTruePi_, meBarrelTruePiNoPID);

  meEndcapTruePi_ = meEndcapTruePiAsPi;
  incrementME(meEndcapTruePi_, meEndcapTruePiAsK);
  incrementME(meEndcapTruePi_, meEndcapTruePiAsP);
  incrementME(meEndcapTruePi_, meEndcapTruePiNoPID);

  meBarrelTrueK_ = meBarrelTrueKAsPi;
  incrementME(meBarrelTrueK_, meBarrelTrueKAsK);
  incrementME(meBarrelTrueK_, meBarrelTrueKAsP);
  incrementME(meBarrelTrueK_, meBarrelTrueKNoPID);

  meEndcapTrueK_ = meEndcapTrueKAsPi;
  incrementME(meEndcapTrueK_, meEndcapTrueKAsK);
  incrementME(meEndcapTrueK_, meEndcapTrueKAsP);
  incrementME(meEndcapTrueK_, meEndcapTrueKNoPID);

  meBarrelTrueP_ = meBarrelTruePAsPi;
  incrementME(meBarrelTrueP_, meBarrelTruePAsK);
  incrementME(meBarrelTrueP_, meBarrelTruePAsP);
  incrementME(meBarrelTrueP_, meBarrelTruePNoPID);

  meEndcapTrueP_ = meEndcapTruePAsPi;
  incrementME(meEndcapTrueP_, meEndcapTruePAsK);
  incrementME(meEndcapTrueP_, meEndcapTruePAsP);
  incrementME(meEndcapTrueP_, meEndcapTruePNoPID);

  meBarrelAsPi_ = meBarrelTruePiAsPi;
  incrementME(meBarrelAsPi_, meBarrelTrueKAsPi);
  incrementME(meBarrelAsPi_, meBarrelTruePAsPi);

  meEndcapAsPi_ = meEndcapTruePiAsPi;
  incrementME(meEndcapAsPi_, meEndcapTrueKAsPi);
  incrementME(meEndcapAsPi_, meEndcapTruePAsPi);

  meBarrelAsK_ = meBarrelTruePiAsK;
  incrementME(meBarrelAsK_, meBarrelTrueKAsK);
  incrementME(meBarrelAsK_, meBarrelTruePAsK);

  meEndcapAsK_ = meEndcapTruePiAsK;
  incrementME(meEndcapAsK_, meEndcapTrueKAsK);
  incrementME(meEndcapAsK_, meEndcapTruePAsK);

  meBarrelAsP_ = meBarrelTruePiAsP;
  incrementME(meBarrelAsP_, meBarrelTrueKAsP);
  incrementME(meBarrelAsP_, meBarrelTruePAsP);

  meEndcapAsP_ = meEndcapTruePiAsP;
  incrementME(meEndcapAsP_, meEndcapTrueKAsP);
  incrementME(meEndcapAsP_, meEndcapTruePAsP);

  meBarrelNoPID_ = meBarrelTruePiNoPID;
  incrementME(meBarrelNoPID_, meBarrelTrueKNoPID);
  incrementME(meBarrelNoPID_, meBarrelTruePNoPID);

  meEndcapNoPID_ = meEndcapTruePiNoPID;
  incrementME(meEndcapNoPID_, meEndcapTrueKNoPID);
  incrementME(meEndcapNoPID_, meEndcapTruePNoPID);

  meBarrelPIDPiAsPiEff_ = ibook.book1D("BarrelPIDPiAsPiEff","Barrel True pi as pi id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDPiAsPiEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiAsPi, meBarrelMatchedp, meBarrelPIDPiAsPiEff_);

  meBarrelPIDPiAsKEff_ = ibook.book1D("BarrelPIDPiAsKEff","Barrel True pi as k id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDPiAsKEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiAsK, meBarrelMatchedp, meBarrelPIDPiAsPiEff_);

  meBarrelPIDPiAsPEff_ = ibook.book1D("BarrelPIDPiAsPEff","Barrel True pi as p id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDPiAsPEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiAsP, meBarrelMatchedp, meBarrelPIDPiAsPEff_);

  meBarrelPIDPiNoPIDEff_ = ibook.book1D("BarrelPIDPiNoPIDEff","Barrel True pi no PID id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDPiNoPIDEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiNoPID, meBarrelMatchedp, meBarrelPIDPiNoPIDEff_);

  meBarrelPIDKAsPiEff_ = ibook.book1D("BarrelPIDKAsPiEff","Barrel True k as pi id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDKAsPiEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiAsPi, meBarrelMatchedp, meBarrelPIDKAsPiEff_);

  meBarrelPIDKAsKEff_ = ibook.book1D("BarrelPIDKAsKEff","Barrel True k as k id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDKAsKEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiAsK, meBarrelMatchedp, meBarrelPIDKAsPiEff_);

  meBarrelPIDKAsPEff_ = ibook.book1D("BarrelPIDKAsPEff","Barrel True k as p id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDKAsPEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiAsP, meBarrelMatchedp, meBarrelPIDKAsPEff_);

  meBarrelPIDKNoPIDEff_ = ibook.book1D("BarrelPIDKNoPIDEff","Barrel True k no PID id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDKNoPIDEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiNoPID, meBarrelMatchedp, meBarrelPIDKNoPIDEff_);

  meBarrelPIDPAsPiEff_ = ibook.book1D("BarrelPIDPAsPiEff","Barrel True p as pi id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDPAsPiEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiAsPi, meBarrelMatchedp, meBarrelPIDPAsPiEff_);

  meBarrelPIDPAsKEff_ = ibook.book1D("BarrelPIDPAsKEff","Barrel True p as k id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDPAsKEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiAsK, meBarrelMatchedp, meBarrelPIDPAsPiEff_);

  meBarrelPIDPAsPEff_ = ibook.book1D("BarrelPIDPAsPEff","Barrel True p as p id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDPAsPEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiAsP, meBarrelMatchedp, meBarrelPIDPAsPEff_);

  meBarrelPIDPNoPIDEff_ = ibook.book1D("BarrelPIDPNoPIDEff","Barrel True p no PID id. fraction VS P;P [GeV]",meBarrelMatchedp->getNbinsX(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmin(),meBarrelMatchedp->getTH1()->GetXaxis()->GetXmax());
  meBarrelPIDPNoPIDEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meBarrelTruePiNoPID, meBarrelMatchedp, meBarrelPIDPNoPIDEff_);


  meEndcapPIDPiAsPiEff_ = ibook.book1D("EndcapPIDPiAsPiEff","Endcap True pi as pi id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDPiAsPiEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiAsPi, meEndcapMatchedp, meEndcapPIDPiAsPiEff_);

  meEndcapPIDPiAsKEff_ = ibook.book1D("EndcapPIDPiAsKEff","Endcap True pi as k id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDPiAsKEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiAsK, meEndcapMatchedp, meEndcapPIDPiAsPiEff_);

  meEndcapPIDPiAsPEff_ = ibook.book1D("EndcapPIDPiAsPEff","Endcap True pi as p id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDPiAsPEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiAsP, meEndcapMatchedp, meEndcapPIDPiAsPEff_);

  meEndcapPIDPiNoPIDEff_ = ibook.book1D("EndcapPIDPiNoPIDEff","Endcap True pi no PID id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDPiNoPIDEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiNoPID, meEndcapMatchedp, meEndcapPIDPiNoPIDEff_);

  meEndcapPIDKAsPiEff_ = ibook.book1D("EndcapPIDKAsPiEff","Endcap True k as pi id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDKAsPiEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiAsPi, meEndcapMatchedp, meEndcapPIDKAsPiEff_);

  meEndcapPIDKAsKEff_ = ibook.book1D("EndcapPIDKAsKEff","Endcap True k as k id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDKAsKEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiAsK, meEndcapMatchedp, meEndcapPIDKAsPiEff_);

  meEndcapPIDKAsPEff_ = ibook.book1D("EndcapPIDKAsPEff","Endcap True k as p id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDKAsPEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiAsP, meEndcapMatchedp, meEndcapPIDKAsPEff_);

  meEndcapPIDKNoPIDEff_ = ibook.book1D("EndcapPIDKNoPIDEff","Endcap True k no PID id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDKNoPIDEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiNoPID, meEndcapMatchedp, meEndcapPIDKNoPIDEff_);

  meEndcapPIDPAsPiEff_ = ibook.book1D("EndcapPIDPAsPiEff","Endcap True p as pi id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDPAsPiEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiAsPi, meEndcapMatchedp, meEndcapPIDPAsPiEff_);

  meEndcapPIDPAsKEff_ = ibook.book1D("EndcapPIDPAsKEff","Endcap True p as k id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDPAsKEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiAsK, meEndcapMatchedp, meEndcapPIDPAsPiEff_);

  meEndcapPIDPAsPEff_ = ibook.book1D("EndcapPIDPAsPEff","Endcap True p as p id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDPAsPEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiAsP, meEndcapMatchedp, meEndcapPIDPAsPEff_);

  meEndcapPIDPNoPIDEff_ = ibook.book1D("EndcapPIDPNoPIDEff","Endcap True p no PID id. fraction VS P;P [GeV]",meEndcapMatchedp->getNbinsX(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmin(),meEndcapMatchedp->getTH1()->GetXaxis()->GetXmax());
  meEndcapPIDPNoPIDEff_->getTH1()->SetMinimum(0.);
  computeEfficiency1D(meEndcapTruePiNoPID, meEndcapMatchedp, meEndcapPIDPNoPIDEff_);

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ----------
void Primary4DVertexHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/Vertices/");

  descriptions.add("Primary4DVertexPostProcessor", desc);
}

DEFINE_FWK_MODULE(Primary4DVertexHarvester);
