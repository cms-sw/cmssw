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
  const std::string folder_;

  // --- Histograms
  MonitorElement* meMVAPtSelEff_;
  MonitorElement* meMVAEtaSelEff_;
  MonitorElement* meMVAPtMatchEff_;
  MonitorElement* meMVAEtaMatchEff_;
};

// ------------ constructor and destructor --------------
Primary4DVertexHarvester::Primary4DVertexHarvester(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {}

Primary4DVertexHarvester::~Primary4DVertexHarvester() {}

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

  if (!meMVATrackEffPtTot || !meMVATrackMatchedEffPtTot || !meMVATrackMatchedEffPtMtd || !meMVATrackEffEtaTot ||
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
  meMVAEtaSelEff_ = ibook.book1D("MVAEtaSelEff",
                                 "Track selected efficiency VS Eta;Eta;Efficiency",
                                 meMVATrackEffEtaTot->getNbinsX(),
                                 meMVATrackEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                 meMVATrackEffEtaTot->getTH1()->GetXaxis()->GetXmax());
  meMVAPtMatchEff_ = ibook.book1D("MVAPtMatchEff",
                                  "Track matched to GEN efficiency VS Pt;Pt [GeV];Efficiency",
                                  meMVATrackMatchedEffPtTot->getNbinsX(),
                                  meMVATrackMatchedEffPtTot->getTH1()->GetXaxis()->GetXmin(),
                                  meMVATrackMatchedEffPtTot->getTH1()->GetXaxis()->GetXmax());
  meMVAEtaMatchEff_ = ibook.book1D("MVAEtaMatchEff",
                                   "Track matched to GEN efficiency VS Eta;Eta;Efficiency",
                                   meMVATrackMatchedEffEtaTot->getNbinsX(),
                                   meMVATrackMatchedEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                                   meMVATrackMatchedEffEtaTot->getTH1()->GetXaxis()->GetXmax());

  meMVAPtSelEff_->getTH1()->SetMinimum(0.);
  meMVAEtaSelEff_->getTH1()->SetMinimum(0.);
  meMVAPtMatchEff_->getTH1()->SetMinimum(0.);
  meMVAEtaMatchEff_->getTH1()->SetMinimum(0.);

  for (int ibin = 1; ibin <= meMVATrackEffPtTot->getNbinsX(); ibin++) {
    double eff = meMVATrackMatchedEffPtTot->getBinContent(ibin) / meMVATrackEffPtTot->getBinContent(ibin);
    double bin_err = sqrt((meMVATrackMatchedEffPtTot->getBinContent(ibin) *
                           (meMVATrackEffPtTot->getBinContent(ibin) - meMVATrackMatchedEffPtTot->getBinContent(ibin))) /
                          pow(meMVATrackEffPtTot->getBinContent(ibin), 3));
    if (meMVATrackEffPtTot->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meMVAPtSelEff_->setBinContent(ibin, eff);
    meMVAPtSelEff_->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meMVATrackEffEtaTot->getNbinsX(); ibin++) {
    double eff = meMVATrackMatchedEffEtaTot->getBinContent(ibin) / meMVATrackEffEtaTot->getBinContent(ibin);
    double bin_err =
        sqrt((meMVATrackMatchedEffEtaTot->getBinContent(ibin) *
              (meMVATrackEffEtaTot->getBinContent(ibin) - meMVATrackMatchedEffEtaTot->getBinContent(ibin))) /
             pow(meMVATrackEffEtaTot->getBinContent(ibin), 3));
    if (meMVATrackEffEtaTot->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meMVAEtaSelEff_->setBinContent(ibin, eff);
    meMVAEtaSelEff_->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meMVATrackMatchedEffPtTot->getNbinsX(); ibin++) {
    double eff = meMVATrackMatchedEffPtMtd->getBinContent(ibin) / meMVATrackMatchedEffPtTot->getBinContent(ibin);
    double bin_err =
        sqrt((meMVATrackMatchedEffPtMtd->getBinContent(ibin) *
              (meMVATrackMatchedEffPtTot->getBinContent(ibin) - meMVATrackMatchedEffPtMtd->getBinContent(ibin))) /
             pow(meMVATrackMatchedEffPtTot->getBinContent(ibin), 3));
    if (meMVATrackMatchedEffPtTot->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meMVAPtMatchEff_->setBinContent(ibin, eff);
    meMVAPtMatchEff_->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meMVATrackMatchedEffEtaTot->getNbinsX(); ibin++) {
    double eff = meMVATrackMatchedEffEtaMtd->getBinContent(ibin) / meMVATrackMatchedEffEtaTot->getBinContent(ibin);
    double bin_err =
        sqrt((meMVATrackMatchedEffEtaMtd->getBinContent(ibin) *
              (meMVATrackMatchedEffEtaTot->getBinContent(ibin) - meMVATrackMatchedEffEtaMtd->getBinContent(ibin))) /
             pow(meMVATrackMatchedEffEtaTot->getBinContent(ibin), 3));
    if (meMVATrackMatchedEffEtaTot->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meMVAEtaMatchEff_->setBinContent(ibin, eff);
    meMVAEtaMatchEff_->setBinError(ibin, bin_err);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ----------
void Primary4DVertexHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/Vertices/");

  descriptions.add("Primary4DVertexPostProcessor", desc);
}

DEFINE_FWK_MODULE(Primary4DVertexHarvester);
