#include <string>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

class BtlRecoHarvester : public DQMEDHarvester {
public:
  explicit BtlRecoHarvester(const edm::ParameterSet& iConfig);
  ~BtlRecoHarvester() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  const std::string folder_;

  // --- Histograms
  MonitorElement* meBtlEtaEff_;
  MonitorElement* meBtlPhiEff_;
  MonitorElement* meBtlPtEff_;
};

// ------------ constructor and destructor --------------
BtlRecoHarvester::BtlRecoHarvester(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {}

BtlRecoHarvester::~BtlRecoHarvester() {}

// ------------ endjob tasks ----------------------------
void BtlRecoHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& igetter) {
  // --- Get the monitoring histograms
  MonitorElement* meTrackEffEtaTot = igetter.get(folder_ + "TrackEffEtaTot");
  MonitorElement* meTrackEffPhiTot = igetter.get(folder_ + "TrackEffPhiTot");
  MonitorElement* meTrackEffPtTot = igetter.get(folder_ + "TrackEffPtTot");
  MonitorElement* meTrackEffEtaMtd = igetter.get(folder_ + "TrackEffEtaMtd");
  MonitorElement* meTrackEffPhiMtd = igetter.get(folder_ + "TrackEffPhiMtd");
  MonitorElement* meTrackEffPtMtd = igetter.get(folder_ + "TrackEffPtMtd");

  if (!meTrackEffEtaTot || !meTrackEffPhiTot || !meTrackEffPtTot || !meTrackEffEtaMtd || !meTrackEffPhiMtd ||
      !meTrackEffPtMtd) {
    edm::LogError("BtlRecoHarvester") << "Monitoring histograms not found!" << std::endl;
    return;
  }

  // --- Book  histograms
  ibook.cd(folder_);
  meBtlEtaEff_ = ibook.book1D("BtlEtaEff",
                              " Track Efficiency VS Eta;#eta;Efficiency",
                              meTrackEffEtaTot->getNbinsX(),
                              meTrackEffEtaTot->getTH1()->GetXaxis()->GetXmin(),
                              meTrackEffEtaTot->getTH1()->GetXaxis()->GetXmax());
  meBtlPhiEff_ = ibook.book1D("BtlPhiEff",
                              "Track Efficiency VS Phi;#phi;Efficiency [rad]",
                              meTrackEffPhiTot->getNbinsX(),
                              meTrackEffPhiTot->getTH1()->GetXaxis()->GetXmin(),
                              meTrackEffPhiTot->getTH1()->GetXaxis()->GetXmax());
  meBtlPtEff_ = ibook.book1D("BtlPtEff",
                             "Track Efficiency VS Pt;Pt;Efficiency [GeV]",
                             meTrackEffPtTot->getNbinsX(),
                             meTrackEffPtTot->getTH1()->GetXaxis()->GetXmin(),
                             meTrackEffPtTot->getTH1()->GetXaxis()->GetXmax());
  meBtlEtaEff_->getTH1()->SetMinimum(0.);
  meBtlPhiEff_->getTH1()->SetMinimum(0.);
  meBtlPtEff_->getTH1()->SetMinimum(0.);

  // --- Calculate efficiency
  for (int ibin = 1; ibin <= meTrackEffEtaTot->getNbinsX(); ibin++) {
    double eff = meTrackEffEtaMtd->getBinContent(ibin) / meTrackEffEtaTot->getBinContent(ibin);
    double bin_err = sqrt((meTrackEffEtaMtd->getBinContent(ibin) *
                           (meTrackEffEtaTot->getBinContent(ibin) - meTrackEffEtaMtd->getBinContent(ibin))) /
                          pow(meTrackEffEtaTot->getBinContent(ibin), 3));
    if (meTrackEffEtaTot->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meBtlEtaEff_->setBinContent(ibin, eff);
    meBtlEtaEff_->setBinError(ibin, bin_err);
  }
  for (int ibin = 1; ibin <= meTrackEffPhiTot->getNbinsX(); ibin++) {
    double eff = meTrackEffPhiMtd->getBinContent(ibin) / meTrackEffPhiTot->getBinContent(ibin);
    double bin_err = sqrt((meTrackEffPhiMtd->getBinContent(ibin) *
                           (meTrackEffPhiTot->getBinContent(ibin) - meTrackEffPhiMtd->getBinContent(ibin))) /
                          pow(meTrackEffPhiTot->getBinContent(ibin), 3));
    if (meTrackEffPhiTot->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meBtlPhiEff_->setBinContent(ibin, eff);
    meBtlPhiEff_->setBinError(ibin, bin_err);
  }
  for (int ibin = 1; ibin <= meTrackEffPtTot->getNbinsX(); ibin++) {
    double eff = meTrackEffPtMtd->getBinContent(ibin) / meTrackEffPtTot->getBinContent(ibin);
    double bin_err = sqrt((meTrackEffPtMtd->getBinContent(ibin) *
                           (meTrackEffPtTot->getBinContent(ibin) - meTrackEffPtMtd->getBinContent(ibin))) /
                          pow(meTrackEffPtTot->getBinContent(ibin), 3));
    if (meTrackEffPtTot->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meBtlPtEff_->setBinContent(ibin, eff);
    meBtlPtEff_->setBinError(ibin, bin_err);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ----------
void BtlRecoHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/Reco/");

  descriptions.add("btlRecoPostProcessor", desc);
}

DEFINE_FWK_MODULE(BtlRecoHarvester);
