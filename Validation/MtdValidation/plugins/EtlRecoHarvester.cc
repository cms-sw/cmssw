#include <string>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

class EtlRecoHarvester : public DQMEDHarvester {
public:
  explicit EtlRecoHarvester(const edm::ParameterSet& iConfig);
  ~EtlRecoHarvester() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  const std::string folder_;

  // --- Histograms
  MonitorElement* meEtlEtaEff_[2];
  MonitorElement* meEtlPhiEff_[2];
  MonitorElement* meEtlPtEff_[2];
};

// ------------ constructor and destructor --------------
EtlRecoHarvester::EtlRecoHarvester(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {}

EtlRecoHarvester::~EtlRecoHarvester() {}

// ------------ endjob tasks ----------------------------
void EtlRecoHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& igetter) {
  // --- Get the monitoring histograms
  MonitorElement* meTrackEffEtaTotZneg = igetter.get(folder_ + "TrackEffEtaTotZneg");
  MonitorElement* meTrackEffPhiTotZneg = igetter.get(folder_ + "TrackEffPhiTotZneg");
  MonitorElement* meTrackEffPtTotZneg = igetter.get(folder_ + "TrackEffPtTotZneg");
  MonitorElement* meTrackEffEtaMtdZneg = igetter.get(folder_ + "TrackEffEtaMtdZneg");
  MonitorElement* meTrackEffPhiMtdZneg = igetter.get(folder_ + "TrackEffPhiMtdZneg");
  MonitorElement* meTrackEffPtMtdZneg = igetter.get(folder_ + "TrackEffPtMtdZneg");
  MonitorElement* meTrackEffEtaTotZpos = igetter.get(folder_ + "TrackEffEtaTotZpos");
  MonitorElement* meTrackEffPhiTotZpos = igetter.get(folder_ + "TrackEffPhiTotZpos");
  MonitorElement* meTrackEffPtTotZpos = igetter.get(folder_ + "TrackEffPtTotZpos");
  MonitorElement* meTrackEffEtaMtdZpos = igetter.get(folder_ + "TrackEffEtaMtdZpos");
  MonitorElement* meTrackEffPhiMtdZpos = igetter.get(folder_ + "TrackEffPhiMtdZpos");
  MonitorElement* meTrackEffPtMtdZpos = igetter.get(folder_ + "TrackEffPtMtdZpos");

  if (!meTrackEffEtaTotZneg || !meTrackEffPhiTotZneg || !meTrackEffPtTotZneg || !meTrackEffEtaMtdZneg ||
      !meTrackEffPhiMtdZneg || !meTrackEffPtMtdZneg || !meTrackEffEtaTotZpos || !meTrackEffPhiTotZpos ||
      !meTrackEffPtTotZpos || !meTrackEffEtaMtdZpos || !meTrackEffPhiMtdZpos || !meTrackEffPtMtdZpos) {
    edm::LogError("EtlRecoHarvester") << "Monitoring histograms not found!" << std::endl;
    return;
  }

  // --- Book  histograms
  ibook.cd(folder_);
  meEtlEtaEff_[0] = ibook.book1D("EtlEtaEffZneg",
                                 " Track Efficiency VS Eta (-Z);#eta;Efficiency",
                                 meTrackEffEtaTotZneg->getNbinsX(),
                                 meTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                 meTrackEffEtaTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff_[0] = ibook.book1D("EtlPhiEffZneg",
                                 "Track Efficiency VS Phi (-Z);#phi [rad];Efficiency",
                                 meTrackEffPhiTotZneg->getNbinsX(),
                                 meTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                 meTrackEffPhiTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff_[0] = ibook.book1D("EtlPtEffZneg",
                                "Track Efficiency VS Pt (-Z);Pt [GeV];Efficiency",
                                meTrackEffPtTotZneg->getNbinsX(),
                                meTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmin(),
                                meTrackEffPtTotZneg->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff_[1] = ibook.book1D("EtlEtaEffZpos",
                                 " Track Efficiency VS Eta (+Z);#eta;Efficiency",
                                 meTrackEffEtaTotZpos->getNbinsX(),
                                 meTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                 meTrackEffEtaTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPhiEff_[1] = ibook.book1D("EtlPhiEffZpos",
                                 "Track Efficiency VS Phi (+Z);#phi [rad];Efficiency",
                                 meTrackEffPhiTotZpos->getNbinsX(),
                                 meTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                 meTrackEffPhiTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlPtEff_[1] = ibook.book1D("EtlPtEffZpos",
                                "Track Efficiency VS Pt (+Z);Pt [GeV];Efficiency",
                                meTrackEffPtTotZpos->getNbinsX(),
                                meTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmin(),
                                meTrackEffPtTotZpos->getTH1()->GetXaxis()->GetXmax());
  meEtlEtaEff_[0]->getTH1()->SetMinimum(0.);
  meEtlPhiEff_[0]->getTH1()->SetMinimum(0.);
  meEtlPtEff_[0]->getTH1()->SetMinimum(0.);
  meEtlEtaEff_[1]->getTH1()->SetMinimum(0.);
  meEtlPhiEff_[1]->getTH1()->SetMinimum(0.);
  meEtlPtEff_[1]->getTH1()->SetMinimum(0.);

  // --- Calculate efficiency
  for (int ibin = 1; ibin <= meTrackEffEtaTotZneg->getNbinsX(); ibin++) {
    double eff = meTrackEffEtaMtdZneg->getBinContent(ibin) / meTrackEffEtaTotZneg->getBinContent(ibin);
    double bin_err = sqrt((meTrackEffEtaMtdZneg->getBinContent(ibin) *
                           (meTrackEffEtaTotZneg->getBinContent(ibin) - meTrackEffEtaMtdZneg->getBinContent(ibin))) /
                          pow(meTrackEffEtaTotZneg->getBinContent(ibin), 3));
    if (meTrackEffEtaTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlEtaEff_[0]->setBinContent(ibin, eff);
    meEtlEtaEff_[0]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meTrackEffEtaTotZpos->getNbinsX(); ibin++) {
    double eff = meTrackEffEtaMtdZpos->getBinContent(ibin) / meTrackEffEtaTotZpos->getBinContent(ibin);
    double bin_err = sqrt((meTrackEffEtaMtdZpos->getBinContent(ibin) *
                           (meTrackEffEtaTotZpos->getBinContent(ibin) - meTrackEffEtaMtdZpos->getBinContent(ibin))) /
                          pow(meTrackEffEtaTotZpos->getBinContent(ibin), 3));
    if (meTrackEffEtaTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlEtaEff_[1]->setBinContent(ibin, eff);
    meEtlEtaEff_[1]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meTrackEffPhiTotZneg->getNbinsX(); ibin++) {
    double eff = meTrackEffPhiMtdZneg->getBinContent(ibin) / meTrackEffPhiTotZneg->getBinContent(ibin);
    double bin_err = sqrt((meTrackEffPhiMtdZneg->getBinContent(ibin) *
                           (meTrackEffPhiTotZneg->getBinContent(ibin) - meTrackEffPhiMtdZneg->getBinContent(ibin))) /
                          pow(meTrackEffPhiTotZneg->getBinContent(ibin), 3));
    if (meTrackEffPhiTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPhiEff_[0]->setBinContent(ibin, eff);
    meEtlPhiEff_[0]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meTrackEffPhiTotZpos->getNbinsX(); ibin++) {
    double eff = meTrackEffPhiMtdZpos->getBinContent(ibin) / meTrackEffPhiTotZpos->getBinContent(ibin);
    double bin_err = sqrt((meTrackEffPhiMtdZpos->getBinContent(ibin) *
                           (meTrackEffPhiTotZpos->getBinContent(ibin) - meTrackEffPhiMtdZpos->getBinContent(ibin))) /
                          pow(meTrackEffPhiTotZpos->getBinContent(ibin), 3));
    if (meTrackEffPhiTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPhiEff_[1]->setBinContent(ibin, eff);
    meEtlPhiEff_[1]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meTrackEffPtTotZneg->getNbinsX(); ibin++) {
    double eff = meTrackEffPtMtdZneg->getBinContent(ibin) / meTrackEffPtTotZneg->getBinContent(ibin);
    double bin_err = sqrt((meTrackEffPtMtdZneg->getBinContent(ibin) *
                           (meTrackEffPtTotZneg->getBinContent(ibin) - meTrackEffPtMtdZneg->getBinContent(ibin))) /
                          pow(meTrackEffPtTotZneg->getBinContent(ibin), 3));
    if (meTrackEffPtTotZneg->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPtEff_[0]->setBinContent(ibin, eff);
    meEtlPtEff_[0]->setBinError(ibin, bin_err);
  }

  for (int ibin = 1; ibin <= meTrackEffPtTotZpos->getNbinsX(); ibin++) {
    double eff = meTrackEffPtMtdZpos->getBinContent(ibin) / meTrackEffPtTotZpos->getBinContent(ibin);
    double bin_err = sqrt((meTrackEffPtMtdZpos->getBinContent(ibin) *
                           (meTrackEffPtTotZpos->getBinContent(ibin) - meTrackEffPtMtdZpos->getBinContent(ibin))) /
                          pow(meTrackEffPtTotZpos->getBinContent(ibin), 3));
    if (meTrackEffPtTotZpos->getBinContent(ibin) == 0) {
      eff = 0;
      bin_err = 0;
    }
    meEtlPtEff_[1]->setBinContent(ibin, eff);
    meEtlPtEff_[1]->setBinError(ibin, bin_err);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ----------
void EtlRecoHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/ETL/Reco/");

  descriptions.add("etlRecoPostProcessor", desc);
}

DEFINE_FWK_MODULE(EtlRecoHarvester);
