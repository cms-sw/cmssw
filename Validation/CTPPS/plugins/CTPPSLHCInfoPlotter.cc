/****************************************************************************
 * Authors:
 *   Jan Kašpar
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "TFile.h"
#include "TH1D.h"

//----------------------------------------------------------------------------------------------------

class CTPPSLHCInfoPlotter : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSLHCInfoPlotter(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  std::string lhcInfoLabel_;

  std::string outputFile_;

  TH1D *h_beamEnergy_;
  TH1D *h_xangle_;
  TH1D *h_betaStar_;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSLHCInfoPlotter::CTPPSLHCInfoPlotter(const edm::ParameterSet &iConfig)
    : lhcInfoLabel_(iConfig.getParameter<std::string>("lhcInfoLabel")),
      outputFile_(iConfig.getParameter<string>("outputFile")),

      h_beamEnergy_(new TH1D("h_beamEnergy", ";beam energy   (GeV)", 81, -50., 8050.)),
      h_xangle_(new TH1D("h_xangle", ";(half) crossing angle   (#murad)", 201, -0.5, 200.5)),
      h_betaStar_(new TH1D("h_betaStar", ";#beta^{*}   (m)", 101, -0.005, 1.005)) {}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoPlotter::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::ESHandle<LHCInfo> hLHCInfo;
  iSetup.get<LHCInfoRcd>().get(lhcInfoLabel_, hLHCInfo);

  h_beamEnergy_->Fill(hLHCInfo->energy());
  h_xangle_->Fill(hLHCInfo->crossingAngle());
  h_betaStar_->Fill(hLHCInfo->betaStar());
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoPlotter::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  h_beamEnergy_->Write();
  h_xangle_->Write();
  h_betaStar_->Write();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSLHCInfoPlotter);
