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

#include "CondTools/RunInfo/interface/LHCInfoCombined.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"

//----------------------------------------------------------------------------------------------------

class CTPPSLHCInfoPlotter : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSLHCInfoPlotter(const edm::ParameterSet &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  const edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcInfoToken_;
  const edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd> lhcInfoPerLSToken_;
  const edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd> lhcInfoPerFillToken_;
  const bool useNewLHCInfo_;

  std::string outputFile_;

  TH1D *h_beamEnergy_;
  TH1D *h_xangle_;
  TH1D *h_betaStar_;
  TH2D *h2_betaStar_vs_xangle_;

  TH1D *h_fill_;
  TH1D *h_run_;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSLHCInfoPlotter::CTPPSLHCInfoPlotter(const edm::ParameterSet &iConfig)
    : lhcInfoToken_(esConsumes(ESInputTag("", iConfig.getParameter<std::string>("lhcInfoLabel")))),
      lhcInfoPerLSToken_(esConsumes(ESInputTag("", iConfig.getParameter<std::string>("lhcInfoPerLSLabel")))),
      lhcInfoPerFillToken_(esConsumes(ESInputTag("", iConfig.getParameter<std::string>("lhcInfoPerFillLabel")))),
      useNewLHCInfo_(iConfig.getParameter<bool>("useNewLHCInfo")),
      outputFile_(iConfig.getParameter<string>("outputFile")),

      h_beamEnergy_(new TH1D("h_beamEnergy", ";beam energy   (GeV)", 81, -50., 8050.)),
      h_xangle_(new TH1D("h_xangle", ";(half) crossing angle   (#murad)", 201, -0.5, 200.5)),
      h_betaStar_(new TH1D("h_betaStar", ";#beta^{*}   (m)", 101, -0.005, 1.005)),
      h2_betaStar_vs_xangle_(new TH2D("h2_betaStar_vs_xangle",
                                      ";(half) crossing angle   (#murad);#beta^{*}   (m)",
                                      201,
                                      -0.5,
                                      200.5,
                                      101,
                                      -0.005,
                                      1.005)),

      h_fill_(new TH1D("h_fill", ";fill", 4001, 3999.5, 8000.5)),
      h_run_(new TH1D("h_run", ";run", 6000, 270E3, 330E3)) {}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoPlotter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("lhcInfoLabel", "")->setComment("label of the LHCInfo record");
  desc.add<std::string>("lhcInfoPerLSLabel", "")->setComment("label of the LHCInfoPerLS record");
  desc.add<std::string>("lhcInfoPerFillLabel", "")->setComment("label of the LHCInfoPerFill record");
  desc.add<bool>("useNewLHCInfo", false)->setComment("flag whether to use new LHCInfoPer* records or old LHCInfo");

  desc.add<std::string>("outputFile", "")->setComment("output file");

  descriptions.add("ctppsLHCInfoPlotter", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoPlotter::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  LHCInfoCombined lhcInfoCombined(iSetup, lhcInfoPerLSToken_, lhcInfoPerFillToken_, lhcInfoToken_, useNewLHCInfo_);

  h_beamEnergy_->Fill(lhcInfoCombined.energy);
  h_xangle_->Fill(lhcInfoCombined.crossingAngle());
  h_betaStar_->Fill(lhcInfoCombined.betaStarX);  //adjust accordingly to run period
  // h_betaStar_->Fill(lhcInfoCombined.betaStarY);
  h2_betaStar_vs_xangle_->Fill(lhcInfoCombined.crossingAngle(),
                               lhcInfoCombined.betaStarX);  //adjust accordingly to run period
  // h2_betaStar_vs_xangle_->Fill(lhcInfoCombined.crossingAngle(), lhcInfoCombined.betaStarY);

  h_fill_->Fill(lhcInfoCombined.fillNumber);
  h_run_->Fill(iEvent.id().run());
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoPlotter::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  h_beamEnergy_->Write();
  h_xangle_->Write();
  h_betaStar_->Write();
  h2_betaStar_vs_xangle_->Write();

  h_fill_->Write();
  h_run_->Write();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSLHCInfoPlotter);
