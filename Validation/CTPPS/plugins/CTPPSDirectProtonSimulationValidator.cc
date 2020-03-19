/****************************************************************************
 *
 * This is a part of CTPPS validation software
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"

#include <map>

//----------------------------------------------------------------------------------------------------

class CTPPSDirectProtonSimulationValidator : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSDirectProtonSimulationValidator(const edm::ParameterSet&);

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> simuTracksToken_;
  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> recoTracksToken_;

  std::string outputFile_;

  struct RPPlots {
    std::unique_ptr<TH2D> h2_xr_vs_xs, h2_yr_vs_ys;
    std::unique_ptr<TH1D> h_de_x, h_de_y;

    RPPlots()
        : h2_xr_vs_xs(new TH2D("", ";x_simu   (mm);x_reco   (mm)", 100, -10., +10., 100, -10, +10.)),
          h2_yr_vs_ys(new TH2D("", "y_simu   (mm);y_reco   (mm)", 100, -10., +10., 100, -10, +10.)),
          h_de_x(new TH1D("", ";x   (mm)", 200, -100E-3, +100E-3)),
          h_de_y(new TH1D("", ";y   (mm)", 200, -100E-3, +100E-3)) {}

    void fill(double simu_x, double simu_y, double reco_x, double reco_y) {
      h2_xr_vs_xs->Fill(simu_x, reco_x);
      h2_yr_vs_ys->Fill(simu_y, reco_y);

      h_de_x->Fill(reco_x - simu_x);
      h_de_y->Fill(reco_y - simu_y);
    }

    void write() const {
      h2_xr_vs_xs->Write("h2_xr_vs_xs");
      h2_yr_vs_ys->Write("h2_yr_vs_ys");
      h_de_x->Write("h_de_x");
      h_de_y->Write("h_de_y");
    }
  };

  std::map<unsigned int, RPPlots> rpPlots_;
};

//----------------------------------------------------------------------------------------------------

CTPPSDirectProtonSimulationValidator::CTPPSDirectProtonSimulationValidator(const edm::ParameterSet& iConfig)
    : simuTracksToken_(consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("simuTracksTag"))),
      recoTracksToken_(consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("recoTracksTag"))),
      outputFile_(iConfig.getParameter<std::string>("outputFile")) {}

//----------------------------------------------------------------------------------------------------

void CTPPSDirectProtonSimulationValidator::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  // get input
  edm::Handle<CTPPSLocalTrackLiteCollection> simuTracks;
  iEvent.getByToken(simuTracksToken_, simuTracks);

  edm::Handle<CTPPSLocalTrackLiteCollection> recoTracks;
  iEvent.getByToken(recoTracksToken_, recoTracks);

  // process tracks
  for (const auto& simuTrack : *simuTracks) {
    const CTPPSDetId rpId(simuTrack.rpId());
    for (const auto& recoTrack : *recoTracks) {
      if (simuTrack.rpId() == recoTrack.rpId()) {
        unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();
        rpPlots_[rpDecId].fill(simuTrack.x(), simuTrack.y(), recoTrack.x(), recoTrack.y());
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSDirectProtonSimulationValidator::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  for (const auto& it : rpPlots_) {
    gDirectory = f_out->mkdir(Form("RP %u", it.first));
    it.second.write();
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSDirectProtonSimulationValidator);
