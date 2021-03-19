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

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "TFile.h"
#include "TH1D.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class CTPPSHepMCDistributionPlotter : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSHepMCDistributionPlotter(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  edm::EDGetTokenT<edm::HepMCProduct> tokenHepMC_;
  edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcInfoESToken_;
  std::string outputFile_;

  std::unique_ptr<TH1D> h_vtx_x_, h_vtx_y_, h_vtx_z_, h_vtx_t_;
  std::unique_ptr<TH1D> h_xi_, h_th_x_, h_th_y_;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;
using namespace HepMC;

//----------------------------------------------------------------------------------------------------

CTPPSHepMCDistributionPlotter::CTPPSHepMCDistributionPlotter(const edm::ParameterSet &iConfig)
    : tokenHepMC_(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("tagHepMC"))),
      lhcInfoESToken_(esConsumes(ESInputTag("", iConfig.getParameter<std::string>("lhcInfoLabel")))),
      outputFile_(iConfig.getParameter<string>("outputFile")),

      h_vtx_x_(new TH1D("h_vtx_x", ";vtx_x   (mm)", 100, 0., 0.)),
      h_vtx_y_(new TH1D("h_vtx_y", ";vtx_y   (mm)", 100, 0., 0.)),
      h_vtx_z_(new TH1D("h_vtx_z", ";vtx_z   (mm)", 100, 0., 0.)),
      h_vtx_t_(new TH1D("h_vtx_t", ";vtx_t   (mm)", 100, 0., 0.)),

      h_xi_(new TH1D("h_xi", ";#xi", 100, 0., 0.30)),
      h_th_x_(new TH1D("h_th_x", ";#theta^{*}_{x}", 100, -300E-6, +300E-6)),
      h_th_y_(new TH1D("h_th_y", ";#theta^{*}_{y}", 100, -300E-6, +300E-6)) {}

//----------------------------------------------------------------------------------------------------

void CTPPSHepMCDistributionPlotter::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get conditions
  const auto &lhcInfo = iSetup.getData(lhcInfoESToken_);

  // get input
  edm::Handle<edm::HepMCProduct> hHepMC;
  iEvent.getByToken(tokenHepMC_, hHepMC);
  HepMC::GenEvent *hepMCEvent = (HepMC::GenEvent *)hHepMC->GetEvent();

  // plot vertices
  for (HepMC::GenEvent::vertex_iterator vit = hepMCEvent->vertices_begin(); vit != hepMCEvent->vertices_end(); ++vit) {
    const auto pos = (*vit)->position();
    h_vtx_x_->Fill(pos.x());
    h_vtx_y_->Fill(pos.y());
    h_vtx_z_->Fill(pos.z());
    h_vtx_t_->Fill(pos.t());
  }

  // extract protons
  for (auto it = hepMCEvent->particles_begin(); it != hepMCEvent->particles_end(); ++it) {
    const auto &part = *it;

    // accept only stable non-beam protons
    if (part->pdg_id() != 2212)
      continue;

    if (part->status() != 1)
      continue;

    if (part->is_beam())
      continue;

    const auto &mom = part->momentum();
    const double p_nom = lhcInfo.energy();

    if (mom.rho() / p_nom < 0.7)
      continue;

    const double xi_simu = (p_nom - mom.e()) / p_nom;
    const double th_x_simu = mom.x() / mom.rho();
    const double th_y_simu = mom.y() / mom.rho();

    h_xi_->Fill(xi_simu);
    h_th_x_->Fill(th_x_simu);
    h_th_y_->Fill(th_y_simu);
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSHepMCDistributionPlotter::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  h_vtx_x_->Write();
  h_vtx_y_->Write();
  h_vtx_z_->Write();
  h_vtx_t_->Write();

  h_xi_->Write();
  h_th_x_->Write();
  h_th_y_->Write();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSHepMCDistributionPlotter);
