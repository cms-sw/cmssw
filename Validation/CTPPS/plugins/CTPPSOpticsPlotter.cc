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

#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

#include "TFile.h"
#include "TGraph.h"

#include <map>
#include <string>

//----------------------------------------------------------------------------------------------------

class CTPPSOpticsPlotter : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSOpticsPlotter(const edm::ParameterSet&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  edm::ESGetToken<LHCInterpolatedOpticalFunctionsSetCollection, CTPPSInterpolatedOpticsRcd> opticsESToken_;

  unsigned int rpId_45_N_, rpId_45_F_;
  unsigned int rpId_56_N_, rpId_56_F_;

  std::string outputFile_;

  struct RPPlots {
    std::unique_ptr<TGraph> g_v_x_vs_xi, g_L_x_vs_xi, g_x_D_vs_xi;
    std::unique_ptr<TGraph> g_v_y_vs_xi, g_L_y_vs_xi, g_y_D_vs_xi;
    std::unique_ptr<TGraph> h_y_vs_x_disp;

    RPPlots()
        : g_v_x_vs_xi(new TGraph),
          g_L_x_vs_xi(new TGraph),
          g_x_D_vs_xi(new TGraph),
          g_v_y_vs_xi(new TGraph),
          g_L_y_vs_xi(new TGraph),
          g_y_D_vs_xi(new TGraph),
          h_y_vs_x_disp(new TGraph) {}

    void write() const {
      g_v_x_vs_xi->SetTitle(";xi;v_{x}");
      g_v_x_vs_xi->Write("g_v_x_vs_xi");

      g_L_x_vs_xi->SetTitle(";xi;L_{x}   (cm)");
      g_L_x_vs_xi->Write("g_L_x_vs_xi");

      g_x_D_vs_xi->SetTitle(";xi;x_{D}   (cm)");
      g_x_D_vs_xi->Write("g_x_D_vs_xi");

      g_v_y_vs_xi->SetTitle(";xi;v_{y}");
      g_v_y_vs_xi->Write("g_v_y_vs_xi");

      g_L_y_vs_xi->SetTitle(";xi;L_{y}   (cm)");
      g_L_y_vs_xi->Write("g_L_y_vs_xi");

      g_y_D_vs_xi->SetTitle(";xi;y_{D}   (cm)");
      g_y_D_vs_xi->Write("g_y_D_vs_xi");

      h_y_vs_x_disp->SetTitle(";x   (cm);y   (cm)");
      h_y_vs_x_disp->Write("h_y_vs_x_disp");
    }
  };

  std::map<unsigned int, RPPlots> rp_plots_;

  struct ArmPlots {
    unsigned int id_N, id_F;

    std::unique_ptr<TGraph> g_de_x_vs_x_disp, g_de_y_vs_x_disp;

    ArmPlots() : g_de_x_vs_x_disp(new TGraph), g_de_y_vs_x_disp(new TGraph) {}

    void write() const {
      g_de_x_vs_x_disp->SetTitle(";x_N   (cm);x_F - x_N   (cm)");
      g_de_x_vs_x_disp->Write("g_de_x_vs_x_disp");

      g_de_y_vs_x_disp->SetTitle(";x_N   (cm);y_F - y_N   (cm)");
      g_de_y_vs_x_disp->Write("g_de_y_vs_x_disp");
    }
  };

  std::map<unsigned int, ArmPlots> arm_plots_;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSOpticsPlotter::CTPPSOpticsPlotter(const edm::ParameterSet& iConfig)
    : opticsESToken_(esConsumes(ESInputTag("", iConfig.getParameter<std::string>("opticsLabel")))),

      rpId_45_N_(iConfig.getParameter<unsigned int>("rpId_45_N")),
      rpId_45_F_(iConfig.getParameter<unsigned int>("rpId_45_F")),
      rpId_56_N_(iConfig.getParameter<unsigned int>("rpId_56_N")),
      rpId_56_F_(iConfig.getParameter<unsigned int>("rpId_56_F")),

      outputFile_(iConfig.getParameter<string>("outputFile")) {
  arm_plots_[0].id_N = rpId_45_N_;
  arm_plots_[0].id_F = rpId_45_F_;

  arm_plots_[1].id_N = rpId_56_N_;
  arm_plots_[1].id_F = rpId_56_F_;
}

//----------------------------------------------------------------------------------------------------

void CTPPSOpticsPlotter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // stop if plots already made
  if (!rp_plots_.empty())
    return;

  // get conditions
  const auto& opticalFunctions = iSetup.getData(opticsESToken_);

  // stop if conditions invalid
  if (opticalFunctions.empty())
    return;

  // make per-RP plots
  for (const auto& it : opticalFunctions) {
    CTPPSDetId rpId(it.first);
    unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

    auto& pl = rp_plots_[rpDecId];

    LHCInterpolatedOpticalFunctionsSet::Kinematics k_in_beam = {0., 0., 0., 0., 0.};
    LHCInterpolatedOpticalFunctionsSet::Kinematics k_out_beam;
    it.second.transport(k_in_beam, k_out_beam);

    const double vtx_ep = 1E-4;  // cm
    const double th_ep = 1E-6;   // rad

    for (double xi = 0.; xi < 0.30001; xi += 0.001) {
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_in_xi = {0., 0., 0., 0., xi};
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_out_xi;
      it.second.transport(k_in_xi, k_out_xi);

      LHCInterpolatedOpticalFunctionsSet::Kinematics k_in_xi_vtx_x = {vtx_ep, 0., 0., 0., xi};
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_out_xi_vtx_x;
      it.second.transport(k_in_xi_vtx_x, k_out_xi_vtx_x);

      LHCInterpolatedOpticalFunctionsSet::Kinematics k_in_xi_th_x = {0., th_ep, 0., 0., xi};
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_out_xi_th_x;
      it.second.transport(k_in_xi_th_x, k_out_xi_th_x);

      LHCInterpolatedOpticalFunctionsSet::Kinematics k_in_xi_vtx_y = {0., 0., vtx_ep, 0., xi};
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_out_xi_vtx_y;
      it.second.transport(k_in_xi_vtx_y, k_out_xi_vtx_y);

      LHCInterpolatedOpticalFunctionsSet::Kinematics k_in_xi_th_y = {0., 0., 0., th_ep, xi};
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_out_xi_th_y;
      it.second.transport(k_in_xi_th_y, k_out_xi_th_y);

      int idx = pl.g_v_x_vs_xi->GetN();

      pl.g_v_x_vs_xi->SetPoint(idx, xi, (k_out_xi_vtx_x.x - k_out_xi.x) / vtx_ep);
      pl.g_L_x_vs_xi->SetPoint(idx, xi, (k_out_xi_th_x.x - k_out_xi.x) / th_ep);
      pl.g_x_D_vs_xi->SetPoint(idx, xi, k_out_xi.x - k_out_beam.x);

      pl.g_v_y_vs_xi->SetPoint(idx, xi, (k_out_xi_vtx_y.y - k_out_xi.y) / vtx_ep);
      pl.g_L_y_vs_xi->SetPoint(idx, xi, (k_out_xi_th_y.y - k_out_xi.y) / th_ep);
      pl.g_y_D_vs_xi->SetPoint(idx, xi, k_out_xi.y - k_out_beam.y);

      pl.h_y_vs_x_disp->SetPoint(idx, k_out_xi.x - k_out_beam.x, k_out_xi.y - k_out_beam.y);
    }
  }

  // make per-arm plots
  for (const auto& ap : arm_plots_) {
    // find optics objects
    const LHCInterpolatedOpticalFunctionsSet *opt_N = nullptr, *opt_F = nullptr;

    for (const auto& it : opticalFunctions) {
      CTPPSDetId rpId(it.first);
      unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

      if (rpDecId == ap.second.id_N)
        opt_N = &it.second;
      if (rpDecId == ap.second.id_F)
        opt_F = &it.second;
    }

    if (!opt_N || !opt_F) {
      edm::LogError("CTPPSOpticsPlotter::analyze") << "Cannot find optics objects for arm " << ap.first;
      continue;
    }

    LHCInterpolatedOpticalFunctionsSet::Kinematics k_in_beam = {0., 0., 0., 0., 0.};

    LHCInterpolatedOpticalFunctionsSet::Kinematics k_out_beam_N, k_out_beam_F;
    opt_N->transport(k_in_beam, k_out_beam_N);
    opt_F->transport(k_in_beam, k_out_beam_F);

    for (double xi = 0.; xi < 0.30001; xi += 0.001) {
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_in_xi = {0., 0., 0., 0., xi};

      LHCInterpolatedOpticalFunctionsSet::Kinematics k_out_xi_N, k_out_xi_F;
      opt_N->transport(k_in_xi, k_out_xi_N);
      opt_F->transport(k_in_xi, k_out_xi_F);

      int idx = ap.second.g_de_x_vs_x_disp->GetN();

      ap.second.g_de_x_vs_x_disp->SetPoint(
          idx, k_out_xi_N.x - k_out_beam_N.x, (k_out_xi_F.x - k_out_beam_F.x) - (k_out_xi_N.x - k_out_beam_N.x));
      ap.second.g_de_y_vs_x_disp->SetPoint(
          idx, k_out_xi_N.x - k_out_beam_N.x, (k_out_xi_F.y - k_out_beam_F.y) - (k_out_xi_N.y - k_out_beam_N.y));
    }
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSOpticsPlotter::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  for (const auto& p : rp_plots_) {
    gDirectory = f_out->mkdir(Form("%u", p.first));
    p.second.write();
  }

  for (const auto& p : arm_plots_) {
    gDirectory = f_out->mkdir(Form("arm %u", p.first));
    p.second.write();
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSOpticsPlotter);
