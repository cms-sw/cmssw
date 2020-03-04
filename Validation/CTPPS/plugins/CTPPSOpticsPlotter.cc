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
#include "CondFormats/CTPPSReadoutObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

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

  std::string opticsLabel_;

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
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSOpticsPlotter::CTPPSOpticsPlotter(const edm::ParameterSet& iConfig)
    : opticsLabel_(iConfig.getParameter<std::string>("opticsLabel")),
      outputFile_(iConfig.getParameter<string>("outputFile")) {}

//----------------------------------------------------------------------------------------------------

void CTPPSOpticsPlotter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // stop if plots already made
  if (!rp_plots_.empty())
    return;

  // get conditions
  edm::ESHandle<LHCInterpolatedOpticalFunctionsSetCollection> hOpticalFunctions;
  iSetup.get<CTPPSInterpolatedOpticsRcd>().get(opticsLabel_, hOpticalFunctions);

  // stop if conditions invalid
  if (hOpticalFunctions->empty())
    return;

  // make plots
  for (const auto& it : *hOpticalFunctions) {
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
}

//----------------------------------------------------------------------------------------------------

void CTPPSOpticsPlotter::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  for (const auto& p : rp_plots_) {
    gDirectory = f_out->mkdir(Form("%u", p.first));
    p.second.write();
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSOpticsPlotter);
