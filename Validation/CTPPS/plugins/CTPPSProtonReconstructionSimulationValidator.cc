/****************************************************************************
 * Authors:
 *   Jan Kašpar
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TGraphErrors.h"

#include <map>
#include <string>

//----------------------------------------------------------------------------------------------------

class CTPPSProtonReconstructionSimulationValidator : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSProtonReconstructionSimulationValidator(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  void fillPlots(unsigned int meth_idx,
                 unsigned int idx,
                 const reco::ForwardProton &rec_pr,
                 const HepMC::FourVector &vtx,
                 const HepMC::FourVector &mom,
                 const LHCInfo &lhcInfo);

  edm::EDGetTokenT<edm::HepMCProduct> tokenHepMCBeforeSmearing_;
  edm::EDGetTokenT<edm::HepMCProduct> tokenHepMCAfterSmearing_;

  edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsSingleRP_;
  edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsMultiRP_;

  std::string lhcInfoLabel_;

  std::string outputFile_;

  struct PlotGroup {
    std::unique_ptr<TH1D> h_de_xi;
    std::unique_ptr<TProfile> p_de_xi_vs_xi_simu;
    std::unique_ptr<TH2D> h_xi_reco_vs_xi_simu;

    std::unique_ptr<TH1D> h_de_th_x;
    std::unique_ptr<TProfile> p_de_th_x_vs_xi_simu;

    std::unique_ptr<TH1D> h_de_th_y;
    std::unique_ptr<TProfile> p_de_th_y_vs_xi_simu;

    std::unique_ptr<TH1D> h_de_vtx_y;
    std::unique_ptr<TProfile> p_de_vtx_y_vs_xi_simu;

    std::unique_ptr<TH1D> h_de_t;
    std::unique_ptr<TProfile> p_de_t_vs_xi_simu, p_de_t_vs_t_simu;

    PlotGroup()
        : h_de_xi(new TH1D("", ";#xi_{reco} - #xi_{simu}", 100, 0., 0.)),
          p_de_xi_vs_xi_simu(new TProfile("", ";#xi_{simu};#xi_{reco} - #xi_{simu}", 19, 0.015, 0.205)),
          h_xi_reco_vs_xi_simu(new TH2D("", ";#xi_{simu};#xi_{reco}", 100, 0., 0.30, 100, 0., 0.30)),

          h_de_th_x(new TH1D("", ";#theta_{x,reco} - #theta_{x,simu}", 100, 0., 0.)),
          p_de_th_x_vs_xi_simu(new TProfile("", ";#xi_{simu};#theta_{x,reco} - #theta_{x,simu}", 19, 0.015, 0.205)),

          h_de_th_y(new TH1D("", ";#theta_{y,reco} - #theta_{y,simu}", 100, 0., 0.)),
          p_de_th_y_vs_xi_simu(new TProfile("", ";#xi_{simu};#theta_{y,reco} - #theta_{y,simu}", 19, 0.015, 0.205)),

          h_de_vtx_y(new TH1D("", ";vtx_{y,reco} - vtx_{y,simu}   (mm)", 100, 0., 0.)),
          p_de_vtx_y_vs_xi_simu(new TProfile("", ";#xi_{simu};vtx_{y,reco} - vtx_{y,simu} (mm)", 19, 0.015, 0.205)),

          h_de_t(new TH1D("", ";t_{reco} - t_{simu}", 100, -1., +1.)),
          p_de_t_vs_xi_simu(new TProfile("", ";xi_{simu};t_{reco} - t_{simu}", 19, 0.015, 0.205)),
          p_de_t_vs_t_simu(new TProfile("", ";t_{simu};t_{reco} - t_{simu}", 20, 0., 5.)) {}

    static TGraphErrors profileToRMSGraph(TProfile *p, const char *name = "") {
      TGraphErrors gr_err;
      gr_err.SetName(name);

      for (int bi = 1; bi <= p->GetNbinsX(); ++bi) {
        double c = p->GetBinCenter(bi);
        double w = p->GetBinWidth(bi);

        double N = p->GetBinEntries(bi);
        double Sy = p->GetBinContent(bi) * N;
        double Syy = p->GetSumw2()->At(bi);

        double si_sq = Syy / N - Sy * Sy / N / N;
        double si = (si_sq >= 0.) ? sqrt(si_sq) : 0.;
        double si_unc_sq = si_sq / 2. / N;  // Gaussian approximation
        double si_unc = (si_unc_sq >= 0.) ? sqrt(si_unc_sq) : 0.;

        int idx = gr_err.GetN();
        gr_err.SetPoint(idx, c, si);
        gr_err.SetPointError(idx, w / 2., si_unc);
      }

      return gr_err;
    }

    void write() const {
      h_xi_reco_vs_xi_simu->Write("h_xi_reco_vs_xi_simu");
      h_de_xi->Write("h_de_xi");
      p_de_xi_vs_xi_simu->Write("p_de_xi_vs_xi_simu");
      profileToRMSGraph(p_de_xi_vs_xi_simu.get(), "g_rms_de_xi_vs_xi_simu").Write();

      h_de_th_x->Write("h_de_th_x");
      p_de_th_x_vs_xi_simu->Write("p_de_th_x_vs_xi_simu");
      profileToRMSGraph(p_de_th_x_vs_xi_simu.get(), "g_rms_de_th_x_vs_xi_simu").Write();

      h_de_th_y->Write("h_de_th_y");
      p_de_th_y_vs_xi_simu->Write("p_de_th_y_vs_xi_simu");
      profileToRMSGraph(p_de_th_y_vs_xi_simu.get(), "g_rms_de_th_y_vs_xi_simu").Write();

      h_de_vtx_y->Write("h_de_vtx_y");
      p_de_vtx_y_vs_xi_simu->Write("p_de_vtx_y_vs_xi_simu");
      profileToRMSGraph(p_de_vtx_y_vs_xi_simu.get(), "g_rms_de_vtx_y_vs_xi_simu").Write();

      h_de_t->Write("h_de_t");
      p_de_t_vs_xi_simu->Write("p_de_t_vs_xi_simu");
      profileToRMSGraph(p_de_t_vs_xi_simu.get(), "g_rms_de_t_vs_xi_simu").Write();
      p_de_t_vs_t_simu->Write("p_de_t_vs_t_simu");
      profileToRMSGraph(p_de_t_vs_t_simu.get(), "g_rms_de_t_vs_t_simu").Write();
    }
  };

  std::map<unsigned int, std::map<unsigned int, PlotGroup>> plots_;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;
using namespace HepMC;

//----------------------------------------------------------------------------------------------------

CTPPSProtonReconstructionSimulationValidator::CTPPSProtonReconstructionSimulationValidator(
    const edm::ParameterSet &iConfig)
    : tokenHepMCBeforeSmearing_(
          consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("tagHepMCBeforeSmearing"))),
      tokenHepMCAfterSmearing_(
          consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("tagHepMCAfterSmearing"))),
      tokenRecoProtonsSingleRP_(
          consumes<reco::ForwardProtonCollection>(iConfig.getParameter<InputTag>("tagRecoProtonsSingleRP"))),
      tokenRecoProtonsMultiRP_(
          consumes<reco::ForwardProtonCollection>(iConfig.getParameter<InputTag>("tagRecoProtonsMultiRP"))),
      lhcInfoLabel_(iConfig.getParameter<std::string>("lhcInfoLabel")),
      outputFile_(iConfig.getParameter<string>("outputFile")) {}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionSimulationValidator::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get conditions
  edm::ESHandle<LHCInfo> hLHCInfo;
  iSetup.get<LHCInfoRcd>().get(lhcInfoLabel_, hLHCInfo);

  // get input
  edm::Handle<edm::HepMCProduct> hHepMCBeforeSmearing;
  iEvent.getByToken(tokenHepMCBeforeSmearing_, hHepMCBeforeSmearing);
  HepMC::GenEvent *hepMCEventBeforeSmearing = (HepMC::GenEvent *)hHepMCBeforeSmearing->GetEvent();

  edm::Handle<edm::HepMCProduct> hHepMCAfterSmearing;
  iEvent.getByToken(tokenHepMCAfterSmearing_, hHepMCAfterSmearing);
  HepMC::GenEvent *hepMCEventAfterSmearing = (HepMC::GenEvent *)hHepMCAfterSmearing->GetEvent();

  Handle<reco::ForwardProtonCollection> hRecoProtonsSingleRP;
  iEvent.getByToken(tokenRecoProtonsSingleRP_, hRecoProtonsSingleRP);

  Handle<reco::ForwardProtonCollection> hRecoProtonsMultiRP;
  iEvent.getByToken(tokenRecoProtonsMultiRP_, hRecoProtonsMultiRP);

  // extract vertex position
  bool vertex_set = false;
  FourVector vtx;
  for (auto it = hepMCEventAfterSmearing->vertices_begin(); it != hepMCEventAfterSmearing->vertices_end(); ++it) {
    if (vertex_set) {
      LogError("CTPPSProtonReconstructionSimulationValidator") << "Multiple vertices found.";
      return;
    }

    vertex_set = true;
    vtx = (*it)->position();
  }

  // extract forward protons
  bool proton_45_set = false;
  bool proton_56_set = false;
  FourVector mom_45, mom_56;

  for (auto it = hepMCEventBeforeSmearing->particles_begin(); it != hepMCEventBeforeSmearing->particles_end(); ++it) {
    const auto &part = *it;

    // accept only stable non-beam protons
    if (part->pdg_id() != 2212)
      continue;

    if (part->status() != 1)
      continue;

    if (part->is_beam())
      continue;

    const auto &mom = part->momentum();

    if (mom.e() < 4500.)
      continue;

    if (mom.z() > 0) {
      // 45
      if (proton_45_set) {
        LogError("CTPPSProtonReconstructionSimulationValidator") << "Found multiple protons in sector 45.";
        return;
      }

      proton_45_set = true;
      mom_45 = mom;
    } else {
      // 56
      if (proton_56_set) {
        LogError("CTPPSProtonReconstructionSimulationValidator") << "Found multiple protons in sector 56.";
        return;
      }

      proton_56_set = true;
      mom_56 = mom;
    }
  }

  // do comparison
  for (const auto &handle : {hRecoProtonsSingleRP, hRecoProtonsMultiRP}) {
    for (const auto &rec_pr : *handle) {
      if (!rec_pr.validFit())
        continue;

      unsigned int idx;

      bool mom_set = false;
      FourVector mom;

      if (rec_pr.lhcSector() == reco::ForwardProton::LHCSector::sector45) {
        idx = 0;
        mom_set = proton_45_set;
        mom = mom_45;
      } else {
        idx = 1;
        mom_set = proton_56_set;
        mom = mom_56;
      }

      if (!mom_set)
        continue;

      unsigned int meth_idx = 1234;

      if (rec_pr.method() == reco::ForwardProton::ReconstructionMethod::singleRP) {
        meth_idx = 0;

        CTPPSDetId rpId((*rec_pr.contributingLocalTracks().begin())->rpId());
        idx = 100 * rpId.arm() + 10 * rpId.station() + rpId.rp();
      }

      if (rec_pr.method() == reco::ForwardProton::ReconstructionMethod::multiRP)
        meth_idx = 1;

      fillPlots(meth_idx, idx, rec_pr, vtx, mom, *hLHCInfo);
    }
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionSimulationValidator::fillPlots(unsigned int meth_idx,
                                                             unsigned int idx,
                                                             const reco::ForwardProton &rec_pr,
                                                             const HepMC::FourVector &vtx,
                                                             const HepMC::FourVector &mom,
                                                             const LHCInfo &lhcInfo) {
  const double p_nom = lhcInfo.energy();
  const double xi_simu = (p_nom - mom.rho()) / p_nom;
  const double th_x_simu = mom.x() / mom.rho();
  const double th_y_simu = mom.y() / mom.rho();
  const double vtx_y_simu = vtx.y();
  const double th_simu = sqrt(th_x_simu * th_x_simu + th_y_simu * th_y_simu);
  const double t_simu = -reco::ForwardProton::calculateT(p_nom, mom.rho(), th_simu);

  const double xi_reco = rec_pr.xi();
  const double th_x_reco = rec_pr.thetaX();
  const double th_y_reco = rec_pr.thetaY();
  const double vtx_y_reco = rec_pr.vertex().y() * 10.;  // conversion: cm --> mm
  const double t_reco = -rec_pr.t();

  auto &plt = plots_[meth_idx][idx];

  plt.h_xi_reco_vs_xi_simu->Fill(xi_simu, xi_reco);
  plt.h_de_xi->Fill(xi_reco - xi_simu);
  plt.p_de_xi_vs_xi_simu->Fill(xi_simu, xi_reco - xi_simu);

  plt.h_de_th_x->Fill(th_x_reco - th_x_simu);
  plt.p_de_th_x_vs_xi_simu->Fill(xi_simu, th_x_reco - th_x_simu);

  plt.h_de_th_y->Fill(th_y_reco - th_y_simu);
  plt.p_de_th_y_vs_xi_simu->Fill(xi_simu, th_y_reco - th_y_simu);

  plt.h_de_vtx_y->Fill(vtx_y_reco - vtx_y_simu);
  plt.p_de_vtx_y_vs_xi_simu->Fill(xi_simu, vtx_y_reco - vtx_y_simu);

  plt.h_de_t->Fill(t_reco - t_simu);
  plt.p_de_t_vs_xi_simu->Fill(xi_simu, t_reco - t_simu);
  plt.p_de_t_vs_t_simu->Fill(t_simu, t_reco - t_simu);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionSimulationValidator::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  for (const auto &mit : plots_) {
    const char *method = (mit.first == 0) ? "single rp" : "multi rp";
    TDirectory *d_method = f_out->mkdir(method);

    for (const auto &eit : mit.second) {
      gDirectory = d_method->mkdir(Form("%i", eit.first));
      eit.second.write();
    }
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstructionSimulationValidator);
