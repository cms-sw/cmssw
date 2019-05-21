/****************************************************************************
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com)
****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

#include "TFile.h"
#include "TGraphErrors.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"

//----------------------------------------------------------------------------------------------------

class CTPPSProtonReconstructionPlotter : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSProtonReconstructionPlotter(const edm::ParameterSet &);
  ~CTPPSProtonReconstructionPlotter() override {}

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  void endJob() override;

  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tokenTracks_;
  edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsSingleRP_;
  edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsMultiRP_;

  unsigned int rpId_45_N_, rpId_45_F_;
  unsigned int rpId_56_N_, rpId_56_F_;

  std::string outputFile_;

  signed int maxNonEmptyEvents_;

  static void profileToRMSGraph(TProfile *p, TGraphErrors *g) {
    for (int bi = 1; bi <= p->GetNbinsX(); ++bi) {
      double c = p->GetBinCenter(bi);

      double N = p->GetBinEntries(bi);
      double Sy = p->GetBinContent(bi) * N;
      double Syy = p->GetSumw2()->At(bi);

      double si_sq = Syy / N - Sy * Sy / N / N;
      double si = (si_sq >= 0.) ? sqrt(si_sq) : 0.;
      double si_unc_sq = si_sq / 2. / N;  // Gaussian approximation
      double si_unc = (si_unc_sq >= 0.) ? sqrt(si_unc_sq) : 0.;

      int idx = g->GetN();
      g->SetPoint(idx, c, si);
      g->SetPointError(idx, 0., si_unc);
    }
  }

  struct SingleRPPlots {
    std::unique_ptr<TH1D> h_xi;
    std::unique_ptr<TH2D> h2_th_y_vs_xi;
    std::unique_ptr<TProfile> p_th_y_vs_xi;

    SingleRPPlots()
        : h_xi(new TH1D("", ";#xi", 100, 0., 0.3)),
          h2_th_y_vs_xi(new TH2D("", ";#xi;#theta_{y}   (rad)", 100, 0., 0.3, 100, -500E-6, +500E-6)),
          p_th_y_vs_xi(new TProfile("", ";#xi;#theta_{y}   (rad)", 100, 0., 0.3)) {}

    void fill(const reco::ForwardProton &p) {
      if (p.validFit()) {
        h_xi->Fill(p.xi());

        const double th_y = p.thetaY();
        h2_th_y_vs_xi->Fill(p.xi(), th_y);
        p_th_y_vs_xi->Fill(p.xi(), th_y);
      }
    }

    void write() const {
      h_xi->Write("h_xi");

      h2_th_y_vs_xi->Write("h2_th_y_vs_xi");
      p_th_y_vs_xi->Write("p_th_y_vs_xi");
    }
  };

  std::map<unsigned int, SingleRPPlots> singleRPPlots_;

  struct MultiRPPlots {
    std::unique_ptr<TH1D> h_xi, h_th_x, h_th_y, h_vtx_y, h_t_unif, h_t, h_chi_sq, h_chi_sq_norm;
    std::unique_ptr<TH1D> h_t_xi_range1, h_t_xi_range2, h_t_xi_range3;
    std::unique_ptr<TH1D> h_n_tracking_RPs, h_n_timing_RPs;
    std::unique_ptr<TH2D> h2_th_x_vs_xi, h2_th_y_vs_xi, h2_vtx_y_vs_xi, h2_t_vs_xi;
    std::unique_ptr<TProfile> p_th_x_vs_xi, p_th_y_vs_xi, p_vtx_y_vs_xi;

    MultiRPPlots()
        : h_xi(new TH1D("", ";#xi", 100, 0., 0.3)),
          h_th_x(new TH1D("", ";#theta_{x}   (rad)", 100, -500E-6, +500E-6)),
          h_th_y(new TH1D("", ";#theta_{y}   (rad)", 100, -500E-6, +500E-6)),
          h_vtx_y(new TH1D("", ";vtx_{y}   (cm)", 100, -2., +2.)),
          h_chi_sq(new TH1D("", ";#chi^{2}", 100, 0., 0.)),
          h_chi_sq_norm(new TH1D("", ";#chi^{2}/ndf", 100, 0., 5.)),
          h_n_tracking_RPs(new TH1D("", ";n of tracking RPs", 4, -0.5, +3.5)),
          h_n_timing_RPs(new TH1D("", ";n of timing RPs", 4, -0.5, +3.5)),
          h2_th_x_vs_xi(new TH2D("", ";#xi;#theta_{x}   (rad)", 100, 0., 0.3, 100, -500E-6, +500E-6)),
          h2_th_y_vs_xi(new TH2D("", ";#xi;#theta_{y}   (rad)", 100, 0., 0.3, 100, -500E-6, +500E-6)),
          h2_vtx_y_vs_xi(new TH2D("", ";#xi;vtx_{y}   (cm)", 100, 0., 0.3, 100, -500E-3, +500E-3)),
          p_th_x_vs_xi(new TProfile("", ";#xi;#theta_{x}   (rad)", 100, 0., 0.3)),
          p_th_y_vs_xi(new TProfile("", ";#xi;#theta_{y}   (rad)", 100, 0., 0.3)),
          p_vtx_y_vs_xi(new TProfile("", ";#xi;vtx_{y}   (cm)", 100, 0., 0.3)) {
      std::vector<double> v_t_bin_edges;
      for (double t = 0; t <= 5.;) {
        v_t_bin_edges.push_back(t);
        const double de_t = 0.05 + 0.09 * t + 0.02 * t * t;
        t += de_t;
      }
      h_t_unif.reset(new TH1D("", ";|t|   (GeV^2)", 100, 0., 5.));
      h_t.reset(new TH1D("", ";|t|   (GeV^2)", v_t_bin_edges.size() - 1, v_t_bin_edges.data()));
      h_t_xi_range1.reset(new TH1D("", ";|t|   (GeV^2)", v_t_bin_edges.size() - 1, v_t_bin_edges.data()));
      h_t_xi_range2.reset(new TH1D("", ";|t|   (GeV^2)", v_t_bin_edges.size() - 1, v_t_bin_edges.data()));
      h_t_xi_range3.reset(new TH1D("", ";|t|   (GeV^2)", v_t_bin_edges.size() - 1, v_t_bin_edges.data()));
      h2_t_vs_xi.reset(
          new TH2D("", ";#xi;|t|   (GeV^2)", 100, 0., 0.3, v_t_bin_edges.size() - 1, v_t_bin_edges.data()));
    }

    void fill(const reco::ForwardProton &p) {
      if (!p.validFit())
        return;

      unsigned int n_tracking_RPs = 0, n_timing_RPs = 0;
      for (const auto &tr : p.contributingLocalTracks()) {
        CTPPSDetId detId(tr->getRPId());
        if (detId.subdetId() == CTPPSDetId::sdTrackingStrip || detId.subdetId() == CTPPSDetId::sdTrackingPixel)
          n_tracking_RPs++;
        if (detId.subdetId() == CTPPSDetId::sdTimingDiamond || detId.subdetId() == CTPPSDetId::sdTimingFastSilicon)
          n_timing_RPs++;
      }

      const double th_x = p.thetaX();
      const double th_y = p.thetaY();
      const double mt = -p.t();

      h_chi_sq->Fill(p.chi2());
      if (p.ndof() > 0)
        h_chi_sq_norm->Fill(p.normalizedChi2());

      h_n_tracking_RPs->Fill(n_tracking_RPs);
      h_n_timing_RPs->Fill(n_timing_RPs);

      h_xi->Fill(p.xi());

      h_th_x->Fill(th_x);
      h_th_y->Fill(th_y);

      h_vtx_y->Fill(p.vertex().y());

      h_t_unif->Fill(mt);
      h_t->Fill(mt);
      if (p.xi() > 0.04 && p.xi() < 0.07)
        h_t_xi_range1->Fill(mt);
      if (p.xi() > 0.07 && p.xi() < 0.10)
        h_t_xi_range2->Fill(mt);
      if (p.xi() > 0.10 && p.xi() < 0.13)
        h_t_xi_range3->Fill(mt);

      h2_th_x_vs_xi->Fill(p.xi(), th_x);
      h2_th_y_vs_xi->Fill(p.xi(), th_y);
      h2_vtx_y_vs_xi->Fill(p.xi(), p.vertex().y());
      h2_t_vs_xi->Fill(p.xi(), mt);

      p_th_x_vs_xi->Fill(p.xi(), th_x);
      p_th_y_vs_xi->Fill(p.xi(), th_y);
      p_vtx_y_vs_xi->Fill(p.xi(), p.vertex().y());
    }

    void write() const {
      h_chi_sq->Write("h_chi_sq");
      h_chi_sq_norm->Write("h_chi_sq_norm");

      h_n_tracking_RPs->Write("h_n_tracking_RPs");
      h_n_timing_RPs->Write("h_n_timing_RPs");

      h_xi->Write("h_xi");

      h_th_x->Write("h_th_x");
      h2_th_x_vs_xi->Write("h2_th_x_vs_xi");
      p_th_x_vs_xi->Write("p_th_x_vs_xi");
      auto g_th_x_RMS_vs_xi = std::make_unique<TGraphErrors>();
      profileToRMSGraph(p_th_x_vs_xi.get(), g_th_x_RMS_vs_xi.get());
      g_th_x_RMS_vs_xi->Write("g_th_x_RMS_vs_xi");

      h_th_y->Write("h_th_y");
      h2_th_y_vs_xi->Write("h2_th_y_vs_xi");
      p_th_y_vs_xi->Write("p_th_y_vs_xi");
      auto g_th_y_RMS_vs_xi = std::make_unique<TGraphErrors>();
      profileToRMSGraph(p_th_y_vs_xi.get(), g_th_y_RMS_vs_xi.get());
      g_th_y_RMS_vs_xi->Write("g_th_y_RMS_vs_xi");

      h_vtx_y->Write("h_vtx_y");
      h2_vtx_y_vs_xi->Write("h2_vtx_y_vs_xi");
      p_vtx_y_vs_xi->Write("p_vtx_y_vs_xi");
      auto g_vtx_y_RMS_vs_xi = std::make_unique<TGraphErrors>();
      profileToRMSGraph(p_vtx_y_vs_xi.get(), g_vtx_y_RMS_vs_xi.get());
      g_vtx_y_RMS_vs_xi->Write("g_vtx_y_RMS_vs_xi");

      h_t->Scale(1., "width");

      h_t_unif->Write("h_t_unif");
      h_t->Write("h_t");
      h_t_xi_range1->Write("h_t_xi_range1");
      h_t_xi_range2->Write("h_t_xi_range2");
      h_t_xi_range3->Write("h_t_xi_range3");

      h2_t_vs_xi->Write("h2_t_vs_xi");
    }
  };

  std::map<unsigned int, MultiRPPlots> multiRPPlots_;

  struct SingleMultiCorrelationPlots {
    std::unique_ptr<TH2D> h2_xi_mu_vs_xi_si;
    std::unique_ptr<TH1D> h_xi_diff_mu_si, h_xi_diff_si_mu;

    std::unique_ptr<TH2D> h2_xi_diff_si_mu_vs_xi_mu;
    std::unique_ptr<TProfile> p_xi_diff_si_mu_vs_xi_mu;

    std::unique_ptr<TH2D> h2_th_y_mu_vs_th_y_si;

    SingleMultiCorrelationPlots()
        : h2_xi_mu_vs_xi_si(new TH2D("", ";#xi_{single};#xi_{multi}", 100, 0., 0.3, 100, 0., 0.3)),
          h_xi_diff_mu_si(new TH1D("", ";#xi_{multi} - #xi_{single}", 100, -0.1, +0.1)),
          h_xi_diff_si_mu(new TH1D("", ";#xi_{single} - #xi_{multi}", 100, -0.1, +0.1)),
          h2_xi_diff_si_mu_vs_xi_mu(
              new TH2D("", ";#xi_{multi};#xi_{single} - #xi_{multi}", 100, 0., 0.3, 100, -0.10, 0.10)),
          p_xi_diff_si_mu_vs_xi_mu(new TProfile("", ";#xi_{multi};#xi_{single} - #xi_{multi}", 100, 0., 0.3)),
          h2_th_y_mu_vs_th_y_si(
              new TH2D("", ";#theta^{*}_{y,si};#theta^{*}_{y,mu}", 100, -500E-6, +500E-6, 100, -500E-6, +500E-6)) {}

    void fill(const reco::ForwardProton &p_single, const reco::ForwardProton &p_multi) {
      if (p_single.validFit() && p_multi.validFit()) {
        h2_xi_mu_vs_xi_si->Fill(p_single.xi(), p_multi.xi());
        h_xi_diff_mu_si->Fill(p_multi.xi() - p_single.xi());
        h_xi_diff_si_mu->Fill(p_single.xi() - p_multi.xi());

        h2_xi_diff_si_mu_vs_xi_mu->Fill(p_multi.xi(), p_single.xi() - p_multi.xi());
        p_xi_diff_si_mu_vs_xi_mu->Fill(p_multi.xi(), p_single.xi() - p_multi.xi());

        const double th_y_si = p_single.thetaY();
        const double th_y_mu = p_multi.thetaY();

        h2_th_y_mu_vs_th_y_si->Fill(th_y_si, th_y_mu);
      }
    }

    void write() const {
      h2_xi_mu_vs_xi_si->Write("h2_xi_mu_vs_xi_si");
      h_xi_diff_mu_si->Write("h_xi_diff_mu_si");
      h_xi_diff_si_mu->Write("h_xi_diff_si_mu");

      h2_xi_diff_si_mu_vs_xi_mu->Write("h2_xi_diff_si_mu_vs_xi_mu");
      p_xi_diff_si_mu_vs_xi_mu->Write("p_xi_diff_si_mu_vs_xi_mu");

      h2_th_y_mu_vs_th_y_si->Write("h2_th_y_mu_vs_th_y_si");
    }
  };

  std::map<unsigned int, SingleMultiCorrelationPlots> singleMultiCorrelationPlots_;

  struct ArmCorrelationPlots {
    std::unique_ptr<TH1D> h_xi_si_diffNF;
    std::unique_ptr<TProfile> p_xi_si_diffNF_vs_xi_mu;

    std::unique_ptr<TH1D> h_th_y_si_diffNF;
    std::unique_ptr<TProfile> p_th_y_si_diffNF_vs_xi_mu;

    ArmCorrelationPlots()
        : h_xi_si_diffNF(new TH1D("", ";#xi_{sF} - #xi_{sN}", 100, -0.02, +0.02)),
          p_xi_si_diffNF_vs_xi_mu(new TProfile("", ";#xi_{m};#xi_{sF} - #xi_{sN}", 100, 0., 0.3)),
          h_th_y_si_diffNF(new TH1D("", ";#theta_{y,sF} - #theta_{y,sN}", 100, -100E-6, +100E-6)),
          p_th_y_si_diffNF_vs_xi_mu(new TProfile("", ";#xi_{m};#theta_{y,sF} - #theta_{y,sN}", 100, 0., 0.3)) {}

    void fill(const reco::ForwardProton &p_s_N, const reco::ForwardProton &p_s_F, const reco::ForwardProton &p_m) {
      if (!p_s_N.validFit() || !p_s_F.validFit() || !p_m.validFit())
        return;

      const double th_y_s_N = p_s_N.thetaY();
      const double th_y_s_F = p_s_F.thetaY();

      h_xi_si_diffNF->Fill(p_s_F.xi() - p_s_N.xi());
      p_xi_si_diffNF_vs_xi_mu->Fill(p_m.xi(), p_s_F.xi() - p_s_N.xi());

      h_th_y_si_diffNF->Fill(th_y_s_F - th_y_s_N);
      p_th_y_si_diffNF_vs_xi_mu->Fill(p_m.xi(), th_y_s_F - th_y_s_N);
    }

    void write() const {
      h_xi_si_diffNF->Write("h_xi_si_diffNF");
      p_xi_si_diffNF_vs_xi_mu->Write("p_xi_si_diffNF_vs_xi_mu");

      h_th_y_si_diffNF->Write("h_th_y_si_diffNF");
      p_th_y_si_diffNF_vs_xi_mu->Write("p_th_y_si_diffNF_vs_xi_mu");
    }
  };

  std::map<unsigned int, ArmCorrelationPlots> armCorrelationPlots_;

  std::unique_ptr<TProfile> p_x_L_diffNF_vs_x_L_N_, p_x_R_diffNF_vs_x_R_N_;
  std::unique_ptr<TProfile> p_y_L_diffNF_vs_y_L_N_, p_y_R_diffNF_vs_y_R_N_;

  signed int n_non_empty_events_;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSProtonReconstructionPlotter::CTPPSProtonReconstructionPlotter(const edm::ParameterSet &ps)
    : tokenTracks_(consumes<CTPPSLocalTrackLiteCollection>(ps.getParameter<edm::InputTag>("tagTracks"))),
      tokenRecoProtonsSingleRP_(
          consumes<reco::ForwardProtonCollection>(ps.getParameter<InputTag>("tagRecoProtonsSingleRP"))),
      tokenRecoProtonsMultiRP_(
          consumes<reco::ForwardProtonCollection>(ps.getParameter<InputTag>("tagRecoProtonsMultiRP"))),

      rpId_45_N_(ps.getParameter<unsigned int>("rpId_45_N")),
      rpId_45_F_(ps.getParameter<unsigned int>("rpId_45_F")),
      rpId_56_N_(ps.getParameter<unsigned int>("rpId_56_N")),
      rpId_56_F_(ps.getParameter<unsigned int>("rpId_56_F")),

      outputFile_(ps.getParameter<string>("outputFile")),
      maxNonEmptyEvents_(ps.getUntrackedParameter<signed int>("maxNonEmptyEvents", -1)),

      p_x_L_diffNF_vs_x_L_N_(new TProfile("p_x_L_diffNF_vs_x_L_N", ";x_{LN};x_{LF} - x_{LN}", 100, 0., +20.)),
      p_x_R_diffNF_vs_x_R_N_(new TProfile("p_x_R_diffNF_vs_x_R_N", ";x_{RN};x_{RF} - x_{RN}", 100, 0., +20.)),

      p_y_L_diffNF_vs_y_L_N_(new TProfile("p_y_L_diffNF_vs_y_L_N", ";y_{LN};y_{LF} - y_{LN}", 100, -20., +20.)),
      p_y_R_diffNF_vs_y_R_N_(new TProfile("p_y_R_diffNF_vs_y_R_N", ";y_{RN};y_{RF} - y_{RN}", 100, -20., +20.)),

      n_non_empty_events_(0) {}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionPlotter::analyze(const edm::Event &event, const edm::EventSetup &) {
  // get input
  edm::Handle<CTPPSLocalTrackLiteCollection> hTracks;
  event.getByToken(tokenTracks_, hTracks);

  Handle<reco::ForwardProtonCollection> hRecoProtonsSingleRP;
  event.getByToken(tokenRecoProtonsSingleRP_, hRecoProtonsSingleRP);

  Handle<reco::ForwardProtonCollection> hRecoProtonsMultiRP;
  event.getByToken(tokenRecoProtonsMultiRP_, hRecoProtonsMultiRP);

  if (!hRecoProtonsSingleRP->empty())
    n_non_empty_events_++;

  if (maxNonEmptyEvents_ > 0 && n_non_empty_events_ > maxNonEmptyEvents_)
    throw cms::Exception("CTPPSProtonReconstructionPlotter") << "Number of non empty events reached maximum.";

  // track plots
  const CTPPSLocalTrackLite *tr_L_N = nullptr;
  const CTPPSLocalTrackLite *tr_L_F = nullptr;
  const CTPPSLocalTrackLite *tr_R_N = nullptr;
  const CTPPSLocalTrackLite *tr_R_F = nullptr;

  for (const auto &tr : *hTracks) {
    CTPPSDetId rpId(tr.getRPId());
    unsigned int decRPId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

    if (decRPId == rpId_45_N_)
      tr_L_N = &tr;
    if (decRPId == rpId_45_F_)
      tr_L_F = &tr;
    if (decRPId == rpId_56_N_)
      tr_R_N = &tr;
    if (decRPId == rpId_56_F_)
      tr_R_F = &tr;
  }

  if (tr_L_N && tr_L_F) {
    p_x_L_diffNF_vs_x_L_N_->Fill(tr_L_N->getX(), tr_L_F->getX() - tr_L_N->getX());
    p_y_L_diffNF_vs_y_L_N_->Fill(tr_L_N->getY(), tr_L_F->getY() - tr_L_N->getY());
  }

  if (tr_R_N && tr_R_F) {
    p_x_R_diffNF_vs_x_R_N_->Fill(tr_R_N->getX(), tr_R_F->getX() - tr_R_N->getX());
    p_y_R_diffNF_vs_y_R_N_->Fill(tr_R_N->getY(), tr_R_F->getY() - tr_R_N->getY());
  }

  // make single-RP-reco plots
  for (const auto &proton : *hRecoProtonsSingleRP) {
    CTPPSDetId rpId((*proton.contributingLocalTracks().begin())->getRPId());
    unsigned int decRPId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();
    singleRPPlots_[decRPId].fill(proton);
  }

  // make multi-RP-reco plots
  for (const auto &proton : *hRecoProtonsMultiRP) {
    CTPPSDetId rpId((*proton.contributingLocalTracks().begin())->getRPId());
    unsigned int armId = rpId.arm();
    multiRPPlots_[armId].fill(proton);
  }

  // make correlation plots
  for (const auto &proton_s : *hRecoProtonsSingleRP) {
    for (const auto &proton_m : *hRecoProtonsMultiRP) {
      // only compare object from the same arm
      CTPPSDetId rpId_s((*proton_s.contributingLocalTracks().begin())->getRPId());
      CTPPSDetId rpId_m((*proton_m.contributingLocalTracks().begin())->getRPId());

      if (rpId_s.arm() != rpId_m.arm())
        continue;

      // build index
      const unsigned int idx = rpId_s.arm() * 1000 + rpId_s.station() * 100 + rpId_s.rp() * 10 + rpId_m.arm();

      // fill plots
      singleMultiCorrelationPlots_[idx].fill(proton_s, proton_m);
    }
  }

  // arm correlation plots
  const reco::ForwardProton *p_arm0_s_N = nullptr, *p_arm0_s_F = nullptr, *p_arm0_m = nullptr;
  const reco::ForwardProton *p_arm1_s_N = nullptr, *p_arm1_s_F = nullptr, *p_arm1_m = nullptr;

  for (const auto &proton : *hRecoProtonsSingleRP) {
    CTPPSDetId rpId((*proton.contributingLocalTracks().begin())->getRPId());
    const unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

    if (rpDecId == rpId_45_N_)
      p_arm0_s_N = &proton;
    if (rpDecId == rpId_45_F_)
      p_arm0_s_F = &proton;

    if (rpDecId == rpId_56_N_)
      p_arm1_s_N = &proton;
    if (rpDecId == rpId_56_F_)
      p_arm1_s_F = &proton;
  }

  for (const auto &proton : *hRecoProtonsMultiRP) {
    CTPPSDetId rpId((*proton.contributingLocalTracks().begin())->getRPId());
    const unsigned int arm = rpId.arm();

    if (arm == 0)
      p_arm0_m = &proton;
    if (arm == 1)
      p_arm1_m = &proton;
  }

  if (p_arm0_s_N && p_arm0_s_F && p_arm0_m)
    armCorrelationPlots_[0].fill(*p_arm0_s_N, *p_arm0_s_F, *p_arm0_m);

  if (p_arm1_s_N && p_arm1_s_F && p_arm1_m)
    armCorrelationPlots_[1].fill(*p_arm1_s_N, *p_arm1_s_F, *p_arm1_m);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionPlotter::endJob() {
  LogInfo("CTPPSProtonReconstructionPlotter") << "n_non_empty_events = " << n_non_empty_events_;

  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  p_x_L_diffNF_vs_x_L_N_->Write();
  p_x_R_diffNF_vs_x_R_N_->Write();

  p_y_L_diffNF_vs_y_L_N_->Write();
  p_y_R_diffNF_vs_y_R_N_->Write();

  TDirectory *d_singleRPPlots = f_out->mkdir("singleRPPlots");
  for (const auto &it : singleRPPlots_) {
    gDirectory = d_singleRPPlots->mkdir(Form("rp%u", it.first));
    it.second.write();
  }

  TDirectory *d_multiRPPlots = f_out->mkdir("multiRPPlots");
  for (const auto &it : multiRPPlots_) {
    gDirectory = d_multiRPPlots->mkdir(Form("arm%u", it.first));
    it.second.write();
  }

  TDirectory *d_singleMultiCorrelationPlots = f_out->mkdir("singleMultiCorrelationPlots");
  for (const auto &it : singleMultiCorrelationPlots_) {
    unsigned int si_rp = it.first / 10;
    unsigned int mu_arm = it.first % 10;

    gDirectory = d_singleMultiCorrelationPlots->mkdir(Form("si_rp%u_mu_arm%u", si_rp, mu_arm));
    it.second.write();
  }

  TDirectory *d_armCorrelationPlots = f_out->mkdir("armCorrelationPlots");
  for (const auto &it : armCorrelationPlots_) {
    gDirectory = d_armCorrelationPlots->mkdir(Form("arm%u", it.first));
    it.second.write();
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstructionPlotter);
