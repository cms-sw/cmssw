/****************************************************************************
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com)
****************************************************************************/

#include <memory>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

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
  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geometryESToken_;

  unsigned int rpId_45_N_, rpId_45_F_;
  unsigned int rpId_56_N_, rpId_56_F_;

  struct AssociationCuts {
    double ti_tr_min;
    double ti_tr_max;

    void load(const edm::ParameterSet &ps) {
      ti_tr_min = ps.getParameter<double>("ti_tr_min");
      ti_tr_max = ps.getParameter<double>("ti_tr_max");
    }
  };

  std::map<unsigned int, AssociationCuts> association_cuts_;

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

  void CalculateTimingTrackingDistance(const reco::ForwardProton &proton,
                                       const CTPPSLocalTrackLite &tr,
                                       const CTPPSGeometry &geometry,
                                       double &x_tr,
                                       double &x_ti,
                                       double &de_x,
                                       double &de_x_unc);

  struct SingleRPPlots {
    std::unique_ptr<TH1D> h_multiplicity;
    std::unique_ptr<TH1D> h_xi;
    std::unique_ptr<TH2D> h2_th_y_vs_xi;
    std::unique_ptr<TProfile> p_th_y_vs_xi;

    std::map<unsigned int, TH1D *> m_h_xi_nTracks;
    std::unique_ptr<TH1D> h_xi_n1f1;

    SingleRPPlots()
        : h_multiplicity(new TH1D("", ";reconstructed protons", 11, -0.5, 10.5)),
          h_xi(new TH1D("", ";#xi", 100, 0., 0.3)),
          h2_th_y_vs_xi(new TH2D("", ";#xi;#theta_{y}   (rad)", 100, 0., 0.3, 100, -500E-6, +500E-6)),
          p_th_y_vs_xi(new TProfile("", ";#xi;#theta_{y}   (rad)", 100, 0., 0.3)),
          h_xi_n1f1(new TH1D("", ";#xi", 100, 0., 0.3)) {
      for (unsigned int n = 2; n <= 10; ++n)
        m_h_xi_nTracks[n] = new TH1D(*h_xi);
    }

    void fill(const reco::ForwardProton &p, unsigned int nTracks, bool n1f1) {
      if (p.validFit()) {
        h_xi->Fill(p.xi());

        const double th_y = p.thetaY();
        h2_th_y_vs_xi->Fill(p.xi(), th_y);
        p_th_y_vs_xi->Fill(p.xi(), th_y);

        auto it = m_h_xi_nTracks.find(nTracks);
        if (it != m_h_xi_nTracks.end())
          it->second->Fill(p.xi());

        if (n1f1)
          h_xi_n1f1->Fill(p.xi());
      }
    }

    void write() const {
      h_multiplicity->Write("h_multiplicity");
      h_xi->Write("h_xi");

      h2_th_y_vs_xi->Write("h2_th_y_vs_xi");
      p_th_y_vs_xi->Write("p_th_y_vs_xi");

      TDirectory *d_top = gDirectory;

      gDirectory = d_top->mkdir("h_xi_nTracks");
      for (const auto p : m_h_xi_nTracks) {
        char buf[100];
        sprintf(buf, "h_xi_nTracks_%u", p.first);
        p.second->Write(buf);
      }

      gDirectory = d_top;

      h_xi_n1f1->Write("h_xi_n1f1");
    }
  };

  std::map<unsigned int, SingleRPPlots> singleRPPlots_;

  struct MultiRPPlots {
    std::unique_ptr<TH1D> h_multiplicity;
    std::unique_ptr<TH1D> h_xi, h_th_x, h_th_y, h_vtx_y, h_t_unif, h_t, h_chi_sq, h_log_chi_sq, h_chi_sq_norm;
    std::unique_ptr<TH1D> h_t_xi_range1, h_t_xi_range2, h_t_xi_range3;
    std::unique_ptr<TH1D> h_time, h_time_unc;
    std::unique_ptr<TProfile> p_time_unc_vs_x_ClCo, p_time_unc_vs_xi;
    std::unique_ptr<TH1D> h_n_contrib_tracking_tracks, h_n_contrib_timing_tracks;
    std::unique_ptr<TH2D> h2_th_x_vs_xi, h2_th_y_vs_xi, h2_vtx_y_vs_xi, h2_t_vs_xi;
    std::unique_ptr<TProfile> p_th_x_vs_xi, p_th_y_vs_xi, p_vtx_y_vs_xi;

    std::unique_ptr<TH2D> h2_timing_tracks_vs_prot_mult;

    std::map<unsigned int, TH1D *> m_h_xi_nTracks;
    std::unique_ptr<TH1D> h_xi_n1f1;

    std::unique_ptr<TH2D> h2_x_timing_vs_x_tracking_ClCo;

    std::unique_ptr<TH1D> h_de_x_timing_vs_tracking, h_de_x_rel_timing_vs_tracking, h_de_x_match_timing_vs_tracking;
    std::unique_ptr<TH1D> h_de_x_timing_vs_tracking_ClCo, h_de_x_rel_timing_vs_tracking_ClCo,
        h_de_x_match_timing_vs_tracking_ClCo;

    std::unique_ptr<TH2D> h2_y_vs_x_tt0_ClCo, h2_y_vs_x_tt1_ClCo, h2_y_vs_x_ttm_ClCo;

    MultiRPPlots()
        : h_multiplicity(new TH1D("", ";reconstructed protons per event", 11, -0.5, 10.5)),
          h_xi(new TH1D("", ";#xi", 100, 0., 0.3)),
          h_th_x(new TH1D("", ";#theta_{x}   (rad)", 250, -500E-6, +500E-6)),
          h_th_y(new TH1D("", ";#theta_{y}   (rad)", 500, -1000E-6, +1000E-6)),
          h_vtx_y(new TH1D("", ";vtx_{y}   (cm)", 100, -100E-3, +100E-3)),
          h_chi_sq(new TH1D("", ";#chi^{2}", 100, 0., 10.)),
          h_log_chi_sq(new TH1D("", ";log_{10} #chi^{2}", 100, -20., 5.)),
          h_chi_sq_norm(new TH1D("", ";#chi^{2}/ndf", 100, 0., 5.)),
          h_time(new TH1D("", ";time   (ns)", 100, -2., +2.)),
          h_time_unc(new TH1D("", ";time unc   (ns)", 100, -1., +1.)),
          p_time_unc_vs_x_ClCo(new TProfile("", ";x_tracking   (mm);time unc   (ns)", 100, 0., 30.)),
          p_time_unc_vs_xi(new TProfile("", ";xi;time unc   (ns)", 100, 0., 0.3)),
          h_n_contrib_tracking_tracks(new TH1D("", ";n of contrib. tracking tracks per reco proton", 4, -0.5, +3.5)),
          h_n_contrib_timing_tracks(new TH1D("", ";n of contrib. timing tracks per reco proton", 4, -0.5, +3.5)),
          h2_th_x_vs_xi(new TH2D("", ";#xi;#theta_{x}   (rad)", 100, 0., 0.3, 100, -500E-6, +500E-6)),
          h2_th_y_vs_xi(new TH2D("", ";#xi;#theta_{y}   (rad)", 100, 0., 0.3, 100, -500E-6, +500E-6)),
          h2_vtx_y_vs_xi(new TH2D("", ";#xi;vtx_{y}   (cm)", 100, 0., 0.3, 100, -100E-3, +100E-3)),
          p_th_x_vs_xi(new TProfile("", ";#xi;#theta_{x}   (rad)", 100, 0., 0.3)),
          p_th_y_vs_xi(new TProfile("", ";#xi;#theta_{y}   (rad)", 100, 0., 0.3)),
          p_vtx_y_vs_xi(new TProfile("", ";#xi;vtx_{y}   (cm)", 100, 0., 0.3)),
          h2_timing_tracks_vs_prot_mult(
              new TH2D("", ";reco protons per event;timing tracks per event", 11, -0.5, 10.5, 11, -0.5, 10.5)),
          h_xi_n1f1(new TH1D("", ";#xi", 100, 0., 0.3)),

          h2_x_timing_vs_x_tracking_ClCo(
              new TH2D("", ";x_tracking   (mm);x_timing   (mm)", 100, 0., 20., 100, 0., 20.)),
          h_de_x_timing_vs_tracking(new TH1D("", ";#Delta x   (mm)", 200, -1., +1.)),
          h_de_x_rel_timing_vs_tracking(new TH1D("", ";#Delta x / #sigma(x)", 200, -20., +20.)),
          h_de_x_match_timing_vs_tracking(new TH1D("", ";match between tracking and timing tracks", 2, -0.5, +1.5)),
          h_de_x_timing_vs_tracking_ClCo(new TH1D("", ";#Delta x   (mm)", 200, -1., +1.)),
          h_de_x_rel_timing_vs_tracking_ClCo(new TH1D("", ";#Delta x / #sigma(x)", 200, -20., +20.)),
          h_de_x_match_timing_vs_tracking_ClCo(
              new TH1D("", ";match between tracking and timing tracks", 2, -0.5, +1.5)),

          h2_y_vs_x_tt0_ClCo(new TH2D("", ";x   (mm);y   (mm)", 100, -5., 25., 100, -15., +15.)),
          h2_y_vs_x_tt1_ClCo(new TH2D("", ";x   (mm);y   (mm)", 100, -5., 25., 100, -15., +15.)),
          h2_y_vs_x_ttm_ClCo(new TH2D("", ";x   (mm);y   (mm)", 100, -5., 25., 100, -15., +15.)) {
      std::vector<double> v_t_bin_edges;
      for (double t = 0; t <= 5.;) {
        v_t_bin_edges.push_back(t);
        const double de_t = 0.05 + 0.09 * t + 0.02 * t * t;
        t += de_t;
      }
      h_t_unif = std::make_unique<TH1D>("", ";|t|   (GeV^2)", 100, 0., 5.);
      h_t = std::make_unique<TH1D>("", ";|t|   (GeV^2)", v_t_bin_edges.size() - 1, v_t_bin_edges.data());
      h_t_xi_range1 = std::make_unique<TH1D>("", ";|t|   (GeV^2)", v_t_bin_edges.size() - 1, v_t_bin_edges.data());
      h_t_xi_range2 = std::make_unique<TH1D>("", ";|t|   (GeV^2)", v_t_bin_edges.size() - 1, v_t_bin_edges.data());
      h_t_xi_range3 = std::make_unique<TH1D>("", ";|t|   (GeV^2)", v_t_bin_edges.size() - 1, v_t_bin_edges.data());
      h2_t_vs_xi = std::make_unique<TH2D>(
          "", ";#xi;|t|   (GeV^2)", 100, 0., 0.3, v_t_bin_edges.size() - 1, v_t_bin_edges.data());

      for (unsigned int n = 2; n <= 10; ++n)
        m_h_xi_nTracks[n] = new TH1D(*h_xi);
    }

    void fill(const reco::ForwardProton &p, unsigned int nTracks, bool n1f1) {
      if (!p.validFit())
        return;

      unsigned int n_contrib_tracking_tracks = 0, n_contrib_timing_tracks = 0;
      for (const auto &tr : p.contributingLocalTracks()) {
        CTPPSDetId detId(tr->rpId());
        if (detId.subdetId() == CTPPSDetId::sdTrackingStrip || detId.subdetId() == CTPPSDetId::sdTrackingPixel)
          n_contrib_tracking_tracks++;
        if (detId.subdetId() == CTPPSDetId::sdTimingDiamond || detId.subdetId() == CTPPSDetId::sdTimingFastSilicon)
          n_contrib_timing_tracks++;
      }

      const double th_x = p.thetaX();
      const double th_y = p.thetaY();
      const double mt = -p.t();

      h_chi_sq->Fill(p.chi2());
      h_log_chi_sq->Fill(log10(p.chi2()));
      if (p.ndof() > 0)
        h_chi_sq_norm->Fill(p.normalizedChi2());

      h_n_contrib_tracking_tracks->Fill(n_contrib_tracking_tracks);
      h_n_contrib_timing_tracks->Fill(n_contrib_timing_tracks);

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

      if (p.timeError() > 0.) {
        h_time->Fill(p.time());
        h_time_unc->Fill(p.timeError());
        //p_time_unc_vs_x_ClCo filled in ClCo code below
        p_time_unc_vs_xi->Fill(p.xi(), p.timeError());
      }

      h2_th_x_vs_xi->Fill(p.xi(), th_x);
      h2_th_y_vs_xi->Fill(p.xi(), th_y);
      h2_vtx_y_vs_xi->Fill(p.xi(), p.vertex().y());
      h2_t_vs_xi->Fill(p.xi(), mt);

      p_th_x_vs_xi->Fill(p.xi(), th_x);
      p_th_y_vs_xi->Fill(p.xi(), th_y);
      p_vtx_y_vs_xi->Fill(p.xi(), p.vertex().y());

      auto it = m_h_xi_nTracks.find(nTracks);
      if (it != m_h_xi_nTracks.end())
        it->second->Fill(p.xi());

      if (n1f1)
        h_xi_n1f1->Fill(p.xi());
    }

    void write() const {
      h_multiplicity->Write("h_multiplicity");

      h_chi_sq->Write("h_chi_sq");
      h_log_chi_sq->Write("h_log_chi_sq");
      h_chi_sq_norm->Write("h_chi_sq_norm");

      h_n_contrib_tracking_tracks->Write("h_n_contrib_tracking_tracks");
      h_n_contrib_timing_tracks->Write("h_n_contrib_timing_tracks");

      h2_timing_tracks_vs_prot_mult->Write("h2_timing_tracks_vs_prot_mult");

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

      h_time->Write("h_time");
      h_time_unc->Write("h_time_unc");
      p_time_unc_vs_x_ClCo->Write("p_time_unc_vs_x_ClCo");
      p_time_unc_vs_xi->Write("p_time_unc_vs_xi");

      TDirectory *d_top = gDirectory;

      gDirectory = d_top->mkdir("h_xi_nTracks");
      for (const auto p : m_h_xi_nTracks) {
        char buf[100];
        sprintf(buf, "h_xi_nTracks_%u", p.first);
        p.second->Write(buf);
      }

      gDirectory = d_top;

      h_xi_n1f1->Write("h_xi_n1f1");

      h2_x_timing_vs_x_tracking_ClCo->Write("h2_x_timing_vs_x_tracking_ClCo");

      h_de_x_timing_vs_tracking->Write("h_de_x_timing_vs_tracking");
      h_de_x_rel_timing_vs_tracking->Write("h_de_x_rel_timing_vs_tracking");
      h_de_x_match_timing_vs_tracking->Write("h_de_x_match_timing_vs_tracking");

      h_de_x_timing_vs_tracking_ClCo->Write("h_de_x_timing_vs_tracking_ClCo");
      h_de_x_rel_timing_vs_tracking_ClCo->Write("h_de_x_rel_timing_vs_tracking_ClCo");
      h_de_x_match_timing_vs_tracking_ClCo->Write("h_de_x_match_timing_vs_tracking_ClCo");

      h2_y_vs_x_tt0_ClCo->Write("h2_y_vs_x_tt0_ClCo");
      h2_y_vs_x_tt1_ClCo->Write("h2_y_vs_x_tt1_ClCo");
      h2_y_vs_x_ttm_ClCo->Write("h2_y_vs_x_ttm_ClCo");
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
    std::unique_ptr<TH2D> h2_xi_si_diffNF_vs_xi_mu;
    std::unique_ptr<TProfile> p_xi_si_diffNF_vs_xi_mu;

    std::unique_ptr<TH1D> h_th_y_si_diffNF;
    std::unique_ptr<TProfile> p_th_y_si_diffNF_vs_xi_mu;

    ArmCorrelationPlots()
        : h_xi_si_diffNF(new TH1D("", ";#xi_{sF} - #xi_{sN}", 100, -0.02, +0.02)),
          h2_xi_si_diffNF_vs_xi_mu(new TH2D("", ";#xi_{m};#xi_{sF} - #xi_{sN}", 100, 0., 0.3, 100, -0.02, +0.02)),
          p_xi_si_diffNF_vs_xi_mu(new TProfile("", ";#xi_{m};#xi_{sF} - #xi_{sN}", 100, 0., 0.3)),
          h_th_y_si_diffNF(new TH1D("", ";#theta_{y,sF} - #theta_{y,sN}", 100, -100E-6, +100E-6)),
          p_th_y_si_diffNF_vs_xi_mu(new TProfile("", ";#xi_{m};#theta_{y,sF} - #theta_{y,sN}", 100, 0., 0.3)) {}

    void fill(const reco::ForwardProton &p_s_N, const reco::ForwardProton &p_s_F, const reco::ForwardProton &p_m) {
      if (!p_s_N.validFit() || !p_s_F.validFit() || !p_m.validFit())
        return;

      const double th_y_s_N = p_s_N.thetaY();
      const double th_y_s_F = p_s_F.thetaY();

      h_xi_si_diffNF->Fill(p_s_F.xi() - p_s_N.xi());
      h2_xi_si_diffNF_vs_xi_mu->Fill(p_m.xi(), p_s_F.xi() - p_s_N.xi());
      p_xi_si_diffNF_vs_xi_mu->Fill(p_m.xi(), p_s_F.xi() - p_s_N.xi());

      h_th_y_si_diffNF->Fill(th_y_s_F - th_y_s_N);
      p_th_y_si_diffNF_vs_xi_mu->Fill(p_m.xi(), th_y_s_F - th_y_s_N);
    }

    void write() const {
      h_xi_si_diffNF->Write("h_xi_si_diffNF");
      h2_xi_si_diffNF_vs_xi_mu->Write("h2_xi_si_diffNF_vs_xi_mu");
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
      geometryESToken_(esConsumes()),

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

      n_non_empty_events_(0) {
  for (const std::string &sector : {"45", "56"}) {
    const unsigned int arm = (sector == "45") ? 0 : 1;
    association_cuts_[arm].load(ps.getParameterSet("association_cuts_" + sector));
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionPlotter::CalculateTimingTrackingDistance(const reco::ForwardProton &proton,
                                                                       const CTPPSLocalTrackLite &tr_ti,
                                                                       const CTPPSGeometry &geometry,
                                                                       double &x_tr,
                                                                       double &x_ti,
                                                                       double &de_x,
                                                                       double &de_x_unc) {
  // identify tracking-RP tracks
  const auto &tr_i = *proton.contributingLocalTracks().at(0);
  const auto &tr_j = *proton.contributingLocalTracks().at(1);

  const double z_i = geometry.rpTranslation(tr_i.rpId()).z();
  const double z_j = geometry.rpTranslation(tr_j.rpId()).z();

  // interpolation from tracking RPs
  const double z_ti = -geometry.rpTranslation(tr_ti.rpId()).z();  // the minus sign fixes a bug in the diamond geometry
  const double f_i = (z_ti - z_j) / (z_i - z_j), f_j = (z_i - z_ti) / (z_i - z_j);
  const double x_inter = f_i * tr_i.x() + f_j * tr_j.x();
  const double x_inter_unc_sq = f_i * f_i * tr_i.xUnc() * tr_i.xUnc() + f_j * f_j * tr_j.xUnc() * tr_j.xUnc();

  // save distance
  x_tr = x_inter;
  x_ti = tr_ti.x();

  de_x = tr_ti.x() - x_inter;
  de_x_unc = sqrt(tr_ti.xUnc() * tr_ti.xUnc() + x_inter_unc_sq);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionPlotter::analyze(const edm::Event &event, const edm::EventSetup &iSetup) {
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

  // get conditions
  const auto &geometry = iSetup.getData(geometryESToken_);

  // track plots
  const CTPPSLocalTrackLite *tr_L_N = nullptr;
  const CTPPSLocalTrackLite *tr_L_F = nullptr;
  const CTPPSLocalTrackLite *tr_R_N = nullptr;
  const CTPPSLocalTrackLite *tr_R_F = nullptr;
  std::map<unsigned int, unsigned int> armTrackCounter, armTimingTrackCounter;
  std::map<unsigned int, unsigned int> armTrackCounter_N, armTrackCounter_F;

  for (const auto &tr : *hTracks) {
    CTPPSDetId rpId(tr.rpId());
    unsigned int decRPId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

    if (decRPId == rpId_45_N_) {
      tr_L_N = &tr;
      armTrackCounter_N[0]++;
    }
    if (decRPId == rpId_45_F_) {
      tr_L_F = &tr;
      armTrackCounter_F[0]++;
    }
    if (decRPId == rpId_56_N_) {
      tr_R_N = &tr;
      armTrackCounter_N[1]++;
    }
    if (decRPId == rpId_56_F_) {
      tr_R_F = &tr;
      armTrackCounter_F[1]++;
    }

    armTrackCounter[rpId.arm()]++;

    const bool trackerRP =
        (rpId.subdetId() == CTPPSDetId::sdTrackingStrip || rpId.subdetId() == CTPPSDetId::sdTrackingPixel);
    if (!trackerRP)
      armTimingTrackCounter[rpId.arm()]++;
  }

  if (tr_L_N && tr_L_F) {
    p_x_L_diffNF_vs_x_L_N_->Fill(tr_L_N->x(), tr_L_F->x() - tr_L_N->x());
    p_y_L_diffNF_vs_y_L_N_->Fill(tr_L_N->y(), tr_L_F->y() - tr_L_N->y());
  }

  if (tr_R_N && tr_R_F) {
    p_x_R_diffNF_vs_x_R_N_->Fill(tr_R_N->x(), tr_R_F->x() - tr_R_N->x());
    p_y_R_diffNF_vs_y_R_N_->Fill(tr_R_N->y(), tr_R_F->y() - tr_R_N->y());
  }

  // initialise multiplicity counters
  std::map<unsigned int, unsigned int> singleRPMultiplicity, multiRPMultiplicity;
  singleRPMultiplicity[rpId_45_N_] = singleRPMultiplicity[rpId_45_F_] = singleRPMultiplicity[rpId_56_N_] =
      singleRPMultiplicity[rpId_56_F_] = 0;
  multiRPMultiplicity[0] = multiRPMultiplicity[1] = 0;

  // make single-RP-reco plots
  for (const auto &proton : *hRecoProtonsSingleRP) {
    CTPPSDetId rpId((*proton.contributingLocalTracks().begin())->rpId());
    unsigned int decRPId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

    const bool n1f1 = (armTrackCounter_N[rpId.arm()] == 1 && armTrackCounter_F[rpId.arm()] == 1);

    singleRPPlots_[decRPId].fill(proton, armTrackCounter[rpId.arm()], n1f1);

    if (proton.validFit())
      singleRPMultiplicity[decRPId]++;
  }

  for (const auto it : singleRPMultiplicity)
    singleRPPlots_[it.first].h_multiplicity->Fill(it.second);

  // make multi-RP-reco plots
  for (const auto &proton : *hRecoProtonsMultiRP) {
    CTPPSDetId rpId((*proton.contributingLocalTracks().begin())->rpId());
    unsigned int armId = rpId.arm();

    const bool n1f1 = (armTrackCounter_N[armId] == 1 && armTrackCounter_F[armId] == 1);

    multiRPPlots_[armId].fill(proton, armTrackCounter[armId], n1f1);

    if (proton.validFit())
      multiRPMultiplicity[armId]++;
  }

  for (const auto it : multiRPMultiplicity) {
    const auto &pl = multiRPPlots_[it.first];
    pl.h_multiplicity->Fill(it.second);
    pl.h2_timing_tracks_vs_prot_mult->Fill(it.second, armTimingTrackCounter[it.first]);
  }

  // define "clean condition" for each arm
  map<unsigned int, bool> clCo;
  clCo[0] = (singleRPMultiplicity[rpId_45_N_] && singleRPMultiplicity[rpId_45_F_] && multiRPMultiplicity[0] == 1);
  clCo[1] = (singleRPMultiplicity[rpId_56_N_] && singleRPMultiplicity[rpId_56_F_] && multiRPMultiplicity[1] == 1);

  // plot distances between multi-RP protons and timing tracks in the same arm
  for (const auto &proton : *hRecoProtonsMultiRP) {
    if (!proton.validFit())
      continue;

    CTPPSDetId rpId_proton((*proton.contributingLocalTracks().begin())->rpId());
    unsigned int armId = rpId_proton.arm();
    const auto &pl = multiRPPlots_[armId];

    for (const auto &tr : *hTracks) {
      CTPPSDetId rpId_tr(tr.rpId());
      if (rpId_tr.arm() != armId)
        continue;

      const bool trackerRP =
          (rpId_tr.subdetId() == CTPPSDetId::sdTrackingStrip || rpId_tr.subdetId() == CTPPSDetId::sdTrackingPixel);
      if (trackerRP)
        continue;

      double x_tr = -1., x_ti = -1.;
      double de_x = 0., de_x_unc = 0.;
      CalculateTimingTrackingDistance(proton, tr, geometry, x_tr, x_ti, de_x, de_x_unc);

      const double rd = (de_x_unc > 0.) ? de_x / de_x_unc : -1E10;
      const auto &ac = association_cuts_[armId];
      const bool match = (ac.ti_tr_min <= fabs(rd) && fabs(rd) <= ac.ti_tr_max);

      pl.h_de_x_timing_vs_tracking->Fill(de_x);
      pl.h_de_x_rel_timing_vs_tracking->Fill(rd);
      pl.h_de_x_match_timing_vs_tracking->Fill(match ? 1. : 0.);

      if (clCo[armId] && armTimingTrackCounter[armId] == 1) {
        pl.h2_x_timing_vs_x_tracking_ClCo->Fill(x_tr, x_ti);

        pl.h_de_x_timing_vs_tracking_ClCo->Fill(de_x);
        pl.h_de_x_rel_timing_vs_tracking_ClCo->Fill(rd);
        pl.h_de_x_match_timing_vs_tracking_ClCo->Fill(match ? 1. : 0.);

        pl.p_time_unc_vs_x_ClCo->Fill(x_tr, proton.timeError());
      }
    }
  }

  // plot xy maps
  for (const auto &proton : *hRecoProtonsMultiRP) {
    if (!proton.validFit())
      continue;

    CTPPSDetId rpId((*proton.contributingLocalTracks().begin())->rpId());
    unsigned int armId = rpId.arm();
    const auto &pl = multiRPPlots_[armId];
    const auto &nTimingTracks = armTimingTrackCounter[armId];

    if (!clCo[armId])
      continue;

    double x_ref = 0., y_ref = 0.;
    if (armId == 0) {
      x_ref = tr_L_N->x();
      y_ref = tr_L_N->y();
    }
    if (armId == 1) {
      x_ref = tr_R_N->x();
      y_ref = tr_R_N->y();
    }

    if (nTimingTracks == 0)
      pl.h2_y_vs_x_tt0_ClCo->Fill(x_ref, y_ref);
    if (nTimingTracks == 1)
      pl.h2_y_vs_x_tt1_ClCo->Fill(x_ref, y_ref);
    if (nTimingTracks > 1)
      pl.h2_y_vs_x_ttm_ClCo->Fill(x_ref, y_ref);
  }

  // make correlation plots
  for (const auto &proton_m : *hRecoProtonsMultiRP) {
    CTPPSDetId rpId_m((*proton_m.contributingLocalTracks().begin())->rpId());
    unsigned int arm = rpId_m.arm();

    const reco::ForwardProton *p_s_N = nullptr, *p_s_F = nullptr;

    for (const auto &proton_s : *hRecoProtonsSingleRP) {
      // skip if source of single-RP reco not included in multi-RP reco
      const auto key_s = proton_s.contributingLocalTracks()[0].key();
      bool compatible = false;
      for (const auto &tr_m : proton_m.contributingLocalTracks()) {
        if (tr_m.key() == key_s) {
          compatible = true;
          break;
        }
      }

      if (!compatible)
        continue;

      // fill single-multi plots
      CTPPSDetId rpId_s((*proton_s.contributingLocalTracks().begin())->rpId());
      const unsigned int idx = rpId_s.arm() * 1000 + rpId_s.station() * 100 + rpId_s.rp() * 10 + rpId_s.arm();
      singleMultiCorrelationPlots_[idx].fill(proton_s, proton_m);

      // select protons for arm-correlation plots
      const unsigned int rpDecId_s = rpId_s.arm() * 100 + rpId_s.station() * 10 + rpId_s.rp();
      if (rpDecId_s == rpId_45_N_ || rpDecId_s == rpId_56_N_)
        p_s_N = &proton_s;
      if (rpDecId_s == rpId_45_F_ || rpDecId_s == rpId_56_F_)
        p_s_F = &proton_s;
    }

    // fill arm-correlation plots
    if (p_s_N && p_s_F)
      armCorrelationPlots_[arm].fill(*p_s_N, *p_s_F, proton_m);
  }
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
