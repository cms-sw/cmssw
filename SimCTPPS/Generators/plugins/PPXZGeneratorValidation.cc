/****************************************************************************
 *
 * Authors:
 *   Jan Kašpar
 *
 ****************************************************************************/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "DataFormats/ProtonReco/interface/ForwardProton.h"

#include "CLHEP/Vector/LorentzVector.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TGraphErrors.h"

#include "SimCTPPS/Generators/plugins/particle_ids.h"

//----------------------------------------------------------------------------------------------------

class PPXZGeneratorValidation : public edm::one::EDAnalyzer<>
{
  public:
    explicit PPXZGeneratorValidation(const edm::ParameterSet&);
    ~PPXZGeneratorValidation();

  private:
    virtual void beginJob() override;

    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

    virtual void endJob() override;

    edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;
    edm::EDGetTokenT<std::vector<CTPPSLocalTrackLite>> recoTracksToken_;
    edm::EDGetTokenT<std::vector<reco::ForwardProton>> recoProtonsTokenSingleRP_, recoProtonsTokenMultiRP_;

    unsigned int referenceRPDecId_45, referenceRPDecId_56;

    std::string outputFile;

    struct Plots
    {
      TH1D *h_m_Z = NULL;
      TH1D *h_m_XZ;
      TH1D *h_p_z_LAB_2p;

      TH2D *h_xi2_vs_xi1;

      TH1D *h_p_T_X, *h_p_z_X, *h_p_tot_X, *h_theta_X, *h_eta_X;

      TH1D *h_p_T_X_pr1, *h_p_z_X_pr1, *h_p_tot_X_pr1, *h_theta_X_pr1, *h_eta_X_pr1;
      TH1D *h_p_T_X_pr2, *h_p_z_X_pr2, *h_p_tot_X_pr2, *h_theta_X_pr2, *h_eta_X_pr2;

      TH1D *h_p_T_Z, *h_p_z_Z, *h_p_tot_Z, *h_theta_Z, *h_eta_Z;

      TH1D *h_p_T_l_pl, *h_p_z_l_pl, *h_p_tot_l_pl, *h_theta_l_pl, *h_eta_l_pl;
      TH1D *h_p_T_l_mi, *h_p_z_l_mi, *h_p_tot_l_mi, *h_theta_l_mi, *h_eta_l_mi;

      TH1D *h_p_T_l_le, *h_p_z_l_le, *h_p_tot_l_le, *h_theta_l_le, *h_eta_l_le;
      TH1D *h_p_T_l_sl, *h_p_z_l_sl, *h_p_tot_l_sl, *h_theta_l_sl, *h_eta_l_sl;
      TH1D *h_p_T_l_le_cut50;
      TH1D *h_p_T_l_sl_cut50;

      TH1D *h_angle_X_Z, *h_angle_l_pl_l_mi, *h_angle_X_pr1_X_pr2, *h_angle_Z_X_pr1, *h_angle_Z_X_pr2;
      TH1D *h_angleT_X_Z, *h_angleT_l_pl_l_mi, *h_angleT_X_pr1_X_pr2, *h_angleT_Z_X_pr1, *h_angleT_Z_X_pr2;

      TH1D *h_de_xi_single_45, *h_de_xi_single_56;
      TProfile *p_de_xi_vs_xi_single_45, *p_de_xi_vs_xi_single_56;
      TH1D *h_de_xi_multi_45, *h_de_xi_multi_56;
      TProfile *p_de_xi_vs_xi_multi_45, *p_de_xi_vs_xi_multi_56;

      TH1D *h_de_m_X_single, *h_de_m_XZ_single;
      TProfile *p_de_m_vs_m_X_single, *p_de_m_vs_m_XZ_single;
      TH1D *h_de_m_X_multi, *h_de_m_XZ_multi;
      TProfile *p_de_m_vs_m_X_multi, *p_de_m_vs_m_XZ_multi;

      std::vector<double> thresholds;
      std::vector<double> th_counts;

      void init()
      {
        h_m_Z = new TH1D("", ";m_{Z}   (GeV)", 100, 80., 100.);
        h_m_XZ = new TH1D("", ";m_{XZ}   (GeV)", 200, 800., 2000.);
        h_p_z_LAB_2p = new TH1D("", ";p_{z}(2 protons)   (GeV)", 100, -2000., +2000.);

        h_xi2_vs_xi1 = new TH2D("", ";#xi_{1};#xi_{2}", 50., 0., 0.20, 50., 0., 0.20);

        h_p_T_X = new TH1D("", "p_{T}(X)   (GeV)", 100, 0., 180.);
        h_p_z_X = new TH1D("", "p_{z}(X)   (GeV)", 100, -1500., 1500.);
        h_p_tot_X = new TH1D("", "p(X)   (GeV)", 100, 0., 1500.);
        h_theta_X = new TH1D("", "theta(X)", 100, -0.1, 3.3);
        h_eta_X = new TH1D("", "eta(X)", 100, -8., 8.);

        h_p_T_X_pr1 = new TH1D("", "p_{T}(X_pr1)   (GeV)", 100, 0., 800.);
        h_p_z_X_pr1 = new TH1D("", "p_{z}(X_pr1)   (GeV)", 100, -1500., 1500.);
        h_p_tot_X_pr1 = new TH1D("", "p(X_pr1)   (GeV)", 100, 0., 1500.);
        h_theta_X_pr1 = new TH1D("", "theta(X_pr1)", 100, -0.1, 3.3);
        h_eta_X_pr1 = new TH1D("", "eta(X_pr1)", 100, -5., 5.);

        h_p_T_X_pr2 = new TH1D("", "p_{T}(X_pr2)   (GeV)", 100, 0., 800.);
        h_p_z_X_pr2 = new TH1D("", "p_{z}(X_pr2)   (GeV)", 100, -1500., 1500.);
        h_p_tot_X_pr2 = new TH1D("", "p(X_pr2)   (GeV)", 100, 0., 1500.);
        h_theta_X_pr2 = new TH1D("", "theta(X_pr2)", 100, -0.1, 3.3);
        h_eta_X_pr2 = new TH1D("", "eta(X_pr2)", 100, -5., 5.);

        h_p_T_Z = new TH1D("", "p_{T}(Z)   (GeV)", 100, 0., 180.);
        h_p_z_Z = new TH1D("", "p_{z}(Z)   (GeV)", 100, -300., 300.);
        h_p_tot_Z = new TH1D("", "p(Z)   (GeV)", 100, 0., 300.);
        h_theta_Z = new TH1D("", "theta(Z)", 100, -0.1, 3.3);
        h_eta_Z = new TH1D("", "eta(Z)", 100, -5., 5.);

        h_p_T_l_pl = new TH1D("", "p_{T}(l_pl)   (GeV)", 100, 0., 180.);
        h_p_z_l_pl = new TH1D("", "p_{z}(l_pl)   (GeV)", 100, -300., 300.);
        h_p_tot_l_pl = new TH1D("", "p(l_pl)   (GeV)", 100, 0., 300.);
        h_theta_l_pl = new TH1D("", "theta(l_pl)", 100, -0.1, 3.3);
        h_eta_l_pl = new TH1D("", "eta(l_pl)", 100, -5., 5.);

        h_p_T_l_mi = new TH1D("", "p_{T}(l_mi)   (GeV)", 100, 0., 180.);
        h_p_z_l_mi = new TH1D("", "p_{z}(l_mi)   (GeV)", 100, -300., 300.);
        h_p_tot_l_mi = new TH1D("", "p(l_mi)   (GeV)", 100, 0., 300.);
        h_theta_l_mi = new TH1D("", "theta(l_mi)", 100, -0.1, 3.3);
        h_eta_l_mi = new TH1D("", "eta(l_mi)", 100, -5., 5.);

        h_p_T_l_le = new TH1D("", "p_{T}(l_le)   (GeV)", 100, 0., 180.);
        h_p_z_l_le = new TH1D("", "p_{z}(l_le)   (GeV)", 100, -300., 300.);
        h_p_tot_l_le = new TH1D("", "p(l_le)   (GeV)", 100, 0., 300.);
        h_theta_l_le = new TH1D("", "theta(l_le)", 100, -0.1, 3.3);
        h_eta_l_le = new TH1D("", "eta(l_le)", 100, -5., 5.);
        h_p_T_l_le_cut50 = new TH1D("", "p_{T}(l_le)   (GeV)", 100, 0., 180.);

        h_p_T_l_sl = new TH1D("", "p_{T}(l_sl)   (GeV)", 100, 0., 180.);
        h_p_z_l_sl = new TH1D("", "p_{z}(l_sl)   (GeV)", 100, -300., 300.);
        h_p_tot_l_sl = new TH1D("", "p(l_sl)   (GeV)", 100, 0., 300.);
        h_theta_l_sl = new TH1D("", "theta(l_sl)", 100, -0.1, 3.3);
        h_eta_l_sl = new TH1D("", "eta(l_sl)", 100, -5., 5.);
        h_p_T_l_sl_cut50 = new TH1D("", "p_{T}(l_sl)   (GeV)", 100, 0., 180.);

        h_angle_X_Z = new TH1D("", "angle(X, Z)", 100, -1E-3, M_PI + 1E-3);
        h_angle_l_pl_l_mi = new TH1D("", "angle(l_pl, l_mi)", 100, -1E-3, M_PI + 1E-3);
        h_angle_X_pr1_X_pr2 = new TH1D("", "angle(X_pr1, X_pr2)", 100, -1E-3, M_PI + 1E-3);
        h_angle_Z_X_pr1 = new TH1D("", "angle(Z, X_pr1)", 100, -1E-3, M_PI + 1E-3);
        h_angle_Z_X_pr2 = new TH1D("", "angle(Z, X_pr2)", 100, -1E-3, M_PI + 1E-3);

        h_angleT_X_Z = new TH1D("", "angleT(X, Z)", 100, -1E-3, M_PI + 1E-3);
        h_angleT_l_pl_l_mi = new TH1D("", "angleT(l_pl, l_mi)", 100, -1E-3, M_PI + 1E-3);
        h_angleT_X_pr1_X_pr2 = new TH1D("", "angleT(X_pr1, X_pr2)", 100, -1E-3, M_PI + 1E-3);
        h_angleT_Z_X_pr1 = new TH1D("", "angleT(Z, X_pr1)", 100, -1E-3, M_PI + 1E-3);
        h_angleT_Z_X_pr2 = new TH1D("", "angleT(Z, X_pr2)", 100, -1E-3, M_PI + 1E-3);

        h_de_xi_single_45 = new TH1D("", ";#xi_{45,reco} - #xi_{45,simu}", 200, -0.05, 0.05);
        p_de_xi_vs_xi_single_45 = new TProfile("", ";#xi_{45,simu};#xi_{45,reco} - #xi_{45,simu}", 19, 0.015, 0.205);
        h_de_xi_multi_45 = new TH1D("", ";#xi_{45,reco} - #xi_{45,simu}", 200, -0.05, 0.05);
        p_de_xi_vs_xi_multi_45 = new TProfile("", ";#xi_{45,simu};#xi_{45,reco} - #xi_{45,simu}", 19, 0.015, 0.205);

        h_de_xi_single_56 = new TH1D("", ";#xi_{56,reco} - #xi_{56,simu}", 200, -0.05, 0.05);
        p_de_xi_vs_xi_single_56 = new TProfile("", ";#xi_{56,simu};#xi_{56,reco} - #xi_{56,simu}", 19, 0.015, 0.205);
        h_de_xi_multi_56 = new TH1D("", ";#xi_{56,reco} - #xi_{56,simu}", 200, -0.05, 0.05);
        p_de_xi_vs_xi_multi_56 = new TProfile("", ";#xi_{56,simu};#xi_{56,reco} - #xi_{56,simu}", 19, 0.015, 0.205);

        h_de_m_X_single = new TH1D("", ";m_{X,reco} - m_{X,simu}", 200, -500., +500.);
        p_de_m_vs_m_X_single = new TProfile("", ";m_{X,simu};m_{X,reco} - m_{X,simu}", 200, 0., 2000.);
        h_de_m_X_multi = new TH1D("", ";m_{X,reco} - m_{X,simu}", 200, -500., +500.);
        p_de_m_vs_m_X_multi = new TProfile("", ";m_{X,simu};m_{X,reco} - m_{X,simu}", 200, 0., 2000.);

        h_de_m_XZ_single = new TH1D("", ";m_{XZ,reco} - m_{XZ,simu}", 200, -500., +500.);
        p_de_m_vs_m_XZ_single = new TProfile("", ";m_{XZ,simu};m_{XZ,reco} - m_{XZ,simu}", 200, 0., 2000.);
        h_de_m_XZ_multi = new TH1D("", ";m_{XZ,reco} - m_{XZ,simu}", 200, -500., +500.);
        p_de_m_vs_m_XZ_multi = new TProfile("", ";m_{XZ,simu};m_{XZ,reco} - m_{XZ,simu}", 200, 0., 2000.);

        thresholds = { 30., 35., 40., 45., 50., 55. };
        th_counts.resize(5, 0.);
      }

      void fill(const CLHEP::HepLorentzVector &momentum_p1, const CLHEP::HepLorentzVector &momentum_p2,
        const CLHEP::HepLorentzVector &momentum_X,
        const CLHEP::HepLorentzVector &momentum_X_pr1, const CLHEP::HepLorentzVector &momentum_X_pr2,
        const CLHEP::HepLorentzVector &momentum_Z,
        const CLHEP::HepLorentzVector &momentum_l_pl, const CLHEP::HepLorentzVector &momentum_l_mi,
        const reco::ForwardProton &rec_pr_single_45, const reco::ForwardProton &rec_pr_single_56,
        const reco::ForwardProton &rec_pr_multi_45, const reco::ForwardProton &rec_pr_multi_56)
      {
        if (h_m_Z == NULL)
          init();

        // leading and sub-leading lepton
        CLHEP::HepLorentzVector momentum_l_le, momentum_l_sl;
        if (momentum_l_pl.perp() > momentum_l_mi.perp())
        {
          momentum_l_le = momentum_l_pl;
          momentum_l_sl = momentum_l_mi;
        } else {
          momentum_l_le = momentum_l_mi;
          momentum_l_sl = momentum_l_pl;
        }

        CLHEP::Hep3Vector momentumT_X(momentum_X.x(), momentum_X.y(), 0.);
        CLHEP::Hep3Vector momentumT_X_pr1(momentum_X_pr1.x(), momentum_X_pr1.y(), 0.);
        CLHEP::Hep3Vector momentumT_X_pr2(momentum_X_pr2.x(), momentum_X_pr2.y(), 0.);
        CLHEP::Hep3Vector momentumT_Z(momentum_Z.x(), momentum_Z.y(), 0.);
        CLHEP::Hep3Vector momentumT_l_pl(momentum_l_pl.x(), momentum_l_pl.y(), 0.);
        CLHEP::Hep3Vector momentumT_l_mi(momentum_l_mi.x(), momentum_l_mi.y(), 0.);

        h_m_Z->Fill(momentum_Z.m());
        h_m_XZ->Fill((momentum_Z + momentum_X).m());
        h_p_z_LAB_2p->Fill((momentum_p1 + momentum_p2).z());

        const double p_beam = (momentum_p1 + momentum_p2 + momentum_Z + momentum_X).m() / 2.;
        const double xi1 = 1. - momentum_p1.t() / p_beam;
        const double xi2 = 1. - momentum_p2.t() / p_beam;

        const CLHEP::HepLorentzVector momentum_p_simu_45 = (momentum_p1.z() > 0) ? momentum_p1 : momentum_p2;
        const CLHEP::HepLorentzVector momentum_p_simu_56 = (momentum_p1.z() < 0) ? momentum_p1 : momentum_p2;

        const CLHEP::HepLorentzVector momentum_p_reco_single_45(0., 0., +p_beam*(1.-rec_pr_single_45.xi()), p_beam*(1.-rec_pr_single_45.xi()));
        const CLHEP::HepLorentzVector momentum_p_reco_single_56(0., 0., -p_beam*(1.-rec_pr_single_56.xi()), p_beam*(1.-rec_pr_single_56.xi()));

        const CLHEP::HepLorentzVector momentum_p_reco_multi_45(0., 0., +p_beam*(1.-rec_pr_multi_45.xi()), p_beam*(1.-rec_pr_multi_45.xi()));
        const CLHEP::HepLorentzVector momentum_p_reco_multi_56(0., 0., -p_beam*(1.-rec_pr_multi_56.xi()), p_beam*(1.-rec_pr_multi_56.xi()));

        const double xi_simu_45 = 1. - momentum_p_simu_45.t() / p_beam;
        const double xi_simu_56 = 1. - momentum_p_simu_56.t() / p_beam;

        const double xi_reco_single_45 = 1. - momentum_p_reco_single_45.t() / p_beam;
        const double xi_reco_single_56 = 1. - momentum_p_reco_single_56.t() / p_beam;

        const double xi_reco_multi_45 = 1. - momentum_p_reco_multi_45.t() / p_beam;
        const double xi_reco_multi_56 = 1. - momentum_p_reco_multi_56.t() / p_beam;

        const CLHEP::HepLorentzVector momentum_init(0., 0., 0., 2. * p_beam);

        const double m_X_simu = momentum_X.m();
        const double m_X_reco_single = (momentum_init - momentum_p_reco_single_45 - momentum_p_reco_single_56 - momentum_Z).m();
        const double m_X_reco_multi = (momentum_init - momentum_p_reco_multi_45 - momentum_p_reco_multi_56 - momentum_Z).m();

        const double m_XZ_simu = (momentum_Z + momentum_X).m();
        const double m_XZ_reco_single = (momentum_init - momentum_p_reco_single_45 - momentum_p_reco_single_56).m();
        const double m_XZ_reco_multi = (momentum_init - momentum_p_reco_multi_45 - momentum_p_reco_multi_56).m();

        h_xi2_vs_xi1->Fill(xi1, xi2);

        h_p_T_X->Fill(momentum_X.perp());
        h_p_z_X->Fill(momentum_X.z());
        h_p_tot_X->Fill(momentum_X.rho());
        h_theta_X->Fill(momentum_X.theta());
        h_eta_X->Fill(momentum_X.pseudoRapidity());

        h_p_T_X_pr1->Fill(momentum_X_pr1.perp());
        h_p_z_X_pr1->Fill(momentum_X_pr1.z());
        h_p_tot_X_pr1->Fill(momentum_X_pr1.rho());
        h_theta_X_pr1->Fill(momentum_X_pr1.theta());
        h_eta_X_pr1->Fill(momentum_X_pr1.pseudoRapidity());

        h_p_T_X_pr2->Fill(momentum_X_pr2.perp());
        h_p_z_X_pr2->Fill(momentum_X_pr2.z());
        h_p_tot_X_pr2->Fill(momentum_X_pr2.rho());
        h_theta_X_pr2->Fill(momentum_X_pr2.theta());
        h_eta_X_pr2->Fill(momentum_X_pr2.pseudoRapidity());

        h_p_T_Z->Fill(momentum_Z.perp());
        h_p_z_Z->Fill(momentum_Z.z());
        h_p_tot_Z->Fill(momentum_Z.rho());
        h_theta_Z->Fill(momentum_Z.theta());
        h_eta_Z->Fill(momentum_Z.pseudoRapidity());

        h_p_T_l_pl->Fill(momentum_l_pl.perp());
        h_p_z_l_pl->Fill(momentum_l_pl.z());
        h_p_tot_l_pl->Fill(momentum_l_pl.rho());
        h_theta_l_pl->Fill(momentum_l_pl.theta());
        h_eta_l_pl->Fill(momentum_l_pl.pseudoRapidity());

        h_p_T_l_mi->Fill(momentum_l_mi.perp());
        h_p_z_l_mi->Fill(momentum_l_mi.z());
        h_p_tot_l_mi->Fill(momentum_l_mi.rho());
        h_theta_l_mi->Fill(momentum_l_mi.theta());
        h_eta_l_mi->Fill(momentum_l_mi.pseudoRapidity());

        h_p_T_l_le->Fill(momentum_l_le.perp());
        h_p_z_l_le->Fill(momentum_l_le.z());
        h_p_tot_l_le->Fill(momentum_l_le.rho());
        h_theta_l_le->Fill(momentum_l_le.theta());
        h_eta_l_le->Fill(momentum_l_le.pseudoRapidity());

        h_p_T_l_sl->Fill(momentum_l_sl.perp());
        h_p_z_l_sl->Fill(momentum_l_sl.z());
        h_p_tot_l_sl->Fill(momentum_l_sl.rho());
        h_theta_l_sl->Fill(momentum_l_sl.theta());
        h_eta_l_sl->Fill(momentum_l_sl.pseudoRapidity());

        if (momentum_l_le.perp() > 50)
        {
          h_p_T_l_le_cut50->Fill(momentum_l_le.perp());
          h_p_T_l_sl_cut50->Fill(momentum_l_sl.perp());
        }

        h_angle_X_Z->Fill(momentum_X.angle(momentum_Z));
        h_angle_l_pl_l_mi->Fill(momentum_l_pl.angle(momentum_l_mi));
        h_angle_X_pr1_X_pr2->Fill(momentum_X_pr1.angle(momentum_X_pr2));
        h_angle_Z_X_pr1->Fill(momentum_Z.angle(momentum_X_pr1));
        h_angle_Z_X_pr2->Fill(momentum_Z.angle(momentum_X_pr2));

        h_angleT_X_Z->Fill(momentumT_X.angle(momentumT_Z));
        h_angleT_l_pl_l_mi->Fill(momentumT_l_pl.angle(momentumT_l_mi));
        h_angleT_X_pr1_X_pr2->Fill(momentumT_X_pr1.angle(momentumT_X_pr2));
        h_angleT_Z_X_pr1->Fill(momentumT_Z.angle(momentumT_X_pr1));
        h_angleT_Z_X_pr2->Fill(momentumT_Z.angle(momentumT_X_pr2));

        if (rec_pr_single_45.validFit() && rec_pr_single_56.validFit())
        {
          h_de_xi_single_45->Fill(xi_reco_single_45 - xi_simu_45);
          p_de_xi_vs_xi_single_45->Fill(xi_simu_45, xi_reco_single_45 - xi_simu_45);

          h_de_xi_single_56->Fill(xi_reco_single_56 - xi_simu_56);
          p_de_xi_vs_xi_single_56->Fill(xi_simu_56, xi_reco_single_56 - xi_simu_56);

          h_de_m_X_single->Fill(m_X_reco_single - m_X_simu);
          p_de_m_vs_m_X_single->Fill(m_X_simu, m_X_reco_single - m_X_simu);

          h_de_m_XZ_single->Fill(m_XZ_reco_single - m_XZ_simu);
          p_de_m_vs_m_XZ_single->Fill(m_XZ_simu, m_XZ_reco_single - m_XZ_simu);
        }

        if (rec_pr_multi_45.validFit() && rec_pr_multi_56.validFit())
        {
          h_de_xi_multi_45->Fill(xi_reco_multi_45 - xi_simu_45);
          p_de_xi_vs_xi_multi_45->Fill(xi_simu_45, xi_reco_multi_45 - xi_simu_45);

          h_de_xi_multi_56->Fill(xi_reco_multi_56 - xi_simu_56);
          p_de_xi_vs_xi_multi_56->Fill(xi_simu_56, xi_reco_multi_56 - xi_simu_56);

          h_de_m_X_multi->Fill(m_X_reco_multi - m_X_simu);
          p_de_m_vs_m_X_multi->Fill(m_X_simu, m_X_reco_multi - m_X_simu);

          h_de_m_XZ_multi->Fill(m_XZ_reco_multi - m_XZ_simu);
          p_de_m_vs_m_XZ_multi->Fill(m_XZ_simu, m_XZ_reco_multi - m_XZ_simu);
        }

        for (unsigned int thi = 0; thi < thresholds.size(); ++thi)
        {
          if (momentum_l_le.perp() > 55. && momentum_l_sl.perp() > thresholds[thi])
            th_counts[thi] += 1.;
        }
      }

      void write() const
      {
        if (!h_m_Z)
        {
          printf("ERROR in Plots::write > object not initialised.\n");
          return;
        }

        h_m_Z->Write("h_m_Z");
        h_m_XZ->Write("h_m_XZ");
        h_p_z_LAB_2p->Write("h_p_z_LAB_2p");

        h_xi2_vs_xi1->Write("h_xi2_vs_xi1");

        h_p_T_X->Write("h_p_T_X");
        h_p_z_X->Write("h_p_z_X");
        h_p_tot_X->Write("h_p_tot_X");
        h_theta_X->Write("h_theta_X");
        h_eta_X->Write("h_eta_X");

        h_p_T_X_pr1->Write("h_p_T_X_pr1");
        h_p_z_X_pr1->Write("h_p_z_X_pr1");
        h_p_tot_X_pr1->Write("h_p_tot_X_pr1");
        h_theta_X_pr1->Write("h_theta_X_pr1");
        h_eta_X_pr1->Write("h_eta_X_pr1");

        h_p_T_X_pr2->Write("h_p_T_X_pr2");
        h_p_z_X_pr2->Write("h_p_z_X_pr2");
        h_p_tot_X_pr2->Write("h_p_tot_X_pr2");
        h_theta_X_pr2->Write("h_theta_X_pr2");
        h_eta_X_pr2->Write("h_eta_X_pr2");

        h_p_T_Z->Write("h_p_T_Z");
        h_p_z_Z->Write("h_p_z_Z");
        h_p_tot_Z->Write("h_p_tot_Z");
        h_theta_Z->Write("h_theta_Z");
        h_eta_Z->Write("h_eta_Z");

        h_p_T_l_pl->Write("h_p_T_l_pl");
        h_p_z_l_pl->Write("h_p_z_l_pl");
        h_p_tot_l_pl->Write("h_p_tot_l_pl");
        h_theta_l_pl->Write("h_theta_l_pl");
        h_eta_l_pl->Write("h_eta_l_pl");

        h_p_T_l_mi->Write("h_p_T_l_mi");
        h_p_z_l_mi->Write("h_p_z_l_mi");
        h_p_tot_l_mi->Write("h_p_tot_l_mi");
        h_theta_l_mi->Write("h_theta_l_mi");
        h_eta_l_mi->Write("h_eta_l_mi");

        h_p_T_l_le->Write("h_p_T_l_le");
        h_p_z_l_le->Write("h_p_z_l_le");
        h_p_tot_l_le->Write("h_p_tot_l_le");
        h_theta_l_le->Write("h_theta_l_le");
        h_eta_l_le->Write("h_eta_l_le");
        h_p_T_l_le_cut50->Write("h_p_T_l_le_cut50");

        h_p_T_l_sl->Write("h_p_T_l_sl");
        h_p_z_l_sl->Write("h_p_z_l_sl");
        h_p_tot_l_sl->Write("h_p_tot_l_sl");
        h_theta_l_sl->Write("h_theta_l_sl");
        h_eta_l_sl->Write("h_eta_l_sl");
        h_p_T_l_sl_cut50->Write("h_p_T_l_sl_cut50");

        h_angle_X_Z->Write("h_angle_X_Z");
        h_angle_l_pl_l_mi->Write("h_angle_l_pl_l_mi");
        h_angle_X_pr1_X_pr2->Write("h_angle_X_pr1_X_pr2");
        h_angle_Z_X_pr1->Write("h_angle_Z_X_pr1");
        h_angle_Z_X_pr2->Write("h_angle_Z_X_pr2");

        h_angleT_X_Z->Write("h_angleT_X_Z");
        h_angleT_l_pl_l_mi->Write("h_angleT_l_pl_l_mi");
        h_angleT_X_pr1_X_pr2->Write("h_angleT_X_pr1_X_pr2");
        h_angleT_Z_X_pr1->Write("h_angleT_Z_X_pr1");
        h_angleT_Z_X_pr2->Write("h_angleT_Z_X_pr2");

        h_de_xi_single_45->Write("h_de_xi_single_45");
        p_de_xi_vs_xi_single_45->Write("p_de_xi_vs_xi_single_45");
        ProfileToRMSGraph(p_de_xi_vs_xi_single_45, "g_rms_de_xi_vs_xi_single_45")->Write();
        h_de_xi_multi_45->Write("h_de_xi_multi_45");
        p_de_xi_vs_xi_multi_45->Write("p_de_xi_vs_xi_multi_45");
        ProfileToRMSGraph(p_de_xi_vs_xi_multi_45, "g_rms_de_xi_vs_xi_multi_45")->Write();

        h_de_xi_single_56->Write("h_de_xi_single_56");
        p_de_xi_vs_xi_single_56->Write("p_de_xi_vs_xi_single_56");
        ProfileToRMSGraph(p_de_xi_vs_xi_single_56, "g_rms_de_xi_vs_xi_single_56")->Write();
        h_de_xi_multi_56->Write("h_de_xi_multi_56");
        p_de_xi_vs_xi_multi_56->Write("p_de_xi_vs_xi_multi_56");
        ProfileToRMSGraph(p_de_xi_vs_xi_multi_56, "g_rms_de_xi_vs_xi_multi_56")->Write();

        h_de_m_X_single->Write("h_de_m_X_single");
        p_de_m_vs_m_X_single->Write("p_de_m_vs_m_X_single");
        ProfileToRMSGraph(p_de_m_vs_m_X_single, "g_rms_de_m_vs_m_X_single")->Write();
        h_de_m_X_multi->Write("h_de_m_X_multi");
        p_de_m_vs_m_X_multi->Write("p_de_m_vs_m_X_multi");
        ProfileToRMSGraph(p_de_m_vs_m_X_multi, "g_rms_de_m_vs_m_X_multi")->Write();

        h_de_m_XZ_single->Write("h_de_m_XZ_single");
        p_de_m_vs_m_XZ_single->Write("p_de_m_vs_m_XZ_single");
        ProfileToRMSGraph(p_de_m_vs_m_XZ_single, "g_rms_de_m_vs_m_XZ_single")->Write();
        h_de_m_XZ_multi->Write("h_de_m_XZ_multi");
        p_de_m_vs_m_XZ_multi->Write("p_de_m_vs_m_XZ_multi");
        ProfileToRMSGraph(p_de_m_vs_m_XZ_multi, "g_rms_de_m_vs_m_XZ_multi")->Write();

        for (unsigned int thi = 0; thi < thresholds.size(); ++thi)
        {
          printf("    threshold = %.1f, events = %.1f\n", thresholds[thi], th_counts[thi]);
        }
      }

      static TGraphErrors* ProfileToRMSGraph(TProfile *p, const std::string &name = "")
      {
          TGraphErrors *g = new TGraphErrors();
          g->SetName(name.c_str());

          for (int bi = 1; bi <= p->GetNbinsX(); ++bi)
          {
              double c = p->GetBinCenter(bi);
              double w = p->GetBinWidth(bi);

              double N = p->GetBinEntries(bi);
              double Sy = p->GetBinContent(bi) * N;
              double Syy = p->GetSumw2()->At(bi);

              double si_sq = Syy/N - Sy*Sy/N/N;
              double si = (si_sq >= 0.) ? sqrt(si_sq) : 0.;
              double si_unc_sq = si_sq / 2. / N;  // Gaussian approximation
              double si_unc = (si_unc_sq >= 0.) ? sqrt(si_unc_sq) : 0.;

              int idx = g->GetN();
              g->SetPoint(idx, c, si);
              g->SetPointError(idx, w/2., si_unc);
          }

          return g;
      }
    };

    Plots plotsBeforeSimulation, plotsAfterSimulation;

    inline CLHEP::HepLorentzVector convertTo(const HepMC::FourVector &v)
    {
      return CLHEP::HepLorentzVector(v.x(), v.y(), v.z(), v.t());
    }
};

//----------------------------------------------------------------------------------------------------

PPXZGeneratorValidation::PPXZGeneratorValidation( const edm::ParameterSet& iConfig ) :
  hepMCToken_( consumes< edm::HepMCProduct >( iConfig.getParameter<edm::InputTag>( "tagHepMC" ) ) ),
  recoTracksToken_( consumes< std::vector<CTPPSLocalTrackLite> >( iConfig.getParameter<edm::InputTag>( "tagRecoTracks" ) ) ),
  recoProtonsTokenSingleRP_( consumes<std::vector<reco::ForwardProton>>(iConfig.getParameter<edm::InputTag>("tagRecoProtonsSingleRP")) ),
  recoProtonsTokenMultiRP_( consumes<std::vector<reco::ForwardProton>>(iConfig.getParameter<edm::InputTag>("tagRecoProtonsMultiRP")) ),
  referenceRPDecId_45(iConfig.getParameter<unsigned int>("referenceRPDecId_45")),
  referenceRPDecId_56(iConfig.getParameter<unsigned int>("referenceRPDecId_56")),
  outputFile( iConfig.getParameter<std::string>("outputFile") )
{
}

//----------------------------------------------------------------------------------------------------

PPXZGeneratorValidation::~PPXZGeneratorValidation()
{
}

//----------------------------------------------------------------------------------------------------

void PPXZGeneratorValidation::analyze(const edm::Event& iEvent, const edm::EventSetup&)
{
  // get input
  edm::Handle<edm::HepMCProduct> hHepMCProduct;
  iEvent.getByToken(hepMCToken_, hHepMCProduct);

  edm::Handle< std::vector<CTPPSLocalTrackLite> > hRecoTracks;
  iEvent.getByToken(recoTracksToken_, hRecoTracks);

  edm::Handle< std::vector<reco::ForwardProton> > hRecoProtonsSingleRP;
  iEvent.getByToken(recoProtonsTokenSingleRP_, hRecoProtonsSingleRP);

  edm::Handle< std::vector<reco::ForwardProton> > hRecoProtonsMultiRP;
  iEvent.getByToken(recoProtonsTokenMultiRP_, hRecoProtonsMultiRP);

  // process HepMC record
  CLHEP::HepLorentzVector momentum_p1, momentum_p2, momentum_X, momentum_X_pr1, momentum_X_pr2, momentum_Z, momentum_l_pl, momentum_l_mi;

  // loop over event vertices
  auto evt = hHepMCProduct->GetEvent();
  for (auto it_vtx = evt->vertices_begin(); it_vtx != evt->vertices_end(); ++it_vtx)
  {
    auto vtx = *it_vtx;

    // loop over outgoing particles
    for (auto it_part = vtx->particles_out_const_begin(); it_part != vtx->particles_out_const_end(); ++it_part)
    {
      auto part = *it_part;

      if (part->pdg_id() == particleId_p)
      {
        if (part->momentum().z() > 0)
          momentum_p1 = convertTo(part->momentum());
        else
          momentum_p2 = convertTo(part->momentum());
      }

      if (part->pdg_id() == particleId_X)
        momentum_X = convertTo(part->momentum());

      if (part->pdg_id() == particleId_X_pr1)
        momentum_X_pr1 = convertTo(part->momentum());

      if (part->pdg_id() == particleId_X_pr2)
        momentum_X_pr2 = convertTo(part->momentum());

      if (part->pdg_id() == particleId_Z)
        momentum_Z = convertTo(part->momentum());

      if (part->pdg_id() == particleId_e_pl || part->pdg_id() == particleId_mu_pl)
        momentum_l_pl = convertTo(part->momentum());

      if (part->pdg_id() == particleId_e_mi || part->pdg_id() == particleId_mu_mi)
        momentum_l_mi = convertTo(part->momentum());
    }
  }

  // process tracks
  bool protonTrackIn45 = false, protonTrackIn56 = false;
  for (const auto& tr : *hRecoTracks)
  {
    CTPPSDetId rpId(tr.getRPId());
    if (rpId.arm() == 0) protonTrackIn45 = true;
    if (rpId.arm() == 1) protonTrackIn56 = true;
  }

  // process reco protons
  reco::ForwardProton rec_pr_single_45; rec_pr_single_45.setValidFit(false);
  reco::ForwardProton rec_pr_single_56; rec_pr_single_56.setValidFit(false);

  reco::ForwardProton rec_pr_multi_45; rec_pr_multi_45.setValidFit(false);
  reco::ForwardProton rec_pr_multi_56; rec_pr_multi_56.setValidFit(false);

  for (const auto &rec_pr : *hRecoProtonsSingleRP)
  {
    if (! rec_pr.validFit())
      continue;

    CTPPSDetId rpId((*rec_pr.contributingLocalTracks().begin())->getRPId());
    unsigned int rpDecId = 100*rpId.arm() + 10*rpId.station() + rpId.rp();

    if (rpDecId == referenceRPDecId_45)
      rec_pr_single_45 = rec_pr;

    if (rpDecId == referenceRPDecId_56)
      rec_pr_single_56 = rec_pr;
  }

  for (const auto &rec_pr : *hRecoProtonsMultiRP)
  {
    if (! rec_pr.validFit())
      continue;

    if (rec_pr.lhcSector() == reco::ForwardProton::LHCSector::sector45)
      rec_pr_multi_45 = rec_pr;

    if (rec_pr.lhcSector() == reco::ForwardProton::LHCSector::sector56)
      rec_pr_multi_56 = rec_pr;
  }

  // fill plots
  plotsBeforeSimulation.fill(momentum_p1, momentum_p2, momentum_X, momentum_X_pr1, momentum_X_pr2, momentum_Z, momentum_l_pl, momentum_l_mi,
    rec_pr_single_45, rec_pr_single_56, rec_pr_multi_45, rec_pr_multi_56);

  if (protonTrackIn45 && protonTrackIn56)
  {
    plotsAfterSimulation.fill(momentum_p1, momentum_p2, momentum_X, momentum_X_pr1, momentum_X_pr2, momentum_Z, momentum_l_pl, momentum_l_mi,
      rec_pr_single_45, rec_pr_single_56, rec_pr_multi_45, rec_pr_multi_56);
  }
}

//----------------------------------------------------------------------------------------------------

void PPXZGeneratorValidation::beginJob()
{
}

//----------------------------------------------------------------------------------------------------

void PPXZGeneratorValidation::endJob()
{
  TFile *f_out = TFile::Open(outputFile.c_str(), "recreate");
  
  printf("* before simulation\n");
  gDirectory = f_out->mkdir("before simulation");
  plotsBeforeSimulation.write();
  
  printf("* after simulation\n");
  gDirectory = f_out->mkdir("after simulation");
  plotsAfterSimulation.write();

  delete f_out;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PPXZGeneratorValidation);
