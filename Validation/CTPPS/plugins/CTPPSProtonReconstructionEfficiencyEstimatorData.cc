/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Mauricio Thiel
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TF1.h"
#include "TGraph.h"

#include <map>
#include <string>
#include <sstream>

//----------------------------------------------------------------------------------------------------

class CTPPSProtonReconstructionEfficiencyEstimatorData : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSProtonReconstructionEfficiencyEstimatorData(const edm::ParameterSet &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tokenTracks_;
  edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsMultiRP_;
  edm::ESGetToken<LHCInterpolatedOpticalFunctionsSetCollection, CTPPSInterpolatedOpticsRcd> opticsESToken_;

  bool pixelDiscardBXShiftedTracks_;

  double localAngleXMin_, localAngleXMax_, localAngleYMin_, localAngleYMax_;

  unsigned int n_prep_events_;

  unsigned int n_exp_prot_max_;

  std::vector<double> n_sigmas_;

  std::string outputFile_;

  unsigned int verbosity_;

  edm::ESWatcher<CTPPSInterpolatedOpticsRcd> opticsWatcher_;

  struct ArmData {
    unsigned int rpId_N, rpId_F;

    std::shared_ptr<const TSpline3> s_x_to_xi_N;

    unsigned int n_events_with_tracks;

    std::unique_ptr<TH1D> h_de_x, h_de_y;
    std::unique_ptr<TH2D> h2_de_y_vs_de_x;

    double de_x_mean, de_x_sigma;
    double de_y_mean, de_y_sigma;

    std::vector<double> n_sigmas;

    unsigned int n_exp_prot_max;

    struct EffPlots {
      std::unique_ptr<TProfile> p_eff1_vs_x_N;
      std::unique_ptr<TProfile> p_eff1_vs_xi_N;

      std::unique_ptr<TProfile> p_eff2_vs_x_N;
      std::unique_ptr<TProfile> p_eff2_vs_xi_N;

      EffPlots()
          : p_eff1_vs_x_N(new TProfile("", ";x_{N}   (mm);efficiency", 50, 0., 25.)),
            p_eff1_vs_xi_N(new TProfile("", ";#xi_{si,N};efficiency", 50, 0., 0.25)),

            p_eff2_vs_x_N(new TProfile("", ";x_{N}   (mm);efficiency", 50, 0., 25.)),
            p_eff2_vs_xi_N(new TProfile("", ";#xi_{si,N};efficiency", 50, 0., 0.25)) {}

      void write() const {
        p_eff1_vs_x_N->Write("p_eff1_vs_x_N");
        p_eff1_vs_xi_N->Write("p_eff1_vs_xi_N");

        p_eff2_vs_x_N->Write("p_eff2_vs_x_N");
        p_eff2_vs_xi_N->Write("p_eff2_vs_xi_N");
      }
    };

    // (n exp protons, index in n_sigmas) --> plots
    // n exp protons = 0 --> no condition (summary case)
    std::map<unsigned int, std::map<unsigned int, EffPlots>> effPlots;

    ArmData()
        : n_events_with_tracks(0),

          h_de_x(new TH1D("", ";x_{F} - x_{N}   (mm)", 200, -1., +1.)),
          h_de_y(new TH1D("", ";y_{F} - y_{N}   (mm)", 200, -1., +1.)),
          h2_de_y_vs_de_x(new TH2D("", ";x_{F} - x_{N}   (mm);y_{F} - y_{N}   (mm)", 100, -1., +1., 100, -1., +1.)),

          de_x_mean(0.),
          de_x_sigma(0.),
          de_y_mean(0.),
          de_y_sigma(0.) {
      for (unsigned int nep = 0; nep <= n_exp_prot_max; ++nep) {
        for (unsigned int nsi = 0; nsi < n_sigmas.size(); ++nsi) {
          effPlots[nep][nsi] = EffPlots();
        }
      }
    }

    void write() const {
      h_de_x->Write("h_de_x");
      h_de_y->Write("h_de_y");
      h2_de_y_vs_de_x->Write("h2_de_y_vs_de_x");

      char buf[100];

      for (const auto &n_si : n_sigmas) {
        sprintf(buf, "g_de_y_vs_de_x_n_si=%.1f", n_si);
        TGraph *g = new TGraph();
        g->SetName(buf);

        g->SetPoint(0, de_x_mean - n_si * de_x_sigma, de_y_mean - n_si * de_y_sigma);
        g->SetPoint(1, de_x_mean + n_si * de_x_sigma, de_y_mean - n_si * de_y_sigma);
        g->SetPoint(2, de_x_mean + n_si * de_x_sigma, de_y_mean + n_si * de_y_sigma);
        g->SetPoint(3, de_x_mean - n_si * de_x_sigma, de_y_mean + n_si * de_y_sigma);
        g->SetPoint(4, de_x_mean - n_si * de_x_sigma, de_y_mean - n_si * de_y_sigma);

        g->Write();
      }

      TDirectory *d_top = gDirectory;

      for (const auto &nep_p : effPlots) {
        if (nep_p.first == 0)
          sprintf(buf, "exp prot all");
        else
          sprintf(buf, "exp prot %u", nep_p.first);

        TDirectory *d_nep = d_top->mkdir(buf);

        for (const auto &nsi_p : nep_p.second) {
          sprintf(buf, "nsi = %.1f", n_sigmas[nsi_p.first]);

          TDirectory *d_nsi = d_nep->mkdir(buf);

          gDirectory = d_nsi;

          nsi_p.second.write();
        }
      }

      gDirectory = d_top;
    }

    void UpdateOptics(const LHCInterpolatedOpticalFunctionsSetCollection &ofc) {
      const LHCInterpolatedOpticalFunctionsSet *of = nullptr;

      for (const auto &p : ofc) {
        CTPPSDetId rpId(p.first);
        unsigned int decRPId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

        if (decRPId == rpId_N) {
          of = &p.second;
          break;
        }
      }

      if (!of) {
        edm::LogError("ArmData::UpdateOptics") << "Cannot find optical functions for RP " << rpId_N;
        return;
      }

      std::vector<double> xiValues =
          of->getXiValues();  // local copy made since the TSpline constructor needs non-const parameters
      std::vector<double> xDValues = of->getFcnValues()[LHCOpticalFunctionsSet::exd];
      s_x_to_xi_N = std::make_shared<TSpline3>("", xDValues.data(), xiValues.data(), xiValues.size());
    }
  };

  std::map<unsigned int, ArmData> data_;

  std::unique_ptr<TF1> ff_;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSProtonReconstructionEfficiencyEstimatorData::CTPPSProtonReconstructionEfficiencyEstimatorData(
    const edm::ParameterSet &iConfig)
    : tokenTracks_(consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("tagTracks"))),

      tokenRecoProtonsMultiRP_(
          consumes<reco::ForwardProtonCollection>(iConfig.getParameter<InputTag>("tagRecoProtonsMultiRP"))),

      opticsESToken_(esConsumes(ESInputTag("", iConfig.getParameter<std::string>("opticsLabel")))),

      pixelDiscardBXShiftedTracks_(iConfig.getParameter<bool>("pixelDiscardBXShiftedTracks")),

      localAngleXMin_(iConfig.getParameter<double>("localAngleXMin")),
      localAngleXMax_(iConfig.getParameter<double>("localAngleXMax")),
      localAngleYMin_(iConfig.getParameter<double>("localAngleYMin")),
      localAngleYMax_(iConfig.getParameter<double>("localAngleYMax")),

      n_prep_events_(iConfig.getParameter<unsigned int>("n_prep_events")),
      n_exp_prot_max_(iConfig.getParameter<unsigned int>("n_exp_prot_max")),
      n_sigmas_(iConfig.getParameter<std::vector<double>>("n_sigmas")),

      outputFile_(iConfig.getParameter<string>("outputFile")),

      verbosity_(iConfig.getUntrackedParameter<unsigned int>("verbosity")),

      ff_(new TF1("ff", "[0] * exp(- (x-[1])*(x-[1]) / 2 / [2]/[2]) + [4]")) {
  data_[0].n_exp_prot_max = n_exp_prot_max_;
  data_[0].n_sigmas = n_sigmas_;
  data_[0].rpId_N = iConfig.getParameter<unsigned int>("rpId_45_N");
  data_[0].rpId_F = iConfig.getParameter<unsigned int>("rpId_45_F");

  data_[1].n_exp_prot_max = n_exp_prot_max_;
  data_[1].n_sigmas = n_sigmas_;
  data_[1].rpId_N = iConfig.getParameter<unsigned int>("rpId_56_N");
  data_[1].rpId_F = iConfig.getParameter<unsigned int>("rpId_56_F");
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionEfficiencyEstimatorData::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("tagTracks", edm::InputTag())->setComment("input tag for local lite tracks");
  desc.add<edm::InputTag>("tagRecoProtonsMultiRP", edm::InputTag())->setComment("input tag for multi-RP reco protons");

  desc.add<bool>("pixelDiscardBXShiftedTracks", false)
      ->setComment("whether to discard pixel tracks built from BX-shifted planes");

  desc.add<double>("localAngleXMin", -0.03)->setComment("minimal accepted value of local horizontal angle (rad)");
  desc.add<double>("localAngleXMax", +0.03)->setComment("maximal accepted value of local horizontal angle (rad)");
  desc.add<double>("localAngleYMin", -0.04)->setComment("minimal accepted value of local vertical angle (rad)");
  desc.add<double>("localAngleYMax", +0.04)->setComment("maximal accepted value of local vertical angle (rad)");

  desc.add<std::string>("opticsLabel", "")->setComment("label of the optics records");

  desc.add<unsigned int>("n_prep_events", 1000)
      ->setComment("number of preparatory events (to determine de x and de y window)");

  desc.add<unsigned int>("n_exp_prot_max", 5)->setComment("maximum number of expected protons per event and per arm");

  desc.add<std::vector<double>>("n_sigmas", {3., 5., 7.})->setComment("list of n_sigma values");

  desc.add<unsigned int>("rpId_45_N", 0)->setComment("decimal RP id for 45 near");
  desc.add<unsigned int>("rpId_45_F", 0)->setComment("decimal RP id for 45 far");
  desc.add<unsigned int>("rpId_56_N", 0)->setComment("decimal RP id for 56 near");
  desc.add<unsigned int>("rpId_56_F", 0)->setComment("decimal RP id for 56 far");

  desc.add<std::string>("outputFile", "output.root")->setComment("output file name");

  desc.addUntracked<unsigned int>("verbosity", 0)->setComment("verbosity level");

  descriptions.add("ctppsProtonReconstructionEfficiencyEstimatorData", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionEfficiencyEstimatorData::analyze(const edm::Event &iEvent,
                                                               const edm::EventSetup &iSetup) {
  std::ostringstream os;

  // get conditions
  const auto &opticalFunctions = iSetup.getData(opticsESToken_);

  // check optics change
  if (opticsWatcher_.check(iSetup)) {
    data_[0].UpdateOptics(opticalFunctions);
    data_[1].UpdateOptics(opticalFunctions);
  }

  // get input
  edm::Handle<CTPPSLocalTrackLiteCollection> hTracks;
  iEvent.getByToken(tokenTracks_, hTracks);

  Handle<reco::ForwardProtonCollection> hRecoProtonsMultiRP;
  iEvent.getByToken(tokenRecoProtonsMultiRP_, hRecoProtonsMultiRP);

  // pre-selection of tracks
  std::vector<unsigned int> sel_track_idc;
  for (unsigned int idx = 0; idx < hTracks->size(); ++idx) {
    const auto &tr = hTracks->at(idx);

    if (tr.tx() < localAngleXMin_ || tr.tx() > localAngleXMax_ || tr.ty() < localAngleYMin_ ||
        tr.ty() > localAngleYMax_)
      continue;

    if (pixelDiscardBXShiftedTracks_) {
      if (tr.pixelTrackRecoInfo() == CTPPSpixelLocalTrackReconstructionInfo::allShiftedPlanes ||
          tr.pixelTrackRecoInfo() == CTPPSpixelLocalTrackReconstructionInfo::mixedPlanes)
        continue;
    }

    sel_track_idc.push_back(idx);
  }

  // debug print
  if (verbosity_ > 1) {
    os << "* tracks (pre-selected):" << std::endl;

    for (const auto idx : sel_track_idc) {
      const auto &tr = hTracks->at(idx);
      CTPPSDetId rpId(tr.rpId());
      unsigned int decRPId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

      os << "    [" << idx << "] RP=" << decRPId << ", x=" << tr.x() << ", y=" << tr.y() << std::endl;
    }

    os << "* protons:" << std::endl;
    for (unsigned int i = 0; i < hRecoProtonsMultiRP->size(); ++i) {
      const auto &pr = (*hRecoProtonsMultiRP)[i];
      os << "    [" << i << "] track indices: ";
      for (const auto &trr : pr.contributingLocalTracks())
        os << trr.key() << ", ";
      os << std::endl;
    }
  }

  // make de_x and de_y plots
  map<unsigned int, bool> hasTracksInArm;

  for (const auto idx_i : sel_track_idc) {
    const auto &tr_i = hTracks->at(idx_i);
    CTPPSDetId rpId_i(tr_i.rpId());
    unsigned int decRPId_i = rpId_i.arm() * 100 + rpId_i.station() * 10 + rpId_i.rp();

    for (const auto idx_j : sel_track_idc) {
      const auto &tr_j = hTracks->at(idx_j);
      CTPPSDetId rpId_j(tr_j.rpId());
      unsigned int decRPId_j = rpId_j.arm() * 100 + rpId_j.station() * 10 + rpId_j.rp();

      // check whether desired RP combination
      unsigned int arm = 123;
      for (const auto &ap : data_) {
        if (ap.second.rpId_N == decRPId_i && ap.second.rpId_F == decRPId_j)
          arm = ap.first;
      }

      if (arm > 1)
        continue;

      // update flag
      hasTracksInArm[arm] = true;

      // update plots
      auto &ad = data_[arm];
      const double de_x = tr_j.x() - tr_i.x();
      const double de_y = tr_j.y() - tr_i.y();

      if (ad.n_events_with_tracks < n_prep_events_) {
        ad.h_de_x->Fill(de_x);
        ad.h_de_y->Fill(de_y);
      }

      ad.h2_de_y_vs_de_x->Fill(de_x, de_y);
    }
  }

  // update event counter
  for (auto &ap : data_) {
    if (hasTracksInArm[ap.first])
      ap.second.n_events_with_tracks++;
  }

  // if threshold reached do fits
  for (auto &ap : data_) {
    auto &ad = ap.second;

    if (ad.n_events_with_tracks == n_prep_events_) {
      if (ad.de_x_sigma > 0. && ad.de_y_sigma > 0.)
        continue;

      std::vector<std::pair<unsigned int, TH1D *>> m;
      m.emplace_back(0, ad.h_de_x.get());
      m.emplace_back(1, ad.h_de_y.get());

      for (const auto &p : m) {
        double max_pos = -1E100, max_val = -1.;
        for (int bi = 1; bi < p.second->GetNbinsX(); ++bi) {
          const double pos = p.second->GetBinCenter(bi);
          const double val = p.second->GetBinContent(bi);

          if (val > max_val) {
            max_val = val;
            max_pos = pos;
          }
        }

        const double sig = 0.2;

        ff_->SetParameters(max_val, max_pos, sig, 0.);
        p.second->Fit(ff_.get(), "Q", "", max_pos - 3. * sig, max_pos + 3. * sig);
        p.second->Fit(ff_.get(), "Q", "", max_pos - 3. * sig, max_pos + 3. * sig);

        if (p.first == 0) {
          ad.de_x_mean = ff_->GetParameter(1);
          ad.de_x_sigma = fabs(ff_->GetParameter(2));
        }
        if (p.first == 1) {
          ad.de_y_mean = ff_->GetParameter(1);
          ad.de_y_sigma = fabs(ff_->GetParameter(2));
        }
      }

      if (verbosity_) {
        os << "* fitting arm " << ap.first << std::endl
           << "    de_x: mean = " << ad.de_x_mean << ", sigma = " << ad.de_x_sigma << std::endl
           << "    de_y: mean = " << ad.de_y_mean << ", sigma = " << ad.de_y_sigma;
      }
    }
  }

  // data structures for efficiency analysis
  struct ArmEventData {
    std::map<unsigned int, std::set<unsigned int>> matched_track_idc;

    std::set<unsigned int> reco_proton_idc;

    std::map<unsigned int, std::set<unsigned int>> matched_track_with_prot_idc, matched_track_without_prot_idc;
  };

  std::map<unsigned int, ArmEventData> armEventData;

  // determine the number of expected protons
  for (const auto idx_i : sel_track_idc) {
    const auto &tr_i = hTracks->at(idx_i);
    CTPPSDetId rpId_i(tr_i.rpId());
    unsigned int decRPId_i = rpId_i.arm() * 100 + rpId_i.station() * 10 + rpId_i.rp();

    for (const auto idx_j : sel_track_idc) {
      const auto &tr_j = hTracks->at(idx_j);
      CTPPSDetId rpId_j(tr_j.rpId());
      unsigned int decRPId_j = rpId_j.arm() * 100 + rpId_j.station() * 10 + rpId_j.rp();

      // check whether desired RP combination
      unsigned int arm = 123;
      for (const auto &ap : data_) {
        if (ap.second.rpId_N == decRPId_i && ap.second.rpId_F == decRPId_j)
          arm = ap.first;
      }

      if (arm > 1)
        continue;

      // check near-far matching
      auto &ad = data_[arm];
      const double de_x = tr_j.x() - tr_i.x();
      const double de_y = tr_j.y() - tr_i.y();

      for (unsigned int nsi = 0; nsi < ad.n_sigmas.size(); ++nsi) {
        const double n_si = ad.n_sigmas[nsi];
        const bool match_x = fabs(de_x - ad.de_x_mean) < n_si * ad.de_x_sigma;
        const bool match_y = fabs(de_y - ad.de_y_mean) < n_si * ad.de_y_sigma;
        if (match_x && match_y)
          armEventData[arm].matched_track_idc[nsi].insert(idx_i);
      }
    }
  }

  // determine the number of reconstructed protons
  for (unsigned int i = 0; i < hRecoProtonsMultiRP->size(); ++i) {
    const auto &proton = (*hRecoProtonsMultiRP)[i];

    CTPPSDetId rpId((*proton.contributingLocalTracks().begin())->rpId());
    unsigned int arm = rpId.arm();

    if (proton.validFit())
      armEventData[arm].reco_proton_idc.insert(i);
  }

  // compare matched tracks with reco protons
  if (verbosity_ > 1)
    os << "* cmp matched tracks vs. reco protons" << std::endl;

  for (auto &ap : armEventData) {
    auto &ad = data_[ap.first];

    if (verbosity_ > 1)
      os << "    arm " << ap.first << std::endl;

    for (unsigned int nsi = 0; nsi < ad.n_sigmas.size(); ++nsi) {
      if (verbosity_ > 1)
        os << "        nsi = " << nsi << std::endl;

      for (const auto &tri : ap.second.matched_track_idc[nsi]) {
        const auto &track = hTracks->at(tri);

        bool some_proton_matching = false;

        if (verbosity_ > 1)
          os << "            tri = " << tri << std::endl;

        for (const auto &pri : ap.second.reco_proton_idc) {
          const auto &proton = (*hRecoProtonsMultiRP)[pri];

          bool proton_matching = false;

          for (const auto &pr_tr : proton.contributingLocalTracks()) {
            CTPPSDetId rpId(pr_tr->rpId());
            unsigned int decRPId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

            if (decRPId != ad.rpId_N)
              continue;

            const double x = pr_tr->x();
            const double y = pr_tr->y();
            const double th = 1E-3;  // 1 um

            const bool match = (fabs(x - track.x()) < th) && (fabs(y - track.y()) < th);

            if (verbosity_ > 1)
              os << "                pri = " << pri << ": x_tr = " << track.x() << ", x_pr = " << x
                 << ", match = " << match << std::endl;

            if (match) {
              proton_matching = true;
              break;
            }
          }

          if (proton_matching) {
            some_proton_matching = true;
            break;
          }
        }

        if (verbosity_ > 1)
          os << "            --> some_proton_matching " << some_proton_matching << std::endl;

        if (some_proton_matching)
          ap.second.matched_track_with_prot_idc[nsi].insert(tri);
        else
          ap.second.matched_track_without_prot_idc[nsi].insert(tri);
      }
    }
  }

  // debug print
  if (verbosity_ > 1) {
    for (auto &ap : armEventData) {
      auto &ad = data_[ap.first];

      if (ad.de_x_sigma <= 0. && ad.de_y_sigma <= 0.)
        continue;

      os << "* results for arm " << ap.first << std::endl;

      os << "    reco_proton_idc: ";
      for (const auto &idx : ap.second.reco_proton_idc)
        os << idx << ", ";
      os << std::endl;

      for (unsigned int nsi = 0; nsi < ad.n_sigmas.size(); ++nsi) {
        os << "    n_si = " << ad.n_sigmas[nsi] << std::endl;

        os << "        matched_track_idc: ";
        for (const auto &idx : ap.second.matched_track_idc[nsi])
          os << idx << ", ";
        os << std::endl;

        os << "        matched_track_with_prot_idc: ";
        for (const auto &idx : ap.second.matched_track_with_prot_idc[nsi])
          os << idx << ", ";
        os << std::endl;

        os << "        matched_track_without_prot_idc: ";
        for (const auto &idx : ap.second.matched_track_without_prot_idc[nsi])
          os << idx << ", ";
        os << std::endl;
      }
    }
  }

  // update efficiency plots
  for (auto &ap : armEventData) {
    auto &ad = data_[ap.first];

    // stop if sigmas not yet determined
    if (ad.de_x_sigma <= 0. && ad.de_y_sigma <= 0.)
      continue;

    // loop over n_sigma choices
    for (unsigned int nsi = 0; nsi < ad.n_sigmas.size(); ++nsi) {
      const unsigned int n_exp_prot = ap.second.matched_track_idc[nsi].size();
      const unsigned int n_rec_prot = ap.second.reco_proton_idc.size();

      // stop if N(expected protons) out of range
      if (n_exp_prot < 1 || n_exp_prot > ad.n_exp_prot_max)
        continue;

      // update method 1 plots
      const double eff = double(n_rec_prot) / n_exp_prot;

      for (unsigned int tri : ap.second.matched_track_idc[nsi]) {
        const double x_N = hTracks->at(tri).x();
        const double xi_N = ad.s_x_to_xi_N->Eval(x_N * 1E-1);  // conversion mm to cm

        ad.effPlots[0][nsi].p_eff1_vs_x_N->Fill(x_N, eff);
        ad.effPlots[0][nsi].p_eff1_vs_xi_N->Fill(xi_N, eff);

        ad.effPlots[n_exp_prot][nsi].p_eff1_vs_x_N->Fill(x_N, eff);
        ad.effPlots[n_exp_prot][nsi].p_eff1_vs_xi_N->Fill(xi_N, eff);
      }

      // update method 2 plots
      for (const auto &tri : ap.second.matched_track_with_prot_idc[nsi]) {
        const double x_N = hTracks->at(tri).x();
        const double xi_N = ad.s_x_to_xi_N->Eval(x_N * 1E-1);  // conversion mm to cm

        ad.effPlots[0][nsi].p_eff2_vs_x_N->Fill(x_N, 1.);
        ad.effPlots[0][nsi].p_eff2_vs_xi_N->Fill(xi_N, 1.);

        ad.effPlots[n_exp_prot][nsi].p_eff2_vs_x_N->Fill(x_N, 1.);
        ad.effPlots[n_exp_prot][nsi].p_eff2_vs_xi_N->Fill(xi_N, 1.);
      }

      for (const auto &tri : ap.second.matched_track_without_prot_idc[nsi]) {
        const double x_N = hTracks->at(tri).x();
        const double xi_N = ad.s_x_to_xi_N->Eval(x_N * 1E-1);  // conversion mm to cm

        ad.effPlots[0][nsi].p_eff2_vs_x_N->Fill(x_N, 0.);
        ad.effPlots[0][nsi].p_eff2_vs_xi_N->Fill(xi_N, 0.);

        ad.effPlots[n_exp_prot][nsi].p_eff2_vs_x_N->Fill(x_N, 0.);
        ad.effPlots[n_exp_prot][nsi].p_eff2_vs_xi_N->Fill(xi_N, 0.);
      }
    }
  }

  if (verbosity_)
    edm::LogInfo("CTPPSProtonReconstructionEfficiencyEstimatorData") << os.str();
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionEfficiencyEstimatorData::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  for (const auto &ait : data_) {
    char buf[100];
    sprintf(buf, "arm %u", ait.first);
    TDirectory *d_arm = f_out->mkdir(buf);
    gDirectory = d_arm;

    ait.second.write();
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstructionEfficiencyEstimatorData);
