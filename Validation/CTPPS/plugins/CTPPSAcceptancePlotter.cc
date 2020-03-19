/****************************************************************************
 * Authors:
 *   Jan Kašpar
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"

#include <map>
#include <set>
#include <string>

//----------------------------------------------------------------------------------------------------

class CTPPSAcceptancePlotter : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSAcceptancePlotter(const edm::ParameterSet &);

private:
  void beginJob() override {}
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  edm::EDGetTokenT<edm::HepMCProduct> tokenHepMC_;
  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tokenTracks_;

  unsigned int rpId_45_N_, rpId_45_F_, rpId_56_N_, rpId_56_F_;

  std::string outputFile_;

  struct SingleArmPlots {
    std::unique_ptr<TH1D> h_xi_all, h_xi_acc;
    SingleArmPlots() : h_xi_all(new TH1D("", ";#xi", 100, 0., 0.25)), h_xi_acc(new TH1D("", ";#xi", 100, 0., 0.25)) {}

    void fill(double xi, bool acc) {
      h_xi_all->Fill(xi);
      if (acc)
        h_xi_acc->Fill(xi);
    }

    void write() const {
      h_xi_all->Write("h_xi_all");
      h_xi_acc->Write("h_xi_acc");

      auto h_xi_rat = std::make_unique<TH1D>(*h_xi_acc);
      h_xi_rat->Divide(h_xi_all.get());
      h_xi_rat->Write("h_xi_rat");
    }
  };

  std::vector<std::set<unsigned int>> singleArmConfigurations;
  std::map<std::set<unsigned int>, SingleArmPlots> singleArmPlots;

  struct DoubleArmPlots {
    std::unique_ptr<TH1D> h_m_all, h_m_acc;
    std::unique_ptr<TH2D> h2_xi_45_vs_xi_56_all, h2_xi_45_vs_xi_56_acc;
    std::unique_ptr<TH2D> h2_y_vs_m_all, h2_y_vs_m_acc;

    DoubleArmPlots()
        : h_m_all(new TH1D("", ";m   (GeV)", 100, 0., 2500.)),
          h_m_acc(new TH1D("", ";m   (GeV)", 100, 0., 2500.)),
          h2_xi_45_vs_xi_56_all(new TH2D("", ";xi_56;xi_45", 25, 0., 0.25, 25, 0., 0.25)),
          h2_xi_45_vs_xi_56_acc(new TH2D("", ";xi_56;xi_45", 25, 0., 0.25, 25, 0., 0.25)),
          h2_y_vs_m_all(new TH2D("", ";m   (GeV);y", 25, 0., 2500., 25, -1.5, +1.5)),
          h2_y_vs_m_acc(new TH2D("", ";m   (GeV);y", 25, 0., 2500., 25, -1.5, +1.5)) {}

    void fill(double xi_45, double xi_56, bool acc) {
      const double p_nom = 6500.;
      const double m = 2. * p_nom * sqrt(xi_45 * xi_56);
      const double y = log(xi_45 / xi_56) / 2.;

      h_m_all->Fill(m);
      h2_xi_45_vs_xi_56_all->Fill(xi_56, xi_45);
      h2_y_vs_m_all->Fill(m, y);

      if (acc) {
        h_m_acc->Fill(m);
        h2_xi_45_vs_xi_56_acc->Fill(xi_56, xi_45);
        h2_y_vs_m_acc->Fill(m, y);
      }
    }

    void write() const {
      h_m_all->Write("h_m_all");
      h_m_acc->Write("h_m_acc");

      auto h_m_rat = std::make_unique<TH1D>(*h_m_acc);
      h_m_rat->Divide(h_m_all.get());
      h_m_rat->Write("h_m_rat");

      h2_xi_45_vs_xi_56_all->Write("h2_xi_45_vs_xi_56_all");
      h2_xi_45_vs_xi_56_acc->Write("h2_xi_45_vs_xi_56_acc");

      auto h2_xi_45_vs_xi_56_rat = std::make_unique<TH2D>(*h2_xi_45_vs_xi_56_acc);
      h2_xi_45_vs_xi_56_rat->Divide(h2_xi_45_vs_xi_56_all.get());
      h2_xi_45_vs_xi_56_rat->Write("h2_xi_45_vs_xi_56_rat");

      h2_y_vs_m_all->Write("h2_y_vs_m_all");
      h2_y_vs_m_acc->Write("h2_y_vs_m_acc");

      auto h2_y_vs_m_rat = std::make_unique<TH2D>(*h2_y_vs_m_acc);
      h2_y_vs_m_rat->Divide(h2_y_vs_m_all.get());
      h2_y_vs_m_rat->Write("h2_y_vs_m_rat");
    }
  };

  std::vector<std::set<unsigned int>> doubleArmConfigurations;
  std::map<std::set<unsigned int>, DoubleArmPlots> doubleArmPlots;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;
using namespace HepMC;

//----------------------------------------------------------------------------------------------------

CTPPSAcceptancePlotter::CTPPSAcceptancePlotter(const edm::ParameterSet &iConfig)
    : tokenHepMC_(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("tagHepMC"))),
      tokenTracks_(consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("tagTracks"))),
      rpId_45_N_(iConfig.getParameter<unsigned int>("rpId_45_N")),
      rpId_45_F_(iConfig.getParameter<unsigned int>("rpId_45_F")),
      rpId_56_N_(iConfig.getParameter<unsigned int>("rpId_56_N")),
      rpId_56_F_(iConfig.getParameter<unsigned int>("rpId_56_F")),
      outputFile_(iConfig.getParameter<string>("outputFile")) {
  singleArmConfigurations = {
      {rpId_45_N_},
      {rpId_45_F_},
      {rpId_56_N_},
      {rpId_56_F_},
      {rpId_45_N_, rpId_45_F_},
      {rpId_56_N_, rpId_56_F_},
  };

  doubleArmConfigurations = {
      {rpId_45_N_, rpId_56_N_},
      {rpId_45_F_, rpId_56_F_},
      {rpId_45_N_, rpId_45_F_, rpId_56_N_, rpId_56_F_},
  };
}

//----------------------------------------------------------------------------------------------------

void CTPPSAcceptancePlotter::analyze(const edm::Event &iEvent, const edm::EventSetup &) {
  // get input
  edm::Handle<edm::HepMCProduct> hHepMC;
  iEvent.getByToken(tokenHepMC_, hHepMC);
  HepMC::GenEvent *hepMCEvent = (HepMC::GenEvent *)hHepMC->GetEvent();

  edm::Handle<CTPPSLocalTrackLiteCollection> hTracks;
  iEvent.getByToken(tokenTracks_, hTracks);

  // extract protons
  bool proton_45_set = false;
  bool proton_56_set = false;
  FourVector mom_45, mom_56;

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

    if (mom.e() < 4500.)
      continue;

    if (mom.z() > 0) {
      // 45
      if (proton_45_set) {
        LogError("CTPPSAcceptancePlotter") << "Multiple protons found in sector 45.";
        return;
      }

      proton_45_set = true;
      mom_45 = mom;
    } else {
      // 56
      if (proton_56_set) {
        LogError("CTPPSAcceptancePlotter") << "Multiple protons found in sector 56.";
        return;
      }

      proton_56_set = true;
      mom_56 = mom;
    }
  }

  // stop if protons missing
  if (!proton_45_set || !proton_56_set)
    return;

  // calculate xi's
  const double p_nom = 6500.;
  const double xi_45 = (p_nom - mom_45.e()) / p_nom;
  const double xi_56 = (p_nom - mom_56.e()) / p_nom;

  // process tracks
  map<unsigned int, bool> trackPresent;
  for (const auto &trk : *hTracks) {
    CTPPSDetId rpId(trk.rpId());
    unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();
    trackPresent[rpDecId] = true;
  }

  // update plots
  for (const auto rpIds : singleArmConfigurations) {
    bool acc = true;
    signed int arm = -1;
    for (const auto rpId : rpIds) {
      acc &= trackPresent[rpId];
      arm = rpId / 100;
    }

    if (arm < 0)
      continue;

    const double xi = (arm == 0) ? xi_45 : xi_56;

    singleArmPlots[rpIds].fill(xi, acc);
  }

  for (const auto rpIds : doubleArmConfigurations) {
    bool acc = true;
    for (const auto rpId : rpIds)
      acc &= trackPresent[rpId];

    doubleArmPlots[rpIds].fill(xi_45, xi_56, acc);
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSAcceptancePlotter::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  for (const auto &p : singleArmPlots) {
    string dirName;
    for (const auto &rpId : p.first) {
      if (!dirName.empty())
        dirName += ",";
      dirName += Form("%u", rpId);
    }

    gDirectory = f_out->mkdir(dirName.c_str());
    p.second.write();
  }

  for (const auto &p : doubleArmPlots) {
    string dirName;
    for (const auto &rpId : p.first) {
      if (!dirName.empty())
        dirName += ",";
      dirName += Form("%u", rpId);
    }

    gDirectory = f_out->mkdir(dirName.c_str());
    p.second.write();
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSAcceptancePlotter);
