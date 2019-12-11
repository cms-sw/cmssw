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
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TGraphErrors.h"

#include <map>
#include <string>

//----------------------------------------------------------------------------------------------------

class CTPPSProtonReconstructionEfficiencyEstimator : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSProtonReconstructionEfficiencyEstimator(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  edm::EDGetTokenT<edm::HepMCProduct> tokenHepMCAfterSmearing_;

  edm::EDGetTokenT<std::map<int, edm::DetSetVector<TotemRPRecHit>>> tokenStripRecHitsPerParticle_;
  edm::EDGetTokenT<std::map<int, edm::DetSetVector<CTPPSPixelRecHit>>> tokenPixelRecHitsPerParticle_;

  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tracksToken_;

  edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsMultiRP_;

  std::string lhcInfoLabel_;

  unsigned int rpId_45_N_, rpId_45_F_;
  unsigned int rpId_56_N_, rpId_56_F_;

  std::map<unsigned int, unsigned int> rpDecId_near_, rpDecId_far_;

  std::string outputFile_;

  struct PlotGroup {
    std::unique_ptr<TProfile> p_eff_vs_xi;

    std::unique_ptr<TH1D> h_n_part_acc_nr, h_n_part_acc_fr;

    PlotGroup()
        : p_eff_vs_xi(new TProfile("", ";#xi_{simu};efficiency", 19, 0.015, 0.205)),
          h_n_part_acc_nr(new TH1D("", ";n particles in acceptance", 6, -0.5, +5.5)),
          h_n_part_acc_fr(new TH1D("", ";n particles in acceptance", 6, -0.5, +5.5)) {}

    void write() const {
      p_eff_vs_xi->Write("p_eff_vs_xi");
      h_n_part_acc_nr->Write("h_n_part_acc_nr");
      h_n_part_acc_fr->Write("h_n_part_acc_fr");
    }
  };

  std::map<unsigned int, std::map<unsigned int, PlotGroup>> plots_;  // map: arm, n(particles in acceptance) --> plots
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;
using namespace HepMC;

//----------------------------------------------------------------------------------------------------

CTPPSProtonReconstructionEfficiencyEstimator::CTPPSProtonReconstructionEfficiencyEstimator(
    const edm::ParameterSet &iConfig)
    : tokenHepMCAfterSmearing_(
          consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("tagHepMCAfterSmearing"))),
      tokenStripRecHitsPerParticle_(consumes<std::map<int, edm::DetSetVector<TotemRPRecHit>>>(
          iConfig.getParameter<edm::InputTag>("tagStripRecHitsPerParticle"))),
      tokenPixelRecHitsPerParticle_(consumes<std::map<int, edm::DetSetVector<CTPPSPixelRecHit>>>(
          iConfig.getParameter<edm::InputTag>("tagPixelRecHitsPerParticle"))),
      tracksToken_(consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("tagTracks"))),
      tokenRecoProtonsMultiRP_(
          consumes<reco::ForwardProtonCollection>(iConfig.getParameter<InputTag>("tagRecoProtonsMultiRP"))),
      lhcInfoLabel_(iConfig.getParameter<std::string>("lhcInfoLabel")),

      rpId_45_N_(iConfig.getParameter<unsigned int>("rpId_45_N")),
      rpId_45_F_(iConfig.getParameter<unsigned int>("rpId_45_F")),
      rpId_56_N_(iConfig.getParameter<unsigned int>("rpId_56_N")),
      rpId_56_F_(iConfig.getParameter<unsigned int>("rpId_56_F")),

      outputFile_(iConfig.getParameter<string>("outputFile")) {
  rpDecId_near_[0] = rpId_45_N_;
  rpDecId_far_[0] = rpId_45_F_;

  rpDecId_near_[1] = rpId_56_N_;
  rpDecId_far_[1] = rpId_56_F_;

  // book plots
  for (unsigned int arm = 0; arm < 2; ++arm) {
    for (unsigned int np = 1; np <= 5; ++np)
      plots_[arm][np] = PlotGroup();
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionEfficiencyEstimator::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  bool verbosity = false;

  const auto eid = iEvent.id().event();
  //if (eid == 46 || eid == 741 || eid == 1649 || eid == 4223 || eid == 4279)
  //  verbosity = true;

  if (verbosity) {
    printf("--------------------------------------------------\n");
    printf("event %llu\n", eid);
  }

  // get conditions
  edm::ESHandle<LHCInfo> hLHCInfo;
  iSetup.get<LHCInfoRcd>().get(lhcInfoLabel_, hLHCInfo);

  // get input
  edm::Handle<edm::HepMCProduct> hHepMCAfterSmearing;
  iEvent.getByToken(tokenHepMCAfterSmearing_, hHepMCAfterSmearing);
  HepMC::GenEvent *hepMCEventAfterSmearing = (HepMC::GenEvent *)hHepMCAfterSmearing->GetEvent();

  edm::Handle<std::map<int, edm::DetSetVector<TotemRPRecHit>>> hStripRecHitsPerParticle;
  iEvent.getByToken(tokenStripRecHitsPerParticle_, hStripRecHitsPerParticle);

  edm::Handle<std::map<int, edm::DetSetVector<CTPPSPixelRecHit>>> hPixelRecHitsPerParticle;
  iEvent.getByToken(tokenPixelRecHitsPerParticle_, hPixelRecHitsPerParticle);

  edm::Handle<CTPPSLocalTrackLiteCollection> tracks;
  iEvent.getByToken(tracksToken_, tracks);

  Handle<reco::ForwardProtonCollection> hRecoProtonsMultiRP;
  iEvent.getByToken(tokenRecoProtonsMultiRP_, hRecoProtonsMultiRP);

  // buffer for particle information
  struct ParticleInfo {
    unsigned int arm = 2;
    double xi = 0.;
    std::map<unsigned int, unsigned int> recHitsPerRP;
    bool inAcceptanceNear = false, inAcceptanceFar = false, inAcceptance = false;
  };

  std::map<int, ParticleInfo> particleInfo;  // barcode --> info

  // process HepMC
  for (auto it = hepMCEventAfterSmearing->particles_begin(); it != hepMCEventAfterSmearing->particles_end(); ++it) {
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

    ParticleInfo info;

    info.arm = (mom.z() > 0.) ? 0 : 1;

    const double p_nom = hLHCInfo->energy();
    info.xi = (p_nom - mom.rho()) / p_nom;

    particleInfo[part->barcode()] = std::move(info);
  }

  // check acceptance
  for (const auto &pp : *hStripRecHitsPerParticle) {
    const auto barcode = pp.first;

    for (const auto &ds : pp.second) {
      CTPPSDetId detId(ds.detId());
      CTPPSDetId rpId = detId.rpId();
      particleInfo[barcode].recHitsPerRP[rpId] += ds.size();
    }
  }

  for (const auto &pp : *hPixelRecHitsPerParticle) {
    const auto barcode = pp.first;

    for (const auto &ds : pp.second) {
      CTPPSDetId detId(ds.detId());
      CTPPSDetId rpId = detId.rpId();
      particleInfo[barcode].recHitsPerRP[rpId] += ds.size();
    }
  }

  std::map<unsigned int, bool> isStripRPNear, isStripRPFar;

  for (auto &pp : particleInfo) {
    if (verbosity)
      printf("* barcode=%i, arm=%u, xi=%.3f\n", pp.first, pp.second.arm, pp.second.xi);

    for (const auto &rpp : pp.second.recHitsPerRP) {
      CTPPSDetId rpId(rpp.first);
      unsigned int needed_rec_hits = 1000;
      if (rpId.subdetId() == CTPPSDetId::sdTrackingStrip)
        needed_rec_hits = 6;
      if (rpId.subdetId() == CTPPSDetId::sdTrackingPixel)
        needed_rec_hits = 3;

      unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

      if (rpId.subdetId() == CTPPSDetId::sdTrackingStrip) {
        if (rpDecId == rpDecId_near_[rpId.arm()])
          isStripRPNear[rpId.arm()] = true;
        if (rpDecId == rpDecId_far_[rpId.arm()])
          isStripRPFar[rpId.arm()] = true;
      }

      if (rpp.second >= needed_rec_hits) {
        if (rpDecId == rpDecId_near_[rpId.arm()])
          pp.second.inAcceptanceNear = true;
        if (rpDecId == rpDecId_far_[rpId.arm()])
          pp.second.inAcceptanceFar = true;
      }

      if (verbosity)
        printf("    RP %u: %u hits\n", rpDecId, rpp.second);
    }

    pp.second.inAcceptance = pp.second.inAcceptanceNear && pp.second.inAcceptanceFar;

    if (verbosity) {
      printf("    inAcceptance: near=%u, far=%u, global=%u\n",
             pp.second.inAcceptanceNear,
             pp.second.inAcceptanceFar,
             pp.second.inAcceptance);
    }
  }

  // count particles in acceptance
  struct ArmCounter {
    unsigned int near = 0, far = 0, global = 0;
  };
  std::map<unsigned int, ArmCounter> nParticlesInAcceptance;
  for (auto &pp : particleInfo) {
    auto &counter = nParticlesInAcceptance[pp.second.arm];
    if (pp.second.inAcceptanceNear)
      counter.near++;
    if (pp.second.inAcceptanceFar)
      counter.far++;
    if (pp.second.inAcceptance)
      counter.global++;
  }

  // count reconstructed tracks
  std::map<unsigned int, ArmCounter> nReconstructedTracks;
  for (const auto &tr : *tracks) {
    CTPPSDetId rpId(tr.rpId());
    unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

    if (rpDecId == rpDecId_near_[rpId.arm()])
      nReconstructedTracks[rpId.arm()].near++;
    if (rpDecId == rpDecId_far_[rpId.arm()])
      nReconstructedTracks[rpId.arm()].far++;
  }

  // count reconstructed protons
  std::map<unsigned int, unsigned int> nReconstructedProtons;
  for (const auto &pr : *hRecoProtonsMultiRP) {
    if (!pr.validFit())
      continue;

    unsigned int arm = 2;
    if (pr.lhcSector() == reco::ForwardProton::LHCSector::sector45)
      arm = 0;
    if (pr.lhcSector() == reco::ForwardProton::LHCSector::sector56)
      arm = 1;

    nReconstructedProtons[arm]++;
  }

  // fill plots
  for (unsigned int arm = 0; arm < 2; arm++) {
    const auto &npa = nParticlesInAcceptance[arm];
    const auto &nrt = nReconstructedTracks[arm];

    if (verbosity) {
      printf("* arm %u: nRecoProtons=%u (tracks near=%u, far=%u), nAcc=%u (near=%u, far=%u)\n",
             arm,
             nReconstructedProtons[arm],
             nReconstructedTracks[arm].near,
             nReconstructedTracks[arm].far,
             npa.global,
             npa.near,
             npa.far);
    }

    // skip event if no track in global acceptance
    if (npa.global < 1)
      continue;

    const auto &p = plots_[arm][npa.global];

    p.h_n_part_acc_nr->Fill(npa.near);
    p.h_n_part_acc_fr->Fill(npa.far);

    // skip events with some local reconstruction inefficiency
    if (nrt.near != npa.near || nrt.far != npa.far)
      continue;

    const double eff = double(nReconstructedProtons[arm]) / npa.global;

    if (verbosity)
      printf("    eff=%.3f\n", eff);

    for (auto &pp : particleInfo) {
      if (pp.second.arm != arm || !pp.second.inAcceptance)
        continue;

      p.p_eff_vs_xi->Fill(pp.second.xi, eff);
    }
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstructionEfficiencyEstimator::endJob() {
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  for (const auto &ait : plots_) {
    char buf[100];
    sprintf(buf, "arm%u", ait.first);
    TDirectory *d_arm = f_out->mkdir(buf);

    for (const auto &npit : ait.second) {
      sprintf(buf, "%u", npit.first);
      TDirectory *d_np = d_arm->mkdir(buf);
      gDirectory = d_np;

      npit.second.write();
    }
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstructionEfficiencyEstimator);
