#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Validation/RecoParticleFlow/plugins/PFClusterValidation.h"

PFClusterValidation::PFClusterValidation(const edm::ParameterSet& conf) {
  hepMCTok_ = consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"));

  pfClusterECALTok_ =
      consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pflowClusterECAL"));
  pfClusterHCALTok_ =
      consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pflowClusterHCAL"));
  pfClusterHOTok_ = consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pflowClusterHO"));
  pfClusterHFTok_ = consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pflowClusterHF"));
}

PFClusterValidation::~PFClusterValidation() {}

void PFClusterValidation::bookHistograms(DQMStore::IBooker& ibooker,
                                         edm::Run const& irun,
                                         edm::EventSetup const& isetup) {
  constexpr auto size = 100;
  char histo[size];

  constexpr double etaBinsOffset[] = {
      -5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489, -3.314, -3.139, -2.964, -2.853,
      -2.65,  -2.5,   -2.322, -2.172, -2.043, -1.93,  -1.83,  -1.74,  -1.653, -1.566, -1.479, -1.392, -1.305, -1.218,
      -1.131, -1.044, -0.957, -0.879, -0.783, -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087, 0,
      0.087,  0.174,  0.261,  0.348,  0.435,  0.522,  0.609,  0.696,  0.783,  0.879,  0.957,  1.044,  1.131,  1.218,
      1.305,  1.392,  1.479,  1.566,  1.653,  1.74,   1.83,   1.93,   2.043,  2.172,  2.322,  2.5,    2.65,   2.853,
      2.964,  3.139,  3.314,  3.489,  3.664,  3.839,  4.013,  4.191,  4.363,  4.538,  4.716,  4.889,  5.191};
  constexpr int etaBins = std::size(etaBinsOffset) - 1;

  ibooker.setCurrentFolder("ParticleFlow/PFClusterV");

  // These are the single pion scan histos

  strncpy(histo, "emean_vs_eta_E", size);
  emean_vs_eta_E_ = ibooker.bookProfile(histo, histo, etaBins, etaBinsOffset, -100., 2000., " ");
  strncpy(histo, "emean_vs_eta_H", size);
  emean_vs_eta_H_ = ibooker.bookProfile(histo, histo, etaBins, etaBinsOffset, -100., 2000., " ");
  strncpy(histo, "emean_vs_eta_EH", size);
  emean_vs_eta_EH_ = ibooker.bookProfile(histo, histo, etaBins, etaBinsOffset, -100., 2000., " ");

  strncpy(histo, "emean_vs_eta_HF", size);
  emean_vs_eta_HF_ = ibooker.bookProfile(histo, histo, etaBins, etaBinsOffset, -100., 2000., " ");
  strncpy(histo, "emean_vs_eta_HO", size);
  emean_vs_eta_HO_ = ibooker.bookProfile(histo, histo, etaBins, etaBinsOffset, -100., 2000., " ");

  strncpy(histo, "emean_vs_eta_EHF", size);
  emean_vs_eta_EHF_ = ibooker.bookProfile(histo, histo, etaBins, etaBinsOffset, -100., 2000., " ");
  strncpy(histo, "emean_vs_eta_EHFO", size);
  emean_vs_eta_EHFO_ = ibooker.bookProfile(histo, histo, etaBins, etaBinsOffset, -100., 2000., " ");

  // 1D histos

  strncpy(histo, "Ratio_Esummed_ECAL_0", size);
  ratio_Esummed_ECAL_0_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_HCAL_0", size);
  ratio_Esummed_HCAL_0_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_HO_0", size);
  ratio_Esummed_HO_0_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_ECAL_HCAL_0", size);
  ratio_Esummed_ECAL_HCAL_0_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_ECAL_HCAL_HO_0", size);
  ratio_Esummed_ECAL_HCAL_HO_0_ = ibooker.book1D(histo, histo, 50, 0., 5.);

  strncpy(histo, "Ratio_Esummed_ECAL_1", size);
  ratio_Esummed_ECAL_1_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_HCAL_1", size);
  ratio_Esummed_HCAL_1_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_HO_1", size);
  ratio_Esummed_HO_1_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_ECAL_HCAL_1", size);
  ratio_Esummed_ECAL_HCAL_1_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_ECAL_HCAL_HO_1", size);
  ratio_Esummed_ECAL_HCAL_HO_1_ = ibooker.book1D(histo, histo, 50, 0., 5.);

  strncpy(histo, "Ratio_Esummed_ECAL_2", size);
  ratio_Esummed_ECAL_2_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_HCAL_2", size);
  ratio_Esummed_HCAL_2_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_HO_2", size);
  ratio_Esummed_HO_2_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_ECAL_HCAL_2", size);
  ratio_Esummed_ECAL_HCAL_2_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_ECAL_HCAL_HO_2", size);
  ratio_Esummed_ECAL_HCAL_HO_2_ = ibooker.book1D(histo, histo, 50, 0., 5.);

  strncpy(histo, "Ratio_Esummed_ECAL_3", size);
  ratio_Esummed_ECAL_3_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_HCAL_3", size);
  ratio_Esummed_HCAL_3_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_HO_3", size);
  ratio_Esummed_HO_3_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_ECAL_HCAL_3", size);
  ratio_Esummed_ECAL_HCAL_3_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_ECAL_HCAL_HO_3", size);
  ratio_Esummed_ECAL_HCAL_HO_3_ = ibooker.book1D(histo, histo, 50, 0., 5.);

  strncpy(histo, "Ratio_Esummed_ECAL_4", size);
  ratio_Esummed_ECAL_4_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_HCAL_4", size);
  ratio_Esummed_HCAL_4_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_HO_4", size);
  ratio_Esummed_HO_4_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_ECAL_HCAL_4", size);
  ratio_Esummed_ECAL_HCAL_4_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_ECAL_HCAL_HO_4", size);
  ratio_Esummed_ECAL_HCAL_HO_4_ = ibooker.book1D(histo, histo, 50, 0., 5.);

  strncpy(histo, "Ratio_Esummed_HF_5", size);
  ratio_Esummed_HF_5_ = ibooker.book1D(histo, histo, 50, 0., 5.);
  strncpy(histo, "Ratio_Esummed_HF_6", size);
  ratio_Esummed_HF_6_ = ibooker.book1D(histo, histo, 50, 0., 5.);

  strncpy(histo, "Egen_MC", size);
  egen_MC_ = ibooker.book1D(histo, histo, 50, 0, 50);
}

void PFClusterValidation::analyze(edm::Event const& event, edm::EventSetup const& c) {
  double eta_MC = 0.;
  double phi_MC = 0.;
  double energy_MC = 0.;

  edm::Handle<edm::HepMCProduct> hepMC;
  event.getByToken(hepMCTok_, hepMC);
  if (not hepMC.isValid()) {
    edm::LogWarning("PFClusterValidation") << "HepMCProduct not found";
    return;
  }

  // MC particle with highest pt is taken as a direction reference
  double maxPt = -99999.;
  const HepMC::GenEvent* myGenEvent = hepMC->GetEvent();
  for (auto p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p) {
    double phip = (*p)->momentum().phi();
    double etap = (*p)->momentum().eta();
    double pt = (*p)->momentum().perp();
    double energy = (*p)->momentum().e();
    if (pt > maxPt) {
      maxPt = pt;
      energy_MC = energy;
      phi_MC = phip;
      eta_MC = etap;
    }
  }

  egen_MC_->Fill(energy_MC);

  edm::Handle<reco::PFClusterCollection> pfClusterECAL;
  event.getByToken(pfClusterECALTok_, pfClusterECAL);

  edm::Handle<reco::PFClusterCollection> pfClusterHCAL;
  event.getByToken(pfClusterHCALTok_, pfClusterHCAL);

  edm::Handle<reco::PFClusterCollection> pfClusterHO;
  event.getByToken(pfClusterHOTok_, pfClusterHO);

  edm::Handle<reco::PFClusterCollection> pfClusterHF;
  event.getByToken(pfClusterHFTok_, pfClusterHF);

  // sum the energy in a dR cone for each subsystem
  const double Econe = sumEnergy(pfClusterECAL, eta_MC, phi_MC);
  const double Hcone = sumEnergy(pfClusterHCAL, eta_MC, phi_MC);
  const double HOcone = sumEnergy(pfClusterHO, eta_MC, phi_MC);
  const double HFcone = sumEnergy(pfClusterHF, eta_MC, phi_MC);

  if (energy_MC > 0.) {
    if (std::abs(eta_MC) < 0.5) {
      ratio_Esummed_ECAL_0_->Fill(Econe / energy_MC);
      ratio_Esummed_HCAL_0_->Fill(Hcone / energy_MC);
      ratio_Esummed_HO_0_->Fill(HOcone / energy_MC);
      ratio_Esummed_ECAL_HCAL_0_->Fill((Econe + Hcone) / energy_MC);
      ratio_Esummed_ECAL_HCAL_HO_0_->Fill((Econe + Hcone + HOcone) / energy_MC);
    } else if (std::abs(eta_MC) < 1.3 && std::abs(eta_MC) > 0.5) {
      ratio_Esummed_ECAL_1_->Fill(Econe / energy_MC);
      ratio_Esummed_HCAL_1_->Fill(Hcone / energy_MC);
      ratio_Esummed_HO_1_->Fill(HOcone / energy_MC);
      ratio_Esummed_ECAL_HCAL_1_->Fill((Econe + Hcone) / energy_MC);
      ratio_Esummed_ECAL_HCAL_HO_1_->Fill((Econe + Hcone + HOcone) / energy_MC);
    } else if (std::abs(eta_MC) < 2.1 && std::abs(eta_MC) > 1.3) {
      ratio_Esummed_ECAL_2_->Fill(Econe / energy_MC);
      ratio_Esummed_HCAL_2_->Fill(Hcone / energy_MC);
      ratio_Esummed_HO_2_->Fill(HOcone / energy_MC);
      ratio_Esummed_ECAL_HCAL_2_->Fill((Econe + Hcone) / energy_MC);
      ratio_Esummed_ECAL_HCAL_HO_2_->Fill((Econe + Hcone + HOcone) / energy_MC);
    } else if (std::abs(eta_MC) < 2.5 && std::abs(eta_MC) > 2.1) {
      ratio_Esummed_ECAL_3_->Fill(Econe / energy_MC);
      ratio_Esummed_HCAL_3_->Fill(Hcone / energy_MC);
      ratio_Esummed_HO_3_->Fill(HOcone / energy_MC);
      ratio_Esummed_ECAL_HCAL_3_->Fill((Econe + Hcone) / energy_MC);
      ratio_Esummed_ECAL_HCAL_HO_3_->Fill((Econe + Hcone + HOcone) / energy_MC);
    } else if (2.5 < std::abs(eta_MC) && std::abs(eta_MC) < 3.0) {
      ratio_Esummed_ECAL_4_->Fill(Econe / energy_MC);
      ratio_Esummed_HCAL_4_->Fill(Hcone / energy_MC);
      ratio_Esummed_HO_4_->Fill(HOcone / energy_MC);
      ratio_Esummed_ECAL_HCAL_4_->Fill((Econe + Hcone) / energy_MC);
      ratio_Esummed_ECAL_HCAL_HO_4_->Fill((Econe + Hcone + HOcone) / energy_MC);
    } else if (3.0 < std::abs(eta_MC) && std::abs(eta_MC) < 4.0) {
      ratio_Esummed_HF_5_->Fill(HFcone / energy_MC);
    } else if (4.0 < std::abs(eta_MC) && std::abs(eta_MC) < 5.0) {
      ratio_Esummed_HF_6_->Fill(HFcone / energy_MC);
    }
  }

  emean_vs_eta_E_->Fill(eta_MC, Econe);
  emean_vs_eta_H_->Fill(eta_MC, Hcone);
  emean_vs_eta_EH_->Fill(eta_MC, Econe + Hcone);
  emean_vs_eta_HF_->Fill(eta_MC, HFcone);
  emean_vs_eta_HO_->Fill(eta_MC, HOcone);
  emean_vs_eta_EHF_->Fill(eta_MC, Econe + Hcone + HFcone);
  emean_vs_eta_EHFO_->Fill(eta_MC, Econe + Hcone + HFcone + HOcone);
}

double PFClusterValidation::sumEnergy(edm::Handle<reco::PFClusterCollection> const& pfClusters,
                                      double eta,
                                      double phi) {
  if (not pfClusters.isValid())
    return 0.;

  double sum = 0.;
  for (auto pf = pfClusters->begin(); pf != pfClusters->end(); ++pf) {
    if (reco::deltaR2(eta, phi, pf->eta(), pf->phi()) < partR2) {
      sum += pf->energy();
    }
  }

  return sum;
}

DEFINE_FWK_MODULE(PFClusterValidation);
