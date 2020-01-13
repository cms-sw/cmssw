#include <memory>
#include "Validation/RecoEgamma/plugins/PhotonValidatorMiniAOD.h"
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ptr.h"
//#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"
//#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
//#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "TTree.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

// **********************************************************************

PhotonValidatorMiniAOD::PhotonValidatorMiniAOD(const edm::ParameterSet &iConfig)
    : outputFileName_(iConfig.getParameter<std::string>("outputFileName")),
      photonToken_(consumes<edm::View<pat::Photon> >(
          iConfig.getUntrackedParameter<edm::InputTag>("PhotonTag", edm::InputTag("slimmedPhotons")))),
      genpartToken_(consumes<reco::GenParticleCollection>(
          iConfig.getUntrackedParameter<edm::InputTag>("genpartTag", edm::InputTag("prunedGenParticles"))))

{
  parameters_ = iConfig;
}

PhotonValidatorMiniAOD::~PhotonValidatorMiniAOD() {}

void PhotonValidatorMiniAOD::bookHistograms(DQMStore::IBooker &iBooker,
                                            edm::Run const &run,
                                            edm::EventSetup const &es) {
  double eMin = parameters_.getParameter<double>("eMin");
  double eMax = parameters_.getParameter<double>("eMax");
  int eBin = parameters_.getParameter<int>("eBin");

  double etMin = parameters_.getParameter<double>("etMin");
  double etMax = parameters_.getParameter<double>("etMax");
  int etBin = parameters_.getParameter<int>("etBin");

  double resMin = parameters_.getParameter<double>("resMin");
  double resMax = parameters_.getParameter<double>("resMax");
  int resBin = parameters_.getParameter<int>("resBin");

  double etaMin = parameters_.getParameter<double>("etaMin");
  double etaMax = parameters_.getParameter<double>("etaMax");
  int etaBin = parameters_.getParameter<int>("etaBin");

  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int phiBin = parameters_.getParameter<int>("phiBin");

  double r9Min = parameters_.getParameter<double>("r9Min");
  double r9Max = parameters_.getParameter<double>("r9Max");
  int r9Bin = parameters_.getParameter<int>("r9Bin");

  iBooker.setCurrentFolder("EgammaV/PhotonValidatorMiniAOD/Photons");

  h_scEta_[0] = iBooker.book1D("scEta_miniAOD", " SC Eta ", etaBin, etaMin, etaMax);
  h_scPhi_[0] = iBooker.book1D("scPhi_miniAOD", " SC Phi ", phiBin, phiMin, phiMax);

  std::string histname = " ";

  histname = "phoE";
  h_phoE_[0][0] = iBooker.book1D(histname + "All_miniAOD", " Photon Energy: All ecal ", eBin, eMin, eMax);
  h_phoE_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", " Photon Energy: barrel ", eBin, eMin, eMax);
  h_phoE_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", " Photon Energy: Endcap ", eBin, eMin, eMax);

  histname = "phoEt";
  h_phoEt_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", " Photon Transverse Energy: All ecal ", etBin, etMin, etMax);
  h_phoEt_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", " Photon Transverse Energy: Barrel ", etBin, etMin, etMax);
  h_phoEt_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", " Photon Transverse Energy: Endcap ", etBin, etMin, etMax);

  histname = "eRes";
  h_phoERes_[0][0] = iBooker.book1D(
      histname + "All_miniAOD", " Photon E/E_{true}: All ecal;  E/E_{true} (GeV)", resBin, resMin, resMax);
  h_phoERes_[0][1] = iBooker.book1D(
      histname + "Barrel_miniAOD", "Photon E/E_{true}: Barrel; E/E_{true} (GeV)", resBin, resMin, resMax);
  h_phoERes_[0][2] = iBooker.book1D(
      histname + "Endcap_miniAOD", " Photon E/E_{true}: Endcap; E/E_{true} (GeV)", resBin, resMin, resMax);

  histname = "sigmaEoE";
  h_phoSigmaEoE_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "#sigma_{E}/E: All ecal; #sigma_{E}/E", 100, 0., 0.08);
  h_phoSigmaEoE_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "#sigma_{E}/E: Barrel; #sigma_{E}/E", 100, 0., 0.08);
  h_phoSigmaEoE_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "#sigma_{E}/E: Endcap, #sigma_{E}/E", 100, 0., 0.08);

  histname = "r9";
  h_r9_[0][0] = iBooker.book1D(histname + "All_miniAOD", " r9: All Ecal", r9Bin, r9Min, r9Max);
  h_r9_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", " r9: Barrel ", r9Bin, r9Min, r9Max);
  h_r9_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", " r9: Endcap ", r9Bin, r9Min, r9Max);
  histname = "full5x5_r9";
  h_full5x5_r9_[0][0] = iBooker.book1D(histname + "All_miniAOD", " r9: All Ecal", r9Bin, r9Min, r9Max);
  h_full5x5_r9_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", " r9: Barrel ", r9Bin, r9Min, r9Max);
  h_full5x5_r9_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", " r9: Endcap ", r9Bin, r9Min, r9Max);
  histname = "r1";
  h_r1_[0][0] = iBooker.book1D(histname + "All_miniAOD", " e1x5/e5x5: All Ecal", r9Bin, r9Min, r9Max);
  h_r1_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", " e1x5/e5x5: Barrel ", r9Bin, r9Min, r9Max);
  h_r1_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", " e1x5/e5x5: Endcap ", r9Bin, r9Min, r9Max);
  histname = "r2";
  h_r2_[0][0] = iBooker.book1D(histname + "All_miniAOD", " e2x5/e5x5: All Ecal", r9Bin, r9Min, r9Max);
  h_r2_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", " e2x5/e5x5: Barrel ", r9Bin, r9Min, r9Max);
  h_r2_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", " e2x5/e5x5: Endcap ", r9Bin, r9Min, r9Max);
  histname = "hOverE";
  h_hOverE_[0][0] = iBooker.book1D(histname + "All_miniAOD", "H/E: All Ecal", 100, 0., 0.2);
  h_hOverE_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", "H/E: Barrel ", 100, 0., 0.2);
  h_hOverE_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", "H/E: Endcap ", 100, 0., 0.2);
  //
  histname = "newhOverE";
  h_newhOverE_[0][0] = iBooker.book1D(histname + "All_miniAOD", "new H/E: All Ecal", 100, 0., 0.2);
  h_newhOverE_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", "new H/E: Barrel ", 100, 0., 0.2);
  h_newhOverE_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", "new H/E: Endcap ", 100, 0., 0.2);
  //
  histname = "sigmaIetaIeta";
  h_sigmaIetaIeta_[0][0] = iBooker.book1D(histname + "All_miniAOD", "sigmaIetaIeta: All Ecal", 100, 0., 0.1);
  h_sigmaIetaIeta_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", "sigmaIetaIeta: Barrel ", 100, 0., 0.05);
  h_sigmaIetaIeta_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", "sigmaIetaIeta: Endcap ", 100, 0., 0.1);
  histname = "full5x5_sigmaIetaIeta";
  h_full5x5_sigmaIetaIeta_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "Full5x5 sigmaIetaIeta: All Ecal", 100, 0., 0.1);
  h_full5x5_sigmaIetaIeta_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "Full5x5 sigmaIetaIeta: Barrel ", 100, 0., 0.05);
  h_full5x5_sigmaIetaIeta_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "Full5x5 sigmaIetaIeta: Endcap ", 100, 0., 0.1);
  //
  histname = "ecalRecHitSumEtConeDR04";
  h_ecalRecHitSumEtConeDR04_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "ecalRecHitSumEtDR04: All Ecal", etBin, etMin, 20.);
  h_ecalRecHitSumEtConeDR04_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "ecalRecHitSumEtDR04: Barrel ", etBin, etMin, 20.);
  h_ecalRecHitSumEtConeDR04_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "ecalRecHitSumEtDR04: Endcap ", etBin, etMin, 20.);
  histname = "hcalTowerSumEtConeDR04";
  h_hcalTowerSumEtConeDR04_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "hcalTowerSumEtConeDR04: All Ecal", etBin, etMin, 20.);
  h_hcalTowerSumEtConeDR04_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "hcalTowerSumEtConeDR04: Barrel ", etBin, etMin, 20.);
  h_hcalTowerSumEtConeDR04_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "hcalTowerSumEtConeDR04: Endcap ", etBin, etMin, 20.);
  //
  histname = "hcalTowerBcSumEtConeDR04";
  h_hcalTowerBcSumEtConeDR04_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "hcalTowerBcSumEtConeDR04: All Ecal", etBin, etMin, 20.);
  h_hcalTowerBcSumEtConeDR04_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "hcalTowerBcSumEtConeDR04: Barrel ", etBin, etMin, 20.);
  h_hcalTowerBcSumEtConeDR04_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "hcalTowerBcSumEtConeDR04: Endcap ", etBin, etMin, 20.);
  histname = "isoTrkSolidConeDR04";
  h_isoTrkSolidConeDR04_[0][0] =
      iBooker.book1D(histname + "All_miniAOD", "isoTrkSolidConeDR04: All Ecal", etBin, etMin, etMax * 0.1);
  h_isoTrkSolidConeDR04_[0][1] =
      iBooker.book1D(histname + "Barrel_miniAOD", "isoTrkSolidConeDR04: Barrel ", etBin, etMin, etMax * 0.1);
  h_isoTrkSolidConeDR04_[0][2] =
      iBooker.book1D(histname + "Endcap_miniAOD", "isoTrkSolidConeDR04: Endcap ", etBin, etMin, etMax * 0.1);
  histname = "nTrkSolidConeDR04";
  h_nTrkSolidConeDR04_[0][0] = iBooker.book1D(histname + "All_miniAOD", "nTrkSolidConeDR04: All Ecal", 20, 0., 20);
  h_nTrkSolidConeDR04_[0][1] = iBooker.book1D(histname + "Barrel_miniAOD", "nTrkSolidConeDR04: Barrel ", 20, 0., 20);
  h_nTrkSolidConeDR04_[0][2] = iBooker.book1D(histname + "Endcap_miniAOD", "nTrkSolidConeDR04: Endcap ", 20, 0., 20);

  //  Infos from Particle Flow - isolation and ID
  histname = "chargedHadIso";
  h_chHadIso_[0] = iBooker.book1D(histname + "All_miniAOD", "PF chargedHadIso:  All Ecal", etBin, etMin, 20.);
  h_chHadIso_[1] = iBooker.book1D(histname + "Barrel_miniAOD", "PF chargedHadIso:  Barrel", etBin, etMin, 20.);
  h_chHadIso_[2] = iBooker.book1D(histname + "Endcap_miniAOD", "PF chargedHadIso:  Endcap", etBin, etMin, 20.);
  histname = "neutralHadIso";
  h_nHadIso_[0] = iBooker.book1D(histname + "All_miniAOD", "PF neutralHadIso:  All Ecal", etBin, etMin, 20.);
  h_nHadIso_[1] = iBooker.book1D(histname + "Barrel_miniAOD", "PF neutralHadIso:  Barrel", etBin, etMin, 20.);
  h_nHadIso_[2] = iBooker.book1D(histname + "Endcap_miniAOD", "PF neutralHadIso:  Endcap", etBin, etMin, 20.);
  histname = "photonIso";
  h_phoIso_[0] = iBooker.book1D(histname + "All_miniAOD", "PF photonIso:  All Ecal", etBin, etMin, 20.);
  h_phoIso_[1] = iBooker.book1D(histname + "Barrel_miniAOD", "PF photonIso:  Barrel", etBin, etMin, 20.);
  h_phoIso_[2] = iBooker.book1D(histname + "Endcap_miniAOD", "PF photonIso:  Endcap", etBin, etMin, 20.);
}

void PhotonValidatorMiniAOD::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // ********************************************************************************
  using namespace edm;

  // access edm objects
  Handle<View<pat::Photon> > photonsHandle;
  iEvent.getByToken(photonToken_, photonsHandle);
  const auto &photons = *photonsHandle;
  //
  Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(genpartToken_, genParticles);

  for (reco::GenParticleCollection::const_iterator mcIter = genParticles->begin(); mcIter != genParticles->end();
       mcIter++) {
    if (!(mcIter->pdgId() == 22))
      continue;
    if (mcIter->mother() != nullptr && !(mcIter->mother()->pdgId() == 25))
      continue;
    if (fabs(mcIter->eta()) > 2.5)
      continue;
    //       if ( mcIter->pt() < 5) continue;
    //        if ( fabs(mcIter->eta()) > 1.4442 && fabs(mcIter->eta()) < 1.566) continue;

    float mcPhi = mcIter->phi();
    float mcEta = mcIter->eta();
    //mcEta = etaTransformation(mcEta, (*mcPho).primaryVertex().z() );
    float mcEnergy = mcIter->energy();

    double dR = 9999999.;
    float minDr = 10000.;
    int iMatch = -1;
    bool matched = false;
    for (size_t ipho = 0; ipho < photons.size(); ipho++) {
      Ptr<pat::Photon> pho = photons.ptrAt(ipho);

      double dphi = pho->phi() - mcPhi;
      if (std::fabs(dphi) > CLHEP::pi) {
        dphi = dphi < 0 ? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
      }
      double deta = pho->superCluster()->position().eta() - mcEta;

      dR = sqrt(pow((deta), 2) + pow(dphi, 2));
      if (dR < 0.1 && dR < minDr) {
        minDr = dR;
        iMatch = ipho;
      }
    }

    if (iMatch > -1)
      matched = true;
    if (!matched)
      continue;

    Ptr<pat::Photon> matchingPho = photons.ptrAt(iMatch);

    bool phoIsInBarrel = false;
    bool phoIsInEndcap = false;

    float phoEta = matchingPho->superCluster()->position().eta();
    if (fabs(phoEta) < 1.479) {
      phoIsInBarrel = true;
    } else {
      phoIsInEndcap = true;
    }

    float photonE = matchingPho->energy();
    float sigmaEoE = matchingPho->getCorrectedEnergyError(matchingPho->getCandidateP4type()) / matchingPho->energy();
    float photonEt = matchingPho->energy() / cosh(matchingPho->eta());
    //	float photonERegr1 = matchingPho->getCorrectedEnergy(reco::Photon::regression1);
    //float photonERegr2 = matchingPho->getCorrectedEnergy(reco::Photon::regression2);
    float r9 = matchingPho->r9();
    float full5x5_r9 = matchingPho->full5x5_r9();
    float r1 = matchingPho->r1x5();
    float r2 = matchingPho->r2x5();
    float sieie = matchingPho->sigmaIetaIeta();
    float full5x5_sieie = matchingPho->full5x5_sigmaIetaIeta();
    float hOverE = matchingPho->hadronicOverEm();
    float newhOverE = matchingPho->hadTowOverEm();
    float ecalIso = matchingPho->ecalRecHitSumEtConeDR04();
    float hcalIso = matchingPho->hcalTowerSumEtConeDR04();
    float newhcalIso = matchingPho->hcalTowerSumEtBcConeDR04();
    float trkIso = matchingPho->trkSumPtSolidConeDR04();
    float nIsoTrk = matchingPho->nTrkSolidConeDR04();
    // PF related quantities
    float chargedHadIso = matchingPho->chargedHadronIso();
    float neutralHadIso = matchingPho->neutralHadronIso();
    float photonIso = matchingPho->photonIso();
    //	float etOutsideMustache = matchingPho->etOutsideMustache();
    //	int   nClusterOutsideMustache = matchingPho->nClusterOutsideMustache();
    //float pfMVA = matchingPho->pfMVA();

    h_scEta_[0]->Fill(matchingPho->superCluster()->eta());
    h_scPhi_[0]->Fill(matchingPho->superCluster()->phi());

    h_phoE_[0][0]->Fill(photonE);
    h_phoEt_[0][0]->Fill(photonEt);

    h_phoERes_[0][0]->Fill(photonE / mcEnergy);
    h_phoSigmaEoE_[0][0]->Fill(sigmaEoE);

    h_r9_[0][0]->Fill(r9);
    h_full5x5_r9_[0][0]->Fill(full5x5_r9);
    h_r1_[0][0]->Fill(r1);
    h_r2_[0][0]->Fill(r2);

    h_sigmaIetaIeta_[0][0]->Fill(sieie);
    h_full5x5_sigmaIetaIeta_[0][0]->Fill(full5x5_sieie);
    h_hOverE_[0][0]->Fill(hOverE);
    h_newhOverE_[0][0]->Fill(newhOverE);

    h_ecalRecHitSumEtConeDR04_[0][0]->Fill(ecalIso);
    h_hcalTowerSumEtConeDR04_[0][0]->Fill(hcalIso);
    h_hcalTowerBcSumEtConeDR04_[0][0]->Fill(newhcalIso);
    h_isoTrkSolidConeDR04_[0][0]->Fill(trkIso);
    h_nTrkSolidConeDR04_[0][0]->Fill(nIsoTrk);

    //
    h_chHadIso_[0]->Fill(chargedHadIso);
    h_nHadIso_[0]->Fill(neutralHadIso);
    h_phoIso_[0]->Fill(photonIso);

    //
    if (phoIsInBarrel) {
      h_phoE_[0][1]->Fill(photonE);
      h_phoEt_[0][1]->Fill(photonEt);
      h_phoERes_[0][1]->Fill(photonE / mcEnergy);
      h_phoSigmaEoE_[0][1]->Fill(sigmaEoE);

      h_r9_[0][1]->Fill(r9);
      h_full5x5_r9_[0][1]->Fill(full5x5_r9);
      h_r1_[0][1]->Fill(r1);
      h_r2_[0][1]->Fill(r2);
      h_sigmaIetaIeta_[0][1]->Fill(sieie);
      h_full5x5_sigmaIetaIeta_[0][1]->Fill(full5x5_sieie);
      h_hOverE_[0][1]->Fill(hOverE);
      h_newhOverE_[0][1]->Fill(newhOverE);
      h_ecalRecHitSumEtConeDR04_[0][1]->Fill(ecalIso);
      h_hcalTowerSumEtConeDR04_[0][1]->Fill(hcalIso);
      h_hcalTowerBcSumEtConeDR04_[0][1]->Fill(newhcalIso);
      h_isoTrkSolidConeDR04_[0][1]->Fill(trkIso);
      h_nTrkSolidConeDR04_[0][1]->Fill(nIsoTrk);
      h_chHadIso_[1]->Fill(chargedHadIso);
      h_nHadIso_[1]->Fill(neutralHadIso);
      h_phoIso_[1]->Fill(photonIso);
    }
    if (phoIsInEndcap) {
      h_phoE_[0][2]->Fill(photonE);
      h_phoEt_[0][2]->Fill(photonEt);

      h_phoERes_[0][2]->Fill(photonE / mcEnergy);
      h_phoSigmaEoE_[0][2]->Fill(sigmaEoE);
      h_r9_[0][2]->Fill(r9);
      h_full5x5_r9_[0][2]->Fill(full5x5_r9);
      h_r1_[0][2]->Fill(r1);
      h_r2_[0][2]->Fill(r2);
      h_sigmaIetaIeta_[0][2]->Fill(sieie);
      h_full5x5_sigmaIetaIeta_[0][2]->Fill(full5x5_sieie);
      h_hOverE_[0][2]->Fill(hOverE);
      h_newhOverE_[0][2]->Fill(newhOverE);
      h_ecalRecHitSumEtConeDR04_[0][2]->Fill(ecalIso);
      h_hcalTowerSumEtConeDR04_[0][2]->Fill(hcalIso);
      h_hcalTowerBcSumEtConeDR04_[0][2]->Fill(newhcalIso);
      h_isoTrkSolidConeDR04_[0][2]->Fill(trkIso);
      h_nTrkSolidConeDR04_[0][2]->Fill(nIsoTrk);
      h_chHadIso_[2]->Fill(chargedHadIso);
      h_nHadIso_[2]->Fill(neutralHadIso);
      h_phoIso_[2]->Fill(photonIso);
    }

  }  // end loop over gen photons
}
