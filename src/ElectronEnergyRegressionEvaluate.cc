#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyRegressionEvaluate.h"
#include <cmath>
#include <cassert>

#ifndef STANDALONE
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#endif

ElectronEnergyRegressionEvaluate::ElectronEnergyRegressionEvaluate() : 
  fIsInitialized(kFALSE),
  fVersionType(kNoTrkVar),
  forestCorrection_eb(0), 
  forestCorrection_ee(0), 
  forestUncertainty_eb(0), 
  forestUncertainty_ee(0) {
}

ElectronEnergyRegressionEvaluate::~ElectronEnergyRegressionEvaluate() {}
// Destructor does nothing


void ElectronEnergyRegressionEvaluate::initialize(std::string weightsFile, 
                                                  ElectronEnergyRegressionEvaluate::ElectronEnergyRegressionType type) {

  // Loading forest object according to different versions
  TFile file(edm::FileInPath(weightsFile.c_str()).fullPath().c_str());

  forestCorrection_eb = (GBRForest*) file.Get("EBCorrection");
  forestCorrection_ee = (GBRForest*) file.Get("EECorrection");
  forestUncertainty_eb = (GBRForest*) file.Get("EBUncertainty");
  forestUncertainty_ee = (GBRForest*) file.Get("EEUncertainty");
  
  // Just checking
  assert(forestCorrection_eb);
  assert(forestCorrection_ee);
  assert(forestUncertainty_eb);
  assert(forestUncertainty_ee);
    
  // Updating type and marking as initialized
  fVersionType = type;
  fIsInitialized = kTRUE;
}



#ifndef STANDALONE
double ElectronEnergyRegressionEvaluate::calculateRegressionEnergy(const reco::GsfElectron *ele, 
                                                                   EcalClusterLazyTools &myEcalCluster, 
                                                                   const edm::EventSetup &setup,
                                                                   double rho, double nvertices, 
                                                                   bool printDebug) {
  
  if (!fIsInitialized) {
    std::cout << "Error: Electron Energy Regression has not been initialized yet. return 0. \n";
    return 0;
  }

  std::vector<float> vCov = myEcalCluster.localCovariances(*(ele->superCluster()->seed()));
  double spp = 0;
  if (!isnan(vCov[2])) spp = sqrt (vCov[2]);
  double sep;
  if (ele->sigmaIetaIeta()*spp > 0) {
    sep = vCov[1] / (ele->sigmaIetaIeta()*spp);
  } else if (vCov[1] > 0) {
    sep = 1.0;
  } else {
    sep = -1.0;
  }

  //local coordinates
  EcalClusterLocal local;  
  double ietaseed = 0;
  double iphiseed = 0;
  double etacryseed = 0;
  double phicryseed = 0;

  if (ele->superCluster()->seed()->hitsAndFractions().at(0).first.subdetId()==EcalBarrel) {
    float etacry, phicry, thetatilt, phitilt;
    int ieta, iphi;
    local.localCoordsEB(*ele->superCluster()->seed(),setup,etacry,phicry,ieta,iphi,thetatilt,phitilt);
    
    ietaseed = ieta;
    iphiseed = iphi;
    etacryseed = etacry;
    phicryseed = phicry;
  }
  
  if (printDebug) {
    std::cout << "Regression Type: " << fVersionType << std::endl;
    std::cout << "Electron : " << ele->pt() << " " << ele->eta() << " " << ele->phi() << "\n";
  }

  if (fVersionType == kNoTrkVar) {
    return regressionValueNoTrkVar(
                                   ele->superCluster()->rawEnergy(),
                                   ele->superCluster()->eta(),
                                   ele->superCluster()->phi(),
                                   myEcalCluster.e3x3(*ele->superCluster()->seed()) / ele->superCluster()->rawEnergy(),
                                   ele->superCluster()->etaWidth(),
                                   ele->superCluster()->phiWidth(),
                                   ele->superCluster()->clustersSize(),
                                   ele->hadronicOverEm(),
                                   rho,
                                   nvertices,
                                   ele->superCluster()->seed()->eta(),
                                   ele->superCluster()->seed()->phi(),
                                   ele->superCluster()->seed()->energy(),
                                   myEcalCluster.e3x3(*ele->superCluster()->seed()),
                                   myEcalCluster.e5x5(*ele->superCluster()->seed()),
                                   ele->sigmaIetaIeta(),
                                   spp,
                                   sep,
                                   myEcalCluster.eMax(*ele->superCluster()->seed()),
                                   myEcalCluster.e2nd(*ele->superCluster()->seed()),
                                   myEcalCluster.eTop(*ele->superCluster()->seed()),
                                   myEcalCluster.eBottom(*ele->superCluster()->seed()),
                                   myEcalCluster.eLeft(*ele->superCluster()->seed()),
                                   myEcalCluster.eRight(*ele->superCluster()->seed()),
                                   myEcalCluster.e2x5Max(*ele->superCluster()->seed()),
                                   myEcalCluster.e2x5Top(*ele->superCluster()->seed()),
                                   myEcalCluster.e2x5Bottom(*ele->superCluster()->seed()),
                                   myEcalCluster.e2x5Left(*ele->superCluster()->seed()),
                                   myEcalCluster.e2x5Right(*ele->superCluster()->seed()),
                                   ietaseed,
                                   iphiseed,
                                   etacryseed,
                                   phicryseed,
                                   ele->superCluster()->preshowerEnergy() / ele->superCluster()->rawEnergy(),
                                   printDebug
                                   );
  } 
  else if (fVersionType == kNoTrkVarV1) {
    return regressionValueNoTrkVarV1(
                                   ele->superCluster()->rawEnergy(),
                                   ele->superCluster()->eta(),
                                   ele->superCluster()->phi(),
                                   myEcalCluster.e3x3(*ele->superCluster()->seed()) / ele->superCluster()->rawEnergy(),
                                   ele->superCluster()->etaWidth(),
                                   ele->superCluster()->phiWidth(),
                                   ele->superCluster()->clustersSize(),
                                   ele->hadronicOverEm(),
                                   rho,
                                   nvertices,
                                   ele->superCluster()->seed()->eta(),
                                   ele->superCluster()->seed()->phi(),
                                   ele->superCluster()->seed()->energy(),
                                   myEcalCluster.e3x3(*ele->superCluster()->seed()),
                                   myEcalCluster.e5x5(*ele->superCluster()->seed()),
                                   ele->sigmaIetaIeta(),
                                   spp,
                                   sep,
                                   myEcalCluster.eMax(*ele->superCluster()->seed()),
                                   myEcalCluster.e2nd(*ele->superCluster()->seed()),
                                   myEcalCluster.eTop(*ele->superCluster()->seed()),
                                   myEcalCluster.eBottom(*ele->superCluster()->seed()),
                                   myEcalCluster.eLeft(*ele->superCluster()->seed()),
                                   myEcalCluster.eRight(*ele->superCluster()->seed()),
                                   myEcalCluster.e2x5Max(*ele->superCluster()->seed()),
                                   myEcalCluster.e2x5Top(*ele->superCluster()->seed()),
                                   myEcalCluster.e2x5Bottom(*ele->superCluster()->seed()),
                                   myEcalCluster.e2x5Left(*ele->superCluster()->seed()),
                                   myEcalCluster.e2x5Right(*ele->superCluster()->seed()),
                                   ietaseed,
                                   iphiseed,
                                   etacryseed,
                                   phicryseed,
                                   ele->superCluster()->preshowerEnergy() / ele->superCluster()->rawEnergy(),
                                   ele->ecalDrivenSeed(),
                                   printDebug
                                   );
  } 
  else if (fVersionType == kWithTrkVarV1) {
    return regressionValueWithTrkVarV1(
                                     ele->superCluster()->rawEnergy(),
                                     ele->superCluster()->eta(),
                                     ele->superCluster()->phi(),
                                     myEcalCluster.e3x3(*ele->superCluster()->seed()) / ele->superCluster()->rawEnergy(),
                                     ele->superCluster()->etaWidth(),
                                     ele->superCluster()->phiWidth(),
                                     ele->superCluster()->clustersSize(),
                                     ele->hadronicOverEm(),
                                     rho,
                                     nvertices,
                                     ele->superCluster()->seed()->eta(),
                                     ele->superCluster()->seed()->phi(),
                                     ele->superCluster()->seed()->energy(),
                                     myEcalCluster.e3x3(*ele->superCluster()->seed()),
                                     myEcalCluster.e5x5(*ele->superCluster()->seed()),
                                     ele->sigmaIetaIeta(),
                                     spp,
                                     sep,
                                     myEcalCluster.eMax(*ele->superCluster()->seed()),
                                     myEcalCluster.e2nd(*ele->superCluster()->seed()),
                                     myEcalCluster.eTop(*ele->superCluster()->seed()),
                                     myEcalCluster.eBottom(*ele->superCluster()->seed()),
                                     myEcalCluster.eLeft(*ele->superCluster()->seed()),
                                     myEcalCluster.eRight(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Max(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Top(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Bottom(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Left(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Right(*ele->superCluster()->seed()),
                                     ietaseed,
                                     iphiseed,
                                     etacryseed,
                                     phicryseed,
                                     ele->superCluster()->preshowerEnergy() / ele->superCluster()->rawEnergy(),
                                     ele->ecalDrivenSeed(),
                                     ele->trackMomentumAtVtx().R(),
                                     fmax(ele->fbrem(),-1.0),
                                     ele->charge(),
                                     fmin(ele->eSuperClusterOverP(), 20.0),
                                     ele->trackMomentumError(),
                                     ele->correctedEcalEnergyError(),
                                     ele->classification(),                                    
                                     printDebug
                                     );
  } 
  else if (fVersionType == kWithTrkVarV2) {
    return regressionValueWithTrkVarV2(
                                     ele->superCluster()->rawEnergy(),
                                     ele->superCluster()->eta(),
                                     ele->superCluster()->phi(),
                                     myEcalCluster.e3x3(*ele->superCluster()->seed()) / ele->superCluster()->rawEnergy(),
                                     ele->superCluster()->etaWidth(),
                                     ele->superCluster()->phiWidth(),
                                     ele->superCluster()->clustersSize(),
                                     ele->hadronicOverEm(),
                                     rho,
                                     nvertices,
                                     ele->superCluster()->seed()->eta(),
                                     ele->superCluster()->seed()->phi(),
                                     ele->superCluster()->seed()->energy(),
                                     myEcalCluster.e3x3(*ele->superCluster()->seed()),
                                     myEcalCluster.e5x5(*ele->superCluster()->seed()),
                                     ele->sigmaIetaIeta(),
                                     spp,
                                     sep,
                                     myEcalCluster.eMax(*ele->superCluster()->seed()),
                                     myEcalCluster.e2nd(*ele->superCluster()->seed()),
                                     myEcalCluster.eTop(*ele->superCluster()->seed()),
                                     myEcalCluster.eBottom(*ele->superCluster()->seed()),
                                     myEcalCluster.eLeft(*ele->superCluster()->seed()),
                                     myEcalCluster.eRight(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Max(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Top(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Bottom(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Left(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Right(*ele->superCluster()->seed()),
                                     ietaseed,
                                     iphiseed,
                                     etacryseed,
                                     phicryseed,
                                     ele->superCluster()->preshowerEnergy() / ele->superCluster()->rawEnergy(),
                                     ele->ecalDrivenSeed(),
                                     ele->trackMomentumAtVtx().R(),
                                     fmax(ele->fbrem(),-1.0),
                                     ele->charge(),
                                     fmin(ele->eSuperClusterOverP(), 20.0),
                                     ele->trackMomentumError(),
                                     ele->correctedEcalEnergyError(),
                                     ele->classification(),     
                                     fmin(fabs(ele->deltaEtaSuperClusterTrackAtVtx()), 0.6),
                                     ele->deltaPhiSuperClusterTrackAtVtx(),
                                     ele->deltaEtaSeedClusterTrackAtCalo(),
                                     ele->deltaPhiSeedClusterTrackAtCalo(),
                                     ele->gsfTrack()->chi2() / ele->gsfTrack()->ndof(),
                                     (ele->closestCtfTrackRef().isNonnull() ? ele->closestCtfTrackRef()->hitPattern().trackerLayersWithMeasurement() : -1), 
                                     fmin(ele->eEleClusterOverPout(),20.0),
                                     printDebug
                                     );
  } 
  else {
    std::cout << "Warning: Electron Regression Type " << fVersionType << " is not supported. Reverting to default electron momentum.\n"; 
    return ele->p();
  }

  
}

double ElectronEnergyRegressionEvaluate::calculateRegressionEnergyUncertainty(const reco::GsfElectron *ele, 
                                                                              EcalClusterLazyTools &myEcalCluster, 
                                                                              const edm::EventSetup &setup,
                                                                              double rho, double nvertices, 
                                                                              bool printDebug) {
  
  if (!fIsInitialized) {
    std::cout << "Error: Electron Energy Regression has not been initialized yet. return 0. \n";
    return 0;
  }

  std::vector<float> vCov = myEcalCluster.localCovariances(*(ele->superCluster()->seed()));
  double spp = 0;
  if (!isnan(vCov[2])) spp = sqrt (vCov[2]);
  double sep;
  if (ele->sigmaIetaIeta()*spp > 0) {
    sep = vCov[1] / ele->sigmaIetaIeta()*spp;
  } else if (vCov[1] > 0) {
    sep = 1.0;
  } else {
    sep = -1.0;
  }

  //local coordinates
  EcalClusterLocal local;  
  double ietaseed = 0;
  double iphiseed = 0;
  double etacryseed = 0;
  double phicryseed = 0;

  if (ele->superCluster()->seed()->hitsAndFractions().at(0).first.subdetId()==EcalBarrel) {
    float etacry, phicry, thetatilt, phitilt;
    int ieta, iphi;
    local.localCoordsEB(*ele->superCluster()->seed(),setup,etacry,phicry,ieta,iphi,thetatilt,phitilt);
    
    ietaseed = ieta;
    iphiseed = iphi;
    etacryseed = etacry;
    phicryseed = phicry;
  }
  
  if (printDebug) {
    std::cout << "Regression Type: " << fVersionType << std::endl;
    std::cout << "Electron : " << ele->pt() << " " << ele->eta() << " " << ele->phi() << "\n";
  }

  if (fVersionType == kNoTrkVar) {
    return regressionUncertaintyNoTrkVar(
                                         ele->superCluster()->rawEnergy(),
                                         ele->superCluster()->eta(),
                                         ele->superCluster()->phi(),
                                         myEcalCluster.e3x3(*ele->superCluster()->seed()) / ele->superCluster()->rawEnergy(),
                                         ele->superCluster()->etaWidth(),
                                         ele->superCluster()->phiWidth(),
                                         ele->superCluster()->clustersSize(),
                                         ele->hadronicOverEm(),
                                         rho,
                                         nvertices,
                                         ele->superCluster()->seed()->eta(),
                                         ele->superCluster()->seed()->phi(),
                                         ele->superCluster()->seed()->energy(),
                                         myEcalCluster.e3x3(*ele->superCluster()->seed()),
                                         myEcalCluster.e5x5(*ele->superCluster()->seed()),
                                         ele->sigmaIetaIeta(),
                                         spp,
                                         sep,
                                         myEcalCluster.eMax(*ele->superCluster()->seed()),
                                         myEcalCluster.e2nd(*ele->superCluster()->seed()),
                                         myEcalCluster.eTop(*ele->superCluster()->seed()),
                                         myEcalCluster.eBottom(*ele->superCluster()->seed()),
                                         myEcalCluster.eLeft(*ele->superCluster()->seed()),
                                         myEcalCluster.eRight(*ele->superCluster()->seed()),
                                         myEcalCluster.e2x5Max(*ele->superCluster()->seed()),
                                         myEcalCluster.e2x5Top(*ele->superCluster()->seed()),
                                         myEcalCluster.e2x5Bottom(*ele->superCluster()->seed()),
                                         myEcalCluster.e2x5Left(*ele->superCluster()->seed()),
                                         myEcalCluster.e2x5Right(*ele->superCluster()->seed()),
                                         ietaseed,
                                         iphiseed,
                                         etacryseed,
                                         phicryseed,
                                         ele->superCluster()->preshowerEnergy() / ele->superCluster()->rawEnergy(),
                                         printDebug
                                         );
  } 
  else if (fVersionType == kNoTrkVarV1) {
    return regressionUncertaintyNoTrkVarV1(
                                         ele->superCluster()->rawEnergy(),
                                         ele->superCluster()->eta(),
                                         ele->superCluster()->phi(),
                                         myEcalCluster.e3x3(*ele->superCluster()->seed()) / ele->superCluster()->rawEnergy(),
                                         ele->superCluster()->etaWidth(),
                                         ele->superCluster()->phiWidth(),
                                         ele->superCluster()->clustersSize(),
                                         ele->hadronicOverEm(),
                                         rho,
                                         nvertices,
                                         ele->superCluster()->seed()->eta(),
                                         ele->superCluster()->seed()->phi(),
                                         ele->superCluster()->seed()->energy(),
                                         myEcalCluster.e3x3(*ele->superCluster()->seed()),
                                         myEcalCluster.e5x5(*ele->superCluster()->seed()),
                                         ele->sigmaIetaIeta(),
                                         spp,
                                         sep,
                                         myEcalCluster.eMax(*ele->superCluster()->seed()),
                                         myEcalCluster.e2nd(*ele->superCluster()->seed()),
                                         myEcalCluster.eTop(*ele->superCluster()->seed()),
                                         myEcalCluster.eBottom(*ele->superCluster()->seed()),
                                         myEcalCluster.eLeft(*ele->superCluster()->seed()),
                                         myEcalCluster.eRight(*ele->superCluster()->seed()),
                                         myEcalCluster.e2x5Max(*ele->superCluster()->seed()),
                                         myEcalCluster.e2x5Top(*ele->superCluster()->seed()),
                                         myEcalCluster.e2x5Bottom(*ele->superCluster()->seed()),
                                         myEcalCluster.e2x5Left(*ele->superCluster()->seed()),
                                         myEcalCluster.e2x5Right(*ele->superCluster()->seed()),
                                         ietaseed,
                                         iphiseed,
                                         etacryseed,
                                         phicryseed,
                                         ele->superCluster()->preshowerEnergy() / ele->superCluster()->rawEnergy(),
                                         ele->ecalDrivenSeed(),
                                         printDebug
                                         );
  } 
  else if (fVersionType == kWithTrkVarV1) {
    return regressionUncertaintyWithTrkVarV1(
                                     ele->superCluster()->rawEnergy(),
                                     ele->superCluster()->eta(),
                                     ele->superCluster()->phi(),
                                     myEcalCluster.e3x3(*ele->superCluster()->seed()) / ele->superCluster()->rawEnergy(),
                                     ele->superCluster()->etaWidth(),
                                     ele->superCluster()->phiWidth(),
                                     ele->superCluster()->clustersSize(),
                                     ele->hadronicOverEm(),
                                     rho,
                                     nvertices,
                                     ele->superCluster()->seed()->eta(),
                                     ele->superCluster()->seed()->phi(),
                                     ele->superCluster()->seed()->energy(),
                                     myEcalCluster.e3x3(*ele->superCluster()->seed()),
                                     myEcalCluster.e5x5(*ele->superCluster()->seed()),
                                     ele->sigmaIetaIeta(),
                                     spp,
                                     sep,
                                     myEcalCluster.eMax(*ele->superCluster()->seed()),
                                     myEcalCluster.e2nd(*ele->superCluster()->seed()),
                                     myEcalCluster.eTop(*ele->superCluster()->seed()),
                                     myEcalCluster.eBottom(*ele->superCluster()->seed()),
                                     myEcalCluster.eLeft(*ele->superCluster()->seed()),
                                     myEcalCluster.eRight(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Max(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Top(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Bottom(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Left(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Right(*ele->superCluster()->seed()),
                                     ietaseed,
                                     iphiseed,
                                     etacryseed,
                                     phicryseed,
                                     ele->superCluster()->preshowerEnergy() / ele->superCluster()->rawEnergy(),
                                     ele->ecalDrivenSeed(),
                                     ele->trackMomentumAtVtx().R(),
                                     fmax(ele->fbrem(),-1.0),
                                     ele->charge(),
                                     fmin(ele->eSuperClusterOverP(), 20.0),
                                     ele->trackMomentumError(),
                                     ele->correctedEcalEnergyError(),
                                     ele->classification(),                                    
                                     printDebug
                                     );
  }  
  else if (fVersionType == kWithTrkVarV2) {
    return regressionUncertaintyWithTrkVarV2(
                                     ele->superCluster()->rawEnergy(),
                                     ele->superCluster()->eta(),
                                     ele->superCluster()->phi(),
                                     myEcalCluster.e3x3(*ele->superCluster()->seed()) / ele->superCluster()->rawEnergy(),
                                     ele->superCluster()->etaWidth(),
                                     ele->superCluster()->phiWidth(),
                                     ele->superCluster()->clustersSize(),
                                     ele->hadronicOverEm(),
                                     rho,
                                     nvertices,
                                     ele->superCluster()->seed()->eta(),
                                     ele->superCluster()->seed()->phi(),
                                     ele->superCluster()->seed()->energy(),
                                     myEcalCluster.e3x3(*ele->superCluster()->seed()),
                                     myEcalCluster.e5x5(*ele->superCluster()->seed()),
                                     ele->sigmaIetaIeta(),
                                     spp,
                                     sep,
                                     myEcalCluster.eMax(*ele->superCluster()->seed()),
                                     myEcalCluster.e2nd(*ele->superCluster()->seed()),
                                     myEcalCluster.eTop(*ele->superCluster()->seed()),
                                     myEcalCluster.eBottom(*ele->superCluster()->seed()),
                                     myEcalCluster.eLeft(*ele->superCluster()->seed()),
                                     myEcalCluster.eRight(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Max(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Top(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Bottom(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Left(*ele->superCluster()->seed()),
                                     myEcalCluster.e2x5Right(*ele->superCluster()->seed()),
                                     ietaseed,
                                     iphiseed,
                                     etacryseed,
                                     phicryseed,
                                     ele->superCluster()->preshowerEnergy() / ele->superCluster()->rawEnergy(),
                                     ele->ecalDrivenSeed(),
                                     ele->trackMomentumAtVtx().R(),
                                     fmax(ele->fbrem(),-1.0),
                                     ele->charge(),
                                     fmin(ele->eSuperClusterOverP(), 20.0),
                                     ele->trackMomentumError(),
                                     ele->correctedEcalEnergyError(),
                                     ele->classification(),     
                                     fmin(fabs(ele->deltaEtaSuperClusterTrackAtVtx()), 0.6),
                                     ele->deltaPhiSuperClusterTrackAtVtx(),
                                     ele->deltaEtaSeedClusterTrackAtCalo(),
                                     ele->deltaPhiSeedClusterTrackAtCalo(),
                                     ele->gsfTrack()->chi2() / ele->gsfTrack()->ndof(),
                                     (ele->closestCtfTrackRef().isNonnull() ? ele->closestCtfTrackRef()->hitPattern().trackerLayersWithMeasurement() : -1), 
                                     fmin(ele->eEleClusterOverPout(),20.0),
                                     printDebug
                                     );
  }
  else {
    std::cout << "Warning: Electron Regression Type " << fVersionType << " is not supported. Reverting to default electron momentum.\n"; 
    return ele->p();
  }
}
#endif


double ElectronEnergyRegressionEvaluate::regressionValueNoTrkVar(
                                                                 double SCRawEnergy,
                                                                 double scEta,
                                                                 double scPhi,
                                                                 double R9,
                                                                 double etawidth,
                                                                 double phiwidth,
                                                                 double NClusters,
                                                                 double HoE,
                                                                 double rho,
                                                                 double vertices,
                                                                 double EtaSeed,
                                                                 double PhiSeed,
                                                                 double ESeed,
                                                                 double E3x3Seed,
                                                                 double E5x5Seed,
                                                                 double see,
                                                                 double spp,
                                                                 double sep,
                                                                 double EMaxSeed,
                                                                 double E2ndSeed,
                                                                 double ETopSeed,
                                                                 double EBottomSeed,
                                                                 double ELeftSeed,
                                                                 double ERightSeed,
                                                                 double E2x5MaxSeed,
                                                                 double E2x5TopSeed,
                                                                 double E2x5BottomSeed,
                                                                 double E2x5LeftSeed,
                                                                 double E2x5RightSeed,
                                                                 double IEtaSeed,
                                                                 double IPhiSeed,
                                                                 double EtaCrySeed,
                                                                 double PhiCrySeed,
                                                                 double PreShowerOverRaw, 
                                                                 bool   printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if type is correct
  if (!(fVersionType == kNoTrkVar)) {
    std::cout << "Error: Regression VersionType " << fVersionType << " is not supported to use function regressionValueNoTrkVar.\n";
    return 0;
  }

  // Now applying regression according to version and (endcap/barrel)
  float *vals = (fabs(scEta) <= 1.479) ? new float[38] : new float[31];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IEtaSeed;
    vals[31] = IPhiSeed;
    vals[32] = ((int) IEtaSeed)%5;
    vals[33] = ((int) IPhiSeed)%2;
    vals[34] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[35] = ((int) IPhiSeed)%20;
    vals[36] = EtaCrySeed;
    vals[37] = PhiCrySeed;
  }
  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;
  Int_t BinIndex = -1;

  if (fVersionType == kNoTrkVar) {
    if (fabs(scEta) <= 1.479) { 
      regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals); 
      BinIndex = 0;
    }
    else {
      regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
      BinIndex = 1;
    }
  }

  //print debug
  if (printDebug) {    
    if ( fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 38; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 31; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "BinIndex : " << BinIndex << "\n";
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }
  

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionUncertaintyNoTrkVar(
                                                                       double SCRawEnergy,
                                                                       double scEta,
                                                                       double scPhi,
                                                                       double R9,
                                                                       double etawidth,
                                                                       double phiwidth,
                                                                       double NClusters,
                                                                       double HoE,
                                                                       double rho,
                                                                       double vertices,
                                                                       double EtaSeed,
                                                                       double PhiSeed,
                                                                       double ESeed,
                                                                       double E3x3Seed,
                                                                       double E5x5Seed,
                                                                       double see,
                                                                       double spp,
                                                                       double sep,
                                                                       double EMaxSeed,
                                                                       double E2ndSeed,
                                                                       double ETopSeed,
                                                                       double EBottomSeed,
                                                                       double ELeftSeed,
                                                                       double ERightSeed,
                                                                       double E2x5MaxSeed,
                                                                       double E2x5TopSeed,
                                                                       double E2x5BottomSeed,
                                                                       double E2x5LeftSeed,
                                                                       double E2x5RightSeed,
                                                                       double IEtaSeed,
                                                                       double IPhiSeed,
                                                                       double EtaCrySeed,
                                                                       double PhiCrySeed,
                                                                       double PreShowerOverRaw, 
                                                                       bool   printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if type is correct
  if (!(fVersionType == kNoTrkVar)) {
    std::cout << "Error: Regression VersionType " << fVersionType << " is not supported to use function regressionValueNoTrkVar.\n";
    return 0;
  }

  // Now applying regression according to version and (endcap/barrel)
  float *vals = (fabs(scEta) <= 1.479) ? new float[38] : new float[31];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IEtaSeed;
    vals[31] = IPhiSeed;
    vals[32] = ((int) IEtaSeed)%5;
    vals[33] = ((int) IPhiSeed)%2;
    vals[34] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[35] = ((int) IPhiSeed)%20;
    vals[36] = EtaCrySeed;
    vals[37] = PhiCrySeed;
  }
  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;
  Int_t BinIndex = -1;

  if (fVersionType == kNoTrkVar) {
    if (fabs(scEta) <= 1.479) { 
      regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals); 
      BinIndex = 0;
    }
    else {
      regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
      BinIndex = 1;
    }
  }

  //print debug
  if (printDebug) {    
    if (fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 38; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 31; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "BinIndex : " << BinIndex << "\n";
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }
  

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}




double ElectronEnergyRegressionEvaluate::regressionValueNoTrkVarV1(
                                                                 double SCRawEnergy,
                                                                 double scEta,
                                                                 double scPhi,
                                                                 double R9,
                                                                 double etawidth,
                                                                 double phiwidth,
                                                                 double NClusters,
                                                                 double HoE,
                                                                 double rho,
                                                                 double vertices,
                                                                 double EtaSeed,
                                                                 double PhiSeed,
                                                                 double ESeed,
                                                                 double E3x3Seed,
                                                                 double E5x5Seed,
                                                                 double see,
                                                                 double spp,
                                                                 double sep,
                                                                 double EMaxSeed,
                                                                 double E2ndSeed,
                                                                 double ETopSeed,
                                                                 double EBottomSeed,
                                                                 double ELeftSeed,
                                                                 double ERightSeed,
                                                                 double E2x5MaxSeed,
                                                                 double E2x5TopSeed,
                                                                 double E2x5BottomSeed,
                                                                 double E2x5LeftSeed,
                                                                 double E2x5RightSeed,
                                                                 double IEtaSeed,
                                                                 double IPhiSeed,
                                                                 double EtaCrySeed,
                                                                 double PhiCrySeed,
                                                                 double PreShowerOverRaw, 
                                                                 int    IsEcalDriven,
                                                                 bool   printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if type is correct
  if (!(fVersionType == kNoTrkVarV1)) {
    std::cout << "Error: Regression VersionType " << fVersionType << " is not supported to use function regressionValueNoTrkVar.\n";
    return 0;
  }

  // Now applying regression according to version and (endcap/barrel)
  float *vals = (fabs(scEta) <= 1.479) ? new float[39] : new float[32];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = IEtaSeed;
    vals[32] = IPhiSeed;
    vals[33] = ((int) IEtaSeed)%5;
    vals[34] = ((int) IPhiSeed)%2;
    vals[35] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[36] = ((int) IPhiSeed)%20;
    vals[37] = EtaCrySeed;
    vals[38] = PhiCrySeed;
  }
  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;
  Int_t BinIndex = -1;

  if (fVersionType == kNoTrkVarV1) {
    if (fabs(scEta) <= 1.479) { 
      regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals); 
      BinIndex = 0;
    }
    else {
      regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
      BinIndex = 1;
    }
  }

  //print debug
  if (printDebug) {    
    if ( fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 39; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 32; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "BinIndex : " << BinIndex << "\n";
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }
  

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionUncertaintyNoTrkVarV1(
                                                                       double SCRawEnergy,
                                                                       double scEta,
                                                                       double scPhi,
                                                                       double R9,
                                                                       double etawidth,
                                                                       double phiwidth,
                                                                       double NClusters,
                                                                       double HoE,
                                                                       double rho,
                                                                       double vertices,
                                                                       double EtaSeed,
                                                                       double PhiSeed,
                                                                       double ESeed,
                                                                       double E3x3Seed,
                                                                       double E5x5Seed,
                                                                       double see,
                                                                       double spp,
                                                                       double sep,
                                                                       double EMaxSeed,
                                                                       double E2ndSeed,
                                                                       double ETopSeed,
                                                                       double EBottomSeed,
                                                                       double ELeftSeed,
                                                                       double ERightSeed,
                                                                       double E2x5MaxSeed,
                                                                       double E2x5TopSeed,
                                                                       double E2x5BottomSeed,
                                                                       double E2x5LeftSeed,
                                                                       double E2x5RightSeed,
                                                                       double IEtaSeed,
                                                                       double IPhiSeed,
                                                                       double EtaCrySeed,
                                                                       double PhiCrySeed,
                                                                       double PreShowerOverRaw, 
                                                                       int    IsEcalDriven,
                                                                       bool   printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if type is correct
  if (!(fVersionType == kNoTrkVarV1)) {
    std::cout << "Error: Regression VersionType " << fVersionType << " is not supported to use function regressionValueNoTrkVar.\n";
    return 0;
  }

  // Now applying regression according to version and (endcap/barrel)
  float *vals = (fabs(scEta) <= 1.479) ? new float[39] : new float[32];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = IEtaSeed;
    vals[32] = IPhiSeed;
    vals[33] = ((int) IEtaSeed)%5;
    vals[34] = ((int) IPhiSeed)%2;
    vals[35] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[36] = ((int) IPhiSeed)%20;
    vals[37] = EtaCrySeed;
    vals[38] = PhiCrySeed;
  }
  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;
  Int_t BinIndex = -1;

  if (fVersionType == kNoTrkVarV1) {
    if (fabs(scEta) <= 1.479) { 
      regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals); 
      BinIndex = 0;
    }
    else {
      regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
      BinIndex = 1;
    }
  }

  //print debug
  if (printDebug) {    
    if (fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 39; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 32; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "BinIndex : " << BinIndex << "\n";
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }
  

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}



// This option is now deprecated. we keep it only
// for backwards compatibility
double ElectronEnergyRegressionEvaluate::regressionValueWithTrkVar(
                                                                   double electronP, 
                                                                   double SCRawEnergy,                
                                                                   double scEta,
                                                                   double scPhi,
                                                                   double R9,
                                                                   double etawidth,
                                                                   double phiwidth,
                                                                   double NClusters,
                                                                   double HoE,
                                                                   double rho,
                                                                   double vertices,
                                                                   double EtaSeed,
                                                                   double PhiSeed,
                                                                   double ESeed,
                                                                   double E3x3Seed,
                                                                   double E5x5Seed,
                                                                   double see,
                                                                   double spp,
                                                                   double sep,
                                                                   double EMaxSeed,
                                                                   double E2ndSeed,
                                                                   double ETopSeed,
                                                                   double EBottomSeed,
                                                                   double ELeftSeed,
                                                                   double ERightSeed,
                                                                   double E2x5MaxSeed,
                                                                   double E2x5TopSeed,
                                                                   double E2x5BottomSeed,
                                                                   double E2x5LeftSeed,
                                                                   double E2x5RightSeed,
                                                                   double pt,
                                                                   double GsfTrackPIn,
                                                                   double fbrem,
                                                                   double Charge,
                                                                   double EoP,
                                                                   double IEtaSeed,
                                                                   double IPhiSeed,
                                                                   double EtaCrySeed,
                                                                   double PhiCrySeed,
                                                                   double PreShowerOverRaw, 
                                                                   bool printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVar);

  float *vals = (fabs(scEta) <= 1.479) ? new float[43] : new float[36];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = pt;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = IEtaSeed;
    vals[36] = IPhiSeed;
    vals[37] = ((int) IEtaSeed)%5;
    vals[38] = ((int) IPhiSeed)%2;
    vals[39] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[40] = ((int) IPhiSeed)%20;
    vals[41] = EtaCrySeed;
    vals[42] = PhiCrySeed;
  }

  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = pt;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVar) {
    if (fabs(scEta) <= 1.479) regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
    else regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
  }


  //print debug
  if (printDebug) {
    if (scEta <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 43; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 36; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "pt = " << pt << " : SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}




// This option is now deprecated. we keep it only
// for backwards compatibility
double ElectronEnergyRegressionEvaluate::regressionUncertaintyWithTrkVar(
                                                                         double electronP, 
                                                                         double SCRawEnergy,                
                                                                         double scEta,
                                                                         double scPhi,
                                                                         double R9,
                                                                         double etawidth,
                                                                         double phiwidth,
                                                                         double NClusters,
                                                                         double HoE,
                                                                         double rho,
                                                                         double vertices,
                                                                         double EtaSeed,
                                                                         double PhiSeed,
                                                                         double ESeed,
                                                                         double E3x3Seed,
                                                                         double E5x5Seed,
                                                                         double see,
                                                                         double spp,
                                                                         double sep,
                                                                         double EMaxSeed,
                                                                         double E2ndSeed,
                                                                         double ETopSeed,
                                                                         double EBottomSeed,
                                                                         double ELeftSeed,
                                                                         double ERightSeed,
                                                                         double E2x5MaxSeed,
                                                                         double E2x5TopSeed,
                                                                         double E2x5BottomSeed,
                                                                         double E2x5LeftSeed,
                                                                         double E2x5RightSeed,
                                                                         double pt,
                                                                         double GsfTrackPIn,
                                                                         double fbrem,
                                                                         double Charge,
                                                                         double EoP,
                                                                         double IEtaSeed,
                                                                         double IPhiSeed,
                                                                         double EtaCrySeed,
                                                                         double PhiCrySeed,
                                                                         double PreShowerOverRaw, 
                                                                         bool printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVar);

  float *vals = (fabs(scEta) <= 1.479) ? new float[43] : new float[36];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = pt;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = IEtaSeed;
    vals[36] = IPhiSeed;
    vals[37] = ((int) IEtaSeed)%5;
    vals[38] = ((int) IPhiSeed)%2;
    vals[39] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[40] = ((int) IPhiSeed)%20;
    vals[41] = EtaCrySeed;
    vals[42] = PhiCrySeed;
  }

  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = pt;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVar) {
    if (fabs(scEta) <= 1.479) regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
    else regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (scEta <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 43; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 36; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "pt = " << pt << " : SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }


  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}




double ElectronEnergyRegressionEvaluate::regressionValueWithTrkVarV1(
                                                                   double SCRawEnergy,  
                                                                   double scEta,
                                                                   double scPhi,
                                                                   double R9,
                                                                   double etawidth,
                                                                   double phiwidth,
                                                                   double NClusters,
                                                                   double HoE,
                                                                   double rho,
                                                                   double vertices,
                                                                   double EtaSeed,
                                                                   double PhiSeed,
                                                                   double ESeed,
                                                                   double E3x3Seed,
                                                                   double E5x5Seed,
                                                                   double see,
                                                                   double spp,
                                                                   double sep,
                                                                   double EMaxSeed,
                                                                   double E2ndSeed,
                                                                   double ETopSeed,
                                                                   double EBottomSeed,
                                                                   double ELeftSeed,
                                                                   double ERightSeed,
                                                                   double E2x5MaxSeed,
                                                                   double E2x5TopSeed,
                                                                   double E2x5BottomSeed,
                                                                   double E2x5LeftSeed,
                                                                   double E2x5RightSeed,
                                                                   double IEtaSeed,
                                                                   double IPhiSeed,
                                                                   double EtaCrySeed,
                                                                   double PhiCrySeed,
                                                                   double PreShowerOverRaw,            
                                                                   int    IsEcalDriven,
                                                                   double GsfTrackPIn,
                                                                   double fbrem,
                                                                   double Charge,
                                                                   double EoP,
                                                                   double TrackMomentumError,
                                                                   double EcalEnergyError,
                                                                   int    Classification,                           
                                                                   bool printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV1);

  float *vals = (fabs(scEta) <= 1.479) ? new float[46] : new float[39];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = IEtaSeed;
    vals[39] = IPhiSeed;
    vals[40] = ((int) IEtaSeed)%5;
    vals[41] = ((int) IPhiSeed)%2;
    vals[42] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[43] = ((int) IPhiSeed)%20;
    vals[44] = EtaCrySeed;
    vals[45] = PhiCrySeed;
  }

  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV1) {
    if (fabs(scEta) <= 1.479) regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
    else regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
  }


  //print debug
  if (printDebug) {
    if (fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 46; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 39; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}




double ElectronEnergyRegressionEvaluate::regressionUncertaintyWithTrkVarV1(
                                                                         double SCRawEnergy,
                                                                         double scEta,
                                                                         double scPhi,
                                                                         double R9,
                                                                         double etawidth,
                                                                         double phiwidth,
                                                                         double NClusters,
                                                                         double HoE,
                                                                         double rho,
                                                                         double vertices,
                                                                         double EtaSeed,
                                                                         double PhiSeed,
                                                                         double ESeed,
                                                                         double E3x3Seed,
                                                                         double E5x5Seed,
                                                                         double see,
                                                                         double spp,
                                                                         double sep,
                                                                         double EMaxSeed,
                                                                         double E2ndSeed,
                                                                         double ETopSeed,
                                                                         double EBottomSeed,
                                                                         double ELeftSeed,
                                                                         double ERightSeed,
                                                                         double E2x5MaxSeed,
                                                                         double E2x5TopSeed,
                                                                         double E2x5BottomSeed,
                                                                         double E2x5LeftSeed,
                                                                         double E2x5RightSeed,
                                                                         double IEtaSeed,
                                                                         double IPhiSeed,
                                                                         double EtaCrySeed,
                                                                         double PhiCrySeed,
                                                                         double PreShowerOverRaw,
                                                                         int    IsEcalDriven,
                                                                         double GsfTrackPIn,
                                                                         double fbrem,
                                                                         double Charge,
                                                                         double EoP,
                                                                         double TrackMomentumError,
                                                                         double EcalEnergyError,
                                                                         int    Classification,    
                                                                         bool printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV1);

  float *vals = (fabs(scEta) <= 1.479) ? new float[46] : new float[39];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = IEtaSeed;
    vals[39] = IPhiSeed;
    vals[40] = ((int) IEtaSeed)%5;
    vals[41] = ((int) IPhiSeed)%2;
    vals[42] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[43] = ((int) IPhiSeed)%20;
    vals[44] = EtaCrySeed;
    vals[45] = PhiCrySeed;
  }

  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV1) {
    if (fabs(scEta) <= 1.479) regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
    else regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 46; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 39; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << " SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }


  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}





double ElectronEnergyRegressionEvaluate::regressionValueWithTrkVarV1(std::vector<double> &inputvars,                                                                                   
                                                                     bool printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV1);

  // Checking if fVersionType is correct
  assert(inputvars.size() == 42);

  double SCRawEnergy  = inputvars[0];
  double scEta  = inputvars[1];
  double scPhi  = inputvars[2];
  double R9  = inputvars[3];
  double etawidth  = inputvars[4];
  double phiwidth  = inputvars[5];
  double NClusters  = inputvars[6];
  double HoE  = inputvars[7];
  double rho  = inputvars[8];
  double vertices  = inputvars[9];
  double EtaSeed  = inputvars[10];
  double PhiSeed  = inputvars[11];
  double ESeed  = inputvars[12];
  double E3x3Seed  = inputvars[13];
  double E5x5Seed  = inputvars[14];
  double see  = inputvars[15];
  double spp  = inputvars[16];
  double sep  = inputvars[17];
  double EMaxSeed  = inputvars[18];
  double E2ndSeed  = inputvars[19];
  double ETopSeed  = inputvars[20];
  double EBottomSeed  = inputvars[21];
  double ELeftSeed  = inputvars[22];
  double ERightSeed  = inputvars[23];
  double E2x5MaxSeed  = inputvars[24];
  double E2x5TopSeed  = inputvars[25];
  double E2x5BottomSeed  = inputvars[26];
  double E2x5LeftSeed  = inputvars[27];
  double E2x5RightSeed  = inputvars[28];
  double IEtaSeed  = inputvars[29];
  double IPhiSeed  = inputvars[30];
  double EtaCrySeed  = inputvars[31];
  double PhiCrySeed  = inputvars[32];
  double PreShowerOverRaw  = inputvars[33];
  int    IsEcalDriven  = inputvars[34];
  double GsfTrackPIn  = inputvars[35];
  double fbrem  = inputvars[36];
  double Charge  = inputvars[37];
  double EoP  = inputvars[38];
  double TrackMomentumError  = inputvars[39];
  double EcalEnergyError  = inputvars[40];
  int    Classification  = inputvars[41]; 

  float *vals = (fabs(scEta) <= 1.479) ? new float[46] : new float[39];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = IEtaSeed;
    vals[39] = IPhiSeed;
    vals[40] = ((int) IEtaSeed)%5;
    vals[41] = ((int) IPhiSeed)%2;
    vals[42] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[43] = ((int) IPhiSeed)%20;
    vals[44] = EtaCrySeed;
    vals[45] = PhiCrySeed;
  }

  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV1) {
    if (fabs(scEta) <= 1.479) regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
    else regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
  }


  //print debug
  if (printDebug) {
    if (fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 46; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 39; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}




double ElectronEnergyRegressionEvaluate::regressionUncertaintyWithTrkVarV1(std::vector<double> &inputvars,                                                                          
                                                                           bool printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV1);

  // Checking if fVersionType is correct
  assert(inputvars.size() == 42);

  double SCRawEnergy  = inputvars[0];
  double scEta  = inputvars[1];
  double scPhi  = inputvars[2];
  double R9  = inputvars[3];
  double etawidth  = inputvars[4];
  double phiwidth  = inputvars[5];
  double NClusters  = inputvars[6];
  double HoE  = inputvars[7];
  double rho  = inputvars[8];
  double vertices  = inputvars[9];
  double EtaSeed  = inputvars[10];
  double PhiSeed  = inputvars[11];
  double ESeed  = inputvars[12];
  double E3x3Seed  = inputvars[13];
  double E5x5Seed  = inputvars[14];
  double see  = inputvars[15];
  double spp  = inputvars[16];
  double sep  = inputvars[17];
  double EMaxSeed  = inputvars[18];
  double E2ndSeed  = inputvars[19];
  double ETopSeed  = inputvars[20];
  double EBottomSeed  = inputvars[21];
  double ELeftSeed  = inputvars[22];
  double ERightSeed  = inputvars[23];
  double E2x5MaxSeed  = inputvars[24];
  double E2x5TopSeed  = inputvars[25];
  double E2x5BottomSeed  = inputvars[26];
  double E2x5LeftSeed  = inputvars[27];
  double E2x5RightSeed  = inputvars[28];
  double IEtaSeed  = inputvars[29];
  double IPhiSeed  = inputvars[30];
  double EtaCrySeed  = inputvars[31];
  double PhiCrySeed  = inputvars[32];
  double PreShowerOverRaw  = inputvars[33];
  int    IsEcalDriven  = inputvars[34];
  double GsfTrackPIn  = inputvars[35];
  double fbrem  = inputvars[36];
  double Charge  = inputvars[37];
  double EoP  = inputvars[38];
  double TrackMomentumError  = inputvars[39];
  double EcalEnergyError  = inputvars[40];
  int    Classification  = inputvars[41]; 


  float *vals = (fabs(scEta) <= 1.479) ? new float[46] : new float[39];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = IEtaSeed;
    vals[39] = IPhiSeed;
    vals[40] = ((int) IEtaSeed)%5;
    vals[41] = ((int) IPhiSeed)%2;
    vals[42] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[43] = ((int) IPhiSeed)%20;
    vals[44] = EtaCrySeed;
    vals[45] = PhiCrySeed;
  }

  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV1) {
    if (fabs(scEta) <= 1.479) regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
    else regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 46; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 39; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << " SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }


  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}



double ElectronEnergyRegressionEvaluate::regressionValueWithTrkVarV2(
                                                                   double SCRawEnergy,
                                                                   double scEta,
                                                                   double scPhi,
                                                                   double R9,
                                                                   double etawidth,
                                                                   double phiwidth,
                                                                   double NClusters,
                                                                   double HoE,
                                                                   double rho,
                                                                   double vertices,
                                                                   double EtaSeed,
                                                                   double PhiSeed,
                                                                   double ESeed,
                                                                   double E3x3Seed,
                                                                   double E5x5Seed,
                                                                   double see,
                                                                   double spp,
                                                                   double sep,
                                                                   double EMaxSeed,
                                                                   double E2ndSeed,
                                                                   double ETopSeed,
                                                                   double EBottomSeed,
                                                                   double ELeftSeed,
                                                                   double ERightSeed,
                                                                   double E2x5MaxSeed,
                                                                   double E2x5TopSeed,
                                                                   double E2x5BottomSeed,
                                                                   double E2x5LeftSeed,
                                                                   double E2x5RightSeed,
                                                                   double IEtaSeed,
                                                                   double IPhiSeed,
                                                                   double EtaCrySeed,
                                                                   double PhiCrySeed,
                                                                   double PreShowerOverRaw,
                                                                   int    IsEcalDriven,
                                                                   double GsfTrackPIn,
                                                                   double fbrem,
                                                                   double Charge,
                                                                   double EoP,
                                                                   double TrackMomentumError,
                                                                   double EcalEnergyError,
                                                                   int    Classification,           
                                                                   double detaIn,
                                                                   double dphiIn,
                                                                   double detaCalo,
                                                                   double dphiCalo,
                                                                   double GsfTrackChiSqr,
                                                                   double KFTrackNLayers,
                                                                   double ElectronEnergyOverPout,
                                                                   bool printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV2);

  float *vals = (fabs(scEta) <= 1.479) ? new float[53] : new float[46];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = IEtaSeed;
    vals[46] = IPhiSeed;
    vals[47] = ((int) IEtaSeed)%5;
    vals[48] = ((int) IPhiSeed)%2;
    vals[49] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[50] = ((int) IPhiSeed)%20;
    vals[51] = EtaCrySeed;
    vals[52] = PhiCrySeed;
  }

  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV2) {
    if (fabs(scEta) <= 1.479) regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
    else regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
  }


  //print debug
  if (printDebug) {
    if (fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 53; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 46; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}




double ElectronEnergyRegressionEvaluate::regressionUncertaintyWithTrkVarV2(
                                                                         double SCRawEnergy,                
                                                                         double scEta,
                                                                         double scPhi,
                                                                         double R9,
                                                                         double etawidth,
                                                                         double phiwidth,
                                                                         double NClusters,
                                                                         double HoE,
                                                                         double rho,
                                                                         double vertices,
                                                                         double EtaSeed,
                                                                         double PhiSeed,
                                                                         double ESeed,
                                                                         double E3x3Seed,
                                                                         double E5x5Seed,
                                                                         double see,
                                                                         double spp,
                                                                         double sep,
                                                                         double EMaxSeed,
                                                                         double E2ndSeed,
                                                                         double ETopSeed,
                                                                         double EBottomSeed,
                                                                         double ELeftSeed,
                                                                         double ERightSeed,
                                                                         double E2x5MaxSeed,
                                                                         double E2x5TopSeed,
                                                                         double E2x5BottomSeed,
                                                                         double E2x5LeftSeed,
                                                                         double E2x5RightSeed,
                                                                         double IEtaSeed,
                                                                         double IPhiSeed,
                                                                         double EtaCrySeed,
                                                                         double PhiCrySeed,
                                                                         double PreShowerOverRaw, 
                                                                         int    IsEcalDriven,
                                                                         double GsfTrackPIn,
                                                                         double fbrem,
                                                                         double Charge,
                                                                         double EoP,
                                                                         double TrackMomentumError,
                                                                         double EcalEnergyError,
                                                                         int    Classification,
                                                                         double detaIn,
                                                                         double dphiIn,
                                                                         double detaCalo,
                                                                         double dphiCalo,
                                                                         double GsfTrackChiSqr,
                                                                         double KFTrackNLayers,
                                                                         double ElectronEnergyOverPout,
                                                                         bool printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV2);

  float *vals = (fabs(scEta) <= 1.479) ? new float[53] : new float[46];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = IEtaSeed;
    vals[46] = IPhiSeed;
    vals[47] = ((int) IEtaSeed)%5;
    vals[48] = ((int) IPhiSeed)%2;
    vals[49] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[50] = ((int) IPhiSeed)%20;
    vals[51] = EtaCrySeed;
    vals[52] = PhiCrySeed;
  }

  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV2) {
    if (fabs(scEta) <= 1.479) regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
    else regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 53; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 46; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }


  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}




double ElectronEnergyRegressionEvaluate::regressionValueWithTrkVarV2(std::vector<double> &inputvars,
                                                                     bool printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV2);

  // Checking if fVersionType is correct
  assert(inputvars.size() == 49);

  double SCRawEnergy  = inputvars[0];
  double scEta  = inputvars[1];
  double scPhi  = inputvars[2];
  double R9  = inputvars[3];
  double etawidth  = inputvars[4];
  double phiwidth  = inputvars[5];
  double NClusters  = inputvars[6];
  double HoE  = inputvars[7];
  double rho  = inputvars[8];
  double vertices  = inputvars[9];
  double EtaSeed  = inputvars[10];
  double PhiSeed  = inputvars[11];
  double ESeed  = inputvars[12];
  double E3x3Seed  = inputvars[13];
  double E5x5Seed  = inputvars[14];
  double see  = inputvars[15];
  double spp  = inputvars[16];
  double sep  = inputvars[17];
  double EMaxSeed  = inputvars[18];
  double E2ndSeed  = inputvars[19];
  double ETopSeed  = inputvars[20];
  double EBottomSeed  = inputvars[21];
  double ELeftSeed  = inputvars[22];
  double ERightSeed  = inputvars[23];
  double E2x5MaxSeed  = inputvars[24];
  double E2x5TopSeed  = inputvars[25];
  double E2x5BottomSeed  = inputvars[26];
  double E2x5LeftSeed  = inputvars[27];
  double E2x5RightSeed  = inputvars[28];
  double IEtaSeed  = inputvars[29];
  double IPhiSeed  = inputvars[30];
  double EtaCrySeed  = inputvars[31];
  double PhiCrySeed  = inputvars[32];
  double PreShowerOverRaw  = inputvars[33];
  int    IsEcalDriven  = inputvars[34];
  double GsfTrackPIn  = inputvars[35];
  double fbrem  = inputvars[36];
  double Charge  = inputvars[37];
  double EoP  = inputvars[38];
  double TrackMomentumError  = inputvars[39];
  double EcalEnergyError  = inputvars[40];
  int    Classification  = inputvars[41]; 
  double detaIn  = inputvars[42];
  double dphiIn  = inputvars[43];
  double detaCalo  = inputvars[44];
  double dphiCalo  = inputvars[45];
  double GsfTrackChiSqr  = inputvars[46];
  double KFTrackNLayers  = inputvars[47];
  double ElectronEnergyOverPout  = inputvars[48];

  float *vals = (fabs(scEta) <= 1.479) ? new float[53] : new float[46];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = IEtaSeed;
    vals[46] = IPhiSeed;
    vals[47] = ((int) IEtaSeed)%5;
    vals[48] = ((int) IPhiSeed)%2;
    vals[49] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[50] = ((int) IPhiSeed)%20;
    vals[51] = EtaCrySeed;
    vals[52] = PhiCrySeed;
  }

  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV2) {
    if (fabs(scEta) <= 1.479) regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
    else regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
  }


  //print debug
  if (printDebug) {
    if (fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 53; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 46; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}




double ElectronEnergyRegressionEvaluate::regressionUncertaintyWithTrkVarV2(std::vector<double> &inputvars,                                                                        
                                                                           bool printDebug) 
{
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV2);

  // Checking if fVersionType is correct
  assert(inputvars.size() == 49);

  double SCRawEnergy  = inputvars[0];
  double scEta  = inputvars[1];
  double scPhi  = inputvars[2];
  double R9  = inputvars[3];
  double etawidth  = inputvars[4];
  double phiwidth  = inputvars[5];
  double NClusters  = inputvars[6];
  double HoE  = inputvars[7];
  double rho  = inputvars[8];
  double vertices  = inputvars[9];
  double EtaSeed  = inputvars[10];
  double PhiSeed  = inputvars[11];
  double ESeed  = inputvars[12];
  double E3x3Seed  = inputvars[13];
  double E5x5Seed  = inputvars[14];
  double see  = inputvars[15];
  double spp  = inputvars[16];
  double sep  = inputvars[17];
  double EMaxSeed  = inputvars[18];
  double E2ndSeed  = inputvars[19];
  double ETopSeed  = inputvars[20];
  double EBottomSeed  = inputvars[21];
  double ELeftSeed  = inputvars[22];
  double ERightSeed  = inputvars[23];
  double E2x5MaxSeed  = inputvars[24];
  double E2x5TopSeed  = inputvars[25];
  double E2x5BottomSeed  = inputvars[26];
  double E2x5LeftSeed  = inputvars[27];
  double E2x5RightSeed  = inputvars[28];
  double IEtaSeed  = inputvars[29];
  double IPhiSeed  = inputvars[30];
  double EtaCrySeed  = inputvars[31];
  double PhiCrySeed  = inputvars[32];
  double PreShowerOverRaw  = inputvars[33];
  int    IsEcalDriven  = inputvars[34];
  double GsfTrackPIn  = inputvars[35];
  double fbrem  = inputvars[36];
  double Charge  = inputvars[37];
  double EoP  = inputvars[38];
  double TrackMomentumError  = inputvars[39];
  double EcalEnergyError  = inputvars[40];
  int    Classification  = inputvars[41]; 
  double detaIn  = inputvars[42];
  double dphiIn  = inputvars[43];
  double detaCalo  = inputvars[44];
  double dphiCalo  = inputvars[45];
  double GsfTrackChiSqr  = inputvars[46];
  double KFTrackNLayers  = inputvars[47];
  double ElectronEnergyOverPout  = inputvars[48];

  float *vals = (fabs(scEta) <= 1.479) ? new float[53] : new float[46];
  if (fabs(scEta) <= 1.479) {		// Barrel
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = IEtaSeed;
    vals[46] = IPhiSeed;
    vals[47] = ((int) IEtaSeed)%5;
    vals[48] = ((int) IPhiSeed)%2;
    vals[49] = (abs(IEtaSeed)<=25)*(((int)IEtaSeed)%25) + (abs(IEtaSeed)>25)*(((int) (IEtaSeed-25*abs(IEtaSeed)/IEtaSeed))%20);
    vals[50] = ((int) IPhiSeed)%20;
    vals[51] = EtaCrySeed;
    vals[52] = PhiCrySeed;
  }

  else {	// Endcap
    vals[0]  = SCRawEnergy;
    vals[1]  = scEta;
    vals[2]  = scPhi;
    vals[3]  = R9;
    vals[4]  = E5x5Seed/SCRawEnergy;
    vals[5]  = etawidth;
    vals[6]  = phiwidth;
    vals[7]  = NClusters;
    vals[8]  = HoE;
    vals[9]  = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed-scPhi),cos(PhiSeed-scPhi));
    vals[13] = ESeed/SCRawEnergy;
    vals[14] = E3x3Seed/ESeed;
    vals[15] = E5x5Seed/ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed/ESeed;
    vals[20] = E2ndSeed/ESeed;
    vals[21] = ETopSeed/ESeed;
    vals[22] = EBottomSeed/ESeed;
    vals[23] = ELeftSeed/ESeed;
    vals[24] = ERightSeed/ESeed;
    vals[25] = E2x5MaxSeed/ESeed;
    vals[26] = E2x5TopSeed/ESeed;
    vals[27] = E2x5BottomSeed/ESeed;
    vals[28] = E2x5LeftSeed/ESeed;
    vals[29] = E2x5RightSeed/ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError/GsfTrackPIn;
    vals[36] = EcalEnergyError/SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV2) {
    if (fabs(scEta) <= 1.479) regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
    else regressionResult = (SCRawEnergy*(1+PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (fabs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (uint v=0; v < 53; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    else {
      std::cout << "Endcap :";
      for (uint v=0; v < 46; ++v) std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }


  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

