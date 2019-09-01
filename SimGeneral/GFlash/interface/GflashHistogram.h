#ifndef GflashHistogram_H
#define GflashHistogram_H

// created by Soon Yung Jun, Dongwook Jang, 2007/12/07

#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TObject.h>
#include <TProfile.h>
#include <TString.h>
#include <TTree.h>

#include <string>

class GflashHistogram : public TObject {
public:
  static GflashHistogram *instance();

  GflashHistogram();
  ~GflashHistogram() override;

  void bookHistogram(std::string histFileName = "gflash_histogram.root");
  void setStoreFlag(bool flag) { theStoreFlag = flag; }
  bool getStoreFlag() { return theStoreFlag; }

  // We are declaring histograms here as public
  // This is just for convenience.
  // Once we are settled down with all histograms,
  // probably we will make them as private

  // add hitograms here

  // histogram output
  TFile *histFile_;

  // histograms for EM shower model in GflashEMShowerProfile
  TH1F *em_incE;
  TH1F *em_ssp_rho;
  TH1F *em_ssp_z;
  TH1F *em_long;
  TH2F *em_lateral;
  TH1F *em_long_sd;
  TH2F *em_lateral_sd;
  TH1F *em_nSpots_sd;

  // histograms for HAD shower model in GflashHadronShowerModel

  TH1F *preStepPosition;
  TH1F *postStepPosition;
  TH1F *deltaStep;
  TH1F *kineticEnergy;
  TH1F *energyLoss;
  TH1F *energyRatio;

  // histograms for HAD shower model in GflashHadronShowerProfile

  TH1F *rshower;
  TH1F *lateralx;
  TH1F *lateraly;
  TH2F *gfhlongProfile;

  // histograms for Watcher

  TH1F *g4vertexTrack;
  TH2F *g4stepCharge;
  TH1F *g4nSecondary;
  TH1F *g4pidSecondary;
  TH1F *g4energySecondary;
  TH1F *g4energyPi0;
  TH1F *g4energyElectron;
  TH1F *g4energyPhoton;

  TH1F *g4totalEnergySecPi0;
  TH1F *g4totalEnergySecElectron;
  TH1F *g4totalEnergySecPhoton;

  TH1F *g4energyEM;
  TH1F *g4energyHad;
  TH1F *g4energyTotal;
  TH1F *g4energyEMMip;
  TH1F *g4energyHadMip;
  TH1F *g4energyMip;
  TH2F *g4energyEMvsHad;

  TH1F *g4energySensitiveEM;
  TH1F *g4energySensitiveHad;
  TH1F *g4energySensitiveTotal;
  TH1F *g4energyHybridTotal;
  TH1F *g4energySensitiveEMMip;
  TH2F *g4energySensitiveEMvsHad;

  TH2F *g4energyEMProfile;
  TH2F *g4energyHadProfile;
  TH2F *g4energyTotalProfile;
  TH2F *g4energyHybridProfile;

  TH1F *g4ssp;
  TH1F *g4energy;
  TH1F *g4energyLoss;
  TH1F *g4momentum;
  TH1F *g4charge;

  TH2F *g4rshower;
  TH2F *g4rshowerR1;
  TH2F *g4rshowerR2;
  TH2F *g4rshowerR3;
  TH2F *g4lateralXY;
  TH2F *g4lateralRZ;
  TH2F *g4spotXY;
  TH2F *g4spotRZ;
  TH2F *g4spotRZ0;
  TH2F *g4trajectoryXY;
  TH2F *g4trajectoryRZ;

  TH1F *g4stepRho;
  TH1F *g4trajectoryPhi0;

  TH2F *g4longProfile;
  TH2F *g4longDetector;
  TH2F *g4longSensitive;

private:
  static GflashHistogram *instance_;
  bool theStoreFlag;
};

#endif
