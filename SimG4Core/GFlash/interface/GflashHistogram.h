#ifndef SimG4Core_GFlash_GflashHistogram_H
#define SimG4Core_GFlash_GflashHistogram_H

// created by Soon Yung Jun, Dongwook Jang, 2007/12/07

#include <TObject.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TString.h>


class GflashHistogram : public TObject {

 public:

  static GflashHistogram* instance();
  
  GflashHistogram();
  ~GflashHistogram();

  void bookHistogram(TString histFileName="gflash_histogram.root");
  void setStoreFlag(bool flag) { theStoreFlag = flag; }
  bool getStoreFlag() { return theStoreFlag; }

  // We are declaring histograms here as public
  // This is just for convenience.
  // Once we are settled down with all histograms,
  // probably we will make them as private

  // add hitograms here

  // histogram output
  TFile*    histFile_;

  // histograms for EM shower model in GflashEMShowerProfile
  TH1F*     incE_atEcal;
  TH2F*     dEdz;
  TH1F*     dEdz_p;
  TH1F*     dndz_spot;
  TH2F*     rxry;
  TH1F*     dx;
  TH2F*     xdz;
  TH2F*     rzSpots;
  TH1F*     rho_ssp;
  TH1F*     rArm;

  // histograms for HAD shower model in GflashHadronShowerModel

  TH1F*     preStepPosition;
  TH1F*     postStepPosition;
  TH1F*     deltaStep;
  TH1F*     kineticEnergy;
  TH1F*     energyLoss;

  // histograms for HAD shower model in GflashHadronShowerProfile

  TH1F*     rshower;
  TH1F*     lateralx;
  TH1F*     lateraly;

  TH1F*     gfhlong;
  TH1F*     gfhsll;
  TH1F*     gfhssp;
  TH1F*     gfheinc;
  TH2F*     gfhlongProfile;

 private:

  static GflashHistogram* instance_;
  bool   theStoreFlag;

};

#endif
