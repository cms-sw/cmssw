#ifndef HcalTestBeam_HcalTB06Histo_H
#define HcalTestBeam_HcalTB06Histo_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB06Histo
//
/**\class HcalTB06Histo HcalTB06Histo.h SimG4CMS/HcalTestBeam/interface/HcalTB06Histo.h
  
 Description: Histogram handling for Hcal Test Beam 2006 studies
  
 Usage: Sets up histograms and stores in a file
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Oct 10 10:14:34 CEST 2006
//

// system include files
#include <string>
#include <vector>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <TH1D.h>
#include <TH2D.h>
#include <TTree.h>

class HcalTB06Histo {
public:
  // ---------- Constructor and destructor -----------------
  HcalTB06Histo(const edm::ParameterSet &ps);
  virtual ~HcalTB06Histo();

  // ---------- member functions ---------------------------
  void fillPrimary(double energy, double eta, double phi);
  void fillEdep(double etots, double eecals, double ehcals);
  void fillTree(std::vector<double> &ecalo, std::vector<double> &etrig);

private:
  // ---------- Private Data members -----------------------
  bool verbose_, mkTree_;

  TH1D *iniE, *iEta, *iPhi;
  TH1D *edepS, *edecS, *edhcS;
  TH1D *edepN, *edecN, *edhcN, *emhcN;
  TH2D *edehS;
  TTree *tree_;
  double eBeam_, etaBeam_, phiBeam_;
  double edepEC_, edepHB_, edepHO_;
  double noiseEC_, noiseHB_, noiseHO_;
  double edepS1_, edepS2_, edepS3_, edepS4_, edepVC_, edepS7_, edepS8_;
  double mip_;
};

#endif
