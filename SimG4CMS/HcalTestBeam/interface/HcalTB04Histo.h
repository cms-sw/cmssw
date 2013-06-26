#ifndef HcalTestBeam_HcalTB04Histo_H
#define HcalTestBeam_HcalTB04Histo_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB04Histo
//
/**\class HcalTB04Histo HcalTB04Histo.h SimG4CMS/HcalTestBeam/interface/HcalTB04Histo.h
  
 Description: Histogram handling for Hcal Test Beam 2004 studies
  
 Usage: Sets up histograms and stores in a file
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Thu May 18 10:14:34 CEST 2006
// $Id: HcalTB04Histo.h,v 1.5 2013/05/28 17:42:14 gartung Exp $
//
  
// system include files
#include<string>
#include<vector>
 
// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>

class HcalTB04Histo {
   
public:
 
  // ---------- Constructor and destructor -----------------
  HcalTB04Histo(const edm::ParameterSet &ps);
  virtual ~HcalTB04Histo();

  // ---------- member functions ---------------------------
  void fillPrimary(double energy, double eta, double phi);
  void fillEdep(double etots, double eecals, double ehcals, 
		double etotq, double eecalq, double ehcalq);
  void fillTrnsProf(const std::vector<double>& es1, const std::vector<double>& eq1,
                    const std::vector<double>& es2, const std::vector<double>& eq2);
  void fillLongProf(const std::vector<double>& es, const std::vector<double>& eq);
                                                                               
private:

  // ---------- Private Data members -----------------------
  bool                  verbose;
  double                eTotMax, eHcalMax;

  TH1D                  *iniE,  *iEta,  *iPhi;
  TH1D                  *edepS, *edecS, *edhcS, *edepQ, *edecQ, *edhcQ;
  TH2D                  *edehS, *edehQ;
  TProfile              *latse, *latqe, *latsf, *latqf, *lngs, *lngq;
};
 
#endif
