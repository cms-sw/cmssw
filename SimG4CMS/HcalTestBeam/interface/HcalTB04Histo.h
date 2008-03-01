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
// $Id: HcalTB04Histo.h,v 1.2 2006/05/23 10:53:29 sunanda Exp $
//
  
// system include files
#include<string>
#include<vector>
 
// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class HcalTB04Histo {
   
public:
 
  // ---------- Constructor and destructor -----------------
  HcalTB04Histo(const edm::ParameterSet &ps);
  virtual ~HcalTB04Histo();

  // ---------- member functions ---------------------------
  void fillPrimary(double energy, double eta, double phi);
  void fillEdep(double etots, double eecals, double ehcals, 
		double etotq, double eecalq, double ehcalq);
  void fillTrnsProf(std::vector<double> es1, std::vector<double> eq1,
                    std::vector<double> es2, std::vector<double> eq2);
  void fillLongProf(std::vector<double> es, std::vector<double> eq);
                                                                               
private:

  // ---------- Private Data members -----------------------
  std::string           fileName;
  bool                  verbose;
  double                eTotMax, eHcalMax;

  DQMStore              *dbe_;
  MonitorElement        *iniE,  *iEta,  *iPhi;
  MonitorElement        *edepS, *edecS, *edhcS, *edepQ, *edecQ, *edhcQ;
  MonitorElement        *edehS, *edehQ;
  MonitorElement        *latse, *latqe, *latsf, *latqf, *lngs, *lngq;
};
 
#endif
