#ifndef HcalTestBeam_HcalTB02Histo_H
#define HcalTestBeam_HcalTB02Histo_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02Histo
//
/**\class HcalTB02Histo HcalTB02Histo.h SimG4CMS/HcalTestBeam/interface/HcalTB02Histo.h
  
 Description: Histogram handling for Hcal Test Beam 2002 studies
  
 Usage: Sets up histograms and stores in a file
*/
//
// Original Author:  
//         Created:  Thu Sun 21 10:14:34 CEST 2006
// $Id: HcalTB02Histo.h,v 1.1 2006/05/23 10:53:29 sunanda Exp $
//
  
// system include files
#include<string>
#include<vector>
 
// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class HcalTB02Histo {
   
public:
 
  // ---------- Constructor and destructor -----------------
  HcalTB02Histo(const edm::ParameterSet &ps);
  virtual ~HcalTB02Histo();

  // ---------- member functions ---------------------------
  void fillAllTime(float v);
  void fillTransProf(float u, float v);
  void fillProfile(int ilayer, float value);
  float getMean(int ilayer);
  float getRMS (int ilayer);
                                                                               
private:

  // ---------- Private Data members -----------------------
  std::string                   fileName;
  bool                          verbose;

  DQMStore                     *dbe_;
  MonitorElement               *rt_tbTimes, *rt_TransProf;
  std::vector<MonitorElement *> rt_histoProf;
};
 
#endif
