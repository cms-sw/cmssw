// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02Histo
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun May 21 10:14:34 CEST 2006
// $Id: HcalTB02Histo.cc,v 1.4 2010/01/12 07:11:39 hegner Exp $
//
 
// system include files
#include <iostream>
#include <cmath>
 
// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02Histo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
 
//
// constructors and destructor
HcalTB02Histo::HcalTB02Histo(const edm::ParameterSet& ps) :
  rt_tbTimes(0),  rt_TransProf(0) {

  verbose    = ps.getUntrackedParameter<bool>("Verbose", false);
  edm::LogInfo("HcalTBSim") << "HcalTB02Histo:: initialised with o/p file "
			    << fileName << " verbosity " << verbose;

  // Book histograms
  edm::Service<TFileService> tfile;

  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";

  char title[80];
  for (int ilayer=0; ilayer<19; ilayer++) {
    sprintf(title, "Scint. Energy in Layer %d ", ilayer);
    TH1D *h = tfile->make<TH1D>(title, title, 500, 0., 1.5);
    rt_histoProf.push_back(h);
    edm::LogInfo("HcalTBSim") << "HcalTB02Histo:: Initialise Histo " <<title;
  }
  sprintf(title, "All Hit Time slices");
  rt_tbTimes   = tfile->make<TH1D>(title, title, 100, 0., 100.);
  edm::LogInfo("HcalTBSim") << "HcalTB02Histo:: Initialise Histo " << title;
  sprintf(title, "Transv. Shower Profile");
  rt_TransProf = tfile->make<TH2D>(title, title, 100, 0., 1., 1000, 0., 0.5);
  edm::LogInfo("HcalTBSim") << "HcalTB02Histo:: Initialise Histo " << title;
}
 
HcalTB02Histo::~HcalTB02Histo() {
  edm::LogInfo("HcalTBSim") << " Deleting HcalTB02Histo";
}
 
//
// member functions
//
 
void HcalTB02Histo::fillAllTime(float v) {

  LogDebug("HcalTBSim") << "HcalTB02Histo::Fill Time histo with " << v;
  if (rt_tbTimes) {
    rt_tbTimes->Fill(v);
  }
}

void HcalTB02Histo::fillTransProf(float u, float v) {

  LogDebug("HcalTBSim") << "HcalTB02Histo:::Fill Shower Transv. Profile histo"
			<< " with " << u << " and " << v;
  if (rt_TransProf) {
    rt_TransProf->Fill(u,v);
  }
}

void HcalTB02Histo::fillProfile(int ilayer, float value) {

  if (ilayer < (int)(rt_histoProf.size())) {
    rt_histoProf[ilayer]->Fill(value);
    LogDebug("HcalTBSim") << "HcalTB02Histo::Fill profile " << ilayer
			  << " with " << value;
  }
}

float HcalTB02Histo::getMean(int ilayer) {

  if (ilayer < (int)(rt_histoProf.size())) {
    return rt_histoProf[ilayer]->GetMean();
  } else {
    return 0;
  }
}

float HcalTB02Histo::getRMS(int ilayer) {

  if (ilayer < (int)(rt_histoProf.size())) {
    return rt_histoProf[ilayer]->GetRMS();
  } else {
    return 0;
  }
}
