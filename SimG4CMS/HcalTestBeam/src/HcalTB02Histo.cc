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
//

// system include files
#include <iostream>
#include <cmath>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02Histo.h"

//#define EDM_ML_DEBUG
//
// constructors and destructor
HcalTB02Histo::HcalTB02Histo(const edm::ParameterSet& ps) : rt_tbTimes(nullptr), rt_TransProf(nullptr) {
  verbose = ps.getUntrackedParameter<bool>("Verbose", false);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB02Histo:: initialised with o/p file " << fileName << " verbosity " << verbose;
#endif
  // Book histograms
  edm::Service<TFileService> tfile;

  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";

  char title[80];
  for (int ilayer = 0; ilayer < 19; ilayer++) {
    sprintf(title, "Scint. Energy in Layer %d ", ilayer);
    TH1D* h = tfile->make<TH1D>(title, title, 500, 0., 1.5);
    rt_histoProf.push_back(h);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalTBSim") << "HcalTB02Histo:: Initialise Histo " << title;
#endif
  }
  sprintf(title, "All Hit Time slices");
  rt_tbTimes = tfile->make<TH1D>(title, title, 100, 0., 100.);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB02Histo:: Initialise Histo " << title;
#endif
  sprintf(title, "Transv. Shower Profile");
  rt_TransProf = tfile->make<TH2D>(title, title, 100, 0., 1., 1000, 0., 0.5);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB02Histo:: Initialise Histo " << title;
#endif
}

HcalTB02Histo::~HcalTB02Histo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << " Deleting HcalTB02Histo";
#endif
}

//
// member functions
//

void HcalTB02Histo::fillAllTime(float v) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB02Histo::Fill Time histo with " << v;
#endif
  if (rt_tbTimes) {
    rt_tbTimes->Fill(v);
  }
}

void HcalTB02Histo::fillTransProf(float u, float v) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB02Histo:::Fill Shower Transv. Profile histo"
                                << " with " << u << " and " << v;
#endif
  if (rt_TransProf) {
    rt_TransProf->Fill(u, v);
  }
}

void HcalTB02Histo::fillProfile(int ilayer, float value) {
  if (ilayer < (int)(rt_histoProf.size())) {
    rt_histoProf[ilayer]->Fill(value);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalTBSim") << "HcalTB02Histo::Fill profile " << ilayer << " with " << value;
#endif
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
