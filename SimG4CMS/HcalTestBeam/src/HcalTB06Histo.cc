// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB06Histo
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Tue Oct 10 10:14:34 CEST 2006
//
 
// system include files
#include <iostream>
#include <cmath>
 
// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06Histo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
 
//
// constructors and destructor
HcalTB06Histo::HcalTB06Histo(const edm::ParameterSet& ps) :
  iniE(0),  iEta(0), iPhi(0), edepS(0), edecS(0), edhcS(0), edehS(0) {

  verbose    = ps.getUntrackedParameter<bool>("Verbose", false);
  double em1 = ps.getUntrackedParameter<double>("ETtotMax", 400.);
  double em2 = ps.getUntrackedParameter<double>("EHCalMax", 4.0);

  // Book histograms
  edm::Service<TFileService> tfile;

  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  iniE = tfile->make<TH1D>("iniE",  "Incident Energy (GeV)",  4000,   0., em1);
  iEta = tfile->make<TH1D>("iEta",  "Eta at incidence     ",   300,   0.,  3.);
  iPhi = tfile->make<TH1D>("iPhi",  "Phi at incidence     ",   300,  -1.,  1.);
  edepS= tfile->make<TH1D>("edepS", "Energy deposit == Total",4000, 0., em1);
  edecS= tfile->make<TH1D>("edecS", "Energy deposit == ECal ",4000, 0., em1);
  edhcS= tfile->make<TH1D>("edhcS", "Energy deposit == HCal ",4000, 0., em2);
  edehS= tfile->make<TH2D>("edehS", "Hcal vs Ecal", 100,0.,em1, 100, 0.,em2);
}
 
HcalTB06Histo::~HcalTB06Histo() {}
 
//
// member functions
//

void HcalTB06Histo::fillPrimary(double energy, double eta, double phi) {

  LogDebug("HcalTBSim") << "HcalTB06Histo::fillPrimary: Energy " 
			<< energy << " Eta " << eta << " Phi " << phi;
  iniE->Fill(energy);
  iEta->Fill(eta);
  iPhi->Fill(phi);
}

void HcalTB06Histo::fillEdep(double etots, double eecals, double ehcals) { 

  LogDebug("HcalTBSim") << "HcalTB06Histo:::fillEdep: Simulated Total "
			<< etots << " ECal " << eecals << " HCal " << ehcals;
  edepS->Fill(etots);
  edecS->Fill(eecals);
  edhcS->Fill(ehcals);
  edehS->Fill(eecals, ehcals);
}
