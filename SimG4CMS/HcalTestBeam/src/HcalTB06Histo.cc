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
 
//
// constructors and destructor
HcalTB06Histo::HcalTB06Histo(const edm::ParameterSet& ps) :
  dbe_(0), iniE(0),  iEta(0), iPhi(0), edepS(0), edecS(0), edhcS(0), edehS(0) {

  fileName   = ps.getUntrackedParameter<std::string>("FileName", "HcalTB06Histo.root");
  verbose    = ps.getUntrackedParameter<bool>("Verbose", false);
  double em1 = ps.getUntrackedParameter<double>("ETtotMax", 400.);
  double em2 = ps.getUntrackedParameter<double>("EHCalMax", 4.0);

  // DQMServices
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();
  if (dbe_) {
    if (verbose) {
      dbe_->setVerbose(1);
      sleep (3);
      dbe_->showDirStructure();
    } else {
      dbe_->setVerbose(0);
    }
    dbe_->setCurrentFolder("HcalTB06Histo");
    iniE = dbe_->book1D("iniE",  "Incident Energy (GeV)",  4000,   0., em1);
    iEta = dbe_->book1D("iEta",  "Eta at incidence     ",   300,   0.,  3.);
    iPhi = dbe_->book1D("iPhi",  "Phi at incidence     ",   300,  -1.,  1.);
    edepS= dbe_->book1D("edepS", "Energy deposit == Total",4000, 0., em1);
    edecS= dbe_->book1D("edecS", "Energy deposit == ECal ",4000, 0., em1);
    edhcS= dbe_->book1D("edhcS", "Energy deposit == HCal ",4000, 0., em2);
    edehS= dbe_->book2D("edehS", "Hcal vs Ecal", 100,0.,em1, 100, 0.,em2);
  }
}
 
HcalTB06Histo::~HcalTB06Histo() {
  if (dbe_) dbe_->save(fileName);
}
 
//
// member functions
//

void HcalTB06Histo::fillPrimary(double energy, double eta, double phi) {

  LogDebug("HcalTBSim") << "HcalTB06Histo::fillPrimary: Energy " 
			<< energy << " Eta " << eta << " Phi " << phi;
  if (dbe_) {
    iniE->Fill(energy);
    iEta->Fill(eta);
    iPhi->Fill(phi);
  }
}

void HcalTB06Histo::fillEdep(double etots, double eecals, double ehcals) { 

  LogDebug("HcalTBSim") << "HcalTB06Histo:::fillEdep: Simulated Total "
			<< etots << " ECal " << eecals << " HCal " << ehcals;
  if (dbe_) {
    edepS->Fill(etots);
    edecS->Fill(eecals);
    edhcS->Fill(ehcals);
    edehS->Fill(eecals, ehcals);
  }
}
