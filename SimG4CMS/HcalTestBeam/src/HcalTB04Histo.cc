// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB04Histo
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: HcalTB04Histo.cc,v 1.2 2006/06/04 13:59:38 sunanda Exp $
//
 
// system include files
#include <iostream>
#include <cmath>
 
// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB04Histo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
//
// constructors and destructor
HcalTB04Histo::HcalTB04Histo(const edm::ParameterSet& ps) :
  dbe_(0), iniE(0),  iEta(0), iPhi(0), edepS(0), edecS(0), edhcS(0), edepQ(0),
  edecQ(0), edhcQ(0), edehS(0), edehQ(0), latse(0), latqe(0), latsf(0), 
  latqf(0), lngs(0), lngq(0) {

  fileName   = ps.getUntrackedParameter<std::string>("FileName", "HcalTB04Histo.root");
  verbose    = ps.getUntrackedParameter<bool>("Verbose", false);
  double em1 = ps.getUntrackedParameter<double>("ETtotMax", 400.);
  double em2 = ps.getUntrackedParameter<double>("EHCalMax", 4.0);

  // DQMServices
  dbe_ = edm::Service<DQMStore>().operator->();
  if (dbe_) {
    if (verbose) {
      dbe_->setVerbose(1);
      sleep (3);
      dbe_->showDirStructure();
    } else {
      dbe_->setVerbose(0);
    }
    dbe_->setCurrentFolder("HcalTB04Histo");
    iniE = dbe_->book1D("iniE",  "Incident Energy (GeV)",  4000,   0., em1);
    iEta = dbe_->book1D("iEta",  "Eta at incidence     ",   300,   0.,  3.);
    iPhi = dbe_->book1D("iPhi",  "Phi at incidence     ",   300,  -1.,  1.);
    edepS= dbe_->book1D("edepS", "Energy deposit == Total (Simhit)",4000, 0., em1);
    edecS= dbe_->book1D("edecS", "Energy deposit == ECal  (Simhit)",4000, 0., em1);
    edhcS= dbe_->book1D("edhcS", "Energy deposit == HCal  (Simhit)",4000, 0., em2);
    edepQ= dbe_->book1D("edepQ", "Energy deposit == Total (QIE)",   4000, 0., em1);
    edecQ= dbe_->book1D("edecQ", "Energy deposit == ECal  (QIE)",   4000, 0., em1);
    edhcQ= dbe_->book1D("edhcQ", "Energy deposit == HCal  (QIE)",   4000, 0., em2);
    edehS= dbe_->book2D("edehS", "Hcal vs Ecal (Simhit)", 100,0.,em1, 100, 0.,em2);
    edehQ= dbe_->book2D("edehQ", "Hcal vs Ecal (QIE)",    100,0.,em1, 100, 0.,em2);
    latse=  dbe_->bookProfile("latse","Lat Prof (Eta Sim)",10,0.,10.,10,0.,10000000.);
    latqe=  dbe_->bookProfile("latqe","Lat Prof (Eta QIE)",10,0.,10.,10,0.,10000000.);
    latsf=  dbe_->bookProfile("latsf","Lat Prof (Phi Sim)",10,0.,10.,10,0.,10000000.);
    latqf=  dbe_->bookProfile("latqf","Lat Prof (Phi QIE)",10,0.,10.,10,0.,10000000.); 
    lngs =  dbe_->bookProfile("lngs", "Long. Prof (Sim)",  20,0.,20.,10,0.,10000000.); 
    lngq =  dbe_->bookProfile("lngq", "Long. Prof (QIE)",  20,0.,20.,10,0.,10000000.);
  }
}
 
HcalTB04Histo::~HcalTB04Histo() {
  if (dbe_) dbe_->save(fileName);
}
 
//
// member functions
//

void HcalTB04Histo::fillPrimary(double energy, double eta, double phi) {

  LogDebug("HcalTBSim") << "HcalTB04Histo::fillPrimary: Energy " 
			<< energy << " Eta " << eta << " Phi " << phi;
  if (dbe_) {
    iniE->Fill(energy);
    iEta->Fill(eta);
    iPhi->Fill(phi);
  }
}

void HcalTB04Histo::fillEdep(double etots, double eecals, double ehcals, 
			     double etotq, double eecalq, double ehcalq) {

  LogDebug("HcalTBSim") << "HcalTB04Histo:::fillEdep: Simulated Total "
			<< etots << " ECal " << eecals << " HCal " << ehcals
			<< " Digitised Total " << etotq << " ECal " << eecalq
			<< " HCal " << ehcalq;
  if (dbe_) {
    edepS->Fill(etots);
    edecS->Fill(eecals);
    edhcS->Fill(ehcals);
    edepQ->Fill(etotq);
    edecQ->Fill(eecalq);
    edhcQ->Fill(ehcalq);
    edehS->Fill(eecals, ehcals);
    edehQ->Fill(eecalq, ehcalq);
  }
}

void HcalTB04Histo::fillTrnsProf(std::vector<double> es1, 
				 std::vector<double> eq1,
				 std::vector<double> es2, 
				 std::vector<double> eq2) {

  unsigned int n1 = std::min(es1.size(),eq1.size());
  unsigned int n2 = std::min(es2.size(),eq2.size());
  unsigned int n  = std::min(n1,n2);
  for (unsigned int i = 0; i < n; i++) 
    LogDebug("HcalTBSim") << "HcalTB04Histo::fillTrnsProf [" << i
			  << "] SimEta " << es1[i] << " DigEta " << eq1[i]
			  << " SimPhi " << es2[i] << " DigPhi " << eq2[i];
  if (dbe_) {
    for (unsigned int i=0; i<(es1.size()); i++) {
      double tow = i+0.5;
      latse->Fill(tow, es1[i]);
    }
    for (unsigned int i=0; i<(eq1.size()); i++) {
      double tow = i+0.5;
      latqe->Fill(tow, eq1[i]);
    }
    for (unsigned int i=0; i<(es2.size()); i++) {
      double tow = i+0.5;
      latsf->Fill(tow, es2[i]);
    }
    for (unsigned int i=0; i<(eq2.size()); i++) {
      double tow = i+0.5;
      latqf->Fill(tow, eq2[i]);
    }
  }
}

void HcalTB04Histo::fillLongProf(std::vector<double> es, 
				 std::vector<double> eq) {

  unsigned int n = std::min(es.size(),eq.size());
  for (unsigned int i = 0; i < n; i++) 
    LogDebug("HcalTBSim") << "HcalTB04Histo::fillLongProf [" << i
			  << "] Sim " << es[i] << " Dig " << eq[i];
  if (dbe_) {
    for (unsigned int i=0; i<(es.size()); i++) {
      double lay = i+0.5;
      lngs->Fill(lay, es[i]);
    }
    for (unsigned int i=0; i<(eq.size()); i++) {
      double lay = i+0.5;
      lngq->Fill(lay, eq[i]);
    }
  }
}
