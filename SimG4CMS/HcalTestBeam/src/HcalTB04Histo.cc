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
// $Id: HcalTB04Histo.cc,v 1.6 2013/05/28 17:42:14 gartung Exp $
//
 
// system include files
#include <iostream>
#include <cmath>
 
// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB04Histo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
 
//
// constructors and destructor
HcalTB04Histo::HcalTB04Histo(const edm::ParameterSet& ps) :
  iniE(0),  iEta(0), iPhi(0), edepS(0), edecS(0), edhcS(0), edepQ(0),
  edecQ(0), edhcQ(0), edehS(0), edehQ(0), latse(0), latqe(0), latsf(0), 
  latqf(0), lngs(0), lngq(0) {

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
  edepS= tfile->make<TH1D>("edepS", "Energy deposit == Total (Simhit)",4000, 0., em1);
  edecS= tfile->make<TH1D>("edecS", "Energy deposit == ECal  (Simhit)",4000, 0., em1);
  edhcS= tfile->make<TH1D>("edhcS", "Energy deposit == HCal  (Simhit)",4000, 0., em2);
  edepQ= tfile->make<TH1D>("edepQ", "Energy deposit == Total (QIE)",   4000, 0., em1);
  edecQ= tfile->make<TH1D>("edecQ", "Energy deposit == ECal  (QIE)",   4000, 0., em1);
  edhcQ= tfile->make<TH1D>("edhcQ", "Energy deposit == HCal  (QIE)",   4000, 0., em2);
  edehS= tfile->make<TH2D>("edehS", "Hcal vs Ecal (Simhit)", 100,0.,em1, 100, 0.,em2);
  edehQ= tfile->make<TH2D>("edehQ", "Hcal vs Ecal (QIE)",    100,0.,em1, 100, 0.,em2);
  latse=  tfile->make<TProfile>("latse","Lat Prof (Eta Sim)",10,0.,10.);
  latqe=  tfile->make<TProfile>("latqe","Lat Prof (Eta QIE)",10,0.,10.);
  latsf=  tfile->make<TProfile>("latsf","Lat Prof (Phi Sim)",10,0.,10.);
  latqf=  tfile->make<TProfile>("latqf","Lat Prof (Phi QIE)",10,0.,10.); 
  lngs =  tfile->make<TProfile>("lngs", "Long. Prof (Sim)",  20,0.,20.); 
  lngq =  tfile->make<TProfile>("lngq", "Long. Prof (QIE)",  20,0.,20.);
}
 
HcalTB04Histo::~HcalTB04Histo() {}
 
//
// member functions
//

void HcalTB04Histo::fillPrimary(double energy, double eta, double phi) {

  LogDebug("HcalTBSim") << "HcalTB04Histo::fillPrimary: Energy " 
			<< energy << " Eta " << eta << " Phi " << phi;
  iniE->Fill(energy);
  iEta->Fill(eta);
  iPhi->Fill(phi);
}

void HcalTB04Histo::fillEdep(double etots, double eecals, double ehcals, 
			     double etotq, double eecalq, double ehcalq) {

  LogDebug("HcalTBSim") << "HcalTB04Histo:::fillEdep: Simulated Total "
			<< etots << " ECal " << eecals << " HCal " << ehcals
			<< " Digitised Total " << etotq << " ECal " << eecalq
			<< " HCal " << ehcalq;
  edepS->Fill(etots);
  edecS->Fill(eecals);
  edhcS->Fill(ehcals);
  edepQ->Fill(etotq);
  edecQ->Fill(eecalq);
  edhcQ->Fill(ehcalq);
  edehS->Fill(eecals, ehcals);
  edehQ->Fill(eecalq, ehcalq);
}

void HcalTB04Histo::fillTrnsProf(const std::vector<double>& es1, 
				 const std::vector<double>& eq1,
				 const std::vector<double>& es2, 
				 const std::vector<double>& eq2) {

  unsigned int n1 = std::min(es1.size(),eq1.size());
  unsigned int n2 = std::min(es2.size(),eq2.size());
  unsigned int n  = std::min(n1,n2);
  for (unsigned int i = 0; i < n; i++) 
    LogDebug("HcalTBSim") << "HcalTB04Histo::fillTrnsProf [" << i
			  << "] SimEta " << es1[i] << " DigEta " << eq1[i]
			  << " SimPhi " << es2[i] << " DigPhi " << eq2[i];
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

void HcalTB04Histo::fillLongProf(const std::vector<double>& es, 
				 const std::vector<double>& eq) {

  unsigned int n = std::min(es.size(),eq.size());
  for (unsigned int i = 0; i < n; i++) 
    LogDebug("HcalTBSim") << "HcalTB04Histo::fillLongProf [" << i
			  << "] Sim " << es[i] << " Dig " << eq[i];
  for (unsigned int i=0; i<(es.size()); i++) {
    double lay = i+0.5;
    lngs->Fill(lay, es[i]);
  }
  for (unsigned int i=0; i<(eq.size()); i++) {
    double lay = i+0.5;
    lngq->Fill(lay, eq[i]);
  }
}
