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
HcalTB06Histo::HcalTB06Histo(const edm::ParameterSet& ps) {
  verbose_ = ps.getUntrackedParameter<bool>("Verbose", false);
  double em1 = ps.getUntrackedParameter<double>("ETtotMax", 400.);
  double em2 = ps.getUntrackedParameter<double>("EHCalMax", 4.0);
  mkTree_ = ps.getUntrackedParameter<bool>("MakeTree", false);
  eBeam_ = 50.;
  mip_ = ps.getParameter<double>("MIP");
  edm::LogInfo("HcalTBSim") << "Verbose :" << verbose_ << " MakeTree: " << mkTree_ << " EMax: " << em1 << ":" << em2
                            << " MIP " << mip_;

  // Book histograms
  edm::Service<TFileService> tfile;

  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  iniE = tfile->make<TH1D>("iniE", "Incident Energy (GeV)", 4000, 0., em1);
  iEta = tfile->make<TH1D>("iEta", "Eta at incidence     ", 300, 0., 3.);
  iPhi = tfile->make<TH1D>("iPhi", "Phi at incidence     ", 300, -1., 1.);
  edepS = tfile->make<TH1D>("edepS", "Energy deposit == Total", 4000, 0., em1);
  edecS = tfile->make<TH1D>("edecS", "Energy deposit == ECal ", 300, -2., 28.);
  edhcS = tfile->make<TH1D>("edhcS", "Energy deposit == HCal ", 4000, 0., em2);
  edepN = tfile->make<TH1D>("edepN", "Etot/Ebeam   ", 200, -2.5, 2.5);
  edecN = tfile->make<TH1D>("edecN", "Eecal/Ebeam  ", 200, -2.5, 2.5);
  edhcN = tfile->make<TH1D>("edhcN", "Ehcal/Ebeam  ", 200, -2.5, 2.5);
  emhcN = tfile->make<TH1D>("emhcN", "Ehcal/Ebeam MIP in Ecal", 200, -2.5, 2.5);
  edehS = tfile->make<TH2D>("edehS", "Hcal vs Ecal", 100, 0., em1, 100, 0., em2);

  if (mkTree_) {
    tree_ = tfile->make<TTree>("TB06Sim", "TB06Sim");
    tree_->Branch("eBeam_", &eBeam_, "eBeam_/D");
    tree_->Branch("etaBeam_", &etaBeam_, "etaBeam_/D");
    tree_->Branch("phiBeam_", &phiBeam_, "phiBeam_/D");
    tree_->Branch("edepEC_", &edepEC_, "edepEC_/D");
    tree_->Branch("edepHB_", &edepHB_, "edepHB_/D");
    tree_->Branch("edepHO_", &edepHO_, "edepHO_/D");
    tree_->Branch("noiseEC_", &noiseEC_, "noiseEC_/D");
    tree_->Branch("noiseHB_", &noiseHB_, "noiseHB_/D");
    tree_->Branch("noiseHO_", &noiseHO_, "noiseHO_/D");
    tree_->Branch("edepS1_", &edepS1_, "edepS1_/D");
    tree_->Branch("edepS2_", &edepS2_, "edepS2_/D");
    tree_->Branch("edepS3_", &edepS3_, "edepS3_/D");
    tree_->Branch("edepS4_", &edepS4_, "edepS4_/D");
    tree_->Branch("edepVC_", &edepVC_, "edepVC_/D");
    tree_->Branch("edepS7_", &edepS7_, "edepS7_/D");
    tree_->Branch("edepS8_", &edepS8_, "edepS8_/D");
  }
}

HcalTB06Histo::~HcalTB06Histo() {}

//
// member functions
//

void HcalTB06Histo::fillPrimary(double energy, double eta, double phi) {
  if (verbose_)
    edm::LogInfo("HcalTBSim") << "HcalTB06Histo::fillPrimary: Energy " << energy << " Eta " << eta << " Phi " << phi;
  eBeam_ = energy;
  etaBeam_ = eta;
  phiBeam_ = phi;
  iniE->Fill(energy);
  iEta->Fill(eta);
  iPhi->Fill(phi);
}

void HcalTB06Histo::fillEdep(double etots, double eecals, double ehcals) {
  if (verbose_)
    edm::LogInfo("HcalTBSim") << "HcalTB06Histo:::fillEdep: Simulated Total " << etots << " ECal " << eecals << " HCal "
                              << ehcals;
  edepS->Fill(etots);
  edecS->Fill(eecals);
  edhcS->Fill(ehcals);
  edepN->Fill(etots / eBeam_);
  edecN->Fill(eecals / eBeam_);
  edhcN->Fill(ehcals / eBeam_);
  if (eecals <= mip_) {
    emhcN->Fill(etots / eBeam_);
  }
  edehS->Fill(eecals, ehcals);
}

void HcalTB06Histo::fillTree(std::vector<double>& ecalo, std::vector<double>& etrig) {
  if (mkTree_) {
    edepEC_ = ecalo[0];
    noiseEC_ = ecalo[1];
    edepHB_ = ecalo[2];
    noiseHB_ = ecalo[3];
    edepHO_ = ecalo[4];
    noiseHO_ = ecalo[5];
    edepS1_ = etrig[0];
    edepS2_ = etrig[1];
    edepS3_ = etrig[2];
    edepS4_ = etrig[3];
    edepVC_ = etrig[4];
    edepS7_ = etrig[5];
    edepS8_ = etrig[6];
    tree_->Fill();
    if (verbose_)
      edm::LogInfo("HcalTBSim") << "HcalTB06Histo:::fillTree: Energies " << edepEC_ << ":" << noiseEC_ << ":" << edepHB_
                                << ":" << noiseHB_ << ":" << edepHO_ << ":" << noiseHO_ << " Trigger counters "
                                << edepS1_ << ":" << edepS2_ << ":" << edepS3_ << ":" << edepS4_ << ":" << edepVC_
                                << ":" << edepS7_ << ":" << edepS8_;
  }
}
