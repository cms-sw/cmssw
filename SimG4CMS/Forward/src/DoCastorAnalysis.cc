// -*- C++ -*-
//
// Package:     Forward
// Class  :     DoCastorAnalysis
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: P. Katsas
//         Created: 02/2007
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "SimG4CMS/Forward/interface/DoCastorAnalysis.h"
#include "SimG4CMS/Forward/interface/CastorNumberingScheme.h"

#include "TFile.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>

//#define EDM_ML_DEBUG

DoCastorAnalysis::DoCastorAnalysis(const edm::ParameterSet& p) {
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("DoCastorAnalysis");
  verbosity = m_Anal.getParameter<int>("Verbosity");

  TreeFileName = m_Anal.getParameter<std::string>("CastorTreeFileName");

  if (verbosity > 0) {
    edm::LogVerbatim("ForwardSim") << std::endl;
    edm::LogVerbatim("ForwardSim") << "============================================================================";
    edm::LogVerbatim("ForwardSim") << "DoCastorAnalysis:: Initialized as observer";

    edm::LogVerbatim("ForwardSim") << " Castor Tree will be created";
    edm::LogVerbatim("ForwardSim") << " Castor Tree will be in file: " << TreeFileName;
#ifdef EDM_ML_DEBUG
    getchar();
#endif
    edm::LogVerbatim("ForwardSim") << "============================================================================";
    edm::LogVerbatim("ForwardSim") << std::endl;
  }

  edm::LogVerbatim("ForwardSim") << "DoCastorAnalysis: output event root file created";
  TString treefilename = TreeFileName;
  CastorOutputEventFile = new TFile(treefilename, "RECREATE");

  CastorTree = new TTree("Sim", "Sim");

  CastorTree->Branch("simhit_x", "std::vector<double>", &psimhit_x);
  CastorTree->Branch("simhit_y", "std::vector<double>", &psimhit_y);
  CastorTree->Branch("simhit_z", "std::vector<double>", &psimhit_z);

  CastorTree->Branch("simhit_eta", "std::vector<double>", &psimhit_eta);
  CastorTree->Branch("simhit_phi", "std::vector<double>", &psimhit_phi);
  CastorTree->Branch("simhit_energy", "std::vector<double>", &psimhit_energy);

  // CastorTree->Branch("simhit_time","std::vector<double>",&psimhit_time);
  CastorTree->Branch("simhit_sector", "std::vector<int>", &psimhit_sector);
  CastorTree->Branch("simhit_module", "std::vector<int>", &psimhit_module);

  CastorTree->Branch("simhit_etot", &simhit_etot, "simhit_etot/D");
}

DoCastorAnalysis::~DoCastorAnalysis() {
  //destructor of DoCastorAnalysis

  CastorOutputEventFile->cd();
  //-- CastorOutputEventFile->Write();
  CastorTree->Write("", TObject::kOverwrite);
  edm::LogVerbatim("ForwardSim") << "DoCastorAnalysis: Ntuple event written";
#ifdef EDM_ML_DEBUG
  getchar();
#endif
  CastorOutputEventFile->Close();
  edm::LogVerbatim("ForwardSim") << "DoCastorAnalysis: Event file closed";
#ifdef EDM_ML_DEBUG
  getchar();
#endif

  if (verbosity > 0) {
    edm::LogVerbatim("ForwardSim") << std::endl << "DoCastorAnalysis: end of process";
#ifdef EDM_ML_DEBUG
    getchar();
#endif
  }
}

//=================================================================== per EVENT

void DoCastorAnalysis::update(const BeginOfJob* job) { edm::LogVerbatim("ForwardSim") << " Starting new job "; }

//==================================================================== per RUN

void DoCastorAnalysis::update(const BeginOfRun* run) {
  edm::LogVerbatim("ForwardSim") << std::endl << "DoCastorAnalysis: Starting Run";

  // edm::LogVerbatim("ForwardSim") << "DoCastorAnalysis: output event root file created";
  // TString treefilename = TreeFileName;
  // CastorOutputEventFile = new TFile(treefilename,"RECREATE");

  eventIndex = 1;
}

void DoCastorAnalysis::update(const BeginOfEvent* evt) {
  edm::LogVerbatim("ForwardSim") << "DoCastorAnalysis: Processing Event Number: " << eventIndex;
  eventIndex++;
}

//================= End of EVENT ===============

void DoCastorAnalysis::update(const EndOfEvent* evt) {
  // Look for the Hit Collection

  // access to the G4 hit collections
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();

  int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("CastorFI");
  CaloG4HitCollection* theCAFI = (CaloG4HitCollection*)allHC->GetHC(CAFIid);

  CastorNumberingScheme* theCastorNumScheme = new CastorNumberingScheme();

  unsigned int volumeID = 0;
  // std::map<int,float,std::less<int> > themap;

  int nentries = theCAFI->entries();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardSim") << "nentries in CAFI: " << nentries;
  getchar();
#endif

  psimhit_x = &simhit_x;
  psimhit_x->clear();
  psimhit_x->reserve(nentries);

  psimhit_y = &simhit_y;
  psimhit_y->clear();
  psimhit_y->reserve(nentries);

  psimhit_z = &simhit_z;
  psimhit_z->clear();
  psimhit_z->reserve(nentries);

  psimhit_eta = &simhit_eta;
  psimhit_eta->clear();
  psimhit_eta->reserve(nentries);

  psimhit_phi = &simhit_phi;
  psimhit_phi->clear();
  psimhit_phi->reserve(nentries);

  psimhit_energy = &simhit_energy;
  psimhit_energy->clear();
  psimhit_energy->reserve(nentries);

  //psimhit_time=&simhit_time;
  //psimhit_time->clear();
  //psimhit_time->reserve(nentries);

  psimhit_sector = &simhit_sector;
  psimhit_sector->clear();
  psimhit_sector->reserve(nentries);

  psimhit_module = &simhit_module;
  psimhit_module->clear();
  psimhit_module->reserve(nentries);

  simhit_etot = 0;

  if (nentries > 0) {
    for (int ihit = 0; ihit < nentries; ihit++) {
      CaloG4Hit* aHit = (*theCAFI)[ihit];
      volumeID = aHit->getUnitID();

      //themap[volumeID] += aHit->getEnergyDeposit();
      int zside, sector, zmodule;

      theCastorNumScheme->unpackIndex(volumeID, zside, sector, zmodule);

      double energy = aHit->getEnergyDeposit() / GeV;
      //double time     = aHit->getTimeSlice();

      math::XYZPoint pos = aHit->getPosition();
      double theta = pos.theta();
      double eta = -log(tan(theta / 2.));
      double phi = pos.phi();

      psimhit_x->push_back(pos.x());
      psimhit_y->push_back(pos.y());
      psimhit_z->push_back(pos.z());

      psimhit_eta->push_back(eta);
      psimhit_phi->push_back(phi);
      psimhit_energy->push_back(energy);

      // psimhit_time->push_back(time);
      psimhit_sector->push_back(sector);
      psimhit_module->push_back(zmodule);

      simhit_etot += energy;

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("ForwardSim") << "hit " << ihit + 1 << " : x = " << (*psimhit_x)[ihit]
                                     << " , eta =  " << (*psimhit_eta)[ihit] << " , phi = " << (*psimhit_phi)[ihit]
                                     << " , energy = " << (*psimhit_energy)[ihit];
#endif
    }

    //if (debug) edm::LogVerbatim("ForwardSim") << " total energy = " << simhit_etot;
#ifdef EDM_ML_DEBUG
    getchar();
#endif
    CastorTree->Fill();

  }  // nentries > 0
  delete theCastorNumScheme;
}

void DoCastorAnalysis::update(const EndOfRun* run) { ; }

void DoCastorAnalysis::update(const G4Step* aStep) { ; }
