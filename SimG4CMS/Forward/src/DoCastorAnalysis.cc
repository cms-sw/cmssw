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

#define debug 0

DoCastorAnalysis::DoCastorAnalysis(const edm::ParameterSet &p) {

  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("DoCastorAnalysis");
  verbosity    = m_Anal.getParameter<int>("Verbosity");

  TreeFileName = m_Anal.getParameter<std::string>("CastorTreeFileName");

  if (verbosity > 0) {

    std::cout<<std::endl;
    std::cout<<"============================================================================"<<std::endl;
    std::cout << "DoCastorAnalysis:: Initialized as observer"<< std::endl;
    
    std::cout <<" Castor Tree will be created"<< std::endl;
    std::cout <<" Castor Tree will be in file: "<<TreeFileName<<std::endl;
    if(debug) getchar();
  
    std::cout<<"============================================================================"<<std::endl;
    std::cout<<std::endl;
  }

  std::cout << "DoCastorAnalysis: output event root file created"<< std::endl;
  TString treefilename = TreeFileName;
  CastorOutputEventFile = new TFile(treefilename,"RECREATE");

  CastorTree = new TTree("Sim","Sim");

  CastorTree->Branch("simhit_x","std::vector<double>",&psimhit_x);
  CastorTree->Branch("simhit_y","std::vector<double>",&psimhit_y);
  CastorTree->Branch("simhit_z","std::vector<double>",&psimhit_z);

  CastorTree->Branch("simhit_eta","std::vector<double>",&psimhit_eta);
  CastorTree->Branch("simhit_phi","std::vector<double>",&psimhit_phi);
  CastorTree->Branch("simhit_energy","std::vector<double>",&psimhit_energy);

  // CastorTree->Branch("simhit_time","std::vector<double>",&psimhit_time);
  CastorTree->Branch("simhit_sector","std::vector<int>",&psimhit_sector);
  CastorTree->Branch("simhit_module","std::vector<int>",&psimhit_module);

  CastorTree->Branch("simhit_etot",&simhit_etot,"simhit_etot/D");
}

DoCastorAnalysis::~DoCastorAnalysis() {

  //destructor of DoCastorAnalysis
    
  CastorOutputEventFile->cd();
  //-- CastorOutputEventFile->Write();
  CastorTree->Write("",TObject::kOverwrite);
  std::cout << "DoCastorAnalysis: Ntuple event written" << std::endl;
  if(debug) getchar();
  CastorOutputEventFile->Close();
  std::cout << "DoCastorAnalysis: Event file closed" << std::endl;
  if(debug) getchar();

  if (verbosity > 0) {
    std::cout<<std::endl<<"DoCastorAnalysis: end of process"<<std::endl; 
    if(debug) getchar();
  }

}
  
//=================================================================== per EVENT

void DoCastorAnalysis::update(const BeginOfJob * job) {

  std::cout << " Starting new job " << std::endl;
}

//==================================================================== per RUN

void DoCastorAnalysis::update(const BeginOfRun * run) {

  std::cout << std::endl << "DoCastorAnalysis: Starting Run"<< std::endl; 

  // std::cout << "DoCastorAnalysis: output event root file created"<< std::endl;
  // TString treefilename = TreeFileName;
  // CastorOutputEventFile = new TFile(treefilename,"RECREATE");
  
  eventIndex = 1;
}

void DoCastorAnalysis::update(const BeginOfEvent * evt) {
  std::cout << "DoCastorAnalysis: Processing Event Number: "<<eventIndex<< std::endl;
  eventIndex++;
}


//================= End of EVENT ===============

void DoCastorAnalysis::update(const EndOfEvent * evt) {

  // Look for the Hit Collection 

  // access to the G4 hit collections 
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
 
  int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("CastorFI");    
  CaloG4HitCollection* theCAFI = (CaloG4HitCollection*) allHC->GetHC(CAFIid);

  CastorNumberingScheme *theCastorNumScheme = new CastorNumberingScheme();

  unsigned int volumeID=0;
  // std::map<int,float,std::less<int> > themap;
  
  int nentries = theCAFI->entries();
  if(debug) std::cout<<"nentries in CAFI: "<<nentries<<std::endl;
  if(debug) getchar();

  psimhit_x=&simhit_x;
  psimhit_x->clear();
  psimhit_x->reserve(nentries);

  psimhit_y=&simhit_y;
  psimhit_y->clear();
  psimhit_y->reserve(nentries);

  psimhit_z=&simhit_z;
  psimhit_z->clear();
  psimhit_z->reserve(nentries);

  psimhit_eta=&simhit_eta;
  psimhit_eta->clear();
  psimhit_eta->reserve(nentries);
  
  psimhit_phi=&simhit_phi;
  psimhit_phi->clear();
  psimhit_phi->reserve(nentries);
  
  psimhit_energy=&simhit_energy;
  psimhit_energy->clear();
  psimhit_energy->reserve(nentries);
  
  //psimhit_time=&simhit_time;
  //psimhit_time->clear();
  //psimhit_time->reserve(nentries);
  
  psimhit_sector=&simhit_sector;
  psimhit_sector->clear();
  psimhit_sector->reserve(nentries);

  psimhit_module=&simhit_module;
  psimhit_module->clear();
  psimhit_module->reserve(nentries);

  simhit_etot = 0;

  if (nentries > 0) {

    for (int ihit = 0; ihit < nentries; ihit++) {
      CaloG4Hit* aHit = (*theCAFI)[ihit];
      volumeID = aHit->getUnitID();

      //themap[volumeID] += aHit->getEnergyDeposit();
      int zside,sector,zmodule;

      theCastorNumScheme->unpackIndex(volumeID,zside,sector,zmodule);

      double energy   = aHit->getEnergyDeposit()/GeV;
      //double time     = aHit->getTimeSlice();
      
      math::XYZPoint pos  = aHit->getPosition();
      double theta    = pos.theta();
      double   eta    = -log(tan(theta/2.));
      double   phi    = pos.phi();
      
      psimhit_x->push_back(pos.x());
      psimhit_y->push_back(pos.y());
      psimhit_z->push_back(pos.z());
      
      psimhit_eta->push_back(eta);
      psimhit_phi->push_back(phi);
      psimhit_energy->push_back(energy);

      // psimhit_time->push_back(time);
      psimhit_sector->push_back(sector);
      psimhit_module->push_back(zmodule);

      simhit_etot+=energy;
       
      if(debug) std::cout<<"hit "<<ihit+1<<" : x = "<<(*psimhit_x)[ihit]<<" , eta =  "<<(*psimhit_eta)[ihit]
		    <<" , phi = "<<(*psimhit_phi)[ihit]<<" , energy = "<<(*psimhit_energy)[ihit]<<std::endl;
    }

    //if(debug) std::cout<<" total energy = "<<simhit_etot<<std::endl;
    if(debug) getchar();
    CastorTree->Fill();
	
  } // nentries > 0
  delete theCastorNumScheme;
}

void DoCastorAnalysis::update(const EndOfRun * run) {;}

void DoCastorAnalysis::update(const G4Step * aStep) {;}


