//
//
//
// Package:        CastorShowerLibraryProducer
// Program:        CastorShowerLibraryMerger
//
// Implementation
//
// Original Author: Maria Elena Pol (polme@mail.cern.ch)
//                  Luiz Mundim     (mundim@mail.cern.ch)
//         Created: 14/Jun/2010
//
/////////////////////////////////////////////////////////////////////
//
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "TBranchObject.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "TF1.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdlib.h>
#include <cassert>
#include <unistd.h>
#include <string>
#include <vector>

// Classes for shower library Root file
#include "SimDataFormats/CaloHit/interface/CastorShowerLibraryInfo.h"
#include "SimDataFormats/CaloHit/interface/CastorShowerEvent.h"

void Usage();

int main(int argc, char* argv[])
{
  std::vector<std::string>   FilesToMerge;
  std::string                OutPutFile;
  if (argc<3) Usage();
  for(int i=1;i<argc-1;i++) {
     if (access(argv[i],R_OK)==-1) {
        std::cout << "CastorShowerLibraryMerger: File not readable. Existing." << std::endl;
        std::cout << "                          "<< argv[i] << std::endl;
        exit(1);
     }
     FilesToMerge.push_back(std::string(argv[i]));
  }
  if (access(argv[argc-1],R_OK)!=-1) {
     std::cout << "CastorShowerLibraryMerger: Output File already exists. Exiting." << std::endl;
     std::cout << "                           argv[i]" << std::endl;
     exit(1);
  }
  OutPutFile = std::string(argv[argc-1]);

  TFile *of = new TFile(OutPutFile.c_str(),"RECREATE");
  // Check that TFile has been successfully opened
  if (!of->IsOpen()) { 
     //edm::LogError("CastorShowerLibraryMerger") << "Opening " << OutPutFile << " failed";
     throw cms::Exception("Unknown", "CastorShowerLibraryMerger") 
                                   << "Opening of " << OutPutFile << " fails\n";
  } else {
     /*edm::LogInfo*/ std::cout << ("CastorShowerLibraryMerger")  << "Successfully openned " << OutPutFile << " for output.\n"; 
  }
// Create the output tree
  Int_t split = 1;
  Int_t bsize = 64000;
  TTree *outTree = new TTree("CastorCherenkovPhotons","Cherenkov Photons");
  CastorShowerLibraryInfo *emInfo_out   = new CastorShowerLibraryInfo();
  CastorShowerEvent       *emShower     = new CastorShowerEvent();
  CastorShowerLibraryInfo *hadInfo_out  = new CastorShowerLibraryInfo();
  CastorShowerEvent       *hadShower    = new CastorShowerEvent();
  outTree->Branch("emShowerLibInfo.", "CastorShowerLibraryInfo", &emInfo_out, bsize, split);
  outTree->Branch("emParticles.", "CastorShowerEvent", &emShower, bsize, split);
  outTree->Branch("hadShowerLibInfo.", "CastorShowerLibraryInfo", &hadInfo_out, bsize, split);
  outTree->Branch("hadParticles.", "CastorShowerEvent", &hadShower, bsize, split);

// rebuild info branch
  CastorShowerLibraryInfo *emInfo = new CastorShowerLibraryInfo();
  CastorShowerLibraryInfo *hadInfo= new CastorShowerLibraryInfo();
// Check for the TBranch holding EventInfo in "Events" TTree
  std::vector<double> ebin_em;
  std::vector<double> etabin_em;
  std::vector<double> phibin_em;
  std::vector<double> ebin_had;
  std::vector<double> etabin_had;
  std::vector<double> phibin_had;
  int nevt_perbin_e_em=0;
  int nevt_tot_em=0;
  int nevt_perbin_e_had=0;
  int nevt_tot_had=0;
  for(int i = 0;i<int(FilesToMerge.size());i++) {
     TFile in(FilesToMerge.at(i).c_str(),"r");
     TTree* event = (TTree *) in.Get("CastorCherenkovPhotons");
     TBranchObject *emInfo_b  = (TBranchObject *) event->GetBranch("emShowerLibInfo.");
     TBranchObject *hadInfo_b = (TBranchObject *) event->GetBranch("hadShowerLibInfo.");
     if (emInfo_b) {
        emInfo_b->SetAddress(&emInfo);
        emInfo_b->GetEntry(0);
        ebin_em.insert(ebin_em.end(),emInfo->Energy.getBin().begin(),emInfo->Energy.getBin().end());
        if (nevt_perbin_e_em>0) {
           if (nevt_perbin_e_em!=int(emInfo->Energy.getNEvtPerBin())) {
              std::cout << "CastorShowerLibraryMerger: ERROR: Number of events per energy bin not the same. Exiting."
                        << std::endl;
              exit(1);
           }
        }
        else nevt_perbin_e_em = emInfo->Energy.getNEvtPerBin();
        nevt_tot_em+=emInfo->Energy.getNEvts();

        if (emInfo_out->Eta.getBin().size()>0) {
           if (emInfo_out->Eta.getBin()!=emInfo->Eta.getBin()) {
              std::cout << "CastorShowerLibraryMerger: ERROR: Eta bins not the same in all files. Exiting."
                        << std::endl;
              exit(1);
           }
        }
        else {
           emInfo_out->Eta.setBin(emInfo->Eta.getBin());
           emInfo_out->Eta.setNEvtPerBin(emInfo->Eta.getNEvtPerBin());
           emInfo_out->Eta.setNBins(emInfo->Eta.getNBins());
           emInfo_out->Eta.setNEvts(emInfo->Eta.getNEvts());
        }
           
        if (emInfo_out->Phi.getBin().size()>0) {
           if (emInfo_out->Phi.getBin()!=emInfo->Phi.getBin()) {
              std::cout << "CastorShowerLibraryMerger: ERROR: Phi bins not the same in all files. Exiting."
                        << std::endl;
              exit(1);
           }
        }
        else {
           emInfo_out->Phi.setBin(emInfo->Phi.getBin());
           emInfo_out->Phi.setNEvtPerBin(emInfo->Phi.getNEvtPerBin());
           emInfo_out->Phi.setNBins(emInfo->Phi.getNBins());
           emInfo_out->Phi.setNEvts(emInfo->Phi.getNEvts());
        }
     }
     if (hadInfo_b) {
        hadInfo_b->SetAddress(&hadInfo);
        hadInfo_b->GetEntry(0);
        ebin_had.insert(ebin_had.end(),hadInfo->Energy.getBin().begin(),hadInfo->Energy.getBin().end());
        if (nevt_perbin_e_had>0) {
           if (nevt_perbin_e_had!=int(hadInfo->Energy.getNEvtPerBin())) {
              std::cout << "CastorShowerLibraryMerger: ERROR: Number of events per energy bin not the same. Exiting." << std::endl;
              exit(1);
           }
        }
        else nevt_perbin_e_had = hadInfo->Energy.getNEvtPerBin();
        nevt_tot_had+=hadInfo->Energy.getNEvts();
        
        if (hadInfo_out->Eta.getBin().size()>0) {
           if (hadInfo_out->Eta.getBin()!=hadInfo->Eta.getBin()) {
              std::cout << "CastorShowerLibraryMerger: ERROR: Eta bins not the same in all files. Exiting."
                        << std::endl;
              exit(1);
           }
        }
        else {
           hadInfo_out->Eta.setBin(hadInfo->Eta.getBin());
           hadInfo_out->Eta.setNEvtPerBin(hadInfo->Eta.getNEvtPerBin());
           hadInfo_out->Eta.setNBins(hadInfo->Eta.getNBins());
           hadInfo_out->Eta.setNEvts(hadInfo->Eta.getNEvts());
        }
        if (hadInfo_out->Phi.getBin().size()>0) {
           if (hadInfo_out->Phi.getBin()!=hadInfo->Phi.getBin()) {
              std::cout << "CastorShowerLibraryMerger: ERROR: Phi bins not the same in all files. Exiting."
                        << std::endl;
              exit(1);
           }
        }
        else {
           hadInfo_out->Phi.setBin(hadInfo->Phi.getBin());
           hadInfo_out->Phi.setNEvtPerBin(hadInfo->Phi.getNEvtPerBin());
           hadInfo_out->Phi.setNBins(hadInfo->Phi.getNBins());
           hadInfo_out->Phi.setNEvts(hadInfo->Phi.getNEvts());
        }
     }
// put ther new info data into emInfo_out and hadInfo_out
     in.Close();
  }
  emInfo_out->Energy.setBin(ebin_em);
  emInfo_out->Energy.setNEvtPerBin(nevt_perbin_e_em);
  emInfo_out->Energy.setNEvts(nevt_tot_em);
  emInfo_out->Energy.setNBins(ebin_em.size());
  hadInfo_out->Energy.setBin(ebin_had);
  hadInfo_out->Energy.setNEvtPerBin(nevt_perbin_e_had);
  hadInfo_out->Energy.setNEvts(nevt_tot_had);
  hadInfo_out->Energy.setNBins(ebin_had.size());
  
// Loop over events from input files merging them into a new file, sequentially
  for(int i = 0;i<int(FilesToMerge.size());i++) {
     TFile in(FilesToMerge.at(i).c_str());
     TTree* event = (TTree *) in.Get("CastorCherenkovPhotons");
     TBranchObject *emShower_b = (TBranchObject *) event->GetBranch("emParticles.");
     TBranchObject *hadShower_b = (TBranchObject *) event->GetBranch("hadParticles.");
     emShower_b->SetAddress(&emShower);
     hadShower_b->SetAddress(&hadShower);
     int nevents = event->GetEntries();
     for(int n=0;n<nevents;n++) {
        event->GetEntry(n);
        outTree->Fill();
     }
  }
  of->cd();
  outTree->Write();
  of->Close();
}

void Usage() 
{
   std::cout << "Usage: CastorShowerLibraryMerger input_file1 input_file2 [...] output_file" << std::endl;
   exit(1);
}
