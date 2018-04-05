#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include <vector>

// takes "new" form of HF shower library and makes it a v3 version

void convertHFShowerLibrary() {
  TFile *nF=new TFile("david.root","RECREATE","hiThere",1);
  TTree *nT=new TTree("HFSimHits","HF simple tree");
  std::vector<float> *parts=new std::vector<float>();
  std::vector<float> *partsHad=new std::vector<float>();
  nT->Branch("emParticles", &parts);
  nT->GetBranch("emParticles")->SetBasketSize(1);
  nT->Branch("hadParticles", &partsHad);
  nT->GetBranch("hadParticles")->SetBasketSize(1);

  TDirectory *target=gDirectory;
  TFile *oF=TFile::Open("HFShowerLibrary_npmt_noatt_eta4_16en_v3_orig.root");
  TTree *t=(TTree*)oF->Get("HFSimHits");
  TTreeReader fReader(t);
  TTreeReaderArray<Float_t> b1(fReader, "emParticles.position_.fCoordinates.fX");
  TTreeReaderArray<Float_t> b2(fReader, "emParticles.position_.fCoordinates.fY");
  TTreeReaderArray<Float_t> b3(fReader, "emParticles.position_.fCoordinates.fZ");
  TTreeReaderArray<Float_t> b4(fReader, "emParticles.lambda_");
  TTreeReaderArray<Float_t> b5(fReader, "emParticles.time_");

  TTreeReaderArray<Float_t> h1(fReader, "hadParticles.position_.fCoordinates.fX");
  TTreeReaderArray<Float_t> h2(fReader, "hadParticles.position_.fCoordinates.fY");
  TTreeReaderArray<Float_t> h3(fReader, "hadParticles.position_.fCoordinates.fZ");
  TTreeReaderArray<Float_t> h4(fReader, "hadParticles.lambda_");
  TTreeReaderArray<Float_t> h5(fReader, "hadParticles.time_");


  target->cd();
  while ( fReader.Next() ) {
    parts->clear();
    unsigned int s=b1.GetSize();
    parts->resize(5*s);
    for ( unsigned int i=0; i<b1.GetSize(); i++) {
      (*parts)[i]=(b1[i]);
      (*parts)[i+1*s]=(b2[i]);
      (*parts)[i+2*s]=(b3[i]);
      (*parts)[i+3*s]=(b4[i]);
      (*parts)[i+4*s]=(b5[i]);
    }  

    partsHad->clear();
    s=h1.GetSize();
    partsHad->resize(5*s);
    for ( unsigned int i=0; i<h1.GetSize(); i++) {
      (*partsHad)[i]=(h1[i]);
      (*partsHad)[i+1*s]=(h2[i]);
      (*partsHad)[i+2*s]=(h3[i]);
      (*partsHad)[i+3*s]=(h4[i]);
      (*partsHad)[i+4*s]=(h5[i]);
    }  

    nT->Fill();
  }

  nT->Write();
  nF->Close();
}
