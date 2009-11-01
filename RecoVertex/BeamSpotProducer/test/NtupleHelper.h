#ifndef NtupleHelper_h
#define NtupleHelper_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <vector>
#include <iostream>
#include "TObject.h"

#include "global.h"

class NtupleHelper {
public :



  TH1F           *hsx; 
  TH2F           *hd; 
  TH1F           *hsw; 
  TH1F           *hsd; 
  TH1F           *hsigma; 
  //  TH1F           *hsw; 
  TH2F           *hvxy;
  TH2F           *hvxz;
  TH2F           *hdphi;
  TH3F           *hvxyz;
  TH1F           *hpt;   
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leave types
   Float_t         run;
   Float_t         event;
   Double_t         pt;
   Double_t         d0;
   Double_t         phi;
   Double_t         sigmaD;
   Double_t         z0;
   Double_t         sigmaz0;
   Float_t         x;
   Float_t         y;

   UInt_t nPixelHit;
   UInt_t nStripHit;
   Double_t chi2;
   Double_t ndof;
   Double_t eta;
   
   // List of branches
   TBranch        *b_run;   //!
   TBranch        *b_event;   //!
   TBranch        *b_pt;   //!
   TBranch        *b_d0;   //!
   TBranch        *b_phi;   //!
   TBranch        *b_sigmaD;   //!
   TBranch        *b_z0;   //!
   TBranch        *b_sigmaz0;   //!
   TBranch        *b_x;   //!
   TBranch        *b_y;   //!

   TBranch *b_nPixelHit;
   TBranch *b_nStripHit;
   TBranch *b_chi2;
   TBranch *b_ndof;
   TBranch *b_eta;
   
   NtupleHelper(const char* fname ,TTree *tree=0);
   virtual ~NtupleHelper();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual zData    Loop(int maxEvents= 0);
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
   virtual void     Book();
  ClassDef(NtupleHelper,1) 
};

#endif
#ifdef NtupleHelper_cc
NtupleHelper::NtupleHelper(const char* fname,TTree *tree)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject(fname);
      if (!f) {
         f = new TFile(fname);
      }
      tree = (TTree*)gDirectory->Get("mytree");
	  std::cout << "[NtupleHelper] got tree " << std::endl;
	  //tree = (TTree*)gDirectory->Get("mytree"); // for reco ntuple

   }
   Init(tree);
}

NtupleHelper::~NtupleHelper()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t NtupleHelper::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t NtupleHelper::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->IsA() != TChain::Class()) return centry;
   TChain *chain = (TChain*)fChain;
   if (chain->GetTreeNumber() != fCurrent) {
      fCurrent = chain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void NtupleHelper::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses of the tree
   // will be set. It is normaly not necessary to make changes to the
   // generated code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running with PROOF.

   // Set branch addresses
   if (tree == 0) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   //fChain->SetBranchAddress("run",&run);
   //fChain->SetBranchAddress("event",&event);
   fChain->SetBranchAddress("pt",&pt);
   fChain->SetBranchAddress("d0",&d0);
   //fChain->SetBranchAddress("phi",&phi);
   fChain->SetBranchAddress("phi0",&phi);// for reco ntuple
   //fChain->SetBranchAddress("sigmaD",&sigmaD);
   fChain->SetBranchAddress("sigmad0",&sigmaD);// for reco ntuple
   fChain->SetBranchAddress("z0",&z0);
   fChain->SetBranchAddress("sigmaz0",&sigmaz0);
   //fChain->SetBranchAddress("x",&x);
   //fChain->SetBranchAddress("y",&y);

   fChain->SetBranchAddress("nStripHit",&nStripHit);
   fChain->SetBranchAddress("nPixelHit",&nPixelHit);
   fChain->SetBranchAddress("chi2",&chi2);
   fChain->SetBranchAddress("ndof",&ndof);
   fChain->SetBranchAddress("eta",&eta);

   std::cout << "[NtupleHelper] tree initialized " << std::endl;
   Notify();
}

Bool_t NtupleHelper::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. Typically here the branch pointers
   // will be retrieved. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed.

   // Get branch pointers
   //b_run = fChain->GetBranch("run");
   //b_event = fChain->GetBranch("event");
   b_pt = fChain->GetBranch("pt");
   b_d0 = fChain->GetBranch("d0");
   //b_phi = fChain->GetBranch("phi");
   //b_sigmaD = fChain->GetBranch("sigmaD");
   b_phi = fChain->GetBranch("phi0");
   b_sigmaD = fChain->GetBranch("sigmad0");
   
   b_z0 = fChain->GetBranch("z0");
   b_sigmaz0 = fChain->GetBranch("sigmaz0");
   //b_x = fChain->GetBranch("x");
   //b_y = fChain->GetBranch("y");

   b_nPixelHit = fChain->GetBranch("nPixelHit");
   b_nStripHit = fChain->GetBranch("nStripHit");
   b_chi2 = fChain->GetBranch("chi2");
   b_ndof = fChain->GetBranch("ndof");
   b_eta = fChain->GetBranch("eta");

   std::cout << "[NtupleHelper] branches notified" << std::endl;
   return kTRUE;
}

void NtupleHelper::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t NtupleHelper::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef NtupleHelper_cc
