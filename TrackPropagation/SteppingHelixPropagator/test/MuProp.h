//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Wed May  3 14:00:55 2006 by ROOT version 5.10/00c
// from TTree MuProp/MuProp
// found on file: PropagatorDump.test_29.mup.root
//////////////////////////////////////////////////////////

#ifndef MuProp_h
#define MuProp_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH2.h>
#include <TH1.h>
#include <TProfile.h>


#include <map>


class MuProp {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leave types
   Int_t           nPoints;
   Int_t           q[250];   //[nPoints]
   Int_t           pStatus[250][3];   //[nPoints]
   Float_t         p3[250][9];   //[nPoints]
   Float_t         r3[250][9];   //[nPoints]
   Int_t           id[250];   //[nPoints]
   Float_t         p3R[250][3];   //[nPoints]
   Float_t         r3R[250][3];   //[nPoints]
   Float_t         covFlat[250][21];   //[nPoints]
   Int_t           run;
   Int_t           event_;

   // List of branches
   TBranch        *b_nPoints;   //!
   TBranch        *b_q;   //!
   TBranch        *b_pStatus;   //!
   TBranch        *b_p3;   //!
   TBranch        *b_r3;   //!
   TBranch        *b_id;   //!
   TBranch        *b_p3R;   //!
   TBranch        *b_r3R;   //!
   TBranch        *b_covFlat;   //!
   TBranch        *b_run;   //!
   TBranch        *b_event;   //!


   std::map<Int_t,TH1F*> dX_mh1;
   TH1F* dX_fh1(Int_t i){ return dX_mh1[i];}

   std::map<Int_t,TH1F*> dXPull_mh1;
   TH1F* dXPull_fh1(Int_t i){ return dXPull_mh1[i];}

   Int_t idDT(){return idDT_WhSt(0,0);}
   Int_t idCSC(){return idCSC_EnStRi(0,0,0);}

   Int_t idDT_WhSt(Int_t wh, Int_t st) { 
     //new format
     return ((0x2<<28) | (0x1<<25) | ((wh&0x7)<<15) | ((st&0x7)<<22) );
     //old format
     //     return ((0x2<<28) | (0x1<<25) | ((wh&0x7)<<22) | ((st&0x7)<<19) );
   }

   Int_t idCSC_EnStRi(Int_t en, Int_t st, Int_t ri) { 
     return ((0x2<<28) | (0x2<<25) | ((en&0x3)<<16) | ((st&0x7)<<13) | ((ri&0x7)<<10) );
   }
     
   MuProp(TTree *tree=0);
   virtual ~MuProp();
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop(Double_t maxEloss = 1e12);
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef MuProp_cxx
MuProp::MuProp(TTree *tree)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("PropagatorDump.test_29.mup.root");
      if (!f) {
         f = new TFile("PropagatorDump.test_29.mup.root");
      }
      tree = (TTree*)gDirectory->Get("MuProp");

   }
   Init(tree);
}

MuProp::~MuProp()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t MuProp::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t MuProp::LoadTree(Long64_t entry)
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

void MuProp::Init(TTree *tree)
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

   fChain->SetBranchAddress("nPoints",&nPoints);
   fChain->SetBranchAddress("q",q);
   fChain->SetBranchAddress("pStatus",pStatus);
   fChain->SetBranchAddress("p3",p3);
   fChain->SetBranchAddress("r3",r3);
   fChain->SetBranchAddress("id",id);
   fChain->SetBranchAddress("p3R",p3R);
   fChain->SetBranchAddress("r3R",r3R);
   fChain->SetBranchAddress("covFlat",covFlat);
   fChain->SetBranchAddress("run",&run);
   fChain->SetBranchAddress("event_",&event_);
   Notify();
}

Bool_t MuProp::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. Typically here the branch pointers
   // will be retrieved. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed.

   // Get branch pointers
   b_nPoints = fChain->GetBranch("nPoints");
   b_q = fChain->GetBranch("q");
   b_pStatus = fChain->GetBranch("pStatus");
   b_p3 = fChain->GetBranch("p3");
   b_r3 = fChain->GetBranch("r3");
   b_id = fChain->GetBranch("id");
   b_p3R = fChain->GetBranch("p3R");
   b_r3R = fChain->GetBranch("r3R");
   b_covFlat = fChain->GetBranch("covFlat");
   b_run = fChain->GetBranch("run");
   b_event = fChain->GetBranch("event_");

   return kTRUE;
}

void MuProp::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
#endif // #ifdef MuProp_cxx
