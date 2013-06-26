//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Dec 14 14:31:18 2009 by ROOT version 5.18/00a
// from TTree Radtuple/Neutrino Radiation Length analyzer ntuple
// found on file: neutrad.root
//////////////////////////////////////////////////////////

#ifndef MyRad_h
#define MyRad_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>

class MyRad : public TSelector {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain

   // Declaration of leaf types
   Int_t           evt_run;
   Int_t           evt_evtnum;
   Int_t           evt_numfs;
   Int_t           evt_numtp;
   Int_t           trk_pdgid;
   Int_t           trk_nlyrs;
   Float_t         trk_theta;
   Float_t         trk_phi;
   Float_t         trk_eta;
   Float_t         trk_zee;
   Float_t         trk_mom;
   Float_t         trk_eng;
   Int_t           lyr_laynm;
   Float_t         lyr_radln;

   // List of branches
   TBranch        *b_evt;   //!
   TBranch        *b_trk;   //!
   TBranch        *b_lyr;   //!

   MyRad(TTree * /*tree*/ =0) { }
   virtual ~MyRad() { }
   virtual Int_t   Version() const { return 2; }
   virtual void    Begin(TTree *tree);
   virtual void    SlaveBegin(TTree *tree);
   virtual void    Init(TTree *tree);
   virtual Bool_t  Notify();
   virtual Bool_t  Process(Long64_t entry);
   virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) { fInput = input; }
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();

   ClassDef(MyRad,0);
};

#endif

#ifdef MyRad_cxx
void MyRad::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("evt", &evt_run, &b_evt);
   fChain->SetBranchAddress("trk", &trk_pdgid, &b_trk);
   fChain->SetBranchAddress("lyr", &lyr_laynm, &b_lyr);
}

Bool_t MyRad::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

#endif // #ifdef MyRad_cxx
