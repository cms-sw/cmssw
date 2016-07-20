//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Jul 19 18:37:58 2016 by ROOT version 6.06/05
// from TTree TB06Sim/TB06Sim
// found on file: tb_pi_50gevNOECAL.root
//////////////////////////////////////////////////////////

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

// Header file for the classes stored in the TTree if any.

class TB06Analysis {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Fixed size dimensions of array or collections stored in the TTree if any.
  const Int_t kMaxeBeam = 1;
  const Int_t kMaxetaBeam = 1;
  const Int_t kMaxphiBeam = 1;
  const Int_t kMaxedepEC = 1;
  const Int_t kMaxedepHB = 1;
  const Int_t kMaxedepHO = 1;
  const Int_t kMaxnoiseEC = 1;
  const Int_t kMaxnoiseHB = 1;
  const Int_t kMaxnoiseHO = 1;
  const Int_t kMaxedepS1 = 1;
  const Int_t kMaxedepS2 = 1;
  const Int_t kMaxedepS3 = 1;
  const Int_t kMaxedepS4 = 1;
  const Int_t kMaxedepVC = 1;

  // Declaration of leaf types
  Double_t        eBeam_;
  Double_t        etaBeam_;
  Double_t        phiBeam_;
  Double_t        edepEC_;
  Double_t        edepHB_;
  Double_t        edepHO_;
  Double_t        noiseEC_;
  Double_t        noiseHB_;
  Double_t        noiseHO_;
  Double_t        edepS1_;
  Double_t        edepS2_;
  Double_t        edepS3_;
  Double_t        edepS4_;
  Double_t        edepVC_;

  // List of branches
  TBranch        *b_eBeam_;   //!
  TBranch        *b_etaBeam_;   //!
  TBranch        *b_phiBeam_;   //!
  TBranch        *b_edepEC_;   //!
  TBranch        *b_edepHB_;   //!
  TBranch        *b_edepHO_;   //!
  TBranch        *b_noiseEC_;   //!
  TBranch        *b_noiseHB_;   //!
  TBranch        *b_noiseHO_;   //!
  TBranch        *b_edepS1_;   //!
  TBranch        *b_edepS2_;   //!
  TBranch        *b_edepS3_;   //!
  TBranch        *b_edepS4_;   //!
  TBranch        *b_edepVC_;   //!

  TB06Analysis(std::string fname, std::string dirnm);
  virtual ~TB06Analysis();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
};

TB06Analysis::TB06Analysis(std::string fname, std::string dirnm) : fChain(0) {

  TFile      *file = new TFile(fname.c_str());
  TDirectory *dir  = (TDirectory*)file->FindObjectAny(dirnm.c_str());
  TTree      *tree = (TTree*)dir->Get("TB06Sim");
  Init(tree);
}

TB06Analysis::~TB06Analysis() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t TB06Analysis::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t TB06Analysis::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  if (!fChain) return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0) return centry;
  if (fChain->GetTreeNumber() != fCurrent) {
    fCurrent = fChain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void TB06Analysis::Init(TTree *tree) {
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
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("eBeam_", &eBeam_, &b_eBeam_);
  fChain->SetBranchAddress("etaBeam_", &etaBeam_, &b_etaBeam_);
  fChain->SetBranchAddress("phiBeam_", &phiBeam_, &b_phiBeam_);
  fChain->SetBranchAddress("edepEC_", &edepEC_, &b_edepEC_);
  fChain->SetBranchAddress("edepHB_", &edepHB_, &b_edepHB_);
  fChain->SetBranchAddress("edepHO_", &edepHO_, &b_edepHO_);
  fChain->SetBranchAddress("noiseEC_", &noiseEC_, &b_noiseEC_);
  fChain->SetBranchAddress("noiseHB_", &noiseHB_, &b_noiseHB_);
  fChain->SetBranchAddress("noiseHO_", &noiseHO_, &b_noiseHO_);
  fChain->SetBranchAddress("edepS1_", &edepS1_, &b_edepS1_);
  fChain->SetBranchAddress("edepS2_", &edepS2_, &b_edepS2_);
  fChain->SetBranchAddress("edepS3_", &edepS3_, &b_edepS3_);
  fChain->SetBranchAddress("edepS4_", &edepS4_, &b_edepS4_);
  fChain->SetBranchAddress("edepVC_", &edepVC_, &b_edepVC_);
  Notify();
}

Bool_t TB06Analysis::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void TB06Analysis::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t TB06Analysis::Cut(Long64_t ) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void TB06Analysis::Loop() {
  //   In a ROOT session, you can do:
  //      root> .L TB06Analysis.C
  //      root> TB06Analysis t
  //      root> t.GetEntry(12); // Fill t data members with entry number 12
  //      root> t.Show();       // Show values of entry 12
  //      root> t.Show(16);     // Read and show values of entry 16
  //      root> t.Loop();       // Loop on all entries
  //

  //     This is the loop skeleton where:
  //    jentry is the global entry number in the chain
  //    ientry is the entry number in the current Tree
  //  Note that the argument to GetEntry must be:
  //    jentry for TChain::GetEntry
  //    ientry for TTree::GetEntry and TBranch::GetEntry
  //
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  //by  b_branchname->GetEntry(ientry); //read only this branch

  if (fChain == 0) return;

  Long64_t nentries = fChain->GetEntriesFast();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;
  }
}
