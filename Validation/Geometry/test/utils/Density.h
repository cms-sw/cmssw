//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Aug  7 10:01:22 2006 by ROOT version 5.12/00
// from TTree T1/GeometryTest Tree
// found on file: matbdg_tree_TOB.root
//////////////////////////////////////////////////////////

#ifndef Density_h
#define Density_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TString.h>
#include <TGaxis.h>

#include <TTree.h>
#include <iostream>
#include <fstream>

// histograms
// Target
TProfile* prof_density_vs_eta;
//

// logfile
std::ofstream theLogFile;
//

// plot range
double etaMin;
double etaMax;
//

class Density {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain
  
  // Declaration of leave types
  Float_t         ParticleEta;
  Int_t           Nsteps;
  Double_t        InitialX[8000];   //[Nsteps]
  Double_t        InitialY[8000];   //[Nsteps]
  Double_t        InitialZ[8000];   //[Nsteps]
  Double_t        FinalX[8000];   //[Nsteps]
  Double_t        FinalY[8000];   //[Nsteps]
  Double_t        FinalZ[8000];   //[Nsteps]
  Float_t         MaterialDensity[8000];   //[Nsteps]
  
  // List of branches
  TBranch        *b_ParticleEta;   //!
  TBranch        *b_Nsteps;   //!
  TBranch        *b_InitialX;   //!
  TBranch        *b_InitialY;   //!
  TBranch        *b_InitialZ;   //!
  TBranch        *b_FinalX;   //!
  TBranch        *b_FinalY;   //!
  TBranch        *b_FinalZ;   //!
  TBranch        *b_MaterialDensity;   //!
  
  Density(TString fileName);
  virtual ~Density();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);

  
private:
  //
  virtual void helpfulCommands();
  //
  // directory to store images
  TString theDirName;
};

#endif

#ifdef Density_cxx
Density::Density(TString fileName)
{
  // images directory
  theDirName = "Images";
  
  // files
  cout << "*** Open file... " << endl;
  cout << fileName << endl;
  cout << "***" << endl;
  cout << " Output Directory... " << endl;
  cout << theDirName << endl;
  cout << "***" << endl;
  //
  
  // open root files
  TFile* theDetectorFile = new TFile(fileName);
  //
  // get tree
  TTree* theTree = (TTree*)theDetectorFile->Get("T1");
  //
  Init(theTree);
  Book();
  //
  helpfulCommands();
  //
}

Density::~Density()
{
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t Density::GetEntry(Long64_t entry)
{
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}
Long64_t Density::LoadTree(Long64_t entry)
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

void Density::Init(TTree *tree)
{
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normaly not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).
  
  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);
  
  fChain->SetBranchAddress("Particle Eta",    &ParticleEta, &b_ParticleEta);
  fChain->SetBranchAddress("Nsteps", &Nsteps, &b_Nsteps);
  fChain->SetBranchAddress("Initial X", InitialX, &b_InitialX);
  fChain->SetBranchAddress("Initial Y", InitialY, &b_InitialY);
  fChain->SetBranchAddress("Initial Z", InitialZ, &b_InitialZ);
  fChain->SetBranchAddress("Final X",   FinalX,   &b_FinalX);
  fChain->SetBranchAddress("Final Y",   FinalY,   &b_FinalY);
  fChain->SetBranchAddress("Final Z",   FinalZ,   &b_FinalZ);
  fChain->SetBranchAddress("Material Density",   MaterialDensity,   &b_MaterialDensity);
  Notify();
}

Bool_t Density::Notify()
{
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normaly not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  
  return kTRUE;
}

void Density::Show(Long64_t entry)
{
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}
Int_t Density::Cut(Long64_t entry)
{
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void Density::Book(){
  etaBin = 15;
  etaMin = 0.0;
  etaMax = 3.0;
  //
  prof_density_vs_eta = new TProfile("prof_density_vs_eta",
				     "Average Density vs Pseudorapidity;|#eta|;#bar{#rho} [g/cm^{3}]",
				     etaBin,etaMin,etaMax);
}

void Density::MakePlots(TString suffix);

#endif // #ifdef Density_cxx
