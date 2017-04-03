//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Dec 18 22:51:59 2007 by ROOT version 5.14/00
// from TTree h101/HEPEVT
// found on file: dpe10.root
//////////////////////////////////////////////////////////

#ifndef h101_h
#define h101_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TMath.h>
#include <TRandom2.h>
#include <vector>
#include <string>
#include <CLHEP/Vector/ThreeVector.h>
#include <CLHEP/Vector/LorentzVector.h>


struct MADProton
{
	MADProton()
	{
		x=y=px=py=pz=xi=t_0=phi_0=thetax=thetay=xi_0;
	}
  double x,y,px,py,pz,xi,t_0,phi_0,thetax,thetay,xi_0;
};

struct MADProtonPair
{
  MADProton r, l;
  double mass;
};

typedef std::vector<MADProtonPair> MADProtonPairCollection;


class h101 {
  public :
    TTree          *fChain;   //!pointer to the analyzed TTree or TChain
    Int_t           fCurrent; //!current Tree number in a TChain
    
    // Declaration of leave types
    Int_t           Nevhep;
    Int_t           Nhep;
    Int_t           Idhep[15000];   //[Nhep]
    Int_t           Jsmhep[15000];   //[Nhep]
    Int_t           Jsdhep[15000];   //[Nhep]
    Float_t         Phep[15000][5];   //[Nhep]
    Float_t         Vhep[15000][4];   //[Nhep]
    Int_t           Irun;
    Int_t           Ievt;
    Float_t         Weight;
    Float_t         Xsecn;
    Int_t           Ifilter;
    Int_t           Nparam;
    Float_t         Param[2000];   //[Nparam]
    
    // List of branches
    TBranch        *b_Nevhep;   //!
    TBranch        *b_Nhep;   //!
    TBranch        *b_Idhep;   //!
    TBranch        *b_Jsmhep;   //!
    TBranch        *b_Jsdhep;   //!
    TBranch        *b_Phep;   //!
    TBranch        *b_Vhep;   //!
    TBranch        *b_Irun;   //!
    TBranch        *b_Ievt;   //!
    TBranch        *b_Weight;   //!
    TBranch        *b_Xsecn;   //!
    TBranch        *b_Ifilter;   //!
    TBranch        *b_Nparam;   //!
    TBranch        *b_Param;   //!
    
    h101(TTree *tree=0);
    virtual ~h101();
    virtual Int_t    Cut(Long64_t entry);
    virtual Int_t    GetEntry(Long64_t entry);
    virtual Long64_t LoadTree(Long64_t entry);
    virtual void     Init(TTree *tree);
    virtual void     Loop();
    virtual Long64_t GetEntries();
    virtual MADProtonPairCollection GetMADProtonPairs(Long64_t first_event, Long64_t event_no);
    virtual bool GetForwardProton(int direction, MADProton &proton);
    virtual void SmearForwardProtonPair(MADProtonPair &proton_pair);
    virtual void CorrectBeamCoordinateSigns(MADProtonPair &proton_pair);
    virtual void SmearProton(MADProton &proton, double angle, double beta, double factor, const CLHEP::Hep3Vector &axis, const CLHEP::Hep3Vector &dir);
    virtual void WriteMADInputParticles(const MADProtonPairCollection &proton_pair_collection, int direction, std::string file_name);
    virtual Bool_t   Notify();
    virtual void     Show(Long64_t entry = -1);
    
    static const double mp0_;
    static const double E0_;
    static const double p0_;
    
    double Momentum_to_t(double px, double py, double pz) const;  //GeV
    double Momentum_to_ksi(double px, double py, double pz) const;  //GeV
    double Momentum_to_phi(double px, double py, double pz) const;  //rad
    
    static const double BeamPosX_;
    static const double BeamPosY_;
    static const double BeamSigmaX_;
    static const double BeamSigmaY_;
    
    static const double MeanAlpha_;
    static const double SigmaTheta_;
    
    static const double SigmaXi_;
    
    TRandom2 *rand;
};

#endif

#ifdef h101_cxx
h101::h101(TTree *tree)
{
  rand = new TRandom2(736583);
  
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree.
  if (tree == 0) {
    TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("dpe10.root");
    if (!f) {
      f = new TFile("dpe10.root");
    }
    tree = (TTree*)gDirectory->Get("h101");
    
  }
  Init(tree);
}

h101::~h101()
{
  if (!fChain) return;
  delete fChain->GetCurrentFile();
  delete rand;
}

Int_t h101::GetEntry(Long64_t entry)
{
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}
Long64_t h101::LoadTree(Long64_t entry)
{
  // Set the environment to read one entry
  if (!fChain) return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0) return centry;
  if (!fChain->InheritsFrom(TChain::Class()))  return centry;
  TChain *chain = (TChain*)fChain;
  if (chain->GetTreeNumber() != fCurrent) {
    fCurrent = chain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void h101::Init(TTree *tree)
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
  
  fChain->SetBranchAddress("Nevhep", &Nevhep, &b_Nevhep);
  fChain->SetBranchAddress("Nhep", &Nhep, &b_Nhep);
  fChain->SetBranchAddress("Idhep", Idhep, &b_Idhep);
  fChain->SetBranchAddress("Jsmhep", Jsmhep, &b_Jsmhep);
  fChain->SetBranchAddress("Jsdhep", Jsdhep, &b_Jsdhep);
  fChain->SetBranchAddress("Phep", Phep, &b_Phep);
  fChain->SetBranchAddress("Vhep", Vhep, &b_Vhep);
  fChain->SetBranchAddress("Irun", &Irun, &b_Irun);
  fChain->SetBranchAddress("Ievt", &Ievt, &b_Ievt);
  fChain->SetBranchAddress("Weight", &Weight, &b_Weight);
  fChain->SetBranchAddress("Xsecn", &Xsecn, &b_Xsecn);
  fChain->SetBranchAddress("Ifilter", &Ifilter, &b_Ifilter);
  fChain->SetBranchAddress("Nparam", &Nparam, &b_Nparam);
  fChain->SetBranchAddress("Param", Param, &b_Param);
  Notify();
}

Bool_t h101::Notify()
{
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normaly not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  
  return kTRUE;
}

void h101::Show(Long64_t entry)
{
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}
Int_t h101::Cut(Long64_t entry)
{
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}
#endif // #ifdef h101_cxx
