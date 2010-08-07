#ifndef SimpleVertexAnalysis_h
#define SimpleVertexAnalysis_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1F.h>
#include "TString.h"
#include "TCanvas.h"
#include <iostream>
#include <iomanip>
#include <vector>

#define MAXTRACK 120



  /**
   * Root analysis code for TTree produced by SimpleVertexTree
   */

class SimpleVertexAnalysis {
   public :

   SimpleVertexAnalysis(TTree *tree=0);
  /**
   * Constructor for a single file<br>
   * \param filename: The file name
   * \param treeName: The name of the tree
   */
   SimpleVertexAnalysis(TString filename, 
   		TString treeName = "VertexFitter");
  /**
   * Constructor for multiple files, of the type result_1.root<br>
   * \param filename: 	The base of he file name (e.g. "result")
   * \param start:	The number of the first file
   * \param end:	The number of the last file
   * \param treeName: 	The name of the tree
   */
   SimpleVertexAnalysis(TString base, int start, int end, 
   		TString treeName = "VertexFitter");
   ~SimpleVertexAnalysis();

  /**
   * The main method to loop over all the events and fill the histos.
   */
   void vertexLoop();

  /**
   * Sets the limits for the vertex hisotgrams:
   *   aMaxTrans:	vertex resolutions, transverse coordinates (x, y)
   *   aMaxZ: 	vertex resolutions, longitudinal coordinates (z)
   *   aMaxTracks:	number of tracks (Sim, Rec)
   *   aMaxWeights:	sum of weights
   *   aMaxTime:	cpu time per fit
   */

   void vertexHistoLimits(float aMaxTrans = 250.0, float aMaxZ = 250.0, 
     float aMaxTracks = 60.0,  float aMaxWeights = 35.0, float aMaxTime = 0.05);

  /**
   * The main method to loop over all the events and fill the histos.
   */
   void trackLoop();
  /**
   * Method to print all the Vertx plots to a ps file.
   */
   void psVertexResult(TString name);
  /**
   * Method to print all the Vertx plots to a serie of eps files.
   */
   void epsVertexResult(TString name);
  /**
   * Output Vertx plots into 2 canvases
   */
   void plotVertexResult();

  /**
   * Fit of the residual and pull plots with a single Gaussian, and printout of the
   * main results
   * A separate output stream can be provided (parameter out - default is std::cout)
   */
   void singleGaussianVertexResult(ostream &out = std::cout);
  /**
   * Fit of the residual and pull plots with two Gaussians (one for the core, the
   * other for the tails, and printout of the main results
   * A separate output stream can be provided (parameter out - default is std::cout)
   */
   void doubleGaussianVertexResult(ostream &out = std::cout);
  /**
   * Measurement of the coverage of the residual distributions 
   * (50%, 90% and 95% coverage)
   * A separate output stream can be provided (parameter out - default is std::cout)
   */
   void vertexCoverage(ostream &out = std::cout);
  /**
   * Output of Track parameter results into 4 canvases
   */
   void plotTrackResult();
  /**
   * Method to print all the Track parameter result plots to a ps file.
   */
   void psTrackResult(TString name);

  /**
   * Method to produce the TeX line for the statistices table 
   * of the Tables (TeX )
   * A separate output stream can be provided (parameter out - default is std::cout)
   */
   void statTeXResult(ostream &out = std::cout);

  /**
   * Method to produce the TeX line for the X-coordinate table 
   * of the Tables (TeX )
   * A separate output stream can be provided (parameter out - default is std::cout)
   */
   void xTeXResult(ostream &out = std::cout);

  /**
   * Method to produce the TeX line for the Z-coordinate table 
   * of the Tables (TeX )
   * A separate output stream can be provided (parameter out - default is std::cout)
   */
   void zTeXResult(ostream &out = std::cout);

  /**
   * Method to produce the TeX line for the X- and Z-coordinate table 
   * of the Tables (TeX )
   * A separate output stream can be provided (parameter out - default is std::cout)
   */
   void resolutionTeXResult(ostream &out = std::cout);


   TH1F *resX, *resY, *resZ; 
   TH1F *pullX, *pullY, *pullZ;
   TH1F *chiNorm, *chiProbability, *weight, *normWeight, *downWeight;
   TH1F *numberUsedRecTracks, *numberRawRecTracks, *numberSimTracks, *sharedTracks, *ratioSharedTracks, *timing;
   TCanvas *resCanvas, *statCanvas;
   float x_coverage, y_coverage, z_coverage;
   int failure, total;

   TH1F *resRecPt, *resRecPhi, *resRecTheta, *resRecTimp, *resRecLimp;
   TH1F *pullRecPt, *pullRecPhi, *pullRecTheta, *pullRecTimp, *pullRecLimp;
   TH1F *resRefPt, *resRefPhi, *resRefTheta, *resRefTimp, *resRefLimp;
   TH1F *pullRefPt, *pullRefPhi, *pullRefTheta, *pullRefTimp, *pullRefLimp;
   TH1F *pTSim, *etaSim, *pTRec, *etaRec, *pTRef, *etaRef;
   TCanvas *resRecCanvas,*pullRecCanvas,*resRefCanvas,*pullRefCanvas, *distCanvas;
   bool bookedVertexC, bookedTrackC;

   Int_t  Cut(Int_t entry);
   Int_t  GetEntry(Int_t entry);
   Int_t  LoadTree(Int_t entry);
   void   Init(TTree *tree);
   Bool_t Notify();
   void   Show(Int_t entry = -1);
   TString theTreeName;
   Int_t nentries;


   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain
   // Declaration of leave types
   Int_t           vertex;
   Float_t         simPos_X;
   Float_t         simPos_Y;
   Float_t         simPos_Z;
   Float_t         recPos_X;
   Float_t         recPos_Y;
   Float_t         recPos_Z;
   Float_t         recErr_X;
   Float_t         recErr_Y;
   Float_t         recErr_Z;
   Int_t           nbrTrk_Sim;
   Int_t           nbrTrk_Rec;
   Int_t           nbrTrk_Shared;
   Float_t         chiTot;
   Float_t         ndf;
   Float_t         chiProb;
   Float_t         time;
   Int_t           simTracks;
   Int_t           simTrack_recIndex[MAXTRACK];   //[simTracks]
   Float_t         simPar_ptinv[MAXTRACK];   //[simTracks]
   Float_t         simPar_theta[MAXTRACK];   //[simTracks]
   Float_t         simPar_phi[MAXTRACK];   //[simTracks]
   Float_t         simPar_timp[MAXTRACK];   //[simTracks]
   Float_t         simPar_limp[MAXTRACK];   //[simTracks]
   Int_t           recTracks;
   Int_t           recTrack_simIndex[MAXTRACK];   //[recTracks]
   Float_t         recTrack_weight[MAXTRACK];   //[recTracks]
   Float_t         recPar_ptinv[MAXTRACK];   //[recTracks]
   Float_t         recPar_theta[MAXTRACK];   //[recTracks]
   Float_t         recPar_phi[MAXTRACK];   //[recTracks]
   Float_t         recPar_timp[MAXTRACK];   //[recTracks]
   Float_t         recPar_limp[MAXTRACK];   //[recTracks]
   Float_t         refPar_ptinv[MAXTRACK];   //[recTracks]
   Float_t         refPar_theta[MAXTRACK];   //[recTracks]
   Float_t         refPar_phi[MAXTRACK];   //[recTracks]
   Float_t         refPar_timp[MAXTRACK];   //[recTracks]
   Float_t         refPar_limp[MAXTRACK];   //[recTracks]
   Float_t         recErr_ptinv[MAXTRACK];   //[recTracks]
   Float_t         recErr_theta[MAXTRACK];   //[recTracks]
   Float_t         recErr_phi[MAXTRACK];   //[recTracks]
   Float_t         recErr_timp[MAXTRACK];   //[recTracks]
   Float_t         recErr_limp[MAXTRACK];   //[recTracks]
   Float_t         refErr_ptinv[MAXTRACK];   //[recTracks]
   Float_t         refErr_theta[MAXTRACK];   //[recTracks]
   Float_t         refErr_phi[MAXTRACK];   //[recTracks]
   Float_t         refErr_timp[MAXTRACK];   //[recTracks]
   Float_t         refErr_limp[MAXTRACK];   //[recTracks]

   // List of branches
   TBranch        *b_vertex;   //!
   TBranch        *b_simPos;   //!
   TBranch        *b_recPos;   //!
   TBranch        *b_recErr;   //!
   TBranch        *b_nbrTrk;   //!
   TBranch        *b_chiTot;   //!
   TBranch        *b_ndf;   //!
   TBranch        *b_chiProb;   //!
   TBranch        *b_time;   //!
   TBranch        *b_simTracks;   //!
   TBranch        *b_simTrack_recIndex;   //!
   TBranch        *b_simPar_ptinv;   //!
   TBranch        *b_simPar_theta;   //!
   TBranch        *b_simPar_phi;   //!
   TBranch        *b_simPar_timp;   //!
   TBranch        *b_simPar_limp;   //!
   TBranch        *b_recTracks;   //!
   TBranch        *b_recTrack_simIndex;   //!
   TBranch        *b_recTrack_weight;   //!
   TBranch        *b_recPar_ptinv;   //!
   TBranch        *b_recPar_theta;   //!
   TBranch        *b_recPar_phi;   //!
   TBranch        *b_recPar_timp;   //!
   TBranch        *b_recPar_limp;   //!
   TBranch        *b_refPar_ptinv;   //!
   TBranch        *b_refPar_theta;   //!
   TBranch        *b_refPar_phi;   //!
   TBranch        *b_refPar_timp;   //!
   TBranch        *b_refPar_limp;   //!
   TBranch        *b_recErr_ptinv;   //!
   TBranch        *b_recErr_theta;   //!
   TBranch        *b_recErr_phi;   //!
   TBranch        *b_recErr_timp;   //!
   TBranch        *b_recErr_limp;   //!
   TBranch        *b_refErr_ptinv;   //!
   TBranch        *b_refErr_theta;   //!
   TBranch        *b_refErr_phi;   //!
   TBranch        *b_refErr_timp;   //!
   TBranch        *b_refErr_limp;   //!

   TChain* myChain;

private:
  /**
   * Calculates and prints the 50, 90 and 95% coverage for the given vector.
   * Returns the 90% coverage.
   */
  float getCoverage(std::vector<float> &residuals, ostream &out = std::cout);
  void doubleGaussianFit(TH1F *plot, ostream &out = std::cout);
  void epsPlot(TH1F *plot, TString name);

   void bookTrackHisto();
   void deleteTrackHisto();
   void bookVertexHisto();
   void deleteVertexHisto();

  float maxTrans, maxZ, maxTracks, maxWeights, maxTime;
  bool bookedVertex, bookedTrack;
};

#endif

#ifdef SimpleVertexAnalysis_cxx
SimpleVertexAnalysis::SimpleVertexAnalysis(TTree *tree)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   bookedVertex=false;bookedTrack=false;
   bookedVertexC=false;bookedTrackC=false;
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("data/pvr_digi_1.root");
      if (!f) {
         f = new TFile("data/pvr_digi_1.root");
      }
      tree = (TTree*)gDirectory->Get("closest");

   }
   Init(tree);
   vertexHistoLimits();
}

SimpleVertexAnalysis::~SimpleVertexAnalysis()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
  if (bookedVertex) deleteVertexHisto();
   if (bookedTrack) deleteTrackHisto();
}

Int_t SimpleVertexAnalysis::GetEntry(Int_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Int_t SimpleVertexAnalysis::LoadTree(Int_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Int_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->IsA() != TChain::Class()) return centry;
   TChain *chain = (TChain*)fChain;
   if (chain->GetTreeNumber() != fCurrent) {
      fCurrent = chain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void SimpleVertexAnalysis::Init(TTree *tree)
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

   fChain->SetBranchAddress("vertex",&vertex);
   fChain->SetBranchAddress("simPos",&simPos_X);
   fChain->SetBranchAddress("recPos",&recPos_X);
   fChain->SetBranchAddress("recErr",&recErr_X);
   fChain->SetBranchAddress("nbrTrk",&nbrTrk_Sim);
   fChain->SetBranchAddress("chiTot",&chiTot);
   fChain->SetBranchAddress("ndf",&ndf);
   fChain->SetBranchAddress("chiProb",&chiProb);
   fChain->SetBranchAddress("time",&time);
   fChain->SetBranchAddress("simTracks",&simTracks);
   fChain->SetBranchAddress("simTrack_recIndex",simTrack_recIndex);
   fChain->SetBranchAddress("simPar_ptinv",simPar_ptinv);
   fChain->SetBranchAddress("simPar_theta",simPar_theta);
   fChain->SetBranchAddress("simPar_phi",simPar_phi);
   fChain->SetBranchAddress("simPar_timp",simPar_timp);
   fChain->SetBranchAddress("simPar_limp",simPar_limp);
   fChain->SetBranchAddress("recTracks",&recTracks);
   fChain->SetBranchAddress("recTrack_simIndex",recTrack_simIndex);
   fChain->SetBranchAddress("recTrack_weight",recTrack_weight);
   fChain->SetBranchAddress("recPar_ptinv",recPar_ptinv);
   fChain->SetBranchAddress("recPar_theta",recPar_theta);
   fChain->SetBranchAddress("recPar_phi",recPar_phi);
   fChain->SetBranchAddress("recPar_timp",recPar_timp);
   fChain->SetBranchAddress("recPar_limp",recPar_limp);
   fChain->SetBranchAddress("refPar_ptinv",refPar_ptinv);
   fChain->SetBranchAddress("refPar_theta",refPar_theta);
   fChain->SetBranchAddress("refPar_phi",refPar_phi);
   fChain->SetBranchAddress("refPar_timp",refPar_timp);
   fChain->SetBranchAddress("refPar_limp",refPar_limp);
   fChain->SetBranchAddress("recErr_ptinv",recErr_ptinv);
   fChain->SetBranchAddress("recErr_theta",recErr_theta);
   fChain->SetBranchAddress("recErr_phi",recErr_phi);
   fChain->SetBranchAddress("recErr_timp",recErr_timp);
   fChain->SetBranchAddress("recErr_limp",recErr_limp);
   fChain->SetBranchAddress("refErr_ptinv",refErr_ptinv);
   fChain->SetBranchAddress("refErr_theta",refErr_theta);
   fChain->SetBranchAddress("refErr_phi",refErr_phi);
   fChain->SetBranchAddress("refErr_timp",refErr_timp);
   fChain->SetBranchAddress("refErr_limp",refErr_limp);
   Notify();
}

Bool_t SimpleVertexAnalysis::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. Typically here the branch pointers
   // will be retrieved. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed.

   // Get branch pointers
   b_vertex = fChain->GetBranch("vertex");
   b_simPos = fChain->GetBranch("simPos");
   b_recPos = fChain->GetBranch("recPos");
   b_recErr = fChain->GetBranch("recErr");
   b_nbrTrk = fChain->GetBranch("nbrTrk");
   b_chiTot = fChain->GetBranch("chiTot");
   b_ndf = fChain->GetBranch("ndf");
   b_chiProb = fChain->GetBranch("chiProb");
   b_time = fChain->GetBranch("time");
   b_simTracks = fChain->GetBranch("simTracks");
   b_simTrack_recIndex = fChain->GetBranch("simTrack_recIndex");
   b_simPar_ptinv = fChain->GetBranch("simPar_ptinv");
   b_simPar_theta = fChain->GetBranch("simPar_theta");
   b_simPar_phi = fChain->GetBranch("simPar_phi");
   b_simPar_timp = fChain->GetBranch("simPar_timp");
   b_simPar_limp = fChain->GetBranch("simPar_limp");
   b_recTracks = fChain->GetBranch("recTracks");
   b_recTrack_simIndex = fChain->GetBranch("recTrack_simIndex");
   b_recTrack_weight = fChain->GetBranch("recTrack_weight");
   b_recPar_ptinv = fChain->GetBranch("recPar_ptinv");
   b_recPar_theta = fChain->GetBranch("recPar_theta");
   b_recPar_phi = fChain->GetBranch("recPar_phi");
   b_recPar_timp = fChain->GetBranch("recPar_timp");
   b_recPar_limp = fChain->GetBranch("recPar_limp");
   b_refPar_ptinv = fChain->GetBranch("refPar_ptinv");
   b_refPar_theta = fChain->GetBranch("refPar_theta");
   b_refPar_phi = fChain->GetBranch("refPar_phi");
   b_refPar_timp = fChain->GetBranch("refPar_timp");
   b_refPar_limp = fChain->GetBranch("refPar_limp");
   b_recErr_ptinv = fChain->GetBranch("recErr_ptinv");
   b_recErr_theta = fChain->GetBranch("recErr_theta");
   b_recErr_phi = fChain->GetBranch("recErr_phi");
   b_recErr_timp = fChain->GetBranch("recErr_timp");
   b_recErr_limp = fChain->GetBranch("recErr_limp");
   b_refErr_ptinv = fChain->GetBranch("refErr_ptinv");
   b_refErr_theta = fChain->GetBranch("refErr_theta");
   b_refErr_phi = fChain->GetBranch("refErr_phi");
   b_refErr_timp = fChain->GetBranch("refErr_timp");
   b_refErr_limp = fChain->GetBranch("refErr_limp");

   return kTRUE;
}

void SimpleVertexAnalysis::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t SimpleVertexAnalysis::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return entry;
}
#endif // #ifdef SimpleVertexAnalysis_cxx

