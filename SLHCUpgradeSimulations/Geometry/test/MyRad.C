#define MyRad_cxx
// The class definition in MyRad.h has been generated automatically
// by the ROOT utility TTree::MakeSelector(). This class is derived
// from the ROOT class TSelector. For more information on the TSelector
// framework see $ROOTSYS/README/README.SELECTOR or the ROOT User Manual.

// The following methods are defined in this file:
//    Begin():        called every time a loop on the tree starts,
//                    a convenient place to create your histograms.
//    SlaveBegin():   called after Begin(), when on PROOF called only on the
//                    slave servers.
//    Process():      called for each event, in this function you decide what
//                    to read and fill your histograms.
//    SlaveTerminate: called at the end of the loop on the tree, when on PROOF
//                    called only on the slave servers.
//    Terminate():    called at the end of the loop on the tree,
//                    a convenient place to draw/fit your histograms.
//
// To use this file, try the following session on your Tree T:
//
// Root > T->Process("MyRad.C")
// Root > T->Process("MyRad.C","some options")
// Root > T->Process("MyRad.C+")
//

#include "MyRad.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

TH1F *hNtracks;
TH1F *hNlayers;
TH1F *hJlow;
TH1F *hJhgh;

// Layer Numbers for Standard Detector:
// (layers > 99 are cables, supports, etc.)
//   Beampipe:  100
//   PXB:       1,101,2,102,103,3,104,105
//   PXD:       4,5,106,107
//   TIB:       6,7,8,9,108,109
//   TID:       10,11,12,110
//   TOB:       111,13,14,15,16,17,18,112
//   TEC:       19,20,21,22,23,24,25,26,27,
//   Tracker Outside:  113,114,115
//{ PXB=0,PXD=3,TIB=5,TID=9,TOB=12,TEC=18 };

const int nlPlots=43;

TProfile *tpl[nlPlots];
TH1D *hpl[nlPlots];

const int ntpbins=401;
const float xtplow=-4.0;
const float xtphgh=4.0;

map <int,int> transl;

const int mymap[nlPlots] = {100,                                  // Beampipe
                              1,101,  2,102,103,  3,104,105,      // PXB
			      4,  5,106,107,                      // PXD
			      6,  7,  8,  9,108,109,              // TIB
			     10, 11, 12,110,                      // TID
			    111, 13, 14, 15, 16, 17, 18,112,      // TOB
			     19, 20, 21, 22, 23, 24, 25, 26, 27,  // TEC
			    113,114,115};                         // Tracker Outside

const char *hlname[nlPlots] = {"tp100",                                                            // Beampipe
                               "tp001","tp101","tp002","tp102","tp103","tp003","tp104","tp105",    // PXB
			       "tp004","tp005","tp106","tp107",                                    // PXD
			       "tp006","tp007","tp008","tp009","tp108","tp109",                    // TIB
			       "tp010","tp011","tp012","tp110",                                    // TID
			       "tp111","tp013","tp014","tp015","tp016","tp017","tp018","tp112",    // TOB
			       "tp019","tp020","tp021","tp022","tp023","tp024","tp025","tp026","tp027",  // TEC
			       "tp113","tp114","tp115"};                                           // Tracker Outside

const int lcolors[nlPlots] = {9,
			      2,2,2,2,2,2,2,2,
			      3,3,3,3,
			      4,4,4,4,4,4,
			      5,5,5,5,
			      6,6,6,6,6,6,6,6,
			      7,7,7,7,7,7,7,7,7,
			      9,9,9};

THStack hs("hs","Radlen vs. eta");

void MyRad::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   for (int i=0; i<nlPlots; i++) {
     transl[mymap[i]] = i;
   }

   hNtracks=new TH1F("hNtracks","# of Tracks",21,-2.0,18.0);
   hNlayers=new TH1F("hNlayers","# of Layers Crossed",51,0.0,50.0);
   hJlow=new TH1F("hJlow","Low Layers Crossed",31,0.0,30.0);
   hJhgh=new TH1F("hJhgh","Hgh Layers Crossed",31,99.0,129.0);

   for (int i=0; i<nlPlots; i++) {
     tpl[i]=new TProfile(hlname[i],"Radlen vs. eta",ntpbins,xtplow,xtphgh);
   }

}

void MyRad::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

}

Bool_t MyRad::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either MyRad::GetEntry() or TBranch::GetEntry()
   // to read either all or the required parts of the data. When processing
   // keyed objects with PROOF, the object is already loaded and is available
   // via the fObject pointer.
   //
   // This function should contain the "body" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.
   //
   // The processing can be stopped by calling Abort().
   //
   // Use fStatus to set the return value of TTree::Process().
   //
   // The return value is currently not used.

  fChain->GetTree()->GetEntry(entry);

  hNtracks->Fill(evt_numfs);
  hNlayers->Fill(trk_nlyrs);
  hJlow->Fill(lyr_laynm);
  hJhgh->Fill(lyr_laynm);

  int ihist=transl[lyr_laynm];
  tpl[ihist]->Fill(trk_eta,lyr_radln);

  return kTRUE;
}

void MyRad::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

}

void MyRad::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

  TCanvas *c=new TCanvas();

  //c->Divide(2,1);
  //c->cd(1);
  //hJlow->Draw();
  //  c->cd(2);
  //hJhgh->Draw();

  for (int i=0; i<nlPlots; i++) {
    hpl[i] = tpl[i]->ProjectionX();
    hpl[i]->SetFillColor(lcolors[i]);
    hpl[i]->SetLineColor(lcolors[i]);
    hs.Add(hpl[i]);
  }

  hs.Draw("hist");

}
