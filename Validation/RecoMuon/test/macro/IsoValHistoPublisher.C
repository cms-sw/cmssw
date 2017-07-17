#include <vector>
#include <algorithm>
#include "TMath.h"
#include "PlotHelpers.C"

// Uncomment the following line for some extra debug information
// #define DEBUG

void IsoValHistoPublisher(const char* newFile="NEW_FILE",const char* refFile="REF_FILE") {
  cout << ">> Starting IsoValHistoPublisher(" << newFile << "," << refFile << ")..." << endl;

  //====  To be replaced from python ====================

  const char* dataType = "DATATYPE";
  const char* refLabel("REF_LABEL, REF_RELEASE REFSELECTION");
  const char* newLabel("NEW_LABEL, NEW_RELEASE NEWSELECTION");
  
  
  // ==== Initial settings and loads
  //gROOT->ProcessLine(".x HistoCompare_Tracks.C");
  //gROOT ->Reset();
  gROOT ->SetBatch();
  gErrorIgnoreLevel = kWarning; // Get rid of the info messages

  SetGlobalStyle();
  
  
  // ==== Some cleaning... is this needed?  
  delete gROOT->GetListOfFiles()->FindObject(refFile);
  delete gROOT->GetListOfFiles()->FindObject(newFile); 
  

  // ==== Opening files, moving to the right branch and getting the list of sub-branches
  cout << ">> Openning file, moving to the right branch and getting sub-branches..." << endl;

  cout << ">> Finding sources..." << endl;
  TFile* sfile = new TFile(newFile);
  TList* sl = getListOfBranches(dataType, sfile, "MuonIsolationV_inc");
  if (!sl) {
    cout << "ERROR: Could not find keys!!!" << endl;
    cerr << "ERROR: Could not find keys!!!" << endl;
    return;
  }
  TDirectory*  sdir  = gDirectory;
  for (unsigned int i = 0; i < sl->GetEntries(); i++)
    cout << "   + " << sl->At(i)->GetName() << endl;
    

  cout << ">> Finding references..." << endl;
  TFile* rfile = new TFile(refFile);
  TList* rl = getListOfBranches(dataType, rfile, "MuonIsolationV_inc");
  if (!rl) {
    cout << "ERROR: Could not find keys!!!" << endl;
    cerr << "ERROR: Could not find keys!!!" << endl;
    return;
  }
  TDirectory* rdir  = gDirectory;
  for (unsigned int i = 0; i < sl->GetEntries(); i++)
    cout << "   + " << sl->At(i)->GetName() << endl;
  
  
  
  //==== Iterate now over histograms and collections
  cout << ">> Iterating over histograms and collections..." << endl;

  TIter iter_r( rl );
  TIter iter_s( sl );
  TKey* rKey = 0;
  TKey* sKey = 0;
  TString rcollname;
  TString scollname;

  while ( (rKey = (TKey*)iter_r()) ) {
    TString myName = rKey->GetName();
#ifdef DEBUG
    cout << "DEBUG: Checking key " << myName << endl;
#endif
    rcollname = myName;
    sKey = (TKey*)iter_s();
    if (!sKey) continue;
    scollname = sKey->GetName();
    if ( (rcollname != scollname) && (rcollname+"FS" != scollname) && (rcollname != scollname+"FS") ) {
      cerr << "ERROR: Different collection names, please check: " << rcollname << " : " << scollname << endl;
      cout << "ERROR: Different collection names, please check: " << rcollname << " : " << scollname << endl;
      continue;
    }
    
    // ==== Now let's go for the plotting...
    cout << ">> Comparing plots in " << myName << "..." << endl;    
    cerr << ">> Comparing plots in " << myName << "..." << endl;    
    TString newDir("NEW_RELEASE/NEWSELECTION/NEW_LABEL/");
    newDir+=myName;
    gSystem->mkdir(newDir,kTRUE);
    
    bool    logy    [] = {false,   false,  false,      false    };
    bool    doKolmo [] = {true,    true,   true,       true     };
    Double_t norm   [] = {0.,0.,0.,0.};
    //===== Tracker, ECAL Deposits
    const char* plots1  [] = {"sumPt", "emEt", "sumPt_cd", "emEt_cd"};
    Plot4Histograms(newDir + "/muonIso1.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "IsoHistos1", "Tracker, ECAL Deposits",
		    refLabel, newLabel,
		    plots1, 0,
		    logy, doKolmo, norm);
    
    //===== HCAL and HO Isolation Distributions
    const char* plots2  [] = {"hadEt", "hoEt", "hadEt_cd", "hoEt_cd"};
    Plot4Histograms(newDir + "/muonIso2.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "IsoHistos2", "HCAL, HO Deposits",
		    refLabel, newLabel,
		    plots2, 0,
		    logy, doKolmo,norm);
    
    //===== N_Tracks, N_Jets around #mu
    const char* plots3   [] = {"nTracks", "nJets", "nTracks_cd", "nJets_cd"};
    Plot4Histograms(newDir + "/muonIso3.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "IsoHistos3", "Number of tracks, jets around #mu",
		    refLabel, newLabel,
		    plots3, 0,
		    logy, doKolmo,norm);
    
    
    //===== avg Pt, weighted Et around #mu
    const char* plots4   [] = {"avgPt", "weightedEt", "avgPt_cd", "weightedEt_cd"};

    Plot4Histograms(newDir + "/muonIso4.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "IsoHistos4", "Average p_{T}, weighted E_{T} aroun #mu",
		    refLabel, newLabel,
		    plots4, 0,
		    logy, doKolmo,norm);
    
    
    //===== Tracker and CAL deposits vs muon pT
    const char* plots5   [] = {"muonPt_sumPt", "muonPt_emEt", "muonPt_hadEt", "muonPt_hoEt"};
    Double_t  norm2    [] = {-999.,-999.,-999.,-999.};
    Plot4Histograms(newDir + "/muonIso5.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "IsoHistos5", "Trk, CAL Isolations vs. #mu p_{T}",
		    refLabel, newLabel,
		    plots5, 0,
		    logy, doKolmo,norm2);
    
    //===== NTracks, NJets, avgPt, weightedEt vs Muon pT
    const char* plots6   [] = {"muonPt_nTracks", "muonPt_nJets", "muonPt_avgPt", "muonPt_weightedEt"};
    Plot4Histograms(newDir + "/muonIso6.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "IsoHistos6", "Other stuff vs #mu p_{T}",
		    refLabel, newLabel,
		    plots6, 0,
		    logy, doKolmo,norm2);
    
    
    //// Merge pdf histograms together into larger files, and name them based on the collection names
    TString mergefile = "merged_iso.pdf"; // File name where partial pdfs will be merged
    TString destfile  = newDir + "/../" + myName + ".pdf"; // Destination file name
    TString gscommand = "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile="+ mergefile + " "
      +newDir+"/muonIso1.pdf "
      +newDir+"/muonIso2.pdf "
      +newDir+"/muonIso3.pdf "
      +newDir+"/muonIso4.pdf "
      +newDir+"/muonIso5.pdf "
      +newDir+"/muonIso6.pdf "; 
    cout << ">> Merging partial pdfs to " << mergefile << "..." << endl;
#ifdef DEBUG
    cout << "DEBUG: ...with command \"" << gscommand << "\"" << endl;
#endif
    gSystem->Exec(gscommand);
    cout << ">> Moving " << mergefile << " to " << destfile << "..." << endl;
    gSystem->Rename(mergefile, destfile);

    cout << ">> Deleting partial pdf files" << endl;
    gSystem->Exec("rm -rf "+newDir);    
    cout << "   ... Done" << endl;
  }  // end of "while loop"


  cout << ">> Removing the relval files from ROOT before closing..." << endl;
  gROOT->GetListOfFiles()->Remove(sfile);
  gROOT->GetListOfFiles()->Remove(rfile);

#ifdef DEBUG
  cout << "DEBUG: Exiting!" << endl;
  cerr << "DEBUG: Exiting!" << endl;
#endif
}
