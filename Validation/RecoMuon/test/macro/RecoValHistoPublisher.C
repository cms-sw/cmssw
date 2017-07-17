#include <vector>
#include <algorithm>
#include "TMath.h"
#include "PlotHelpers.C"

// Uncomment the following line for some extra debug information
// #define DEBUG

void RecoValHistoPublisher(const char* newFile="NEW_FILE",const char* refFile="REF_FILE") {
  cout << ">> Starting RecoValHistoPublisher(" << newFile << "," << refFile << ")..." << endl;

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
  TList* sl = getListOfBranches(dataType, sfile, "MuonRecoAnalyzer");
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
  TList* rl = getListOfBranches(dataType, rfile, "MuonRecoAnalyzer");
  if (!rl) {
    cout << "ERROR: Could not find keys!!!" << endl;
    cerr << "ERROR: Could not find keys!!!" << endl;
    return;
  }
  TDirectory* rdir  = gDirectory;
  for (unsigned int i = 0; i < sl->GetEntries(); i++)
    cout << "   + " << sl->At(i)->GetName() << endl;


  // Get the number of events for the normalization:
  TH1F *sevt, *revt;
  sdir->GetObject("RecoMuonV/RecoMuon_MuonAssoc_Glb/NMuon",sevt);
  rdir->GetObject("RecoMuonV/RecoMuon_MuonAssoc_Glb/NMuon",revt);
  /*
  if (sevt && revt) {
    if (revt->GetEntries()>0) 
      snorm = sevt->GetEntries()/revt->GetEntries();
    cout << "   + SEntries = " << sevt->GetEntries()
	 << ", REntries = " << revt->GetEntries() 
	 << ", Normalization = " << snorm << endl;
    cout << "   + SIntegral = " << sevt->Integral()
	 << ", RIntegral = " << revt->Integral() 
	 << ", Normalization = " << snorm << endl;
  }
  else {  
    cout << "WARNING: Missing normalization histos!" << endl; 
  }
  */
    Float_t maxPT;
  TString File = newFile;

  if(File.Contains("SingleMuPt10")) {maxPT = 70.;}
  else if (File.Contains("SingleMuPt100")) {maxPT = 400.;}
  else if (File.Contains("SingleMuPt1000") ||File.Contains("WpM")||File.Contains("ZpMM")   ) maxPT=1400.;  else maxPT = 300.;

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
    
    bool resolx = false;
    bool *resol = &resolx;
    bool    logy    [] = {false,   false,  false,      false    };
    bool    doKolmo [] = {true,    true,   true,       true     };
    Double_t minx   [] = {-1E100, -1E100,    5.,   -1E100,    -1E100, -1E100 };
    Double_t maxx   [] = {-1E100, -1E100,maxPT, -1E100,  -1E100, -1E100 };
    Double_t snorm  [] = {0.,0.,0.,0.,0.,0.};
    //===== reco muon distributions: GLB_GLB
    const char* plots1  [] = {"GlbMuon_Glb_eta", "GlbMuon_Glb_phi", "GlbMuon_Glb_pt", "GlbMuon_Glb_chi2OverDf"};   
    const char* plotst1 [] = {"GlobalMuon(GLB) #eta", "GlobalMuon(GLB) #phi", "GlobalMuon(GLB) pT", "GlobalMuon(GLB) #chi^{2}/ndf"};
    Plot4Histograms(newDir + "/muonReco1.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecoHistos1", "Distributions for GlobalMuons (GLB)",
		    refLabel, newLabel,
		    plots1, plotst1,
		    logy, doKolmo, snorm,resol, minx,maxx);
    
    
    //===== reco muon distributions: GLB_STA
    const char* plots2  [] = {"GlbMuon_Sta_eta", "GlbMuon_Sta_phi", "GlbMuon_Sta_pt", "GlbMuon_Sta_chi2OverDf"};   
    const char* plotst2 [] = {"GlobalMuon(STA) #eta", "GlobalMuon(STA) #phi", "GlobalMuon(STA) p_T", "GlobalMuon(STA) #chi^{2}/ndf"};
    Double_t minx1   [] = {-1E100,-1E100,  5.,    -1E100, -1E100 };
    Double_t maxx1   [] = {-1E100, -1E100,maxPT,  -1E100, -1E100 };

    Plot4Histograms(newDir + "/muonReco2.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecoHistos2", "Distributions for GlobalMuons (STA)",
		    refLabel, newLabel,
		    plots2, plotst2,
		    logy, doKolmo, snorm,resol, minx1,maxx1);
    
    
    //===== reco muon distributions: GLB_TK
    const char* plots3  [] = {"GlbMuon_Tk_eta", "GlbMuon_Tk_phi", "GlbMuon_Tk_pt", "GlbMuon_Tk_chi2OverDf"};   
    const char* plotst3 [] = {"GlobalMuon(TK) #eta", "GlobalMuon(TK) #phi", "GlobalMuon(TK) pT", "GlobalMuon(TK) #chi^{2}/ndf"};
    Plot4Histograms(newDir + "/muonReco3.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecoHistos3", "Distributions for GlobalMuons (TK)",
		    refLabel, newLabel,
		    plots3, plotst3,
		    logy, doKolmo, snorm,resol,minx1,maxx1);
    
    
    //===== reco muon distributions: STA
    const char* plots4  [] = {"StaMuon_eta", "StaMuon_phi", "StaMuon_pt", "StaMuon_chi2OverDf"};   
    const char* plotst4 [] = {"StaMuon #eta", "StaMuon #phi", "StaMuon p_T", "StaMuon #chi^{2}/ndf"};
    Plot4Histograms(newDir + "/muonReco4.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecoHistos4", "Distributions for StandAlone Muons",
		    refLabel, newLabel,
		    plots4, plotst4,
		    logy, doKolmo, snorm,resol,minx1,maxx1);
    
    
    //===== reco muon distributions: Tracker Muons
    const char* plots5  [] = {"TkMuon_eta", "TkMuon_phi", "TkMuon_pt", "TkMuon_chi2OverDf"};   
    const char* plotst5 [] = {"TkMuon #eta", "TkMuon #phi", "TkMuon p_T", "TkMuon #chi^{2}/ndf"};
    Plot4Histograms(newDir + "/muonReco5.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecoHistos5", "Distributions for Tracker Muons",
		    refLabel, newLabel,
		    plots5, plotst5,
		    logy, doKolmo, snorm,resol,minx1,maxx1);
    
    
    //// Merge pdf histograms together into larger files, and name them based on the collection names
    TString mergefile = "merged_reco.pdf"; // File name where partial pdfs will be merged
    TString destfile  = newDir + "/../" + myName + ".pdf"; // Destination file name
    TString gscommand = "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile="+ mergefile + " "
      +newDir+"/muonReco1.pdf "
      +newDir+"/muonReco2.pdf "
      +newDir+"/muonReco3.pdf "
      +newDir+"/muonReco4.pdf "
      +newDir+"/muonReco5.pdf ";

    cout << ">> Merging partial pdfs to " << mergefile << "..." << endl;
#ifdef DEBUG
    cout << "DEBUG: ...with command \"" << gscommand << "\"" << endl;
#endif
    gSystem->Exec(gscommand);
    cout << ">> Moving " << mergefile << " to " << destfile << "..." << endl;
    gSystem->Rename(mergefile, destfile);
    
    cout << ">> Deleting partial pdf files" << endl;
    gSystem->Exec("rm -r "+newDir);
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
