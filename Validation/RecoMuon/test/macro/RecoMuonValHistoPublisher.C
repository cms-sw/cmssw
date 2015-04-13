#include <vector>
#include <algorithm>
#include "TMath.h"
#include "PlotHelpers.C"

/////
// Uncomment the following line to get more debuggin output
// #define DEBUG

void RecoMuonValHistoPublisher(const char* newFile="NEW_FILE",const char* refFile="REF_FILE") {
  cout << ">> Starting RecoMuonValHistoPublisher(" << newFile << "," << refFile << ")..." << endl;

  //====  To be replaced from python ====================
  
  const char* dataType = "DATATYPE";
  const char* refLabel("REF_LABEL, REF_RELEASE REFSELECTION");
  const char* newLabel("NEW_LABEL, NEW_RELEASE NEWSELECTION");
  const char* fastSim = "IS_FSIM";


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
  TList* sl = getListOfBranches(dataType, sfile, "RecoMuonV");
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
  TList* rl = getListOfBranches(dataType, rfile, "RecoMuonV");
  if (!rl) {
    cout << "ERROR: Could not find keys!!!" << endl;
    cerr << "ERROR: Could not find keys!!!" << endl;
    return;
  }
  TDirectory* rdir  = gDirectory;
  for (unsigned int i = 0; i < sl->GetEntries(); i++)
    cout << "   + " << sl->At(i)->GetName() << endl;

  Float_t maxPT;
  TString File = newFile;
  if (File.Contains("SingleMuPt1000") ||File.Contains("WpM")||File.Contains("ZpMM")   ) maxPT=1400.;
  else if(File.Contains("SingleMuPt10")) {maxPT = 70.;}
  else if (File.Contains("SingleMuPt100")) {maxPT = 400.;}
  else maxPT = 400.;

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
    Double_t minx   [] = {-1E100, -1E100,    -1E100,   5.,    -1E100, -1E100 };
    Double_t maxx   [] = {-1E100, -1E100,-1E100, maxPT,  -1E100, -1E100 };
 
    Double_t norm   [] = {0.,0.,-999.,-999.,0.,0.}; //Normalize to first histogram

 
   //===== reco muon distributions: GLB
    TString baseh     = Form("RecoMuon_MuonAssoc_Glb%s/",fastSim);
    const char* plots1 [] = {(baseh + "ErrPt").Data(), (baseh + "ErrP").Data(), 
			     (baseh + "ErrPt_vs_Eta_Sigma").Data(), (baseh + "ErrPt_vs_Pt_Sigma").Data()};   
    const char* plotst1[] = {"GlobalMuon(GLB) #Delta p_{T}/p_{T}", "GlobalMuon(GLB) #Delta p/p", 
			     "GlobalMuon(GLB) #Delta p_{T}/p_{T} vs #sigma(#eta)", "GlobalMuon(GLB) #Delta p_{T}/p_{T} vs #sigma(p_{T})"};
    Plot4Histograms(newDir + "/muonRecoGlb.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecHistosGlb", "Distributions for GlobalMuons (GLB)",
		    refLabel, newLabel,
		    plots1, plotst1,
		    logy, doKolmo, norm,resol,minx,maxx);
    
    
    //==== efficiencies and fractions GLB
    const char* plots2 [] = {(baseh + "EffP").Data(), (baseh + "EffEta").Data(), 
			     (baseh + "FractP").Data(), (baseh + "FractEta").Data()};   
    const char* plotst2[] = {"GlobalMuon(GLB) #epsilon vs. p", "GlobalMuon(GLB) #epsilon vs. #eta", 
			     "GlobalMuon(GLB) fraction vs. p", "GlobalMuon(GLB) fraction vs. #eta"};
    Double_t minx1   [] = {5., -1E100,    5.,   -1E100,    -1E100, -1E100 };
    Double_t maxx1   [] = {maxPT, -1E100,maxPT, -1E100,  -1E100, -1E100 };
    Double_t norm2   [] = {-999.,-999.,-999.,-999.,-999.,-999.}; //Normalize to first histogram 
    Plot4Histograms(newDir + "/muonRecoGlbEff.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecEffHistosGlb", "Distributions for GlobalMuons (GLB), efficiencies and fractions",
		    refLabel, newLabel,
		    plots2, plotst2,
		    logy, doKolmo,norm2,resol,minx1,maxx1);
    
    /*
    //===== reco muon distributions: GLBPF
    baseh             = Form("RecoMuon_MuonAssoc_GlbPF%s/",fastSim);
    const char* plots3[]  = {(baseh + "ErrPt").Data(), (baseh + "ErrP").Data(), 
			     (baseh + "ErrPt_vs_Eta_Sigma").Data(), (baseh + "ErrPt_vs_Pt_Sigma").Data()};   
    const char* plotst3[] = {"PFGlobalMuon(GLBPF) #Delta p_{T}/p_{T}", "PFGlobalMuon(GLBPF) #Delta p/p", 
			     "PFGlobalMuon(GLBPF) #Delta p_{T}/p_{T} vs #sigma(#eta)", "PFGlobalMuon(GLBPF) #Delta p_{T}/p_{T} vs #sigma(p_{T})"};
    Plot4Histograms(newDir + "/muonRecoGlbPF.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecHistosGlbPF", "Distributions for PFGlobalMuons (GLBPF)",
		    refLabel, newLabel,
		    plots3, plotst3,
		    logy, doKolmo, norm);
    
    
    //==== efficiencies and fractions GLBPF
    const char* plots4 [] = {(baseh + "EffP").Data(), (baseh + "EffEta").Data(), 
			     (baseh + "FractP").Data(), (baseh + "FractEta").Data()};   
    const char* plotst4[] = {"PFGlobalMuon(GLBPF) #epsilon vs. p", "PFGlobalMuon(GLBPF) #epsilon vs. #eta", 
			     "PFGlobalMuon(GLBPF) fraction vs. p", "PFGlobalMuon(GLBPF) fraction vs. #eta"};
    Plot4Histograms(newDir + "/muonRecoGlbPFEff.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecEffHistosGlbPF", "Distributions for PFGlobalMuons (GLBPF), efficiencies and fractions",
		    refLabel, newLabel,
		    plots4, plotst4,
		    logy, doKolmo, norm);
    */
    
    //===== reco muon distributions: STA
    baseh             = Form("RecoMuon_MuonAssoc_Sta%s/",fastSim);
    const char* plots5 [] = {(baseh + "ErrPt").Data(), (baseh + "ErrP").Data(), 
			     (baseh + "ErrPt_vs_Eta_Sigma").Data(), (baseh + "ErrPt_vs_Pt_Sigma").Data()};   
    const char* plotst5[] = {"StandAloneMuon(STA) #Delta p_{T}/p_{T}", "StandAloneMuon(STA) #Delta p/p", 
			     "StandAloneMuon(STA) #Delta p_{T}/p_{T} vs #sigma(#eta)", "StandAloneMuon(STA) #Delta p_{T}/p_{T} vs #sigma(p_{T})"};
    Plot4Histograms(newDir + "/muonRecoSta.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecHistosSta", "Distributions for StandAloneMuons (STA)",
		    refLabel, newLabel,
		    plots5, plotst5,
		    logy, doKolmo, norm,resol, minx,maxx);
    
    
    
    //==== efficiencies and fractions STA
    const char* plots6 [] = {(baseh + "EffP").Data(), (baseh + "EffEta").Data(), 
			     (baseh + "FractP").Data(), (baseh + "FractEta").Data()};   
    const char* plotst6[] = {"StandAloneMuon(STA) #epsilon vs. p", "StandAloneMuon(STA) #epsilon vs. #eta", 
			     "StandAloneMuon(STA) fraction vs. p", "StandAloneMuon(STA) fraction vs. #eta"};
    Plot4Histograms(newDir + "/muonRecoStaEff.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecEffHistosSta", "Distributions for StandAloneMuons (STA), efficiencies and fractions",
		    refLabel, newLabel,
		    plots6, plotst6,
		    logy, doKolmo, norm2,resol,minx1,maxx1);



   //===== reco muon distributions: TRK
    baseh             = Form("RecoMuon_MuonAssoc_Trk%s/",fastSim);
    const char* plots7 [] = {(baseh + "ErrPt").Data(), (baseh + "ErrP").Data(), 
			     (baseh + "ErrPt_vs_Eta_Sigma").Data(), (baseh + "ErrPt_vs_Pt_Sigma").Data()};   
    const char* plotst7[] = {"TrackerMuon(TRK) #Delta p_{T}/p_{T}", "TrackerMuon(TRK) #Delta p/p", 
			  "TrackerMuon(TRK) #Delta p_{T}/p_{T} vs #sigma(#eta)", "TrackerMuon(TRK) #Delta p_{T}/p_{T} vs #sigma(p_{T})"};
    Plot4Histograms(newDir + "/muonRecoTrk.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecHistosTrk", "Distributions for TrackerMuons (TRK)",
		    refLabel, newLabel,
		    plots7, plotst7,
		    logy, doKolmo, norm,resol,minx,maxx);



   //==== efficiencies and fractions TRK
    const char* plots8 [] = {(baseh + "EffP").Data(), (baseh + "EffEta").Data(), 
			     (baseh + "FractP").Data(), (baseh + "FractEta").Data()};   
    const char* plotst8[] = {"TrackerMuon(TRK) #epsilon vs. p", "TrackerMuon(TRK) #epsilon vs. #eta", 
                         "TrackerMuon(TRK) fraction vs. p", "TrackerMuon(TRK) fraction vs. #eta"};
    Plot4Histograms(newDir + "/muonRecoTrkEff.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecEffHistosTrk", "Distributions for TrackerMuons (TRK), efficiencies and fractions",
		    refLabel, newLabel,
		    plots8, plotst8,
		    logy, doKolmo, norm2,resol,minx1,maxx1);

    
    //
    //===== reco muon distributions: Tight Muons
    //
    baseh             = Form("RecoMuon_MuonAssoc_Tgt%s/",fastSim);
    const char* plots9 [] = {(baseh + "ErrPt").Data(), (baseh + "ErrP").Data(), 
			     (baseh + "ErrPt_vs_Eta_Sigma").Data(), (baseh + "ErrPt_vs_Pt_Sigma").Data()};   
    const char* plotst9[] = {"Tight Muon #Delta p_{T}/p_{T}", "Tight Muon #Delta p/p", 
			     "Tight Muon #Delta p_{T}/p_{T} vs #sigma(#eta)", "Tight Muon #Delta p_{T}/p_{T} vs #sigma(p_{T})"};
    Plot4Histograms(newDir + "/muonRecoTgt.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecHistosTgt", "Distributions for Tight Muons",
		    refLabel, newLabel,
		    plots9, plotst9,
		    logy, doKolmo, norm,resol,minx,maxx);



   //==== efficiencies and fractions Tight Muons
    const char* plots10 [] = {(baseh + "EffP").Data(), (baseh + "EffEta").Data(), 
			      (baseh + "FractP").Data(), (baseh + "FractEta").Data()};   
    const char* plotst10[] = {"Tight Muon #epsilon vs. p", "Tight Muon #epsilon vs. #eta", 
			      "Tight Muon fraction vs. p", "Tight Muon fraction vs. #eta"};
    Plot4Histograms(newDir + "/muonRecoTgtEff.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "RecEffHistosTgt", "Distributions for Tight Muons, efficiencies and fractions",
		    refLabel, newLabel,
		    plots10, plotst10,
		    logy, doKolmo, norm2,resol,minx1,maxx1);
    
    
    
    //
    // Merge pdf histograms together into larger files, and name them based on the collection names
    //
    TString mergefile = "merged_recomuonval.pdf"; // File name where partial pdfs will be merged
    TString destfile  = newDir + "/../" + myName + ".pdf"; // Destination file name
    TString gscommand = "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile="+ mergefile + " "
      +newDir+"/muonRecoGlb.pdf "
      +newDir+"/muonRecoGlbEff.pdf "
      //      +newDir+"/muonRecoGlbPF.pdf "
      //      +newDir+"/muonRecoGlbPFEff.pdf "
      +newDir+"/muonRecoSta.pdf "
      +newDir+"/muonRecoStaEff.pdf "
      +newDir+"/muonRecoTrk.pdf "
      +newDir+"/muonRecoTrkEff.pdf "
      +newDir+"/muonRecoTgt.pdf "
      +newDir+"/muonRecoTgtEff.pdf ";

    cout << ">> Merging partial pdfs to " << mergefile << "..." << endl;
#ifdef DEBUG
    cout << "DEBUG: ...with command \"" << gscommand << "\"" << endl;
#endif
    gSystem->Exec(gscommand);
    cout << ">> Moving " << mergefile << " to " << destfile << "..." << endl;
    gSystem->Rename(mergefile, destfile);
    cout << "   ... Done" << endl;
   
    cout << ">> Deleting partial pdf files" << endl;
    gSystem->Exec("rm -r "+newDir);
    
  }  // end of "while loop"
  
  cout << ">> Removing the relval files from ROOT before closing..." << endl;
  gROOT->GetListOfFiles()->Remove(sfile);
  gROOT->GetListOfFiles()->Remove(rfile);
  
#ifdef DEBUG
  cout << "DEBUG: Exiting!" << endl;
  cerr << "DEBUG: Exiting!" << endl;
#endif
}
