#include <vector>
#include <algorithm>
#include "TMath.h"
#include "PlotHelpers.C"


//Uncomment the following line to get some more output
//#define DEBUG 1

void SeedValHistoPublisher(const char* newFile="NEW_FILE",const char* refFile="REF_FILE") {
  cout << ">> Starting SeedValHistoPublisher(" << newFile << "," << refFile << ")..." << endl;

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
  TList* sl = getListOfBranches(dataType, sfile, "Seeds");
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
  TList* rl = getListOfBranches(dataType, rfile, "Seeds");
  if (!rl) {
    cout << "ERROR: Could not find keys!!!" << endl;
    cerr << "ERROR: Could not find keys!!!" << endl;
    return;
  }
  TDirectory* rdir  = gDirectory;
  for (unsigned int i = 0; i < sl->GetEntries(); i++)
    cout << "   + " << sl->At(i)->GetName() << endl;



 
 

  //==== Get the number of events for the normalization:
  cout << ">> Find out number of events for normalization..." << endl;
  TH1F *sevt, *revt;
  sdir->GetObject("RecoMuonV/RecoMuon_TrackAssoc/Muons/NMuon",sevt);
  rdir->GetObject("RecoMuonV/RecoMuon_TrackAssoc/Muons/NMuon",revt);

  /*  if (sevt && revt) {
    if (revt->GetEntries()>0) 
      norm = sevt->GetEntries()/revt->GetEntries();
  }
  else {  
    cerr << "WARNING: Missing seed normalization histos" << endl; 
    cout << "WARNING: Missing seed normalization histos" << endl; 
  }
  cout << "   + NORM = " << norm << endl;
  */

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

 
    bool    logy    [] = {false,   true,   false,      true    };
    bool    doKolmo [] = {true,    true,   true,       true    };
    Double_t norm   [] = {0.,0.,0.,0.,0.,0.};    
    /*
    const char* plots [] = {"", "", "", ""};
    const char* plotsl[] = {"", "", "", ""};
    Plot4Histograms(newDir + "/muonIso1.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "", "",
		    refLabel, newLabel,
		    plots, plotsl,
		    logy, doKolmo);
    */

    //===== muon seeds plots, first page:
    const char* plots1 [] = {"seedEta_", "seedEtaErr_", "seedPhi_", "seedPhiErr_"};
    const char* plotsl1[] = {"seed #eta", "seed #eta error", "seed #phi", "seed #phi error"};
    Plot4Histograms(newDir + "/muonSeed1.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "Seeds1", "Seeds eta and phi",
		    refLabel, newLabel,
		    plots1, plotsl1,
		    logy, doKolmo, norm);
 

    // ====== muon seeds plots, second page:
    // NOTE: Originally in one page, now split in two pages
    // const char* plots2 [] = {"seedPt_", "seedPtErrOverPt_", "seedPz_", "seedPzErrOverPz_"};
    // const char* plotsl2[] = {"seed P_{T}", "seed P_{T} Err/P_{T}", "seed P_{Z}", "seed P_{Z} Err/P_{Z}"};
    // Plot4Histograms(newDir + "/muonSeed2.pdf",
    // 		    rdir, sdir, 
    // 		    rcollname, scollname,
    // 		    "Seeds2", "Seeds momenta",
    // 		    refLabel, newLabel,
    // 		    plots2, plotsl2,
    // 		    logy, doKolmo, norm);

    // const char* plots3 [] = {"NumberOfRecHitsPerSeed_", "seedPErrOverP_", "", ""};
    // const char* plotsl3[] = {"Nr RecHits per seed", "seed P Err/P", "", ""};
    // Plot4Histograms(newDir + "/muonSeed3.pdf",
    // 		    rdir, sdir, 
    // 		    rcollname, scollname,
    // 		    "Seeds3", "Seeds hits and momentum",
    // 		    refLabel, newLabel,
    // 		    plots3, plotsl3,
    // 		    logy, doKolmo);
    
    bool    logy2   [] = {false, true, false, true, false, true};
    bool    doKolmo2[] = {true,  true, true,  true, true,  true};
    const char* plots2  [] = {"seedPt_", "seedPtErrOverPt_", "seedPz_", "seedPzErrOverPz_",
			      "NumberOfRecHitsPerSeed_", "seedPErrOverP_"};
    const char* plotsl2 [] = {"seed P_{T}", "seed P_{T} Err/P_{T}", "seed P_{Z}", "seed P_{Z} Err/P_{Z}",
			      "Nr RecHits per seed", "seed P Err/P"};
    Plot6Histograms(newDir + "/muonSeed2.pdf",
		    rdir, sdir, 
		    rcollname, scollname,
		    "Seeds2", "Seeds momenta and hits",
		    refLabel, newLabel,
		    plots2, plotsl2,
		    logy2, doKolmo2, norm);
   
 
    //// Merge pdf histograms together into larger files, and name them based on the collection names
    TString mergefile = "merged_seed.pdf"; // File name where partial pdfs will be merged
    TString destfile  = newDir + "/../" + myName + ".pdf"; // Destination file name
    TString gscommand = "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=" + mergefile + " "
      + newDir + "/muonSeed1.pdf "
      + newDir + "/muonSeed2.pdf ";
      //      + newDir + "/muonSeed3.pdf ";
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
