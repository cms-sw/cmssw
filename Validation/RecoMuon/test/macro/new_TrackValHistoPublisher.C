#include <vector>
#include <algorithm>
#include "TMath.h"
#include "macro/new_PlotHelpers.C"

// debugging printouts
bool DEBUG = false;

TList* GetListOfBranches(const char* dataType, TFile* file) {
  if (TString(dataType) == "HLT") {
    file->cd("DQMData/Run 1/HLT/Run summary/Muon/MuonTrack");
  }
  else if (TString(dataType) == "RECO") {
    file->cd("DQMData/Run 1/Muons/Run summary/RecoMuonV/MuonTrack");
  }
  else {
    cout << "ERROR: Data type " << dataType << " not allowed: only RECO and HLT are considered" << endl;
    cerr << "ERROR: Data type " << dataType << " not allowed: only RECO and HLT are considered" << endl;
    return 0;
  }

  TDirectory * dir=gDirectory;
  TList* sl = GetListOfDirectories(dir);
  
  if (sl->GetSize() == 0) {
    cout << "ERROR: No DQM muon reco histos found in NEW file " << endl;
    cerr << "ERROR: No DQM muon reco histos found in NEW file " << endl;
    delete sl;
    return 0;
  }
  
  return sl;
}

void plotOptReset(bool logx[6], bool logy[6], bool doKolmo[6], Double_t norm[6], 
		  Double_t minx[6], Double_t maxx[6], Double_t miny[6], Double_t maxy[6], const char* drawopt[6], 
		  TString plots[6], TString titles[6]) {
  
  for(int i=0; i<6; ++i) {
    logx[i] = false;
    logy[i] = false;
    doKolmo[i] = true;
    norm[i] = -1;
    minx[i] = 0;
    maxx[i] = 0;
    miny[i] = 0;
    maxy[i] = 0;
    drawopt[i] = "";
    plots[i]   = "";
    titles[i]  = "";
  }
}

void new_TrackValHistoPublisher(const char* newFile="NEW_FILE",const char* refFile="REF_FILE") {

  cout << ">> Starting new_TrackValHistoPublisher(" 
       << newFile << "," << refFile << ")..." << endl;

  //====  To be replaced from python ====================
  
  const char* dataType = "DATATYPE";
  const char* refLabel("REF_LABEL, REF_RELEASE REFSELECTION");
  const char* newLabel("NEW_LABEL, NEW_RELEASE NEWSELECTION");


  // ==== Initial settings and loads
  gROOT ->SetBatch();
  gErrorIgnoreLevel = kWarning; // Get rid of the info messages
  SetGlobalStyle();

  // ==== Some cleaning... is this needed?  
  delete gROOT->GetListOfFiles()->FindObject(refFile);
  delete gROOT->GetListOfFiles()->FindObject(newFile); 
  
  // ==== Opening files, moving to the right branch and getting the list of sub-branches
  cout << ">> Opening files, moving to the right branch and getting the list of sub-branches..." << endl;

  cout << ">> Finding new DQM file ..." << endl;
  TFile * sfile = new TFile(newFile);
  TList* sl = GetListOfBranches(dataType, sfile);

  if (!sl) {
    cout << "ERROR: Could not find keys!!!" << endl;
    cerr << "ERROR: Could not find keys!!!" << endl;
    return;
  }
  TDirectory*  sdir  = gDirectory;

  if (DEBUG) {
    for (unsigned int i = 0; i < sl->GetEntries(); i++)
      cout << "   + " << sl->At(i)->GetName() << endl;
  }
  
  cout << ">> Finding reference DQM file ..." << endl;
  TFile * rfile = new TFile(refFile);
  TList* rl = GetListOfBranches(dataType, rfile);

  if (!rl) {
    cout << "ERROR: Could not find keys!!!" << endl;
    cerr << "ERROR: Could not find keys!!!" << endl;
    return;
  }
  TDirectory* rdir  = gDirectory;

  if (DEBUG) {
    for (unsigned int i = 0; i < rl->GetEntries(); i++)
      cout << "   + " << rl->At(i)->GetName() << endl;
  }

  //==== Iterate now over histograms and collections
  cout << ">> Iterating over histograms and collections..." << endl;

  bool logy[6]     = {false,  false,  false,  false,   false,  false  };
  bool logx[6]     = {false,  false,  false,  false,   false,  false  };
  bool doKolmo[6]  = {true,   true,   true,   true,    true,   true };
  Double_t norm[6] =  {-1.,-1.,-1.,-1.,-1.,-1.};  // initial default: do not normalize
  Double_t minx[6] = {0, 0, 0, 0, 0, 0};
  Double_t maxx[6] = {0, 0, 0, 0, 0, 0};
  Double_t miny[6] = {0, 0, 0, 0, 0, 0};
  Double_t maxy[6] = {0, 0, 0, 0, 0, 0};
  const char* drawopt[6] = {"", "", "", "", "", ""};
  TString plots[6]       = {"", "", "", "", "", ""};
  TString titles[6]      = {"", "", "", "", "", ""};

  TString rcollname;
  TString scollname;
  TIter iter_r( rl );
  TIter iter_s( sl );
  TString newDirBase("NEW_RELEASE/NEWSELECTION/NEW_LABEL/");
  TKey* rKey = 0;

  // before CMSSW_10_1_0_pre1 a few collection names were different
  bool NEWcollNames = false;
  TString Ref_CMSSW_Release("REF_RELEASE");
  if (Ref_CMSSW_Release.Contains("CMSSW_9") || Ref_CMSSW_Release.Contains("CMSSW_10_0")) NEWcollNames=true;

  while ( (rKey = (TKey*)iter_r()) ) {
    TString myName = rKey->GetName();
    rcollname = myName;
    if (DEBUG) {
      cout << " Checking collection: " << myName << endl;
      cerr << " Checking collection: " << myName << endl;
    }

    TString myName2 = myName;
    if (NEWcollNames) {
      if (myName=="NEWprobeTrks") myName2="probeTrks";
      else if (myName=="NEWprobeTrks_TkAsso") myName2="probeTrks_TkAsso";
      else if (myName=="NEWseedsOfSTAmuons") myName2="seedsOfSTAmuons";
      else if (myName=="NEWseedsOfDisplacedSTAmuons") myName2="seedsOfDisplacedSTAmuons";
      else if (myName=="NEWcutsRecoTrkMuons") myName2="cutsRecoTrkMuons";
      else if (myName=="NEWextractGemMuons") myName2="extractGemMuons";
      else if (myName=="NEWextractMe0Muons") myName2="extractMe0Muons";
    }
    scollname = myName2;
    
    if (DEBUG) {
      cout << " Comparing " << rcollname << " and " << scollname << endl;
      cerr << " Comparing " << rcollname << " and " << scollname << endl;
    }

    // ==== Now let's go for the plotting...
    TString newDir = newDirBase+myName2;
    cout<<"creating directory: "<<newDir<<endl;
    gSystem->mkdir(newDir,kTRUE);
   
    // efficiency and fake rate Vs eta and phi
    plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
    plots[0]="effic_vs_eta"    ; titles[0]="Efficiency vs #eta";
    plots[1]="fakerate_vs_eta" ; titles[1]="Fake rate vs #eta" ;
    plots[2]="effic_vs_phi"    ; titles[2]="Efficiency vs #phi" ;
    plots[3]="fakerate_vs_phi" ; titles[3]="Fake rate vs #phi" ;
    
    miny[0]=-0.0001;
    miny[1]=-0.0001;
    miny[2]=-0.0001;
    miny[3]=-0.0001;
    
    maxy[0]=1.09;
    maxy[1]=1.09;
    maxy[2]=1.09;
    maxy[3]=1.09;
    
    Plot4Histograms(newDir + "/eff_eta_phi",
		    rdir, sdir, 
		    rcollname, scollname,
		    "eff_eta_phi", "Efficiency vs eta and Vs phi",
		    refLabel, newLabel,
		    plots, titles, drawopt,
		    logy, logx, doKolmo, norm, minx, maxx, miny, maxy);     
    

    // efficiency and fake rate Vs pt
    plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
    plots[0]="effic_vs_pt"    ; titles[0]="Efficiency vs pt";
    plots[1]="fakerate_vs_pt" ; titles[1]="Fake rate vs pt" ;
    plots[2]="num_simul_pT"   ; titles[2]="N of simulated tracks vs pt" ;
    plots[3]="num_reco_pT"    ; titles[3]="N of reco track vs pt" ;

    logx[0]=true;
    logx[1]=true;
    logx[2]=true;
    logx[3]=true;

    drawopt[0]="";
    drawopt[1]="";
    drawopt[2]="hist";
    drawopt[3]="hist";

    norm[0]= -1.;
    norm[1]= -1.;
    norm[2]= 2.;
    norm[3]= 2.;

    miny[0]= -0.0001;
    miny[1]= -0.0001;
    miny[2]= 0.;
    miny[3]= 0.;

    maxy[0]= 1.09;
    maxy[1]= 1.09;
    maxy[2]= 0.;
    maxy[3]= 0.;

    Plot4Histograms(newDir + "/eff_pt",
		    rdir, sdir, 
		    rcollname, scollname,
		    "eff_pt", "Efficiency vs pt and sim,reco distributions",
		    refLabel, newLabel,
		    plots, titles, drawopt,
		    logy, logx, doKolmo, norm, minx, maxx, miny, maxy);     
 

    // efficiency and fake rate vs Number of Hits; Hit multiplicity per track; Ave.N.hits vs eta
    plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
    plots[0]="effic_vs_hit"      ; titles[0]="Efficiency vs Number of hits";
    plots[1]="fakerate_vs_hit"   ; titles[1]="Fake rate vs Number of hits" ;
    plots[2]="nhits"             ; titles[2]="number of hits per track" ;
    plots[3]="nhits_vs_eta_prof" ; titles[3]="mean number of Hits vs eta" ;

    drawopt[0]="";
    drawopt[1]="";
    drawopt[2]="hist";
    drawopt[3]="";

    norm[0]= -1.;
    norm[1]= -1.;
    norm[2]=  0.;
    norm[3]= -1.;

    miny[0]= -0.0001;
    miny[1]= -0.0001;
    miny[2]= 0.;
    miny[3]= 0.;

    maxy[0]= 1.09;
    maxy[1]= 0.;
    maxy[2]= 0.;
    maxy[3]= 0.;

    Plot4Histograms(newDir + "/eff_hits",
		    rdir, sdir, 
		    rcollname, scollname,
		    "eff_hits", "Efficiency vs Number of hits and hit multiplicity per track",
		    refLabel, newLabel,
		    plots, titles, drawopt,
		    logy, logx, doKolmo, norm, minx, maxx, miny, maxy);
    
    // efficiency and fake rate vs PU
    plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
    plots[0]="effic_vs_pu"      ; titles[0]="Efficiency vs n.PU interactions";
    plots[1]="fakerate_vs_pu"   ; titles[1]="Fake rate vs n.PU interactions" ;

    //maxx[0]= 100.;
    //maxx[1]= 100.;

    miny[0]= -0.0001;
    miny[1]=  0.;

    maxy[0]= 1.09;
    maxy[1]= 0.;

    norm[1] = -1;

    Plot4Histograms(newDir + "/eff_pu",
		    rdir, sdir, 
		    rcollname, scollname,
		    "eff_pu", "Efficiency vs n.PU interactions",
		    refLabel, newLabel,
		    plots, titles, drawopt,
		    logy, logx, doKolmo, norm, minx, maxx, miny, maxy);     
    
    // skip other plots for seeds
    if (!scollname.Contains("seeds")) {

    //===== normalized chi2, chi2 probability, ave. norm. chi2 vs eta; ave. pt bias vs eta
    plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
    plots[0]="chi2"              ; titles[0]="Track #chi^{2}";
    plots[1]="chi2prob"          ; titles[1]="Probability of track #chi^{2}";
    plots[2]="chi2_vs_eta_prof"  ; titles[2]="Mean normalized #chi^{2} vs #eta" ;

    drawopt[0]="hist";
    drawopt[1]="hist";
    drawopt[2]="";

    norm[0]= 0.;
    norm[1]= 0.;
    norm[2]= -1.;

    logy[0]=true;
    logy[1]=false;
    logy[2]=false;
    
    Plot4Histograms(newDir + "/chi2",
		    rdir, sdir, 
		    rcollname, scollname,
		    "chi2", "chi2 distributions",
		    refLabel, newLabel,
		    plots, titles, drawopt,
		    logy, logx, doKolmo, norm, minx, maxx, miny, maxy);
    
    //===== pull distributions
    plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
    plots[0]="ptpull"     ; titles[0]="p_{T} Pull";
    plots[1]="qoverppull" ; titles[1]="q/p Pull" ;
    plots[2]="phipull"    ; titles[2]="#phi Pull" ;
    plots[3]="thetapull"  ; titles[3]="#theta Pull" ;
    plots[4]="dxypull"    ; titles[4]="dxy Pull" ;
    plots[5]="dzpull"     ; titles[5]="dz Pull" ;

    logy[0]=true;
    logy[1]=true;
    logy[2]=true;
    logy[3]=true;
    logy[4]=true;
    logy[5]=true;

    drawopt[0]="hist";
    drawopt[1]="hist";
    drawopt[2]="hist";
    drawopt[3]="hist";
    drawopt[4]="hist";
    drawopt[5]="hist";

    norm[0]= 2.;
    norm[1]= 2.;
    norm[2]= 2.;
    norm[3]= 2.;
    norm[4]= 2.;
    norm[5]= 2.;

    Plot6Histograms(newDir + "/pulls",
		    rdir, sdir, 
		    rcollname, scollname,
		    "pulls", "pull distributions",
		    refLabel, newLabel,
		    plots, titles, drawopt,
		    logy, logx, doKolmo, norm, minx, maxx, miny, maxy);     
    
    
    //===== residual distributions (projected on Y-axis from the 2D histos with residuals vs eta)
    plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
    plots[0]="ptres_vs_eta"       ; titles[0]="p_{T} Relative Residual";
    plots[1]="etares_vs_eta"      ; titles[1]="#eta Residual" ;
    plots[2]="phires_vs_eta"      ; titles[2]="#phi Residual" ;
    plots[3]="thetaCotres_vs_eta" ; titles[3]="cot(#theta) Residual" ;
    plots[4]="dxyres_vs_eta"      ; titles[4]="dxy Residual" ;
    plots[5]="dzres_vs_eta"       ; titles[5]="dz Residual" ;

    logy[0]=true;
    logy[1]=true;
    logy[2]=true;
    logy[3]=true;
    logy[4]=true;
    logy[5]=true;

    drawopt[0]="hist";
    drawopt[1]="hist";
    drawopt[2]="hist";
    drawopt[3]="hist";
    drawopt[4]="hist";
    drawopt[5]="hist";

    norm[0]= 2.;
    norm[1]= 2.;
    norm[2]= 2.;
    norm[3]= 2.;
    norm[4]= 2.;
    norm[5]= 2.;

    Plot6Histograms(newDir + "/residuals",
		    rdir, sdir, 
		    rcollname, scollname,
		    "residuals", "residual distributions",
		    refLabel, newLabel,
		    plots, titles, drawopt,
		    logy, logx, doKolmo, norm, minx, maxx, miny, maxy);      
    
      
    //===== resolutions vs eta; pt relative bias vs eta
    plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
    plots[0]="phires_vs_eta_Sigma"      ; titles[0]="width #phi Residual vs #eta";
    plots[1]="thetaCotres_vs_eta_Sigma" ; titles[1]="width cot(#theta) Residual vs #eta" ;
    plots[2]="dxyres_vs_eta_Sigma"      ; titles[2]="width dxy Residual vs #eta" ;
    plots[3]="dzres_vs_eta_Sigma"       ; titles[3]="width dz Residual vs #eta" ;
    plots[4]="ptres_vs_eta_Sigma"       ; titles[4]="width p_{T} Relative Residual vs #eta" ;
    plots[5]="ptres_vs_eta_Mean"        ; titles[5]="mean p_{T} Relative Residual vs #eta" ;

    logy[0]=true;
    logy[1]=true;
    logy[2]=true;
    logy[3]=true;
    logy[4]=true;
    logy[5]=false;
    
    Plot6Histograms(newDir + "/resol_eta",
		    rdir, sdir, 
		    rcollname, scollname,
		    "resol_eta", "resolutions vs eta",
		    refLabel, newLabel,
		    plots, titles, drawopt,
		    logy, logx, doKolmo, norm, minx, maxx, miny, maxy);     

    //===== resolutions vs pt; pt relative bias vs eta
    plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
    plots[0]="phires_vs_pt_Sigma"      ; titles[0]="width #phi Residual vs p_{T}";
    plots[1]="thetaCotres_vs_pt_Sigma" ; titles[1]="width cot(#theta) Residual vs p_{T}" ;
    plots[2]="dxyres_vs_pt_Sigma"      ; titles[2]="width dxy Residual vs p_{T}" ;
    plots[3]="dzres_vs_pt_Sigma"       ; titles[3]="width dz Residual vs p_{T}" ;
    plots[4]="ptres_vs_pt_Sigma"       ; titles[4]="width p_{T} Relative Residual vs p_{T}" ;
    plots[5]="ptres_vs_pt_Mean"        ; titles[5]="mean p_{T} Relative Residual vs p_{T}" ;

    logx[0]=true;
    logx[1]=true;
    logx[2]=true;
    logx[3]=true;
    logx[4]=true;
    logx[5]=true;

    logy[0]=true;
    logy[1]=true;
    logy[2]=true;
    logy[3]=true;
    logy[4]=true;
    logy[5]=false;
    
    Plot6Histograms(newDir + "/resol_pt",
		    rdir, sdir, 
		    rcollname, scollname,
		    "resol_pt", "resolutions vs pt",
		    refLabel, newLabel,
		    plots, titles, drawopt,
		    logy, logx, doKolmo, norm, minx, maxx, miny, maxy);     
    

    // ================= charge misid rate vs eta, pt, n.hits, PU
    plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
    plots[0]="chargeMisId_vs_eta" ; titles[0]="Charge MisId rate vs #eta";
    plots[1]="chargeMisId_vs_pt"  ; titles[1]="Charge MisID rate vs p_{T}" ;
    plots[2]="chargeMisId_vs_hit" ; titles[2]="Charge MisID rate vs number of RecHits" ;
    plots[3]="chargeMisId_vs_pu"  ; titles[3]="Charge MisID rate vs n.PU interactions" ;

    logx[0]=false;
    logx[1]=true;
    logx[2]=false;
    logx[3]=false;

    //maxx[0]= 0.;
    //maxx[1]= 0.;
    //maxx[2]= 0.;
    //maxx[3]= 100.;

    miny[0]= -0.0001;
    miny[1]=  0.;
    miny[2]= -0.0001;
    miny[3]=  0.;

    maxy[0]= 0.;
    maxy[1]= 0.;
    maxy[2]= 0.;
    maxy[3]= 0.;

    Plot4Histograms(newDir + "/chargeMisId",
		    rdir, sdir, 
		    rcollname, scollname,
		    "chargeMisId", "charge misId rate vs eta, pt, nhits, PU",
		    refLabel, newLabel,
		    plots, titles, drawopt,
		    logy, logx, doKolmo, norm, minx, maxx, miny, maxy);     

    } // if (!scollname.Contains("seeds"))

    //// Merge pdf files together and rename the merged pdf after the collection name
    TString mergefile = "merged_plots.pdf"; // File name where partial pdfs will be merged
    TString destfile  = newDir + "/../" + myName + ".pdf"; // Destination file name
    TString gscommand = "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile="  + mergefile + " "
      + newDir + "/eff_eta_phi.pdf "
      + newDir + "/eff_pt.pdf "
      + newDir + "/eff_hits.pdf "
      + newDir + "/eff_pu.pdf "
      + newDir + "/chi2.pdf "
      + newDir + "/pulls.pdf "
      + newDir + "/residuals.pdf "
      + newDir + "/resol_eta.pdf "
      + newDir + "/resol_pt.pdf "
      + newDir + "/chargeMisId.pdf ";

    if (scollname.Contains("seeds"))
      gscommand = "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile="  + mergefile + " "
	+ newDir + "/eff_eta_phi.pdf "
	+ newDir + "/eff_pt.pdf "
	+ newDir + "/eff_hits.pdf "
	+ newDir + "/eff_pu.pdf ";
  
    cout << ">> Merging partial pdfs to " << mergefile << "..." << endl;
    if (DEBUG) cout << " ...with command \"" << gscommand << "\"" << endl;

    gSystem->Exec(gscommand);
    cout << ">> Moving " << mergefile << " to " << destfile << "..." << endl;
    gSystem->Rename(mergefile, destfile);

    cout << ">> Deleting partial pdf files" << endl;
    gSystem->Exec("rm -rf "+newDir+"/*.pdf"); 
    cout << "   ... Done" << endl;
    
    }  // end of "while loop"

  ///////////////////////////////////////////////////////////////////////////////
  // comparison plots of Muon and Track associators on the probeTracks

  TString dir_MABH_vs_TABH = newDirBase + "probeTrks_MABH_vs_TABH";
  gSystem->mkdir(dir_MABH_vs_TABH, kTRUE);

  // in case of HLT or HeavyIons skip the following
  TString new_Sample_Name("NEW_LABEL");

  if (TString(dataType) == "HLT" || new_Sample_Name.Contains("_HI")) {
    cout << ">> Removing the relval files from ROOT before closing..." << endl;
    gROOT->GetListOfFiles()->Remove(sfile);
    gROOT->GetListOfFiles()->Remove(rfile);

    if (DEBUG) {
      cout << " Exiting!" << endl;
      cerr << " Exiting!" << endl;
    }

    return;
  }

  if (DEBUG) {
    cout << " Comparing MuonAssociatorByHits with quickTrackAssociatorByHits on probeTracks (for the new release)" << endl;
    cerr << " Comparing MuonAssociatorByHits with quickTrackAssociatorByHits on probeTracks (for the new release)" << endl;
  }
  
  sfile->cd("DQMData/Run 1/Muons/Run summary/RecoMuonV/MuonTrack");
  sdir = gDirectory;
  rcollname = "probeTrks_TkAsso";
  scollname = "probeTrks";

  // for releases before CMSSW_10_1_0_pre1 and New Muon Validation
  TString New_CMSSW_Release("NEW_RELEASE");
  bool NEWprobeTrksNames = false;
  if (New_CMSSW_Release.Contains("CMSSW_9") || New_CMSSW_Release.Contains("CMSSW_10_0")) NEWprobeTrksNames=true;
  if (NEWprobeTrksNames) {
    rcollname = "NEWprobeTrks_TkAsso";
    scollname = "NEWprobeTrks";
  }

  const char* _refLabel("NEW_LABEL, NEW_RELEASE NEWSELECTION quickTrackAssociatorByHits");
  const char* _newLabel("NEW_LABEL, NEW_RELEASE NEWSELECTION MuonAssociatorByHits");

  // efficiency and fake rate Vs eta and phi
  plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
  plots[0]="effic_vs_eta"    ; titles[0]="Efficiency vs #eta";
  plots[1]="fakerate_vs_eta" ; titles[1]="Fake rate vs #eta" ;
  plots[2]="effic_vs_pt"     ; titles[2]="Efficiency vs pt" ;
  plots[3]="fakerate_vs_pt"  ; titles[3]="Fake rate vs pt" ;

  logx[0]=false;
  logx[1]=false;
  logx[2]=true;
  logx[3]=true;

  miny[0]=-0.0001;
  miny[1]=-0.0001;
  miny[2]=-0.0001;
  miny[3]=-0.0001;
  
  maxy[0]=1.09;
  maxy[1]=0.;
  maxy[2]=1.09;
  maxy[3]=0.;

  Plot4Histograms(dir_MABH_vs_TABH + "/eff_pt_eta",
		  sdir, sdir,
		  rcollname, scollname,
		  "eff_pt_eta_MABHvsTABH", "Efficiency vs eta and pt - Muon vs Track Associator",
		  _refLabel, _newLabel,
		  plots, titles, drawopt,
		  logy, logx, doKolmo, norm, minx, maxx, miny, maxy);

  // efficiency and fake rate Vs N.hits and phi
  plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
  plots[0]="effic_vs_hit"    ; titles[0]="Efficiency vs Number of hits";
  plots[1]="fakerate_vs_hit" ; titles[1]="Fake rate vs Number of hits" ;
  plots[2]="effic_vs_phi"    ; titles[2]="Efficiency vs #phi" ;
  plots[3]="fakerate_vs_phi" ; titles[3]="Fake rate vs #phi" ;

  miny[0]=-0.0001;
  miny[1]=-0.0001;
  miny[2]=-0.0001;
  miny[3]=-0.0001;
  
  maxy[0]=1.09;
  maxy[1]=0.;
  maxy[2]=1.09;
  maxy[3]=0.;
  
  Plot4Histograms(dir_MABH_vs_TABH + "/eff_phi_hits",
		  sdir, sdir,
		  rcollname, scollname,
		  "eff_phi_hits_MABHvsTABH", "Efficiency vs phi and N. hits - Muon vs Track Associator",
		  _refLabel, _newLabel,
		  plots, titles, drawopt,
		  logy, logx, doKolmo, norm, minx, maxx, miny, maxy);
  
  // efficiency and fake rate Vs PU
  plotOptReset(logx,logy,doKolmo,norm,minx,maxx,miny,maxy,drawopt,plots,titles);
  plots[0]="effic_vs_pu"    ; titles[0]="Efficiency vs n.PU interactions";
  plots[1]="fakerate_vs_pu" ; titles[1]="Fake rate vs n.PU interactions" ;
  
  //maxx[0]= 100.;
  //maxx[1]= 100.;
  
  miny[0]= -0.0001;
  miny[1]=  0.;
  
  maxy[0]= 1.09;
  maxy[1]= 0.;

  norm[1] = -1;
  
  PlotNHistograms(dir_MABH_vs_TABH + "/eff_pu",
		  sdir, sdir,
		  rcollname, scollname,
		  "eff_pu_MABHvsTABH", "Efficiency vs N.PU interactions - Muon vs Track Associator",
		  _refLabel, _newLabel,
		  4, plots, titles, drawopt,
		  logy, logx, doKolmo, norm, minx, maxx, miny, maxy);
  
  //// Merge pdf files together and rename the merged pdf after the 
  TString _destfile  = newDirBase + "probeTrks_MABH_vs_TABH" + ".pdf"; // Destination file name
  TString _gscommand = "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile="  + _destfile + " "
    + dir_MABH_vs_TABH + "/eff_pt_eta.pdf "
    + dir_MABH_vs_TABH + "/eff_phi_hits.pdf "
    + dir_MABH_vs_TABH + "/eff_pu.pdf ";
  
  cout << ">> Merging partial pdfs to " << _destfile << "..." << endl;
  if (DEBUG) cout << " ...with command \"" << _gscommand << "\"" << endl;
  gSystem->Exec(_gscommand);
  
  cout << ">> Deleting partial pdf files" << endl;
  gSystem->Exec("rm -rf "+ dir_MABH_vs_TABH +"/eff_*.pdf"); 
  cout << "   ... Done" << endl;
  
  cout << ">> Removing the relval files from ROOT before closing..." << endl;
  gROOT->GetListOfFiles()->Remove(sfile);
  gROOT->GetListOfFiles()->Remove(rfile);
  
  if (DEBUG) {
    cout << " Exiting!" << endl;
    cerr << " Exiting!" << endl;
  }
}

