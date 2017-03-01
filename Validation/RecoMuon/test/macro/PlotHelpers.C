////////////////////////////////////////////////////////////
//
// A group of helper functions used to compare histograms
// and profiles between two different releases
//
////////////////////////////////////////////////////////////

/////
// Some includes
#include "TH1F.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TPad.h"
#include "TPaveLabel.h"
#include <vector>

void NormalizeHistogramsToFirst(TH1* h1, TH1* h2);
void NormalizeHistogramsTo1(TH1* h1, TH1* h2);

/////
// Uncomment the following line to get more debuggin output
// #define DEBUG

////////////////////////////////////////////////////////////
//
// Sets the global style for the whole system.
//
// Note: There might be redundancies in other places so
//       things can be further simplified
//
void SetGlobalStyle() {
  gROOT->SetStyle("Plain");
  gStyle->SetPadGridX(kTRUE);
  gStyle->SetPadGridY(kTRUE);
  gStyle->SetPadRightMargin(0.07);
  gStyle->SetPadLeftMargin(0.13);
  //gStyle->SetTitleXSize(0.07); 
  //gStyle->SetTitleXOffset(0.6); 
  //tyle->SetTitleYSize(0.3);
  //gStyle->SetLabelSize(0.6) 
  //gStyle->SetTextSize(0.5);
}
//
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
//
// This function sets the style for a given histogram
//
// Note: There might be redundancies in other places so
//       things can be further simplified
//
void SetHistogramStyle(TH1* h, Style_t mstyle, Color_t color, Size_t msize = 0.7, Width_t lwidth = 2,
		       Float_t tsize = 0.05, Float_t toffset = 1.2) {
  if (!h)
    return;
  h->SetMarkerStyle(mstyle);
  h->SetMarkerColor(color);
  h->SetMarkerSize(msize);
  h->SetLineColor(color);
  h->SetLineWidth(lwidth);
  h->GetYaxis()->SetTitleSize(tsize);
  h->GetYaxis()->SetTitleOffset(toffset);
}
//
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
//
// This function finds the list of TDirectories in a branch
// that match a given string

TList* GetListOfDirectories(TDirectory* dir, const char* match = 0) {
  TIter  nextkey(dir->GetListOfKeys());
  TList* sl     = new TList();
  TKey*  key    = 0;
  TKey*  oldkey = 0;

  while (( key = (TKey*)nextkey() )) {
    TObject *obj = key->ReadObj();
    if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      TString theName = obj->GetName();
      if (!match) {
	cout << " -> " << theName << endl;
	sl->Add(obj);
      }
      else if (theName.Contains(match)) {
	cout << " -> " << theName << endl;
	sl->Add(obj);
      }
    }
  } // End of while

  return sl;
}

////////////////////////////////////////////////////////////
//
// This function goes to the right branch inside the file
// looking for branches having branchContent.
// It returns a list with all those branches,
//
TList* getListOfBranches(const char* dataType, TFile* file, const char* branchContent) {
  /*
  if (dataType == "HLT") {
    if(file->cd("DQMData/Run 1/HLT")) {
      file->cd("DQMData/Run 1/HLT/Run summary/Muon/MultiTrack");
    }
    else {
      file->cd("DQMData/HLT/Muon/MultiTrack");
    }
  }
  else if (dataType == "RECO") {
    if(file->cd("DQMData/Run 1/RecoMuonV")) {
      file->cd("DQMData/Run 1/RecoMuonV/Run summary/MultiTrack");
    }
    else if(file->cd("DQMData/Run 1/Muons/Run summary/RecoMuonV")) {
      file->cd("DQMData/Run 1/Muons/Run summary/RecoMuonV/MultiTrack");
    }
    else {
      file->cd("DQMData/RecoMuonV/MultiTrack");
    }
  }
  else {
    cout << "ERROR: Data type " << dataType << " not allowed: only RECO and HLT are considered" << endl;
    cerr << "ERROR: Data type " << dataType << " not allowed: only RECO and HLT are considered" << endl;
    return;
  }
  */
  
  if (TString(dataType) == "RECO") {
    if(! file->cd("DQMData/Run 1/Muons/Run summary")) {
      cout << "ERROR: Muon Histos for " << dataType << " not found" << endl;
      return 0;
    }
  }
  else {
    cout << "ERROR: Data type " << dataType << " not allowed: only RECO is considered" << endl;
    return 0;
  }

  TDirectory * dir=gDirectory;
  TList* sl = GetListOfDirectories(dir, branchContent);

  if (sl->GetSize() == 0) {
    cout << "ERROR: No DQM muon reco histos found in NEW file " << endl;
    delete sl;
    return 0;
  }

  return sl;
}
//
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
//
// This function performs a compatibility test between two
// histogram based on the Kolmogorov-Smirnof algorithm. It
// also prints the value in a TPaveLabel at the upper-right
// corner.
// The return value contains the result of the test
//
double KolmogorovTest(TH1 *h1, TH1 *h2){

  double mya_array[1300], myb_array[1300];
  vector<double> mya;
  vector<double> myb;
  
  
  for (int i=0; i<h1->GetNbinsX(); i++){
    mya.push_back(h1->GetBinContent(i+1));
    myb.push_back(h2->GetBinContent(i+1));
  }

  sort(mya.begin(),mya.end());
  sort(myb.begin(),myb.end()); 
  copy(mya.begin(),mya.end(),mya_array);
  copy(myb.begin(),myb.end(),myb_array);
  
  const int nbinsa = h1->GetNbinsX();
  const int nbinsb = h2->GetNbinsX();
   
  double kstest = TMath::KolmogorovTest(nbinsa, mya_array,
					nbinsb, myb_array,
					"UOX");
#ifdef DEBUG
  cout << "DEBUG:   + KS value = " << kstest << endl;
#endif


  // Create text with the value
  TString legend = Form("KS=%4.2f", kstest);

  // Create a pave text to put the value inside

  TPaveLabel* pl = new TPaveLabel(0.79,0.91,0.93,0.96, legend.Data(), "NDC");

  // Tune style
  //pl->SetTextSize(0.04);
  pl->SetLineColor(41);
  pl->SetLineWidth(1);
  pl->SetLineStyle(1);
  pl->SetFillColor(41);
  pl->SetBorderSize(3);

  if (kstest < 0.7)
    pl->SetTextColor(kRed);

  pl->Draw();
  
  return kstest;
}
//
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
//
// This function draws the stat box for the two plots
//
void setStats(TH1* s, TH1* r, double startingY, double startingX = .1, bool fit = false){
  if (startingY<0){
    s->SetStats(0);
    r->SetStats(0);
  } 
  else {
    //gStyle->SetOptStat(1001);
    
    if (fit){
      s->Fit("gaus");
      TF1* f1 = (TF1*) s->GetListOfFunctions()->FindObject("gaus");
      if (f1) {
	f1->SetLineColor(2);
	f1->SetLineWidth(1);
      }
    }
    s->Draw();
    gPad->Update(); 
    TPaveStats* st1 = (TPaveStats*) s->GetListOfFunctions()->FindObject("stats");
    if (st1) {
      if (fit) {st1->SetOptFit(0010);    st1->SetOptStat(1001);}
      st1->SetX1NDC(startingX);
      st1->SetX2NDC(startingX+0.30);
      st1->SetY1NDC(startingY+0.20);
      st1->SetY2NDC(startingY+0.35);
      st1->SetTextColor(2);
    }
    else s->SetStats(0);
    if (fit) {
      r->Fit("gaus");
      TF1* f2 = (TF1*) r->GetListOfFunctions()->FindObject("gaus");
      if (f2) {
	f2->SetLineColor(4);
	f2->SetLineWidth(1);
      }
    }
    r->Draw();
    gPad->Update(); 
    TPaveStats* st2 = (TPaveStats*) r->GetListOfFunctions()->FindObject("stats");
    if (st2) {
      if (fit) {st2->SetOptFit(0010);    st2->SetOptStat(1001);}
      st2->SetX1NDC(startingX);
      st2->SetX2NDC(startingX+0.30);
      st2->SetY1NDC(startingY);
      st2->SetY2NDC(startingY+0.15);
      st2->SetTextColor(4);
    }
    else r->SetStats(0);
  }
}
//
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
//
// Plot a page with several histograms
//
void PlotNHistograms(const TString& pdfFile,
		     TDirectory* rdir, TDirectory* sdir,
		     const TString& rcollname, const TString& scollname,
		     const char* canvasName, const char* canvasTitle,
		     const TString& refLabel, const TString& newLabel,
		     unsigned int nhistos, const char** hnames, const char** htitles = 0,
		     bool* logy = 0, bool* doKolmo = 0, Double_t* norm = 0, bool *resol = 0,
		     Double_t* minx = 0, Double_t* maxx = 0,
		     Double_t* miny = 0, Double_t* maxy = 0) {
#ifdef DEBUG
  cout << "   + Plotting histograms for " << canvasTitle << endl;
  cerr << "   + Plotting histograms for " << canvasTitle << endl;
#endif
  TH1* rh   = 0;
  TH1* sh   = 0;
  // TH2* rh2d = 0;
  // TH2* sh2d = 0;
  // TH1F* rproj = 0;
  // TH1F* sproj = 0;
  //  TString *hnames_tmp = hnames;
#ifdef DEBUG
  cout << "DEBUG: Creating canvas..." << endl;
#endif
  TCanvas* canvas =  0;
  if (nhistos >4)
    canvas = new TCanvas(canvasName, canvasTitle, 1000, 1400);
  else
    canvas = new TCanvas(canvasName, canvasTitle, 1000, 1050);
  canvas->Divide(2,(nhistos+1)/2); //This way we print in 2 columns

#ifdef DEBUG
  cout << "DEBUG: Looping over histograms" << endl;
#endif
  for (unsigned int i = 0; i < nhistos; i++) {
#ifdef DEBUG
    cout << "DEBUG: [" << i << "] histogram name: " << flush << hnames[i] << endl;
    cout << "DEBUG: rdir:  " << rdir << endl;
    cout << "DEBUG: sdir:  " << sdir << endl;
    cout << "DEBUG: rcollname: " << rcollname << endl;
    cout << "DEBUG: scollname: " << scollname << endl;
    cout << "DEBUG: rname: " << (rcollname + "/" + hnames[i]) << endl;
    cout << "DEBUG: sname: " << (scollname + "/" + hnames[i]) << endl;
#endif

    TString hnamesi = hnames[i];
    
    // Skip histogram if no name is provided
    if (hnamesi == "") continue;
    
    // Get Histograms
    // + Reference
    TString hnames_tmp = hnamesi;
    rdir->cd(rcollname);
    TIter next(gDirectory->GetListOfKeys());
    TKey *key;
    while ((key = (TKey*)next())) { 
    
      TObject *obj = key->ReadObj();
      TString temp = obj->GetName();
      if ( (temp.Contains("nhits_vs_eta_pfx")) &&
	   hnamesi.Contains("hits_eta")
	 ) {
	hnames_tmp = temp;      
      }
      if ( (temp.Contains("chi2_vs_eta_pfx")) &&
	   hnamesi.Contains("chi2mean") 
	 ) {
	hnames_tmp = temp;      
      }
    } 
    

    
#ifdef DEBUG
    cout << "DEBUG: Getting object reference samples " << (rcollname + "/" + hnames_tmp) << " from " << rdir << endl;
#endif
    rdir->GetObject(rcollname + "/" + hnames_tmp, rh);
    if (! rh) {
      cout << "WARNING: Could not find a reference histogram or profile named " << hnames_tmp[i] 
	   << " in " << rdir->GetName() << endl;
      cout << "         Skipping" << endl;
      continue;
    }

    //If it is a 2D project it in Y... is this what we always want?
    if (TString(rh->IsA()->GetName()) == "TH2F") {
#ifdef DEBUG
      cout << "DEBUG: It is a TH2F object... project in Y!" << endl;
#endif
      TH1* proj = ((TH2F*) rh)->ProjectionY();
      rh = proj;
    }


    // + Signal
    
    hnames_tmp=hnamesi;
    
    sdir->cd(scollname);
    TIter next2(gDirectory->GetListOfKeys());
    TKey *key2;
    while ((key2 = (TKey*)next2())) { 
    
      TObject *obj = key2->ReadObj();
      TString temp = obj->GetName();
      if ( (temp.Contains("nhits_vs_eta_pfx")) &&
	   hnamesi.Contains("hits_eta")
	 ) {
	hnames_tmp = temp;      
      }
      if ( (temp.Contains("chi2_vs_eta_pfx")) &&
	   hnamesi.Contains("chi2mean")
	 ) {
	hnames_tmp = temp;      
      }
    } 

#ifdef DEBUG
    cout << "DEBUG: Getting object for selected sample " << (scollname + "/" + hnames_tmp) 
	 << " from " << sdir << endl;
#endif
    sdir->GetObject(scollname + "/" + hnames_tmp, sh);
    if (! rh) {
      cout << "WARNING: Could not find a signal histogram or profile named " << hnames_tmp[i] 
	   << " in " << rdir->GetName() << endl;
      cout << "         Skipping" << endl;
      continue;
    }

    //If it is a 2D project it in Y... is this what we always want?
    if (TString(sh->IsA()->GetName()) == "TH2F") {
#ifdef DEBUG
      cout << "DEBUG: " << hnames_tmp[i] << " is a TH2F object... project in Y!" << endl;
#endif
      TH1* proj = ((TH2F*) sh)->ProjectionY();
      sh = proj;
    }

    if(TString(sh->GetName()).Contains(" vs "))norm[i]= -999.;


    // Normalize
    if (norm[i] == -1.)
      NormalizeHistogramsTo1(rh, sh);
    else if (norm[i] == 0.)
      NormalizeHistogramsToFirst(rh,sh);
    else if (norm[i] == -999.){
      cout << "DEBUG: Normalizing histograms to nothing" << "..." << endl;
    }
    /*    else {
#ifdef DEBUG
      cout << "DEBUG: Normalizing histograms to " << norm[i] << "..." << endl;
#endif
      sh->Scale(norm[i]);
      rh->Scale(norm[i]);
      }*/


    // Set styles
#ifdef DEBUG
    cout << "DEBUG: Setting styles..." << endl;
#endif
    SetHistogramStyle(rh, 21, 4);
    SetHistogramStyle(sh, 20, 2);
    //Change titles
    if (htitles) {
      rh->SetTitle(htitles[i]);
      sh->SetTitle(htitles[i]);
      rh->GetYaxis()->SetTitle(htitles[i]);
      sh->GetYaxis()->SetTitle(htitles[i]);
    }
    //Change x axis range
    if (minx) {

      //      if (minx < -1E99) {

      	rh->GetXaxis()->SetRangeUser(minx[i],maxx[i]);
	sh->GetXaxis()->SetRangeUser(minx[i],maxx[i]);

	//}
    }
    //Change y axis range
    if (miny) {
      if (miny[i] < -1E99) {
	rh->GetYaxis()->SetRangeUser(miny[i],maxy[i]);
	sh->GetYaxis()->SetRangeUser(miny[i],maxy[i]);
      }
    }



    // Move to subpad
    canvas->cd(i+1);
    
    // Check Logy
    if (logy) {
      if (logy[i])
	gPad->SetLogy();
	
    }

    // Set stat boxes
#ifdef DEBUG
    cout << "DEBUG: Setting statistics..." << endl;
#endif
    setStats(sh, rh, -1, 0, false);
    
    // Draw histogram
#ifdef DEBUG
    cout << "DEBUG: Drawing histograms..." << endl;
#endif
    if (sh->GetMaximum() > rh->GetMaximum()) {
      sh->Draw();
      rh->Draw("sames");
    }
    else {
      rh->Draw();
      sh->Draw("sames");
    }
    
    // Perform Kolmogorov test if needed
    if (doKolmo) {
#ifdef DEBUG
      cout << "DEBUG: Performing Kolmogorov test..." << endl;
#endif
      if (doKolmo[i]) {
	//	TPad* c1_1 = canvas->GetPad(i+1);
	double kstest = KolmogorovTest(sh,rh);
	if(kstest<0.7)
	  gPad->SetFillColor(kBlue-10);
      }
    }
  } // End loop


   // Draw Legend
#ifdef DEBUG
  cout << "DEBUG: Drawing legend..." << endl;
#endif
  canvas->cd();
  
  TLegend* l = 0;
  if (nhistos > 4)
    l = new TLegend(0.20,0.665,0.80,0.685);
  else
    l = new TLegend(0.20,0.52,0.80,0.55);

  l->SetTextSize(0.011);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(2);
  l->AddEntry(rh,refLabel,"LPF");
  l->AddEntry(sh,newLabel,"LPF");
  l->Draw();
  
  // Print Canvas
  canvas->Print(pdfFile);

  // Clean memory
  // delete l;
  delete canvas;
  cout << "     ... plotted histograms for " << canvasTitle << endl;
}
//
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
//
// Plot a page with 4 histograms
//
void Plot4Histograms(const TString& pdfFile,
		     TDirectory* rdir, TDirectory* sdir,
		     const TString& rcollname, const TString& scollname,
		     const char* canvasName, const char* canvasTitle,
		     const TString& refLabel, const TString& newLabel,
		     const char** hnames, const char** htitles = 0,
		     bool* logy = 0, bool* doKolmo = 0, Double_t* norm = 0,bool *resol = 0,
		     Double_t* minx = 0, Double_t* maxx = 0,
		     Double_t* miny = 0, Double_t* maxy = 0) {
  PlotNHistograms(pdfFile,
		  rdir, sdir,
		  rcollname, scollname,
		  canvasName, canvasTitle,
		  refLabel, newLabel,
		  4, hnames, htitles,
		  logy, doKolmo, norm,resol,minx,maxx,miny,maxy);
}
//
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
//
// Plot a page with 6 histograms
//
void Plot6Histograms(const TString& pdfFile,
		     TDirectory* rdir, TDirectory* sdir,
		     const TString& rcollname, const TString& scollname,
		     const char* canvasName, const char* canvasTitle,
		     const TString& refLabel, const TString& newLabel,
		     const char** hnames, const char** htitles = 0,
		     bool* logy = 0, bool* doKolmo = 0, Double_t* norm = 0,bool *resol=0,
		     Double_t* minx = 0, Double_t* maxx = 0,
		     Double_t* miny = 0, Double_t* maxy = 0) {
  

  PlotNHistograms(pdfFile,
		  rdir, sdir,
		  rcollname, scollname,
		  canvasName, canvasTitle,
		  refLabel, newLabel,
		  6, hnames, htitles,
		  logy, doKolmo, norm,resol,
		  minx, maxx,
		  miny, maxy);
}
//
////////////////////////////////////////////////////////////

void Plot5Histograms(const TString& pdfFile,
		     TDirectory* rdir, TDirectory* sdir,
		     const TString& rcollname, const TString& scollname,
		     const char* canvasName, const char* canvasTitle,
		     const TString& refLabel, const TString& newLabel,
		     const char** hnames, const char** htitles = 0,
		     bool* logy = 0, bool* doKolmo = 0, Double_t* norm = 0,bool *resol=0,
		     Double_t* minx = 0, Double_t* maxx = 0,
		     Double_t* miny = 0, Double_t* maxy = 0) {
  

  PlotNHistograms(pdfFile,
		  rdir, sdir,
		  rcollname, scollname,
		  canvasName, canvasTitle,
		  refLabel, newLabel,
		  5, hnames, htitles,
		  logy, doKolmo, norm,resol,
		  minx, maxx,
		  miny, maxy);
}
//
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
//
// Normalize the two histograms to the entries on the first histogram
//
void NormalizeHistogramsToFirst(TH1* h1, TH1* h2) {
  if (h1==0 || h2==0) return;
  
  if ( h1->Integral() > 0 && h2->Integral() > 0 ){
    Double_t scale2 = h1->Integral()/h2->Integral();
    h2->Scale(scale2);
  }
}
//
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
//
// Normalize the two histograms to unity
//
void NormalizeHistogramsTo1(TH1* h1, TH1* h2) {
  if (!h1 || !h2) return;
  
  if ( h1->Integral() != 0 && h2->Integral() != 0 ) {
    Double_t scale1 = 1.0/h1->Integral();
    Double_t scale2 = 1.0/h2->Integral();
    h1->Scale(scale1);
    h2->Scale(scale2);
  }
}
//
////////////////////////////////////////////////////////////

/*
void NormalizeHistograms(TH1F* h1, Double_t nrm)
{
  if (h1==0) return;
  h1->Scale(nrm);
}


void plotHistoInCanvas(TCanvas* canvas, unsigned int ncanvas,
			TH1F* s, TH1F* r,
			Size_t markersize,
			bool logy = false,
			bool doKolmo = false) {
  if ((s==0) || (r==0)) {
    cerr << "ERROR: Histograms not found!" << endl;
    return;
  }

  s->SetMarkerStyle(20);
  r->SetMarkerStyle(21);
  s->SetMarkerColor(2);
  r->SetMarkerColor(4);
  s->SetMarkerSize(markersize);
  r->SetMarkerSize(markersize);
  s->SetLineColor(2);
  r->SetLineColor(4);
  s->SetLineWidth(2);
  r->SetLineWidth(2);

  //setStats(r,s, startingY, startingX, fit);
  canvas->cd(ncanvas);
  if (logy)
    gPad->SetLogy();
  setStats(s,r, -1, 0, false);
  s->Draw();
  r->Draw("sames");
  if (doKolmo) {
    TPad *c1_1 = canvas->GetPad(ncanvas);
    double kstest = KolmogorovTest(s,r);
    if(kstest<0.7){
      c1_1->SetFillColor(kYellow);}
  }
}


void plotProfileInCanvas(TCanvas* canvas, unsigned int ncanvas,
			 TProfile* s, TProfile* r,
			 Size_t markersize) {
  if ((s==0) || (r==0)) {
    cerr << "ERROR: TProfile not found!" << endl;
    return;
  }

  s->SetMarkerStyle(20);
  r->SetMarkerStyle(21);
  s->SetMarkerColor(2);
  r->SetMarkerColor(4);
  s->SetMarkerSize(markersize);
  r->SetMarkerSize(markersize);
  s->SetLineColor(2);
  r->SetLineColor(4);
  s->SetLineWidth(2);
  r->SetLineWidth(2);

  //setStats(r,s, startingY, startingX, fit);
  canvas->cd(ncanvas);
  setStats(s,r, -1, 0, false);
  s->Draw();
  r->Draw("sames");
  //  double kstest = KolmogorovTest(s1,r1);
}

void plot4histos(TCanvas *canvas, 
		 TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
		 TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
		 TText* te,
		 char * option, 
		 double startingY, double startingX = .1,
		 bool fit = false){
#ifdef DEBUG
  cout << "DEBUG: plot4histos for histos" << endl;
#endif

  canvas->Divide(2,2);
  plotHistoInCanvas(canvas, 1, s1, r1, 0.7, false, true);
  plotHistoInCanvas(canvas, 2, s2, r2, 0.1, false, true);
  plotHistoInCanvas(canvas, 3, s3, r3, 0.7, false, true);
  plotHistoInCanvas(canvas, 4, s4, r4, 0.7, false, true);
}
*/

/*
void plot4histos(TCanvas *canvas,
		 TProfile *s1,TProfile *r1, TProfile *s2,TProfile *r2,
		 TProfile *s3,TProfile *r3, TProfile *s4,TProfile *r4,
		 TText* te,
		 char * option, 
		 double startingY, double startingX = .1,
		 bool fit = false){

  cout << "DEBUG: plot4histos for profiles" << endl;

  canvas->Divide(2,2);
  
  plotProfileInCanvas(canvas, 1, s1, r1, 0.7);
  plotProfileInCanvas(canvas, 2, s2, r2, 0.1);
  plotProfileInCanvas(canvas, 3, s3, r3, 0.7);
  plotProfileInCanvas(canvas, 4, s4, r4, 0.7);
  
}

void plot6histos(TCanvas *canvas, 
		 TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
		 TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
		 TH1F *s5,TH1F *r5, TH1F *s6,TH1F *r6,
		 TText* te,
		 char * option, 
		 double startingY, double startingX = .1,
		 bool fit = false){
  canvas->Divide(2,3);

  plotHistoInCanvas(canvas, 1, s1, r1, 0.7, false, true);
  plotHistoInCanvas(canvas, 2, s2, r2, 0.1, true,  true);
  plotHistoInCanvas(canvas, 3, s3, r3, 0.7, false, true);
  plotHistoInCanvas(canvas, 4, s4, r4, 0.7, true,  true);
  plotHistoInCanvas(canvas, 5, s5, r5, 0.7, false, true);
  plotHistoInCanvas(canvas, 6, s6, r6, 0.7, true,  true);

}
*/
