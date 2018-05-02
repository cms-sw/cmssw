////////////////////////////////////////////////////////////
//
// A group of helper functions used to compare histograms
// and profiles between two different releases
//
////////////////////////////////////////////////////////////

#include "TH1F.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TPad.h"
#include "TPaveLabel.h"
#include <vector>

void NormalizeHistogramsToFirst(TH1* h1, TH1* h2);
void NormalizeHistogramsToOne(TH1* h1, TH1* h2);
void NormalizeHistogramsAsDensity(TH1* h1, TH1* h2);
TH1* PlotRatiosHistograms(TH1* h1, TH1* h2);

int ratioCounter = 0;

// debugging printouts
bool DEBUGP = false;

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
  gStyle->SetOptStat(0);
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
  h->GetXaxis()->SetLabelFont(63);
  h->GetXaxis()->SetLabelSize(14); // labels will be 14 pixels
  h->GetYaxis()->SetLabelFont(63);
  h->GetYaxis()->SetLabelSize(14);
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
  if (DEBUGP) cout << "   + KS value = " << kstest << endl;

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
 		     unsigned int nhistos, const TString* hnames,
		     const TString* htitles, const char** drawopt,
 		     bool* logy = 0,  bool* logx = 0, bool* doKolmo = 0, Double_t* norm = 0,
		     Double_t* minx = 0, Double_t* maxx = 0,
		     Double_t* miny = 0, Double_t* maxy = 0) {

  if (DEBUGP) {
    cout << "   + Plotting histograms for " << canvasTitle << endl;
    cerr << "   + Plotting histograms for " << canvasTitle << endl;
  }

  TH1* rh   = 0;
  TH1* sh   = 0;

  TCanvas* canvas =  0;
  if (nhistos >4)
    canvas = new TCanvas(canvasName, canvasTitle, 1000, 1400);
  else
    canvas = new TCanvas(canvasName, canvasTitle, 1000, 1050);

  canvas->Draw();
  canvas->Divide(2,(nhistos+1)/2); //This way we print in 2 columns

  for (unsigned int i = 0; i < nhistos; i++) {

    if (DEBUGP) cout << " [" << i << "] histogram name: " << flush << hnames[i] << endl;

    //draw option for the new histogram
    TString drawoption = drawopt[i]; 

    // Skip histogram if no name is provided
    if (hnames[i] == "") continue;
    
    // Get Histograms
    // + Reference release
    rdir->cd(rcollname);
    
    if (DEBUGP) cout << " Getting object for reference sample " << (rcollname + "/" + hnames[i]) << endl;

    rdir->GetObject(rcollname + "/" + hnames[i], rh);
    if (! rh) {
      cout << "WARNING: Could not find a reference histogram or profile named " << hnames[i]
	   << " in " << rdir->GetName() << endl;
      cout << "         Skipping" << endl;
      continue;
    }

    //If it is a 2D project it in Y... is this what we always want?
    if (TString(rh->IsA()->GetName()) == "TH2F") {

      if (DEBUGP) cout << " It is a TH2F object... project in Y!" << endl;

      TH1* proj = ((TH2F*) rh)->ProjectionY();
      rh = proj;
    }

    // + New release    
    sdir->cd(scollname);

    if (DEBUGP) cout << " Getting object for target sample " << (scollname + "/" + hnames[i]) << endl;

    sdir->GetObject(scollname + "/" + hnames[i], sh);
    if (! sh) {
      cout << "WARNING: Could not find a signal histogram or profile named " << hnames[i] 
	   << " in " << sdir->GetName() << endl;
      cout << "         Skipping" << endl;
      continue;
    }

    //If it is a 2D project it in Y... is this what we always want?
    if (TString(sh->IsA()->GetName()) == "TH2F") {

      if (DEBUGP) cout << hnames[i] << " is a TH2F object... project in Y!" << endl;

      TH1* proj = ((TH2F*) sh)->ProjectionY();
      sh = proj;
    }

    // Set styles

    if (DEBUGP) cout << " Setting style..." << endl;

    SetHistogramStyle(rh, 21, 4);
    SetHistogramStyle(sh, 20, 2);

    //Change titles
    if (htitles) {
      rh->SetTitle(htitles[i]);
      sh->SetTitle(htitles[i]);
      rh->GetYaxis()->SetTitle(htitles[i]);
      sh->GetYaxis()->SetTitle(htitles[i]);
    }

    // SET X AXIS RANGE in plots
    Bool_t ChangeXRange = false ;
    Double_t Xleft = rh->GetXaxis()->GetXmin();
    Double_t Xright = rh->GetXaxis()->GetXmax(); 

    if (DEBUGP) cout << "ref histo Xleft, Xright = "<< Xleft << ", "<< Xright  << endl;
   
    if (sh->GetXaxis()->GetXmin() < rh->GetXaxis()->GetXmin()) {
      Xleft = sh->GetXaxis()->GetXmin();
      ChangeXRange = true;
      if (DEBUGP) cout << "automatic reset MIN (new < ref) Xleft = "<< Xleft << endl;
    }
    if (sh->GetXaxis()->GetXmax() > rh->GetXaxis()->GetXmax()) {
      Xright = sh->GetXaxis()->GetXmax();
      ChangeXRange = true;
      if (DEBUGP) cout << "automatic reset MAX (new > ref) Xright = "<< Xright << endl;
    }

    if (minx[i]!=0) {
      ChangeXRange = true ;
      Xleft = minx[i];
      if (DEBUGP) cout << "user reset Xleft = "<< Xleft << endl;
    }
    
    if (maxx[i]!=0) {
      ChangeXRange = true ;
      Xright = maxx[i];
      if (DEBUGP) cout << "user reset Xright = "<< Xleft << endl;
    }
        
    if (ChangeXRange) {
      if (DEBUGP) {
	cout << "Ref histo Xmin, Xmax = "<< rh->GetXaxis()->GetXmin()  << ", " << rh->GetXaxis()->GetXmax() <<endl;
	cout << "New histo Xmin, Xmax = "<< sh->GetXaxis()->GetXmin()  << ", " << sh->GetXaxis()->GetXmax() <<endl;
      }
      
      rh->GetXaxis()->SetRangeUser(Xleft,Xright);
      sh->GetXaxis()->SetRangeUser(Xleft,Xright);      

      if (DEBUGP) {
	cout << "reset Ref histo Xmin, Xmax = "<< rh->GetXaxis()->GetXmin()  << ", " << rh->GetXaxis()->GetXmax() <<endl;
	cout << "reset New histo Xmin, Xmax = "<< sh->GetXaxis()->GetXmin()  << ", " << sh->GetXaxis()->GetXmax() <<endl;
	cout << "resetting Ref and New histo Xleft, Xright = "<< Xleft << ", " << Xright <<endl;
      }
    }

    // ===============================================================================================
    // Normalize
    if (norm[i] < 0.) ; 
      // Default: do not normalize at all !
    else if (norm[i] == 0.)
      NormalizeHistogramsToFirst(rh,sh);
    else if (norm[i] == 1.)
      NormalizeHistogramsToOne(rh,sh);
    else if (norm[i] == 2.)
      NormalizeHistogramsAsDensity(rh,sh);
    // ===============================================================================================

    // ===============================================================================================
    // SET Y AXIS RANGE in plots
    //
    // MINIMUM
    //
    Double_t Ybottom;

    // if user-defined range force it !
    if (miny[i]!=0) {
      Ybottom = miny[i];
      if (DEBUGP) cout << "setting Minimum Y to user defined value: "<< miny[i] << endl;
    }
    else if (logy[i]) {
      // automatic setting for log scale
      Double_t yminr = rh->GetMinimum(0.); // min value larger than zero
      Double_t ymins = sh->GetMinimum(0.);
      Ybottom = yminr < ymins ? yminr*0.5 : ymins*0.5;
      if (DEBUGP) cout << "LOG scale, yminr, ymins: "<<yminr<<", "<<ymins <<"  ==>> Ybottom = "<<Ybottom<< endl;
    }
    else {
      // automatic setting for linear scale
      Double_t yminr = rh->GetMinimum(); // min value larger than zero
      Double_t ymins = sh->GetMinimum();
      Ybottom = yminr < ymins ? yminr-0.1*abs(yminr) : ymins-0.1*abs(ymins) ;
      // limit the scale to -1,+1 for relative pt bias to avoid failing fits
      if ((hnames[i] == "ptres_vs_eta_Mean") && (Ybottom <-1.)) Ybottom = -1.;
      if ((hnames[i] == "ptres_vs_pt_Mean") && (Ybottom <-1.)) Ybottom = -1.;
      if (DEBUGP) cout << "LIN scale, yminr, ymins: "<<yminr<<", "<<ymins <<"  ==>> Ybottom = "<<Ybottom<< endl;    
    }

    ///////////////////
    // MAXIMUM
    //
    Double_t Ytop;

    // if user-defined range force it !
    if (maxy[i]!=0) {
      Ytop = maxy[i];
      if (DEBUGP) cout << "setting Maximum Y to user defined value: "<< maxy[i] << endl;
    }
    else {
      Double_t ymaxr = rh->GetMaximum(); // max value
      Double_t ymaxs = sh->GetMaximum();
      Ytop = ymaxr > ymaxs ? ymaxr : ymaxs ;
      // automatic setting for log scale
      if (logy[i]) {
	Ytop = Ytop*2;
	if (DEBUGP) cout << "LOG scale, ymaxr, ymaxs: "<<ymaxr<<", "<<ymaxs <<"  ==>> Ytop = "<<Ytop<< endl;
      }
      else {
	Ytop = Ytop+0.1*abs(Ytop);
	// limit the scale to -1,+1 for relative pt bias to avoid failing fits
	if ((hnames[i] == "ptres_vs_eta_Mean") && (Ytop >1.)) Ytop = 1.;
	if ((hnames[i] == "ptres_vs_pt_Mean") && (Ytop >1.)) Ytop = 1.;
	if (DEBUGP) cout << "LIN scale, ymaxr, ymaxs: "<<ymaxr<<", "<<ymaxs <<"  ==>> Ytop = "<<Ytop<< endl;     
      }
    }

    // +++++++++++++++++++++++++++++++++++++++++
    rh->GetYaxis()->SetRangeUser(Ybottom,Ytop);
    sh->GetYaxis()->SetRangeUser(Ybottom,Ytop); 
    // +++++++++++++++++++++++++++++++++++++++++     

    // Move to subpad
    canvas->cd(i+1);
    
    TPad* pad1 = NULL;
    TPad* pad2 = NULL;

    pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
    pad2 = new TPad("pad2", "pad2", 0, 0.0, 1, 0.3);

    pad1->SetTopMargin   (0.08);
    pad1->SetBottomMargin(0.01);
    pad1->Draw();

    pad2->SetTopMargin   (0.05);
    pad2->SetBottomMargin(0.45);
    pad2->Draw();// Set stat boxes                                                                      
    pad1->cd();

    // Check Logy                                                                                       
    if (logy[i]) gPad->SetLogy();
    if (logx[i]) {gPad->SetLogx(); pad2->SetLogx();}

    // Set stat boxes

    if (DEBUGP) cout << " Setting statistics..." << endl;
    setStats(sh, rh, -1, 0, false);
   
    // /////////// DRAW  histograms //////////////////////////////////////
    //
    // FIRST plot: reference (blue)   SECOND plot: new (red)
    if (DEBUGP) cout << " Drawing histograms..." << endl;

    if (ChangeXRange) {
      sh->Draw(drawoption);
      rh->Draw("same"+drawoption);
      sh->Draw("same"+drawoption);
    }
    else {
      rh->Draw(drawoption);
      sh->Draw("same"+drawoption);
    }

    // Perform Kolmogorov test if needed
    if (doKolmo) {
      if (doKolmo[i]) {
	if (DEBUGP) cout << " Performing Kolmogorov test..." << endl;
	//	TPad* c1_1 = canvas->GetPad(i+1);
	double kstest = KolmogorovTest(sh,rh);
	if(kstest<0.7)
	  gPad->SetFillColor(kBlue-10);
      }
    }
 
    pad2->cd();

    TH1* ratioplot = PlotRatiosHistograms(rh, sh);
    SetHistogramStyle(ratioplot, 21, 4);
    ratioplot->Draw("ep");
 } // End loop
  
   // Draw Legend

  if (DEBUGP) cout << " Drawing legend..." << endl;

  canvas->cd();
  
  TLegend* l = 0;
  if (nhistos > 4)
    l = new TLegend(0.20,0.665,0.80,0.685);
  else
    l = new TLegend(0.20,0.50,0.80,0.53);

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
  canvas->SaveAs(pdfFile+".pdf");
  canvas->SaveAs(pdfFile+".png");

  // Clean memory
  // delete l;
  delete canvas;
  if (DEBUGP) cout << "     ... plotted histograms for " << canvasTitle << endl;
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
		     const TString* hnames, const TString* htitles, const char** drawopt,
		     bool* logy = 0, bool* logx = 0, bool* doKolmo = 0, Double_t* norm = 0,
		     Double_t* minx = 0, Double_t* maxx = 0,
		     Double_t* miny = 0, Double_t* maxy = 0) {
  PlotNHistograms(pdfFile,
		  rdir, sdir,
		  rcollname, scollname,
		  canvasName, canvasTitle,
		  refLabel, newLabel,
		  4, hnames, htitles, drawopt,
		  logy, logx, doKolmo, norm, minx, maxx, miny, maxy);

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
		     const TString* hnames, const TString* htitles, const char** drawopt,
		     bool* logy = 0, bool* logx = 0, bool* doKolmo = 0, Double_t* norm = 0,
		     Double_t* minx = 0, Double_t* maxx = 0,
		     Double_t* miny = 0, Double_t* maxy = 0) {
  

  PlotNHistograms(pdfFile,
		  rdir, sdir,
		  rcollname, scollname,
		  canvasName, canvasTitle,
		  refLabel, newLabel,
		  6, hnames, htitles, drawopt,
		  logy, logx, doKolmo, norm,
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
		     const TString* hnames, const TString* htitles, const char** drawopt,
		     bool* logy = 0,  bool* logx = 0, bool* doKolmo = 0, Double_t* norm = 0,
		     Double_t* minx = 0, Double_t* maxx = 0,
		     Double_t* miny = 0, Double_t* maxy = 0) {
  

  PlotNHistograms(pdfFile,
		  rdir, sdir,
		  rcollname, scollname,
		  canvasName, canvasTitle,
		  refLabel, newLabel,
		  5, hnames, htitles, drawopt,
		  logy, logx, doKolmo, norm,
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
void NormalizeHistogramsToOne(TH1* h1, TH1* h2) {
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

// Normalize the two histograms as probability density functions:
// Normalize areas to unity and consider the bin width
//
void NormalizeHistogramsAsDensity(TH1* h1, TH1* h2) {
  if (h1==0 || h2==0) return;
  if (h1->Integral() != 0 && h2->Integral() != 0 ) {
    Double_t scale1 = 1.0/h1->Integral();
    Double_t scale2 = 1.0/h2->Integral();
    h1->Scale(scale1, "width");
    h2->Scale(scale2, "width");
  }
}
///////////////////////////////////////////////////////////
// 
// ratio plot from the two histograms
//  
/////////////////////////////////////////////////////////

TH1* PlotRatiosHistograms(TH1* h1, TH1* h2){

  ++ratioCounter;

  Int_t nbinsx = h1->GetNbinsX();

  Double_t xmin = h1->GetBinLowEdge(0);
  Double_t xmax = h1->GetBinLowEdge(nbinsx+1);

  TH1F* h_ratio = new TH1F(Form("h_ratio_%d", ratioCounter), "", nbinsx, xmin, xmax);

  for (Int_t ibin=1; ibin<=nbinsx; ibin++) {

    Float_t h1Value = h1->GetBinContent(ibin);
    Float_t h2Value = h2->GetBinContent(ibin);

    Float_t h1Error = h1->GetBinError(ibin);
    Float_t h2Error = h2->GetBinError(ibin);

    Float_t ratioVal = 999;
    Float_t ratioErr = 999;

    if (h2Value > 0) {
      ratioVal = h1Value / h2Value;
      ratioErr = h1Error / h2Value;
    }

    h_ratio->SetBinContent(ibin, ratioVal);
    h_ratio->SetBinError  (ibin, ratioErr);

  }

  h_ratio->SetTitle("");
  h_ratio->GetYaxis()->SetTitle("");
  h_ratio->GetYaxis()->SetRangeUser(0.4, 1.6);

  return h_ratio;
}
