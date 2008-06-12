#include <string>

void plotMerge(char name[6]="Hits0", char rootfile[6]="ecHit", int files=5,
	       int bin=1, bool logy=true, bool doprob=false, bool debug=false){

  char fname[20];
  sprintf (fname, "%s_1.root", rootfile);
  setTDRStyle();
  TFile *file = TFile::Open(fname);
  TDirectory *d1 = (TDirectory*)file->Get("analyzer");
  TH1F *h1 = (TH1F*)d1->Get(name);
  TH1F *hist(h1);
  std::cout << "Entries at loop 1: " << hist->GetEntries() << "\n";

  for (int i=2; i<=files; i++) {
    sprintf (fname, "%s_%d.root", rootfile, i); 
    TFile *file2 = TFile::Open(fname);
    TDirectory *d2 = (TDirectory*)file2->Get("analyzer");
    TH1F *h2 = (TH1F*)d2->Get(name);
    hist->Add(h2);
    if (debug) std::cout << "Entries at loop " << i << ": " << hist->GetEntries() << "\n";
    file2->Close();
  }
  TCanvas *myc = new TCanvas("name","",800,600);
  if (logy) gPad->SetLogy(1);
  hist->GetYaxis()->SetTitle("Events");
  hist->Rebin(bin);
  hist->Draw();

  if (doprob) {
    int    nbin = hist->GetXaxis()->GetNbins();
    double xmin = hist->GetXaxis()->GetXmin();
    double xmax = hist->GetXaxis()->GetXmax();
    double entry= hist->GetEntries();
    double scale= 1.0;
    if (entry > 0) scale = 1./entry;
    char title[60];
    std::string names(name), namx;
    namx.assign(names,0,4);
    char namx0[6];
    sprintf (namx0, "%s", namx.c_str());
    cout << name << " " << names << " " << namx << " " << namx0 << "\n";
    if      (namx == "E1T0") sprintf (title, "E1 (GeV)");
    else if (namx == "E1T1") sprintf (title, "E1 (GeV) (t < 400 ns)");
    else if (namx == "E9T0") sprintf (title, "E9 (GeV)");
    else if (namx == "E9T1") sprintf (title, "E9 (GeV) (t < 400 ns)");
    else                     sprintf (title, "Unknown X");
    TH1F *h0 = new TH1F("Prob", title, nbin, xmin, xmax);
    h0->GetXaxis()->SetTitle(title);
    h0->GetYaxis()->SetTitle("Probability");
    double sum = 0;
    for (int i=1; i<=nbin; i++) {
      double xb = hist->GetBinContent(i);
      sum += xb;
      double yb = (1.-scale*sum);
      double xc = ((i-0.5)*xmax+(nbin-i+0.5)*xmin)/((double)(nbin));
      h0->SetBinContent(i,yb);
      std::cout << i << " x " << xc << " prob " << yb << "\n";
    }
    myc = new TCanvas("Prob","",800,600);
    if (logy) gPad->SetLogy(1);
    h0->Draw();
  }
}

void setTDRStyle() {
  TStyle *tdrStyle = new TStyle("tdrStyle","Style for P-TDR");

// For the canvas:
  tdrStyle->SetCanvasBorderMode(0);
  tdrStyle->SetCanvasColor(kWhite);
  tdrStyle->SetCanvasDefH(600); //Height of canvas
  tdrStyle->SetCanvasDefW(600); //Width of canvas
  tdrStyle->SetCanvasDefX(0);   //POsition on screen
  tdrStyle->SetCanvasDefY(0);

// For the Pad:
  tdrStyle->SetPadBorderMode(0);
  tdrStyle->SetPadColor(kWhite);
  tdrStyle->SetPadGridX(false);
  tdrStyle->SetPadGridY(false);
  tdrStyle->SetGridColor(0);
  tdrStyle->SetGridStyle(3);
  tdrStyle->SetGridWidth(1);

// For the frame:
  tdrStyle->SetFrameBorderMode(0);
  tdrStyle->SetFrameBorderSize(1);
  tdrStyle->SetFrameFillColor(0);
  tdrStyle->SetFrameFillStyle(0);
  tdrStyle->SetFrameLineColor(1);
  tdrStyle->SetFrameLineStyle(1);
  tdrStyle->SetFrameLineWidth(1);

//For the date:
  tdrStyle->SetOptDate(0);

// For the statistics box:
  tdrStyle->SetOptStat(1111111);

// For the Global title:

  tdrStyle->SetOptTitle(0);
  tdrStyle->SetTitleFont(42);
  tdrStyle->SetTitleColor(1);
  tdrStyle->SetTitleTextColor(1);
  tdrStyle->SetTitleFillColor(10);
  tdrStyle->SetTitleFontSize(0.05);

// For the axis titles:

  tdrStyle->SetTitleColor(1, "XYZ");
  tdrStyle->SetTitleFont(42, "XYZ");
  tdrStyle->SetTitleSize(0.04, "XYZ");
  tdrStyle->SetTitleXOffset(0.8);
  tdrStyle->SetTitleYOffset(0.8);

// For the axis labels:

  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(42, "XYZ");
  tdrStyle->SetLabelOffset(0.007, "XYZ");
  tdrStyle->SetLabelSize(0.03, "XYZ");

// For the axis:

  tdrStyle->SetAxisColor(1, "XYZ");
  tdrStyle->SetStripDecimals(kTRUE);
  tdrStyle->SetTickLength(0.03, "XYZ");
  tdrStyle->SetNdivisions(510, "XYZ");
  tdrStyle->SetPadTickX(1); 
  tdrStyle->SetPadTickY(1);

  tdrStyle->cd();

}
