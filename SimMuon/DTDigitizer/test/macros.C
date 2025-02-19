/** 
 * A collection of simple ROOT macros
 *
 * N. Amapane 2002-2004
 */






/*
 * Draw 2-D plots superimposed to their profiles
 *
 * 2003 NCA
 */

// Draw a 2-D plot within the specified Y range and superimpose its X profile
void plotAndProfileX (TH2* h2, float min, float max, bool profile=false) {
  gPad->SetGrid(1,1);
  gStyle->SetGridColor(15);
  h2->GetYaxis()->SetRangeUser(min,max);
  h2->Draw();
  if (profile) {
    TProfile* prof = h2->ProfileX();
    prof->SetMarkerColor(2);
    prof->SetLineColor(2);
    prof->Draw("same");
  }
  TLine * l = new TLine(h2->GetXaxis()->GetXmin(),0,h2->GetXaxis()->GetXmax(),0);
  l->SetLineColor(3);
  l->Draw();
}

// void plotAndProfileY (TH2* h2, float min, float max) {
//   h2->GetYaxis()->SetRangeUser(min,max);
//   h2->Draw();
//   TProfile* prof = h2->ProfileY();
//   prof->SetMarkerStyle(8);
//   prof->SetMarkerSize(0.7);
//   prof->SetMarkerColor(2);
//   prof->SetLineColor(2);
//   prof->Draw("same");
// }







/*
 * Draw and format a fitted histogram 
 *
 * 2003 NCA
 */

// Fit a histogram with a gaussian and draw it in the specified range.
void drawGFit(TH1 * h1, float min, float max){
  drawGFit(h1, min, max, min, max);
  gPad->Draw();
}

// Fit a histogram in the range (minfit, maxfit) with a gaussian and
// draw it in the range (min, max)
void drawGFit(TH1 * h1, float min, float max, float minfit, float maxfit) {
  static int i = 0;
  i++;
  gPad->SetGrid(1,1);
  gStyle->SetGridColor(15);

  h1->GetXaxis()->SetRangeUser(min,max);
  TF1* g1 = new TF1(TString("g")+i,"gaus",minfit,maxfit);
  g1->SetLineColor(2);
  g1->SetLineWidth(2);
  h1->Fit(g1,"R");
//   TPaveStats *st = (TPaveStats*)h1->GetListOfFunctions()->FindObject("stats");
//   st->SetX2NDC(0.905);
//   st->SetY2NDC(0.905);

}







/*
 * Create a new TCanvas setting its properties
 *
 * 2003 NCA 
 */

// Specify name, title, x/y divisions, form or x,y sizes.
// If no name is specified, a new name is generated automatically
TCanvas * newCanvas(TString name="", TString title="",
                     Int_t xdiv=0, Int_t ydiv=0, Int_t form = 1, Int_t w=-1){

  static int i = 1;

  if (name == "") {
    name = TString("Canvas ") + i;
    i++;
  }

  if (title == "") title = name;

  if (w<0) {
    TCanvas * c = new TCanvas(name,title, form);
  } else {
    TCanvas * c = new TCanvas(name,title,form,w);
  }

  if (xdiv*ydiv!=0) c->Divide(xdiv,ydiv);
  c->cd(1);
  return c;
}

// Create a new canvas with an automatic generated name and the specified 
// divisions and form
TCanvas * newCanvas(Int_t xdiv, Int_t ydiv, Int_t form = 1) {
  return newCanvas("","",xdiv,ydiv,form);
}

// Create a new canvas with an automatic generated name and the specified 
// form
TCanvas * newCanvas(Int_t form = 1)
{
  return newCanvas(0,0,form);
}

// ...without specifying the title...
TCanvas * newCanvas(TString name, Int_t xdiv=0, Int_t ydiv=0, Int_t form = 1,
                    Int_t w=-1) {
  return newCanvas(name, name,xdiv,ydiv,form,w);
}

// ...without specifying title and divisions.
TCanvas * newCanvas(TString name, Int_t form = 1, Int_t w=-1)
{
  return newCanvas(name, name, 0,0,form,w);
}






/*
 * Print all open canvases to PS or EPS files.
 *
 * 2003 NCA 
 */

// Print all canvases in a single PS file
void printCanvasesPS(TString name){
  TPostScript * ps = new TPostScript(name,112);
  TIter iter(gROOT->GetListOfCanvases());

  TCanvas *c;

  while( (c = (TCanvas *)iter()) )
    {
      cout << "Printing " << c->GetName() << endl;
      ps->NewPage();
      c->Draw();
    }
  cout << " File " << name << " was created" << endl;
  ps->Close();
}

// Print all canvases in separate EPS files
void printCanvasesEps(){
  TIter iter(gROOT->GetListOfCanvases());
  TCanvas *c;
  while( (c = (TCanvas *)iter()) ) {
    c->Print(0,"eps");
  }
}

// Print all canvases in separate EPS files (another way)
void printCanvasesEps2() {
  gROOT->GetListOfCanvases()->Print("eps");
}


// Print all canvases in separate EPS files
void printCanvases(TString type="eps"){
  TIter iter(gROOT->GetListOfCanvases());
  TCanvas *c;
  while( (c = (TCanvas *)iter()) ) {
    c->Print(0,type);
  }
}






/*
 * Define different TStyles; use them with:
 * getStyle->cd();
 *
 * 2003 NCA
 */

TStyle * getStyle(TString name="myStyle")
{
  TStyle *theStyle;

  if ( name == "myStyle" ) {
    theStyle = new TStyle("myStyle", "myStyle");
    //    theStyle->SetOptStat(0);
    theStyle->SetPadBorderMode(0);
    theStyle->SetCanvasBorderMode(0);
    theStyle->SetPadColor(0);
    theStyle->SetCanvasColor(0);
    theStyle->SetMarkerStyle(8);
    theStyle->SetMarkerSize(0.7);
    //   theStyle->SetTextFont(132);
    //   theStyle->SetTitleFont(132);
    theStyle->SetTitleBorderSize(1);
    theStyle->SetPalette(1);

// } else if { ... 

  } else {
    // Avoid modifying the default style!
    theStyle = gStyle;
  }
  return theStyle;
}
