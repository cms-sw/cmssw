#include "Riostream.h"
#include "TROOT.h"
#include "TClass.h"
#include "TLatex.h"
#include "TMath.h"
#include "TVirtualPad.h"
#include "TVirtualPS.h"
#include "TLegend.h"
//------------------------------------------------------------------------------
// Setting the Style of TH1F
//------------------------------------------------------------------------------
void SetStyleh1(TH1*   histo,
		int    fillColor  = 1,
		int    fillStyle  = 0,
		int    lineWidth  = 1,
		//char*  xname       = "",
		//char*  yname       = "",
                TString  xname       = "",
		TString  yname       = "",
		double xmin       = 0,
		double xmax       = 0) {

  histo -> SetFillColor(fillColor);
  histo -> SetFillStyle(fillStyle);
  histo -> SetLineColor(fillColor);
  histo -> SetLineWidth(lineWidth);
  histo -> GetXaxis()->SetTitle(xname);
  histo -> GetXaxis()->SetTitleOffset(1.1);
  histo -> GetYaxis()->SetTitle(yname);
  histo -> GetYaxis()->SetTitleOffset(1.2);

  if (xmin && xmax)
    histo->GetXaxis()->SetRangeUser(xmin,xmax);

  return;

}

//------------------------------------------------------------------------------
// Setting the Style of TH2F
//------------------------------------------------------------------------------
void SetStyleh2(TH2*   histo,
		int    markerColor = 1,
		int    markerSize  = 1,
		//char*  xname       = "",
		//char*  yname       = "",
                TString  xname       = "",
		TString  yname       = "",
		double xmin        = 0,
		double xmax        = 0,
		double ymin        = 0,
		double ymax        = 0) {

  histo -> SetMarkerColor(markerColor);
  histo -> SetMarkerSize(markerSize);
  histo -> GetXaxis()->SetTitle(xname);
  histo -> GetXaxis()->SetTitleOffset(1.1);
  histo -> GetYaxis()->SetTitle(yname);
  histo -> GetYaxis()->SetTitleOffset(1.2);
  if (xmin && xmax)
    histo->GetXaxis()->SetRangeUser(xmin,xmax);
  if (ymin && ymax)
    histo->GetYaxis()->SetRangeUser(ymin,ymax);

  return;

}

//------------------------------------------------------------------------------
// Print the title of a histogram
//------------------------------------------------------------------------------
void PrintIt(TPad *pad, TString title)//char *title)
{
  TLatex *latex = new TLatex();
  latex->SetTextFont(  42);
  latex->SetTextSize(0.05);

  // Get the most recent changes
  pad->Update();

  double xmin = pad->GetUxmin();
  double xmax = pad->GetUxmax();
  double ymin = pad->GetUymin();
  double ymax = pad->GetUymax();
 
  double xpos = xmin + 0.50*(xmax - xmin);
  double ypos = ymax + 0.05*(ymax - ymin);

 

  latex->SetTextAlign(22);
  latex->DrawLatex(xpos,ypos,title);
}


void PrintItLog(TPad *pad, TString title)//char *title)
{
  TLatex *latex = new TLatex();
  latex->SetTextFont(  42);
  latex->SetTextSize(0.05);

  // Get the most recent changes
  pad->Update();

  double xmin = pad->GetUxmin();
  double xmax = pad->GetUxmax();
  //double ymin = pad->GetUymin();
  double ymax = pad->GetUymax();

  //cout << "xmin = " << xmin << endl;
  //cout << "xmax = " << xmax << endl;
  //cout << "ymin = " << ymin << endl;
  //cout << "ymax = " << ymax << endl;
  //cout << "TMath::Power(10,ymax) = " << TMath::Power(10,ymax) << endl; 

  double xpos = xmin + 0.50*(xmax - xmin);
  double ypos = 1.5*TMath::Power(10,ymax);// + TMath::*(ymax - ymin);
  
  latex->SetTextAlign(22);
  latex->DrawLatex(xpos,ypos,title);
}

//------------------------------------------------------------------------------
// Set Margin
//------------------------------------------------------------------------------
void SetMargin(TPad* pad,
	       double left   = 0.149103, 
	       double right  = 0.153587,
	       double bottom = 0.13    ,
	       double top    = 0.08    ) 
{

  pad -> SetLeftMargin  (left  );
  pad -> SetRightMargin (right );
  pad -> SetBottomMargin(bottom);
  pad -> SetTopMargin   (top   ); 


}

//------------------------------------------------------------------------------
// Set TLegend
//------------------------------------------------------------------------------
TLegend *SetLegend(float x0, float y0, float x1, float y1) {
  TLegend *leg = new TLegend(x0,y0,x1,y1);
  leg->SetFillStyle (0);            
  leg->SetBorderSize(0);
  leg->SetTextFont(42);
  leg->SetTextSize(0.035);
  return leg;
}

//------------------------------------------------------------------------------
// SequentialLabel
//------------------------------------------------------------------------------
void SequentialLabelX(TH1* histo,
		      int iStart=0) { 

  for (int iBin=0; iBin < histo->GetNbinsX(); iBin++){
    char title[200];
    sprintf(title,"%d",iStart);
    histo->GetXaxis()->SetBinLabel(iBin+1,title);
    iStart++;
  }
}

void SequentialLabelY(TH1* histo,
		      int iStart=0) { 

  for (int iBin=0; iBin < histo->GetNbinsY(); iBin++){
    char title[200];
    sprintf(title,"%d",iStart);
    histo->GetYaxis()->SetBinLabel(iBin+1,title);
    iStart++;
  }
}


// //------------------------------------------------------------------------------
// // Compare Two TH1 histos
// //------------------------------------------------------------------------------
// void compareHistos(TH1F* histo1,
//                    TH1F* histo2,
//                    TCanvas* canvas,
//                    bool isNorm = false) {
// 
//   if (!histo1->GetEntries()){
//     cout << histo1->GetName() << " has no entries: do not draw it\n";
//     return;
//   }
// 
//   if (!histo2->GetEntries()){
//     cout << histo2->GetName() << " has no entries: do not draw it \n";
//     return;
//   }
// 
//   if (histo1->GetMaximum()/histo1->GetEntries() > histo2->GetMaximum()/histo2->GetEntries()){
//     if (isNorm) {
//       histo1->DrawNormalized("",    1);
// 
//       // get stats box
//       canvas->Update(); 
//       TPaveStats *statsbox = (TPaveStats*)canvas->GetPrimitive("stats");
//       // get (default) y position of stats box (histo h1)
//       double y1 = statsbox->GetY1NDC(); // (lower) y start position of stats box
//       double y2 = statsbox->GetY2NDC(); // (upper) y start position of stats box
//    
//       // set new position of stats box (histo h1)
//       double newy1 = 2 * y1 - y2;   // new (lower) y start position of stats box
//       double newy2 = y1;            // new (upper) y start position of stats box
//    
//       statsbox->SetY1NDC(newy1);    //set new y start position
//       statsbox->SetY2NDC(newy2);    //set new y end position
//    
//       statsbox->SetTextColor(histo2->GetFillColor());
// 
//       histo2->DrawNormalized("sames",1);
//     }
//     else {
//       histo1->Draw();
// 
//       // get stats box
//       canvas->Update(); 
//       TPaveStats *statsbox = (TPaveStats*)canvas->GetPrimitive("stats");
// 
//       // get (default) y position of stats box (histo h1)
//       double y1 = statsbox->GetY1NDC(); // (lower) y start position of stats box
//       double y2 = statsbox->GetY2NDC(); // (upper) y start position of stats box
// 
//       // set new position of stats box (histo h1)
//       double newy1 = 2 * y1 - y2;   // new (lower) y start position of stats box
//       double newy2 = y1;            // new (upper) y start position of stats box
//       statsbox->SetY1NDC(newy1);    //set new y start position
//       statsbox->SetY2NDC(newy2);    //set new y end position
// 
//       statsbox->SetTextColor(histo2->GetFillColor());
// 
//       histo2->Draw("sames");
//     }
//   }
//   else {
//     if (isNorm) {
//       histo2->DrawNormalized("",    1);
// 
//       // get stats box
//       canvas->Update(); 
//       TPaveStats *statsbox = (TPaveStats*)canvas->GetPrimitive("stats");
// 
//       // get (default) y position of stats box (histo h1)
//       double y1 = statsbox->GetY1NDC(); // (lower) y start position of stats box
//       double y2 = statsbox->GetY2NDC(); // (upper) y start position of stats box
// 
//       // set new position of stats box (histo h1)
//       double newy1 = 2 * y1 - y2;   // new (lower) y start position of stats box
//       double newy2 = y1;            // new (upper) y start position of stats box
//       statsbox->SetY1NDC(newy1);    //set new y start position
//       statsbox->SetY2NDC(newy2);    //set new y end position
// 
//       statsbox->SetTextColor(histo2->GetFillColor());
// 
//       histo1->DrawNormalized("sames",1);
//     }
//     else {
//       histo2->Draw();
// 
//       // get stats box
//       canvas->Update(); 
//       TPaveStats *statsbox = (TPaveStats*)canvas->GetPrimitive("stats");
// 
//       // get (default) y position of stats box (histo h1)
//       double y1 = statsbox->GetY1NDC(); // (lower) y start position of stats box
//       double y2 = statsbox->GetY2NDC(); // (upper) y start position of stats box
// 
//       // set new position of stats box (histo h1)
//       double newy1 = 2 * y1 - y2;   // new (lower) y start position of stats box
//       double newy2 = y1;            // new (upper) y start position of stats box
//       statsbox->SetY1NDC(newy1);    //set new y start position
//       statsbox->SetY2NDC(newy2);    //set new y end position
// 
//       statsbox->SetTextColor(histo2->GetFillColor());
// 
//       histo1->Draw("sames");
//     }
//   }
// }
