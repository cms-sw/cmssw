#include "TStyle.h"
#include <string>

void setTDRStyle();

void plotValid(char name[10]="Hit34", int logy=0, int mode=0, 
	       float xoff=0.55, float yoff=0.15, int save=0) {

  char title[80];
  if      (name == "Hit01") sprintf (title, "Number of Hits in HCal");
  else if (name == "Hit02") sprintf (title, "Hits with wrong Detector ID"); 
  else if (name == "Hit03") sprintf (title, "Hits with wrong Subdet");
  else if (name == "Hit04") sprintf (title, "Hits with wrong ID");
  else if (name == "Hit05") sprintf (title, "Number of Hits in HB");
  else if (name == "Hit06") sprintf (title, "Number of Hits in HE");
  else if (name == "Hit07") sprintf (title, "Number of Hits in HO");
  else if (name == "Hit08") sprintf (title, "Number of Hits in HF");
  else if (name == "Hit09") sprintf (title, "Detector ID");
  else if (name == "Hit10") sprintf (title, "Subdetectors in HCal");
  else if (name == "Hit11") sprintf (title, "Depths in HCal");
  else if (name == "Hit12") sprintf (title, "#eta in HCal");
  else if (name == "Hit13") sprintf (title, "#phi in HCal");
  else if (name == "Hit14") sprintf (title, "Energy in HCal");
  else if (name == "Hit15") sprintf (title, "Time in HCal");
  else if (name == "Hit16") sprintf (title, "Time in HCal (E wtd)");
  else if (name == "Hit17") sprintf (title, "Depths in HB");
  else if (name == "Hit18") sprintf (title, "Depths in HE");
  else if (name == "Hit19") sprintf (title, "Depths in HO");
  else if (name == "Hit20") sprintf (title, "Depths in HF");
  else if (name == "Hit21") sprintf (title, "#eta in HB");
  else if (name == "Hit22") sprintf (title, "#eta in HE");
  else if (name == "Hit23") sprintf (title, "#eta in HO");
  else if (name == "Hit24") sprintf (title, "#eta in HF");
  else if (name == "Hit25") sprintf (title, "#phi in HB");
  else if (name == "Hit26") sprintf (title, "#phi in HE");
  else if (name == "Hit27") sprintf (title, "#phi in HO");
  else if (name == "Hit28") sprintf (title, "#phi in HF");
  else if (name == "Hit29") sprintf (title, "Energy in HB");
  else if (name == "Hit30") sprintf (title, "Energy in HE");
  else if (name == "Hit31") sprintf (title, "Energy in HO");
  else if (name == "Hit32") sprintf (title, "Energy in HF");
  else if (name == "Hit33") sprintf (title, "Time in HB");
  else if (name == "Hit34") sprintf (title, "Time in HE");
  else if (name == "Hit35") sprintf (title, "Time in HO");
  else if (name == "Hit36") sprintf (title, "Time in HF");
  else if (name == "Lay01") sprintf (title, "Layer number of the Hits");
  else if (name == "Lay02") sprintf (title, "#eta of the Hits");
  else if (name == "Lay03") sprintf (title, "#phi of the Hits");
  else if (name == "Lay04") sprintf (title, "Energy of the Hits");
  else if (name == "Lay05") sprintf (title, "Depth  of the Hits");
  else if (name == "Lay06") sprintf (title, "Time of the Hits");
  else if (name == "Lay07") sprintf (title, "Time (wtd) of Hits");
  else if (name == "Lay08") sprintf (title, "#phi vs. #eta of the Hits");
  else if (name == "Lay09") sprintf (title, "Hit in Ecal");
  else if (name == "Lay10") sprintf (title, "Hit in Hcal");
  else if (name == "Lay11") sprintf (title, "Total Hits");
  else if (name == "Lay12") sprintf (title, "Energy per layer");
  else if (name == "Lay13") sprintf (title, "Lonitudinal Shower Profile");
  else if (name == "Lay14") sprintf (title, "Energy per depth");
  else if (name == "Lay15") sprintf (title, "Total Energy");
  else if (name == "Lay16") sprintf (title, "Energy in HO");
  else if (name == "Lay17") sprintf (title, "Energy in HB/HE");
  else if (name == "Lay18") sprintf (title, "Energy in HF (Long)");
  else if (name == "Lay19") sprintf (title, "Energy in HF (Short)");
  else if (name == "Lay20") sprintf (title, "EM   energy in HF");
  else if (name == "Lay21") sprintf (title, "Had. energy in HF");
  else if (name == "Layl0") sprintf (title, "Energy deposit in Layer 1");
  else if (name == "Layl1") sprintf (title, "Energy deposit in Layer 2");
  else if (name == "Layl2") sprintf (title, "Energy deposit in Layer 3");
  else if (name == "Layl3") sprintf (title, "Energy deposit in Layer 4");
  else if (name == "Layl4") sprintf (title, "Energy deposit in Layer 5");
  else if (name == "Layl5") sprintf (title, "Energy deposit in Layer 6");
  else if (name == "Layl6") sprintf (title, "Energy deposit in Layer 7");
  else if (name == "Layl7") sprintf (title, "Energy deposit in Layer 8");
  else if (name == "Layl8") sprintf (title, "Energy deposit in Layer 9");
  else if (name == "Layl9") sprintf (title, "Energy deposit in Layer 10");
  else if (name == "Layl10")sprintf (title, "Energy deposit in Layer 11");
  else if (name == "Layl11")sprintf (title, "Energy deposit in Layer 12");
  else if (name == "Layl12")sprintf (title, "Energy deposit in Layer 13");
  else if (name == "Layl13")sprintf (title, "Energy deposit in Layer 14");
  else if (name == "Layl14")sprintf (title, "Energy deposit in Layer 15");
  else if (name == "Layl15")sprintf (title, "Energy deposit in Layer 16");
  else if (name == "Layl16")sprintf (title, "Energy deposit in Layer 17");
  else if (name == "Layl17")sprintf (title, "Energy deposit in Layer 18");
  else if (name == "Layl18")sprintf (title, "Energy deposit in Layer 19");
  else if (name == "Layl19")sprintf (title, "Energy deposit in Layer 20");
  else if (name == "Layl20")sprintf (title, "Energy deposit in Depth 1");
  else if (name == "Layl21")sprintf (title, "Energy deposit in Depth 2");
  else if (name == "Layl22")sprintf (title, "Energy deposit in Depth 3");
  else if (name == "Layl23")sprintf (title, "Energy deposit in Depth 4");
  else if (name == "Layl24")sprintf (title, "Energy deposit in Depth 5");
  else if (name == "NxN01") sprintf (title, "Energy in ECal (NxN)r");
  else if (name == "NxN02") sprintf (title, "Energy in HCal (NxN)r");
  else if (name == "NxN03") sprintf (title, "Energy in HO (NxN)r");
  else if (name == "NxN04") sprintf (title, "Energy Total (NxN)r");
  else if (name == "NxN05") sprintf (title, "Energy in ECal (NxN)");
  else if (name == "NxN06") sprintf (title, "Energy in HCal (NxN)");
  else if (name == "NxN07") sprintf (title, "Energy in HO (NxN)");
  else if (name == "NxN08") sprintf (title, "Energy Total (NxN)");
  else if (name == "NxN09") sprintf (title, "Energy of Hits in (NxN)");
  else if (name == "NxN10") sprintf (title, "Time   of Hits in (NxN)");
  else if (name == "NxN11") sprintf (title, "Dist.  of Hits in (NxN)");
  else if (name == "Jet01") sprintf (title, "R of Hits");
  else if (name == "Jet02") sprintf (title, "T of Hits");
  else if (name == "Jet03") sprintf (title, "E of Hits");
  else if (name == "Jet04") sprintf (title, "Ecal Energy (First Jet)");
  else if (name == "Jet05") sprintf (title, "Hcal Energy (First Jet)");
  else if (name == "Jet06") sprintf (title, "Ho   Energy (First Jet)");
  else if (name == "Jet07") sprintf (title, "Total Energy(First Jet)");
  else if (name == "Jet08") sprintf (title, "Energy in Hcal vs  Ecal");
  else if (name == "Jet09") sprintf (title, "Delta #eta");
  else if (name == "Jet10") sprintf (title, "Delta #phi");
  else if (name == "Jet11") sprintf (title, "Delta R");
  else if (name == "Jet12") sprintf (title, "Di-jet mass");
  else if (name == "Jet13") sprintf (title, "Jet Energy");
  else if (name == "Jet14") sprintf (title, "Jet #eta");
  else if (name == "Jet15") sprintf (title, "Jet #phi");
  else                      sprintf (title, "Unknown");

  if (title != "Unknown") {
    char file1[50], file2[50], file3[50], file4[50], file5[50], file6[50];
    char lego1[12], lego2[12], lego3[12], lego4[12], lego5[20], lego6[20];
    sprintf (file1, "valid_HF7.root"); sprintf (lego1, "CMSSW_1_3_3");
    sprintf (file2, "valid_HF8.root"); sprintf (lego2, "CMSSW_1_4_1");
    sprintf (file3, "valid_HF5.root"); sprintf (lego3, "Old Format");
    sprintf (file4, "valid_HF6.root"); sprintf (lego4, "New Format");
    sprintf (file5, "valid_HF9.root"); sprintf (lego5, "G3 File");
    sprintf (file6, "valid_HF0.root"); sprintf (lego6, "G3 File (old)");
    char fileps[50];
    sprintf (fileps, "%s.eps", name);
    cout << fileps << "\n";

    setTDRStyle();
    TCanvas *myc = new TCanvas("myc","",800,600);
    if (logy > 0) gPad->SetLogy(1);

    TFile* File1 = new TFile(file1);
    TDirectory *d1 = (TDirectory*)File1->Get("DQMData/HcalHitValidation");
    TH1F* hist1 = (TH1F*) d1->Get(name);
    hist1->SetLineStyle(1);
    hist1->SetLineWidth(3);
    hist1->SetLineColor(2);
    TFile* File2 = new TFile(file2);
    TDirectory *d2 = (TDirectory*)File2->Get("DQMData/HcalHitValidation");
    TH1F* hist2 = (TH1F*) d2->Get(name);
    hist2->SetLineStyle(2);
    hist2->SetLineWidth(3);
    hist2->SetLineColor(4);
    TFile* File3 = new TFile(file3);
    TDirectory *d3 = (TDirectory*)File3->Get("DQMData/HcalHitValidation");
    TH1F* hist3 = (TH1F*) d3->Get(name);
    hist3->SetLineStyle(3);
    hist3->SetLineWidth(3);
    hist3->SetLineColor(8);
    TFile* File4 = new TFile(file4);
    TDirectory *d4 = (TDirectory*)File4->Get("DQMData/HcalHitValidation");
    TH1F* hist4 = (TH1F*) d4->Get(name);
    hist4->SetLineStyle(4);
    hist4->SetLineWidth(3);
    hist4->SetLineColor(1);
    TFile* File5 = new TFile(file5);
    TDirectory *d5 = (TDirectory*)File5->Get("DQMData/HcalHitValidation");
    TH1F* hist5 = (TH1F*) d5->Get(name);
    hist5->SetLineStyle(5);
    hist5->SetLineWidth(3);
    hist5->SetLineColor(6);
    TFile* File6 = new TFile(file6);
    TDirectory *d6 = (TDirectory*)File6->Get("DQMData/HcalHitValidation");
    TH1F* hist6 = (TH1F*) d6->Get(name);
    hist6->SetLineStyle(6);
    hist6->SetLineWidth(3);
    hist6->SetLineColor(3);

    if (mode == 1) {
      hist1->GetXaxis()->SetTitle(title);
      hist1->Draw();
      hist2->Draw("same");
      hist3->Draw("same");
      hist4->Draw("same");
      hist5->Draw("same");
      hist6->Draw("same");
    } else if (mode == 2) {
      hist2->GetXaxis()->SetTitle(title);
      hist2->Draw();
      hist1->Draw("same");
      hist3->Draw("same");
      hist4->Draw("same");
      hist5->Draw("same");
      hist6->Draw("same");
    } else if (mode == 3) {
      hist3->GetXaxis()->SetTitle(title);
      hist3->Draw();
      hist1->Draw("same");
      hist2->Draw("same");
      hist4->Draw("same");
      hist5->Draw("same");
      hist6->Draw("same");
    } else if (mode == 5) {
      hist5->GetXaxis()->SetTitle(title);
      hist5->Draw();
      hist1->Draw("same");
      hist2->Draw("same");
      hist3->Draw("same");
      hist4->Draw("same");
      hist6->Draw("same");
    } else if (mode == 6) {
      hist6->GetXaxis()->SetTitle(title);
      hist6->Draw();
      hist1->Draw("same");
      hist2->Draw("same");
      hist3->Draw("same");
      hist4->Draw("same");
      hist5->Draw("same");
    } else if (mode < 1) {
      hist4->GetXaxis()->SetTitle(title);
      hist4->Draw();
      hist1->Draw("same");
      hist2->Draw("same");
      hist3->Draw("same");
    } else {
      hist4->GetXaxis()->SetTitle(title);
      hist4->Draw();
      hist1->Draw("same");
      hist2->Draw("same");
      hist3->Draw("same");
      hist5->Draw("same");
      hist6->Draw("same");
    }

    float xmax = xoff+0.3;
    float ymax = yoff+0.2;
    leg1 = new TLegend(xoff,yoff,xmax,ymax);
    leg1->AddEntry(hist1,lego1,"F");
    leg1->AddEntry(hist2,lego2,"F");
    leg1->AddEntry(hist3,lego3,"F");
    leg1->AddEntry(hist4,lego4,"F");
    if (mode > 0) {
      leg1->AddEntry(hist5,lego5,"F");
      leg1->AddEntry(hist6,lego6,"F");
    }
    leg1->Draw();

    if (save > 0) myc->SaveAs(fileps);

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
  // tdrStyle->SetPadBorderSize(Width_t size = 1);
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
  // tdrStyle->SetDateX(Float_t x = 0.01);
  // tdrStyle->SetDateY(Float_t y = 0.01);

// For the statistics box:
  tdrStyle->SetOptFile(0);
  tdrStyle->SetOptStat(0); // To display the mean and RMS:   SetOptStat("mr");

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
  tdrStyle->SetTitleSize(0.06, "XYZ");
  // tdrStyle->SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
  // tdrStyle->SetTitleYSize(Float_t size = 0.02);
  tdrStyle->SetTitleXOffset(0.7);
  tdrStyle->SetTitleYOffset(0.7);
  // tdrStyle->SetTitleOffset(1.1, "Y"); // Another way to set the Offset

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
  tdrStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  tdrStyle->SetPadTickY(1);

  tdrStyle->cd();

}
