#include "root_prefs.h"

void initStyle( TStyle * sty )
{
  sty->SetCanvasBorderMode(0);
  sty->SetOptStat(1110);
  sty->SetStatW(0.22);
  sty->SetStatH(0.17);
  sty->SetStatX(0.993);
  sty->SetStatY(0.993);
  sty->SetStatFormat("6.4g");
  sty->SetOptStat("emuo");
  sty->SetStatColor(0);
  sty->SetTitleBorderSize(1);
  sty->SetTitleOffset(1, "Y");
  sty->SetTitleOffset(0.8, "X");
  sty->SetTitleSize(0.05, "XY");
  sty->SetNdivisions(507, "XY");
  sty->SetPadBorderMode(0);
  sty->SetPadBottomMargin(0.13);
  sty->SetPadLeftMargin(0.1);
  sty->SetPadRightMargin(0.18);
  sty->SetPadTopMargin(0.1);
  sty->SetFrameBorderMode(0);
  sty->SetPadTickX(1);
  sty->SetPadTickY(1);
  sty->SetOptLogy(0);
  sty->SetPadColor(0);
  sty->SetTitleFillColor(0);
  sty->SetFrameBorderSize(0);
  sty->SetPadBorderSize(0);
  sty->SetCanvasBorderMode(0);
  sty->SetPadTickX(1);
  sty->SetPadTickY(1);
  sty->SetOptLogy(0);
  sty->SetPadColor(0);
  sty->SetTitleFillColor(0);
  sty->SetFrameBorderSize(0);
  sty->SetPadBorderSize(0);
  sty->SetCanvasBorderMode(0);
  sty->SetCanvasColor(0);
  sty->SetCanvasDefW(616);
  sty->SetCanvasDefH(820);
  sty->SetPalette(1);
}

void SetupTowerDisplay(TH2F *hist)
{
  hist->SetStats(kFALSE);
  hist->SetXTitle("ieta");
  hist->SetYTitle("iphi");
  hist->GetXaxis()->CenterTitle();
  hist->GetYaxis()->CenterTitle();
  hist->GetXaxis()->SetNdivisions(65);
  hist->GetXaxis()->SetLabelColor(0);
  hist->GetXaxis()->SetTickLength(.78);
  hist->GetXaxis()->SetTitleOffset(0.95);
  hist->GetYaxis()->SetNdivisions(72);
  hist->GetYaxis()->SetLabelColor(0);
  hist->GetYaxis()->SetTitleOffset(0.85);
  TText *yLabel = new TText();
  TLine *pLine = new TLine();
  pLine->SetLineStyle(1);
  pLine->SetLineColor(1);
  pLine->SetLineWidth(1);
  yLabel->SetTextAlign(22);
  yLabel->SetTextSize(0.015);
  char phi_num[3];
  char eta_num[3];
  TText *xLabel = new TText();
  xLabel->SetTextSize(0.015);
  xLabel->SetTextAlign(22);
  for (Int_t i=1; i<73; ++i)
    {
      sprintf(phi_num,"%d",i);
      if(TMath::Abs(i%2)==1) {yLabel->DrawText(-33,0.5+i,phi_num);}
      else {yLabel->DrawText(-34.5,0.5+i,phi_num);}
      pLine->DrawLine(-32,i,33,i);
    }
  for (Int_t i=-32; i<33;++i)
    {
      sprintf(eta_num,"%d",i);
      if(TMath::Abs(i%2)==0) {xLabel->DrawText(0.5+i,-0.5,eta_num);}
      else {xLabel->DrawText(0.5+i,-2,eta_num);}
      pLine->DrawLine(i,1,i,72);
    }
}

void SetStatus(TH1F* hist, string status)
{
  hist->SetFillStyle(1001);
  cout << "Status = " << status;
  if (status=="GOOD")
    {
      hist->SetFillColor(kGreen);
      cout << "\nshould be green\n";
    }
  else if (status=="BAD")
    {
      hist->SetFillColor(kRed);
      cout << "\nshould be red\n";
    }
  else if (status=="UNCHECKED")
    {
      hist->SetFillColor(kBlue);
      cout << "\nshould be blue\n";
    }
  else
    {
      hist->SetFillColor(14);
      cout << "\nshould be grey\n";
    }
}

void SetupTitle(TH1F* hist, char* xtitle, char* ytitle)
{
  hist->GetXaxis()->SetTitle(xtitle);
  hist->GetXaxis()->CenterTitle();
  hist->GetYaxis()->SetTitle(ytitle);
  hist->GetYaxis()->CenterTitle();
}  

void SetupTitle(TProfile* hist, char* xtitle, char* ytitle)
{
  hist->GetXaxis()->SetTitle(xtitle);
  hist->GetXaxis()->CenterTitle();
  hist->GetYaxis()->SetTitle(ytitle);
  hist->GetYaxis()->CenterTitle();
}

void SetupTitle(TH2F* hist, char* xtitle, char* ytitle)
{
  hist->GetXaxis()->SetTitle(xtitle);
  hist->GetXaxis()->CenterTitle();
  hist->GetYaxis()->SetTitle(ytitle);
  hist->GetYaxis()->CenterTitle();
}

index_map::index_map()
{
  ifstream in;
  in.open("index_map.dat");
  if  (!in)
    {
      cerr << "Unable to open index_map.dat\n";
      exit(1);
    }
  int tpgindex, vecindex;
  while (in >> tpgindex >> vecindex)
    {
      ntpg2nvec[tpgindex] = vecindex;
      nvec2ntpg[vecindex] = tpgindex;
    }
  in.close();
}

index_map::~index_map() {}

