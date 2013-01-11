
#include <vector>

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TChain.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TPaveText.h"
#include "tdrstyle.C"
#include "TProfile.h"

#include "../../ICHEP_Analysis/Analysis_Global.h"
#include "../../ICHEP_Analysis/Analysis_PlotFunction.h"
#include "../../ICHEP_Analysis/Analysis_Samples.h"

void MakePlot()
{
   setTDRStyle();
   gStyle->SetPadTopMargin   (0.06);
   gStyle->SetPadBottomMargin(0.15);
   gStyle->SetPadRightMargin (0.03);
   gStyle->SetPadLeftMargin  (0.07);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505);

   TCanvas* c1;
   TH1D** Histos = new TH1D*[10];
   std::vector<string> legend;

   TFile* DataFile = new TFile("pictures/Histos_Data12.root");
   TFile* MCFile   = new TFile("pictures/Histos_MC_DYToMuMu_MC_8TeV_DYToMuMu.root");

   TH1D* Data_TOF                = (TH1D*)GetObjectFromPath(DataFile, "TOF");
   TH1D* MC_TOF                  = (TH1D*)GetObjectFromPath(MCFile,   "TOF");

   TH1D* Data_TOFDT                = (TH1D*)GetObjectFromPath(DataFile, "TOFDT");
   TH1D* MC_TOFDT                  = (TH1D*)GetObjectFromPath(MCFile,   "TOFDT");

   TH1D* Data_TOFCSC             = (TH1D*)GetObjectFromPath(DataFile, "TOFCSC");
   TH1D* MC_TOFCSC               = (TH1D*)GetObjectFromPath(MCFile,   "TOFCSC");

   TH1D* Data_Vertex                = (TH1D*)GetObjectFromPath(DataFile, "Vertex");
   TH1D* MC_Vertex                  = (TH1D*)GetObjectFromPath(MCFile,   "Vertex");

   TH1D* Data_VertexDT                = (TH1D*)GetObjectFromPath(DataFile, "VertexDT");
   TH1D* MC_VertexDT                  = (TH1D*)GetObjectFromPath(MCFile,   "VertexDT");

   TH1D* Data_VertexCSC             = (TH1D*)GetObjectFromPath(DataFile, "VertexCSC");
   TH1D* MC_VertexCSC               = (TH1D*)GetObjectFromPath(MCFile,   "VertexCSC");


   TProfile* Data_TOFVsEta                = (TProfile*)GetObjectFromPath(DataFile, "TOFVsEta");
   TProfile* MC_TOFVsEta                  = (TProfile*)GetObjectFromPath(MCFile,   "TOFVsEta");

   TProfile* Data_TOFVsEtaDT                = (TProfile*)GetObjectFromPath(DataFile, "DTTOFVsEta");
   TProfile* MC_TOFVsEtaDT                  = (TProfile*)GetObjectFromPath(MCFile,   "DTTOFVsEta");

   TProfile* Data_TOFVsEtaCSC             = (TProfile*)GetObjectFromPath(DataFile, "CSCTOFVsEta");
   TProfile* MC_TOFVsEtaCSC               = (TProfile*)GetObjectFromPath(MCFile,   "CSCTOFVsEta");

   TProfile* Data_TOFVsPhi                = (TProfile*)GetObjectFromPath(DataFile, "TOFVsPhi");
   TProfile* MC_TOFVsPhi                  = (TProfile*)GetObjectFromPath(MCFile,   "TOFVsPhi");

   TProfile* Data_TOFVsPhiDT                = (TProfile*)GetObjectFromPath(DataFile, "DTTOFVsPhi");
   TProfile* MC_TOFVsPhiDT                  = (TProfile*)GetObjectFromPath(MCFile,   "DTTOFVsPhi");

   TProfile* Data_TOFVsPhiCSC             = (TProfile*)GetObjectFromPath(DataFile, "CSCTOFVsPhi");
   TProfile* MC_TOFVsPhiCSC               = (TProfile*)GetObjectFromPath(MCFile,   "CSCTOFVsPhi");

   TProfile* Data_TOFVsPt                = (TProfile*)GetObjectFromPath(DataFile, "TOFVsPt");
   TProfile* MC_TOFVsPt                  = (TProfile*)GetObjectFromPath(MCFile,   "TOFVsPt");

   TProfile* Data_TOFVsPtDT                = (TProfile*)GetObjectFromPath(DataFile, "DTTOFVsPt"); Data_TOFVsPtDT->Rebin(4);
   TProfile* MC_TOFVsPtDT                  = (TProfile*)GetObjectFromPath(MCFile,   "DTTOFVsPt"); MC_TOFVsPtDT->Rebin(4);

   TProfile* Data_TOFVsPtCSC             = (TProfile*)GetObjectFromPath(DataFile, "CSCTOFVsPt");
   TProfile* MC_TOFVsPtCSC               = (TProfile*)GetObjectFromPath(MCFile,   "CSCTOFVsPt");

   TProfile* Data_VertexVsEta                = (TProfile*)GetObjectFromPath(DataFile, "VertexVsEta");
   TProfile* MC_VertexVsEta                  = (TProfile*)GetObjectFromPath(MCFile,   "VertexVsEta");

   TProfile* Data_VertexVsEtaDT                = (TProfile*)GetObjectFromPath(DataFile, "DTVertexVsEta");
   TProfile* MC_VertexVsEtaDT                  = (TProfile*)GetObjectFromPath(MCFile,   "DTVertexVsEta");

   TProfile* Data_VertexVsEtaCSC             = (TProfile*)GetObjectFromPath(DataFile, "CSCVertexVsEta");
   TProfile* MC_VertexVsEtaCSC               = (TProfile*)GetObjectFromPath(MCFile,   "CSCVertexVsEta");

   TProfile* Data_VertexVsPhi                = (TProfile*)GetObjectFromPath(DataFile, "VertexVsPhi");
   TProfile* MC_VertexVsPhi                  = (TProfile*)GetObjectFromPath(MCFile,   "VertexVsPhi");

   TProfile* Data_VertexVsPhiDT                = (TProfile*)GetObjectFromPath(DataFile, "DTVertexVsPhi");
   TProfile* MC_VertexVsPhiDT                  = (TProfile*)GetObjectFromPath(MCFile,   "DTVertexVsPhi");

   TProfile* Data_VertexVsPhiCSC             = (TProfile*)GetObjectFromPath(DataFile, "CSCVertexVsPhi");
   TProfile* MC_VertexVsPhiCSC               = (TProfile*)GetObjectFromPath(MCFile,   "CSCVertexVsPhi");

   TProfile* Data_VertexVsPt                = (TProfile*)GetObjectFromPath(DataFile, "VertexVsPt");
   TProfile* MC_VertexVsPt                  = (TProfile*)GetObjectFromPath(MCFile,   "VertexVsPt");

   TProfile* Data_VertexVsPtDT                = (TProfile*)GetObjectFromPath(DataFile, "DTVertexVsPt");
   TProfile* MC_VertexVsPtDT                  = (TProfile*)GetObjectFromPath(MCFile,   "DTVertexVsPt");

   TProfile* Data_VertexVsPtCSC             = (TProfile*)GetObjectFromPath(DataFile, "CSCVertexVsPt");
   TProfile* MC_VertexVsPtCSC               = (TProfile*)GetObjectFromPath(MCFile,   "CSCVertexVsPt");

   gStyle->SetOptStat("");
   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOF; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); legend.push_back("Data");
   Histos[1] = MC_TOF;   if(Histos[1]->Integral()>0) Histos[1]->Scale(1./Histos[1]->Integral()); legend.push_back("MC");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "",0.5, 1.5, 0, 0);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOF_Comp");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFDT; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); legend.push_back("Data");
   Histos[1] = MC_TOFDT;   if(Histos[1]->Integral()>0) Histos[1]->Scale(1./Histos[1]->Integral()); legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1", "1/#beta", "", 0.5, 1.5, 0, 0);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFDT_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFCSC; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); legend.push_back("Data");
   Histos[1] = MC_TOFCSC;   if(Histos[1]->Integral()>0) Histos[1]->Scale(1./Histos[1]->Integral()); legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0.5, 1.5, 0, 0);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFCSC_Comp");
   delete c1;

   gStyle->SetOptStat("emr");
   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOF; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); Histos[0]->SetStats(1); legend.push_back("Data");
   Histos[0]->Fit("gaus");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0.5, 1.5, 0, 0);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOF_Comp_DataFit");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = MC_TOF; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); Histos[0]->SetStats(1); legend.push_back("Data");
   Histos[0]->Fit("gaus");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0.5, 1.5, 0, 0);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOF_Comp_MCFit");
   c1->SetLogy(1);
   delete c1;



   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFDT; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); Histos[0]->SetStats(1); legend.push_back("Data");
   Histos[0]->Fit("gaus");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0.5, 1.5, 0, 0);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFDT_Comp_DataFit");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = MC_TOFDT; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); Histos[0]->SetStats(1); legend.push_back("Data");
   Histos[0]->Fit("gaus");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0.5, 1.5, 0, 0);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFDT_Comp_MCFit");
   c1->SetLogy(1);
   delete c1;
   gStyle->SetOptStat("");

   gStyle->SetOptStat("emr");
   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFCSC; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); Histos[0]->SetStats(1); legend.push_back("Data");
   Histos[0]->Fit("gaus");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0.5, 1.5, 0, 0);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFCSC_Comp_DataFit");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = MC_TOFCSC; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); Histos[0]->SetStats(1); legend.push_back("Data");
   Histos[0]->Fit("gaus");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0.5, 1.5, 0, 0);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFCSC_Comp_MCFit");
   c1->SetLogy(1);
   delete c1;
   gStyle->SetOptStat("");


   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_Vertex; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); legend.push_back("Data");
   Histos[1] = MC_Vertex;   if(Histos[1]->Integral()>0) Histos[1]->Scale(1./Histos[1]->Integral()); legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Vertex Time [ns]", "", 0,0, 0, 0);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Vertex_Comp");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_VertexDT; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); legend.push_back("Data");
   Histos[1] = MC_VertexDT;   if(Histos[1]->Integral()>0) Histos[1]->Scale(1./Histos[1]->Integral()); legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Vertex Time [ns]", "", 0,0, 0, 0);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","VertexDT_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_VertexCSC; if(Histos[0]->Integral()>0) Histos[0]->Scale(1./Histos[0]->Integral()); legend.push_back("Data");
   Histos[1] = MC_VertexCSC;   if(Histos[1]->Integral()>0) Histos[1]->Scale(1./Histos[1]->Integral()); legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "P1",  "Vertex Time [ns]", "", 0,0, 0, 0);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","VertexCSC_Comp");
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFVsEta; legend.push_back("Data");
   Histos[1] = MC_TOFVsEta;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#eta", "", 0, 0, 0.7, 1.1);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFVsEta_Comp");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   TProfile* DiffTOFVsEta = (TProfile*)Data_TOFVsEta->Clone("Data_TOFVsEta");
   for(int j=0; j<Data_TOFVsEta->GetNbinsX()+1; j++) {
     DiffTOFVsEta->SetBinContent(j, Data_TOFVsEta->GetBinContent(j)-MC_TOFVsEta->GetBinContent(j));
     double error = sqrt(Data_TOFVsEta->GetBinError(j)*Data_TOFVsEta->GetBinError(j) + MC_TOFVsEta->GetBinError(j)*MC_TOFVsEta->GetBinError(j));
     DiffTOFVsEta->SetBinContent(j, error);
   }

   Histos[0] = DiffTOFVsEta; legend.push_back("");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#eta", "1/#Beta Data - MC", 0, 0, -0.1, 0.1);
   //DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFVsEta_Diff");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFVsEtaDT; legend.push_back("Data");
   Histos[1] = MC_TOFVsEtaDT;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#eta", "", 0, 0, 0.9, 1.1);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFVsEtaDT_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFVsEtaCSC; legend.push_back("Data");
   Histos[1] = MC_TOFVsEtaCSC;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "P1",  "#eta", "", 0, 0, 0.7, 1.1);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFVsEtaCSC_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFVsPt; legend.push_back("Data");
   Histos[1] = MC_TOFVsPt;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "P_{t}", "", 0, 0, 0.7, 1.1);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFVsPt_Comp");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFVsPtDT; legend.push_back("Data");
   Histos[1] = MC_TOFVsPtDT;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "P_{t}", "", 0, 0, 0.9, 1.1);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFVsPtDT_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFVsPtCSC; legend.push_back("Data");
   Histos[1] = MC_TOFVsPtCSC;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "P1",  "P_{t}", "", 0, 0, 0.7, 1.1);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFVsPtCSC_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFVsPhi; legend.push_back("Data");
   Histos[1] = MC_TOFVsPhi;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#phi", "", 0, 0, 0.7, 1.1);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFVsPhi_Comp");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFVsPhiDT; legend.push_back("Data");
   Histos[1] = MC_TOFVsPhiDT;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "#phi", "", 0, 0, 0.9, 1.1);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFVsPhiDT_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_TOFVsPhiCSC; legend.push_back("Data");
   Histos[1] = MC_TOFVsPhiCSC;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "P1",  "#phi", "", 0, 0, 0.7, 1.1);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","TOFVsPhiCSC_Comp");
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_VertexVsEta; legend.push_back("Data");
   Histos[1] = MC_VertexVsEta;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0,0, -6, 2);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","VertexVsEta_Comp");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_VertexVsEtaDT; legend.push_back("Data");
   Histos[1] = MC_VertexVsEtaDT;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0,0, -6, 2);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","VertexVsEtaDT_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_VertexVsEtaCSC; legend.push_back("Data");
   Histos[1] = MC_VertexVsEtaCSC;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "P1",  "1/#beta", "", 0,0, -6, 2);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","VertexVsEtaCSC_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_VertexVsPt; legend.push_back("Data");
   Histos[1] = MC_VertexVsPt;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0,0, -6, 2);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","VertexVsPt_Comp");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_VertexVsPtDT; legend.push_back("Data");
   Histos[1] = MC_VertexVsPtDT;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0,0, -6, 2);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","VertexVsPtDT_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_VertexVsPtCSC; legend.push_back("Data");
   Histos[1] = MC_VertexVsPtCSC;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "P1",  "1/#beta", "", 0,0, -6, 2);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","VertexVsPtCSC_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_VertexVsPhi; legend.push_back("Data");
   Histos[1] = MC_VertexVsPhi;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0,0, -6, 2);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","VertexVsPhi_Comp");
   c1->SetLogy(1);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_VertexVsPhiDT; legend.push_back("Data");
   Histos[1] = MC_VertexVsPhiDT;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "", 0,0, -6, 2);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","VertexVsPhiDT_Comp");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600); legend.clear();
   Histos[0] = Data_VertexVsPhiCSC; legend.push_back("Data");
   Histos[1] = MC_VertexVsPhiCSC;   legend.push_back("MC");

   DrawSuperposedHistos((TH1**)Histos, legend, "P1",  "1/#beta", "", 0,0, -6, 2);
   DrawLegend((TObject**)Histos,legend,"","P");
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","VertexVsPhiCSC_Comp");
   delete c1;


}


