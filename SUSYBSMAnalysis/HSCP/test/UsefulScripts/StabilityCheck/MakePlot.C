
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
   TObject** Histos = new TObject*[10];
   std::vector<string> legend;

   TFile* InputFile = new TFile("pictures/Histos.root");

   TProfile* Any_PtProf                = (TProfile*)GetObjectFromPath(InputFile, "AnyPtProf");
   TProfile* SingleMu_PtProf           = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMuPtProf");
//   TProfile* DoubleMu_PtProf           = (TProfile*)GetObjectFromPath(InputFile, "HscpPathDoubleMuPtProf");
   TProfile* PFMet_PtProf              = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetPtProf");
//   TProfile* CaloMet_PtProf            = (TProfile*)GetObjectFromPath(InputFile, "HscpPathCaloMetPtProf");

   TProfile* Any_dEdxProf              = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxProf");
   TProfile* SingleMu_dEdxProf         = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxProf");
//   TProfile* DoubleMu_dEdxProf         = (TProfile*)GetObjectFromPath(InputFile, "HscpPathDoubleMudEdxProf");
   TProfile* PFMet_dEdxProf            = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetdEdxProf");
//   TProfile* CaloMet_dEdxProf          = (TProfile*)GetObjectFromPath(InputFile, "HscpPathCaloMetdEdxProf");

   TProfile* Any_dEdxMProf              = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxMProf");
   TProfile* SingleMu_dEdxMProf         = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxMProf");
//   TProfile* DoubleMu_dEdxMProf         = (TProfile*)GetObjectFromPath(InputFile, "HscpPathDoubleMudEdxMProf");
   TProfile* PFMet_dEdxMProf            = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetdEdxMProf");
//   TProfile* CaloMet_dEdxMProf          = (TProfile*)GetObjectFromPath(InputFile, "HscpPathCaloMetdEdxMProf");


   TProfile* Any_TOFProf               = (TProfile*)GetObjectFromPath(InputFile, "AnyTOFProf");
   TProfile* SingleMu_TOFProf          = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMuTOFProf");
//   TProfile* DoubleMu_TOFProf          = (TProfile*)GetObjectFromPath(InputFile, "HscpPathDoubleMuTOFProf");
   TProfile* PFMet_TOFProf             = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetTOFProf");
//   TProfile* CaloMet_TOFProf           = (TProfile*)GetObjectFromPath(InputFile, "HscpPathCaloMetTOFProf");


   TProfile* Any_TOFDTProf               = (TProfile*)GetObjectFromPath(InputFile, "AnyTOFDTProf");
   TProfile* SingleMu_TOFDTProf          = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMuTOFDTProf");
//   TProfile* DoubleMu_TOFDTProf          = (TProfile*)GetObjectFromPath(InputFile, "HscpPathDoubleMuTOFDTProf");
   TProfile* PFMet_TOFDTProf             = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetTOFDTProf");
//   TProfile* CaloMet_TOFDTProf           = (TProfile*)GetObjectFromPath(InputFile, "HscpPathCaloMetTOFDTProf");


   TProfile* Any_TOFCSCProf               = (TProfile*)GetObjectFromPath(InputFile, "AnyTOFCSCProf");
   TProfile* SingleMu_TOFCSCProf          = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMuTOFCSCProf");
//   TProfile* DoubleMu_TOFCSCProf          = (TProfile*)GetObjectFromPath(InputFile, "HscpPathDoubleMuTOFCSCProf");
   TProfile* PFMet_TOFCSCProf             = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetTOFCSCProf");
//   TProfile* CaloMet_TOFCSCProf           = (TProfile*)GetObjectFromPath(InputFile, "HscpPathCaloMetTOFCSCProf");



   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_PtProf;                         legend.push_back("Any");
   Histos[1] = SingleMu_PtProf;                    legend.push_back("SingleMu30");
//   Histos[2] = DoubleMu_PtProf;                    legend.push_back("DoubleMu7");
   Histos[2] = PFMet_PtProf;                       legend.push_back("PFMHT150");
//   Histos[4] = CaloMet_PtProf;                     legend.push_back("Met120");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "p_{T} (GeV/c)", 0,0, 0,0);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_Pt");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxProf;                       legend.push_back("Any");
   Histos[1] = SingleMu_dEdxProf;                  legend.push_back("SingleMu30");
// Histos[2] = DoubleMu_dEdxProf;                  legend.push_back("DoubleMu7");
   Histos[2] = PFMet_dEdxProf;                     legend.push_back("PFMHT150");
//   Histos[4] = CaloMet_dEdxProf;                   legend.push_back("Met120");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{as}", 0,0, 0,0.1);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdx");
   delete c1;


   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxMProf;                       legend.push_back("Any");
   Histos[1] = SingleMu_dEdxMProf;                  legend.push_back("SingleMu30");
// Histos[2] = DoubleMu_dEdxMProf;                  legend.push_back("DoubleMu7");
   Histos[2] = PFMet_dEdxMProf;                     legend.push_back("PFMHT150");
//   Histos[4] = CaloMet_dEdxMProf;                   legend.push_back("Met120");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h}", 0,0, 3.0,3.5);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdxM");
   delete c1;



   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_TOFProf;                        legend.push_back("Any");
   Histos[1] = SingleMu_TOFProf;                   legend.push_back("SingleMu30");
// Histos[2] = DoubleMu_TOFProf;                   legend.push_back("DoubleMu7");
   Histos[2] = PFMet_TOFProf;                      legend.push_back("PFMHT150");
//   Histos[4] = CaloMet_TOFProf;                    legend.push_back("Met120");
   
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "1/#beta_{TOF}", 0,0, 1,1.2);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_TOF");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_TOFDTProf;                        legend.push_back("Any");
   Histos[1] = SingleMu_TOFDTProf;                   legend.push_back("SingleMu30");
// Histos[2] = DoubleMu_TOFDTProf;                   legend.push_back("DoubleMu7");
   Histos[2] = PFMet_TOFDTProf;                      legend.push_back("PFMHT150");
//   Histos[4] = CaloMet_TOFDTProf;                    legend.push_back("Met120");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "1/#beta_{TOF_DT}", 0,0, 1,1.2);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_TOFDT");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_TOFCSCProf;                        legend.push_back("Any");
   Histos[1] = SingleMu_TOFCSCProf;                   legend.push_back("SingleMu30");
// Histos[2] = DoubleMu_TOFCSCProf;                   legend.push_back("DoubleMu7");
   Histos[2] = PFMet_TOFCSCProf;                      legend.push_back("PFMHT150");
//   Histos[4] = CaloMet_TOFCSCProf;                    legend.push_back("Met120");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "1/#beta_{TOF_CSC}", 0,0, 1,1.2);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_TOFCSC");
   delete c1;






}



