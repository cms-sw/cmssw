
#include <string>
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


#include "Analysis_CommonFunction.h"
#include "Analysis_Global.h"
#include "Analysis_PlotFunction.h"
#include "Analysis_PlotStructure.h"
#include "Analysis_Samples.h"

using namespace std;

/////////////////////////// FUNCTION DECLARATION /////////////////////////////

void CutFlow(string InputPattern);
void SelectionPlot (string InputPattern, unsigned int CutIndex);
void MassPrediction(string InputPattern, unsigned int CutIndex, string HistoSuffix="Mass");
void PredictionAndControlPlot(string InputPattern, unsigned int CutIndex);
void Make2DPlot_Core(string ResultPattern, unsigned int CutIndex);

int JobIdToIndex(string JobId);



std::vector<stSignal> signals;
std::vector<stMC>     MCsample;

string LegendTitle;

/////////////////////////// CODE PARAMETERS /////////////////////////////

void Analysis_Step5()
{
   setTDRStyle();
   gStyle->SetPadTopMargin   (0.06);
   gStyle->SetPadBottomMargin(0.12);
   gStyle->SetPadRightMargin (0.16);
   gStyle->SetPadLeftMargin  (0.14);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.45);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505);

   GetSignalDefinition(signals);
   GetMCDefinition(MCsample);

   string InputDir;
   std::vector<string> Legends;                 std::vector<string> Inputs;


//   InputDir = "Results/dedxASmi/combined/Eta25/PtMin15/Type0/SplitMode0/WPPt20/WPI20/WPTOF00/";
//   InputDir = "Results/dedxASmi/combined/Eta25/PtMin20/Type2/SplitMode0/WPPt05/WPI05/WPTOF05/";
//   InputDir = "Results/dedxASmi/combined/Eta25/PtMin20/Type2/SplitMode0/WPPt20/WPI20/WPTOF20/";
//   MassPrediction(InputDir); 
//   Make2DPlot_Core(InputDir);
//   SelectionPlot(InputDir);
//   PredictionAndControlPlot(InputDir);


  InputDir = "Results/dedxASmi/combined/Eta25/PtMin25/Type0/";   unsigned int CutIndex = 18;//24;//41
//   InputDir = "Results/dedxASmi/combined/Eta25/PtMin25/Type2/";   unsigned int CutIndex = 18;

//   InputDir = "Results/dedxASmi/combined/Eta25/PtMin25/Type0/";   unsigned int CutIndex = 84;
//   InputDir = "Results/dedxASmi/combined/Eta25/PtMin25/Type2/";   unsigned int CutIndex = 113;


//     Make2DPlot_Core(InputDir,CutIndex);

//   CutFlow(InputDir);
//   SelectionPlot(InputDir, CutIndex);
//   MassPrediction(InputDir, CutIndex, "Mass");
//   MassPrediction(InputDir, CutIndex, "MassTOF");
//   MassPrediction(InputDir, CutIndex, "MassComb");    
   PredictionAndControlPlot(InputDir, CutIndex);
   return;
}



//////////////////////////////////////////////////     CREATE PLOTS OF SELECTION

void CutFlow(string InputPattern){
   string Input     = InputPattern + "Histos.root";
   string SavePath  = InputPattern + "/CutFlow/";
   MakeDirectories(SavePath);

   TFile* InputFile = new TFile(Input.c_str());
   TFile* InputFileData = new TFile((InputPattern + "Histos_Data.root").c_str());

   stPlots DataPlots, MCTrPlots, SignPlots[signals.size()];
   stPlots_InitFromFile(InputFile, DataPlots,"Data", InputFileData);
   stPlots_InitFromFile(InputFile, MCTrPlots,"MCTr", InputFile);
   for(unsigned int s=0;s<signals.size();s++){
      stPlots_InitFromFile(InputFile, SignPlots[s],signals[s].Name, InputFile);
   }

   TH1D*  HCuts_Pt       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
   TH1D*  HCuts_I        = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");
   TH1D*  HCuts_TOF      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");

   for(int CutIndex=0;CutIndex<HCuts_Pt->GetNbinsX();CutIndex++){
      char Buffer[1024]; sprintf(Buffer,"%s/CutFlow_%03i_Pt%03.0f_I%05.3f_TOF%04.3f.txt",SavePath.c_str(),CutIndex,HCuts_Pt->GetBinContent(CutIndex+1),HCuts_I->GetBinContent(CutIndex+1),HCuts_TOF->GetBinContent(CutIndex+1));
      FILE* pFile = fopen(Buffer,"w");
      stPlots_Dump(DataPlots, pFile, CutIndex);
      stPlots_Dump(MCTrPlots, pFile, CutIndex);
      for(unsigned int s=0;s<signals.size();s++){
         if(!signals[s].MakePlot)continue;
         stPlots_Dump(SignPlots[s], pFile, CutIndex);
      }
      fclose(pFile);
   }

   stPlots_Clear(DataPlots);
   stPlots_Clear(MCTrPlots);
   for(unsigned int s=0;s<signals.size();s++){
      stPlots_Clear(SignPlots[s]);
   }
}


void SelectionPlot(string InputPattern, unsigned int CutIndex){

   string LegendTitle = LegendFromType(InputPattern);;

   string Input     = InputPattern + "Histos.root";
   string SavePath  = InputPattern;
   MakeDirectories(SavePath);

   TFile* InputFile = new TFile(Input.c_str());
   TFile* InputFileData = new TFile((InputPattern + "Histos_Data.root").c_str());
 
   stPlots DataPlots, MCTrPlots, SignPlots[signals.size()];
   stPlots_InitFromFile(InputFile, DataPlots,"Data", InputFileData);
   stPlots_InitFromFile(InputFile, MCTrPlots,"MCTr", InputFile);

   for(unsigned int s=0;s<signals.size();s++){
      stPlots_InitFromFile(InputFile, SignPlots[s],signals[s].Name, InputFile);

      if(!signals[s].MakePlot)continue;
//      stPlots_Draw(SignPlots[s], SavePath + "/Selection_" +  signals[s].Name, LegendTitle, CutIndex);
   }

   stPlots_Draw(DataPlots, SavePath + "/Selection_Data", LegendTitle, CutIndex);
//   stPlots_Draw(MCTrPlots, SavePath + "/Selection_MCTr", LegendTitle);

//   stPlots_Draw(SignPlots[SID_GL600 ], SavePath + "/Selection_" +  signals[SID_GL600 ].Name, LegendTitle);
//   stPlots_Draw(SignPlots[SID_GL600N], SavePath + "/Selection_" +  signals[SID_GL600N].Name, LegendTitle);
//   stPlots_Draw(SignPlots[SID_ST300 ], SavePath + "/Selection_" +  signals[SID_ST300 ].Name, LegendTitle);
//   stPlots_Draw(SignPlots[SID_ST300N], SavePath + "/Selection_" +  signals[SID_ST300N].Name, LegendTitle);
//   stPlots_Draw(SignPlots[SID_GS126 ], SavePath + "/Selection_" +  signals[SID_GS126 ].Name, LegendTitle);

//   stPlots_DrawComparison(SavePath + "/Selection_Comp_Data" , LegendTitle, CutIndex, &DataPlots);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_Gluino" , LegendTitle, CutIndex, &DataPlots, &SignPlots[0], &SignPlots[2], &SignPlots[6], &SignPlots[8]);
   return;

   stPlots_DrawComparison(SavePath + "/Selection_Comp_Gluino" , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_GL200 ], &SignPlots[SID_GL500 ], &SignPlots[SID_GL900 ]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_GluinoN", LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_GL200N], &SignPlots[SID_GL500N], &SignPlots[SID_GL900N]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_Stop"   , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_ST200 ], &SignPlots[SID_ST500 ], &SignPlots[SID_ST800 ]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_StopN"  , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_ST200N], &SignPlots[SID_ST500N], &SignPlots[SID_ST800N]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_GMStau" , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_GS126 ], &SignPlots[SID_GS247 ], &SignPlots[SID_GS308 ]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_PPStau" , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_PS126 ], &SignPlots[SID_PS247 ], &SignPlots[SID_PS308 ]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_DCStau" , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_DS121 ], &SignPlots[SID_DS242 ], &SignPlots[SID_DS302 ]);

   stPlots_Clear(DataPlots);
   stPlots_Clear(MCTrPlots);
   for(unsigned int s=0;s<signals.size();s++){
      if(!signals[s].MakePlot)continue;
      stPlots_Clear(SignPlots[s]);
   }

}



 //////////////////////////////////////////////////     CREATE PLOTS OF CONTROLS AND PREDICTION

void PredictionAndControlPlot(string InputPattern, unsigned int CutIndex){
   TCanvas* c1;
   TObject** Histos = new TObject*[10];
   std::vector<string> legend;

   string LegendTitle = LegendFromType(InputPattern);;
   string Input     = InputPattern + "Histos_Data.root";
   string SavePath  = InputPattern;
   MakeDirectories(SavePath);

   TFile* InputFile = new TFile(Input.c_str());
   TH2D* Pred_P                = (TH2D*)GetObjectFromPath(InputFile, "Pred_P");
   TH2D* Pred_I                = (TH2D*)GetObjectFromPath(InputFile, "Pred_I");
   TH2D* Pred_TOF              = (TH2D*)GetObjectFromPath(InputFile, "Pred_TOF");
   TH2D* Data_I                = (TH2D*)GetObjectFromPath(InputFile, "DataD_I");   
   TH2D* Data_P                = (TH2D*)GetObjectFromPath(InputFile, "DataD_P");   
   TH2D* Data_TOF              = (TH2D*)GetObjectFromPath(InputFile, "DataD_TOF"); 

   TH1D*  H_A            = (TH1D*)GetObjectFromPath(InputFile, "H_A");
   TH1D*  H_B            = (TH1D*)GetObjectFromPath(InputFile, "H_B");
   TH1D*  H_C            = (TH1D*)GetObjectFromPath(InputFile, "H_C");
   TH1D*  H_D            = (TH1D*)GetObjectFromPath(InputFile, "H_D");
   TH1D*  H_E            = (TH1D*)GetObjectFromPath(InputFile, "H_E");
   TH1D*  H_F            = (TH1D*)GetObjectFromPath(InputFile, "H_F");
   TH1D*  H_G            = (TH1D*)GetObjectFromPath(InputFile, "H_G");
   TH1D*  H_H            = (TH1D*)GetObjectFromPath(InputFile, "H_H");
   TH1D*  H_P            = (TH1D*)GetObjectFromPath(InputFile, "H_P");

   TH1D*  HCuts_Pt       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
   TH1D*  HCuts_I        = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");
   TH1D*  HCuts_TOF      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");

   TH1D* CtrlPt_S1_Is         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S1_Is" ); CtrlPt_S1_Is ->Rebin(5);
   TH1D* CtrlPt_S1_Im         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S1_Im" ); CtrlPt_S1_Im ->Rebin(1);
   TH1D* CtrlPt_S1_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S1_TOF"); CtrlPt_S1_TOF->Rebin(1);
   TH1D* CtrlPt_S2_Is         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S2_Is" ); CtrlPt_S2_Is ->Rebin(5);
   TH1D* CtrlPt_S2_Im         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S2_Im" ); CtrlPt_S2_Im ->Rebin(1);
   TH1D* CtrlPt_S2_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S2_TOF"); CtrlPt_S2_TOF->Rebin(1);
   TH1D* CtrlPt_S3_Is         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S3_Is" ); CtrlPt_S3_Is ->Rebin(5);
   TH1D* CtrlPt_S3_Im         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S3_Im" ); CtrlPt_S3_Im ->Rebin(1);
   TH1D* CtrlPt_S3_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S3_TOF"); CtrlPt_S3_TOF->Rebin(1);
   TH1D* CtrlPt_S4_Is         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S4_Is" ); CtrlPt_S4_Is ->Rebin(5);
   TH1D* CtrlPt_S4_Im         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S4_Im" ); CtrlPt_S4_Im ->Rebin(1);
   TH1D* CtrlPt_S4_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_S4_TOF"); CtrlPt_S4_TOF->Rebin(1);

   TH1D* CtrlIs_S1_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlIs_S1_TOF"); CtrlIs_S1_TOF->Rebin(1);
   TH1D* CtrlIs_S2_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlIs_S2_TOF"); CtrlIs_S2_TOF->Rebin(1);
   TH1D* CtrlIs_S3_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlIs_S3_TOF"); CtrlIs_S3_TOF->Rebin(1);
   TH1D* CtrlIs_S4_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlIs_S4_TOF"); CtrlIs_S4_TOF->Rebin(1);

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlPt_S1_Is->Integral()>0)CtrlPt_S1_Is->Scale(1/CtrlPt_S1_Is->Integral());
   if(CtrlPt_S2_Is->Integral()>0)CtrlPt_S2_Is->Scale(1/CtrlPt_S2_Is->Integral());
   if(CtrlPt_S3_Is->Integral()>0)CtrlPt_S3_Is->Scale(1/CtrlPt_S3_Is->Integral());
   if(CtrlPt_S4_Is->Integral()>0)CtrlPt_S4_Is->Scale(1/CtrlPt_S4_Is->Integral());
   Histos[0] = CtrlPt_S1_Is;                     legend.push_back(" 25<p_{T}< 35 GeV");
   Histos[1] = CtrlPt_S2_Is;                     legend.push_back(" 35<p_{T}< 50 GeV");
   Histos[2] = CtrlPt_S3_Is;                     legend.push_back(" 50<p_{T}<100 GeV");
   Histos[3] = CtrlPt_S4_Is;                     legend.push_back("100<p_{T}");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend, "arbitrary units", 0,0, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"ControlPt_IsSpectrum");
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlPt_S1_Im->Integral()>0)CtrlPt_S1_Im->Scale(1/CtrlPt_S1_Im->Integral());
   if(CtrlPt_S2_Im->Integral()>0)CtrlPt_S2_Im->Scale(1/CtrlPt_S2_Im->Integral());
   if(CtrlPt_S3_Im->Integral()>0)CtrlPt_S3_Im->Scale(1/CtrlPt_S3_Im->Integral());
   if(CtrlPt_S4_Im->Integral()>0)CtrlPt_S4_Im->Scale(1/CtrlPt_S4_Im->Integral());
   Histos[0] = CtrlPt_S1_Im;                     legend.push_back(" 25<p_{T}< 35 GeV");
   Histos[1] = CtrlPt_S2_Im;                     legend.push_back(" 35<p_{T}< 50 GeV");
   Histos[2] = CtrlPt_S3_Im;                     legend.push_back(" 50<p_{T}<100 GeV");
   Histos[3] = CtrlPt_S4_Im;                     legend.push_back("100<p_{T}");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend, "arbitrary units", 3.0,10, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"ControlPt_ImSpectrum");
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlPt_S1_TOF->Integral()>0)CtrlPt_S1_TOF->Scale(1/CtrlPt_S1_TOF->Integral());
   if(CtrlPt_S2_TOF->Integral()>0)CtrlPt_S2_TOF->Scale(1/CtrlPt_S2_TOF->Integral());
   if(CtrlPt_S3_TOF->Integral()>0)CtrlPt_S3_TOF->Scale(1/CtrlPt_S3_TOF->Integral());
   if(CtrlPt_S4_TOF->Integral()>0)CtrlPt_S4_TOF->Scale(1/CtrlPt_S4_TOF->Integral());
   Histos[0] = CtrlPt_S1_TOF;                    legend.push_back(" 25<p_{T}< 35 GeV");
   Histos[1] = CtrlPt_S2_TOF;                    legend.push_back(" 35<p_{T}< 50 GeV");
   Histos[2] = CtrlPt_S3_TOF;                    legend.push_back(" 50<p_{T}<100 GeV");
   Histos[3] = CtrlPt_S4_TOF;                    legend.push_back("100<p_{T}");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "arbitrary units", 1,2, 0,0); 
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"ControlPt_TOFSpectrum");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlIs_S1_TOF->Integral()>0)CtrlIs_S1_TOF->Scale(1/CtrlIs_S1_TOF->Integral());
   if(CtrlIs_S2_TOF->Integral()>0)CtrlIs_S2_TOF->Scale(1/CtrlIs_S2_TOF->Integral());
   if(CtrlIs_S3_TOF->Integral()>0)CtrlIs_S3_TOF->Scale(1/CtrlIs_S3_TOF->Integral());
   if(CtrlIs_S4_TOF->Integral()>0)CtrlIs_S4_TOF->Scale(1/CtrlIs_S4_TOF->Integral());
   Histos[0] = CtrlIs_S1_TOF;                     legend.push_back("0.0<I_{as}<0.1");
   Histos[1] = CtrlIs_S2_TOF;                     legend.push_back("0.1<I_{as}<0.2");
   Histos[2] = CtrlIs_S3_TOF;                     legend.push_back("0.2<I_{as}<0.3");
   Histos[3] = CtrlIs_S4_TOF;                     legend.push_back("0.3<I_{as}");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "arbitrary units", 1,2, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"ControlIs_TOFSpectrum");
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   c1->SetLogy(true);
   Histos[0] = (TH1D*)(Data_P->ProjectionY("PA",CutIndex+1,CutIndex+1,"o"));   legend.push_back("Observed");
   Histos[1] = (TH1D*)(Pred_P->ProjectionY("PB",CutIndex+1,CutIndex+1,"o"));   legend.push_back("Predicted");
   ((TH1D*)Histos[0])->Scale(1/std::max(((TH1D*)Histos[0])->Integral(),1.0));
   ((TH1D*)Histos[1])->Scale(1/std::max(((TH1D*)Histos[1])->Integral(),1.0));
   ((TH1D*)Histos[0])->Rebin(10);
   ((TH1D*)Histos[1])->Rebin(10);  
   DrawSuperposedHistos((TH1**)Histos, legend, "Hist E1",  "p (Gev/c)", "u.a.", 0,1500, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Prediction_PSpectrum");
   delete Histos[0]; delete Histos[1];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   c1->SetLogy(true);
   Histos[0] = (TH1D*)(Data_I->ProjectionY("IA",CutIndex+1,CutIndex+1,"o"));   legend.push_back("Observed");
   Histos[1] = (TH1D*)(Pred_I->ProjectionY("IB",CutIndex+1,CutIndex+1,"o"));   legend.push_back("Predicted");
   ((TH1D*)Histos[0])->Scale(1/std::max(((TH1D*)Histos[0])->Integral(),1.0));
   ((TH1D*)Histos[1])->Scale(1/std::max(((TH1D*)Histos[1])->Integral(),1.0));
   ((TH1D*)Histos[0])->Rebin(2); 
   ((TH1D*)Histos[1])->Rebin(2);
   DrawSuperposedHistos((TH1**)Histos, legend, "Hist E1",  dEdxM_Legend, "u.a.", 0,15, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Prediction_ISpectrum");
   delete Histos[0]; delete Histos[1];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   c1->SetLogy(true);
   Histos[0] = (TH1D*)(Data_TOF->ProjectionY("TA",CutIndex+1,CutIndex+1,"o"));   legend.push_back("Observed");
   Histos[1] = (TH1D*)(Pred_TOF->ProjectionY("TB",CutIndex+1,CutIndex+1,"o"));   legend.push_back("Predicted");
   ((TH1D*)Histos[0])->Scale(1/std::max(((TH1D*)Histos[0])->Integral(),1.0));
   ((TH1D*)Histos[1])->Scale(1/std::max(((TH1D*)Histos[1])->Integral(),1.0));
   ((TH1D*)Histos[0])->Rebin(2); 
   ((TH1D*)Histos[1])->Rebin(2);
   DrawSuperposedHistos((TH1**)Histos, legend, "Hist E1",  "1/#beta", "u.a.", 0,0, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Prediction_TOFSpectrum");
   delete Histos[0]; delete Histos[1];
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   TH2D* DataVsPred = new TH2D("DataVsPred","DataVsPred",17,30,200,  8,0.05,0.5); 
   TH2D* DataMap    = new TH2D("DataMap"   ,"DataMap"   ,17,30,200,  8,0.05,0.5);
   TH2D* PredMap    = new TH2D("PredMap"   ,"PredMap"   ,17,30,200,  8,0.05,0.5);
   for(unsigned int CutIndex=0;CutIndex<H_P->GetNbinsX();CutIndex++){
      double P    = H_P->GetBinContent(CutIndex+1);
      double D    = H_D->GetBinContent(CutIndex+1);
      double Err  = sqrt( pow(H_P->GetBinError(CutIndex+1),2) + std::max(D,1.0)   );
//      double Err  = sqrt( pow(H_P->GetBinError(CutIndex+1),2) + pow(P*0.1,2)   );
      double NSigma = (D-P)/Err;

      DataMap->SetBinContent(DataVsPred->GetXaxis()->FindBin(HCuts_Pt->GetBinContent(CutIndex+1)), DataVsPred->GetYaxis()->FindBin(HCuts_I->GetBinContent(CutIndex+1)), D);
      PredMap->SetBinContent(DataVsPred->GetXaxis()->FindBin(HCuts_Pt->GetBinContent(CutIndex+1)), DataVsPred->GetYaxis()->FindBin(HCuts_I->GetBinContent(CutIndex+1)), P);


//      if(D==0)continue;
      if(isnan(P))continue;
      if(P<=0){continue;} //Is <=0 only when prediction failed or is not meaningful (i.e. WP=(0,0,0) )
      //if( H_B->GetBinContent(CutIndex+1)>=H_A->GetBinContent(CutIndex+1) ||  H_C->GetBinContent(CutIndex+1)>=H_A->GetBinContent(CutIndex+1))continue;

      printf("CutIndex=%3i Pt>%6.2f  I>%6.2f --> D=%6.2E P=%6.2E+-%6.2E(%6.2f+%6.2f)  (%f Sigma)\n",CutIndex, HCuts_Pt->GetBinContent(CutIndex+1),HCuts_I->GetBinContent(CutIndex+1),D,P,Err,H_P->GetBinError(CutIndex+1),sqrt(D),NSigma);
      DataVsPred->SetBinContent(DataVsPred->GetXaxis()->FindBin(HCuts_Pt->GetBinContent(CutIndex+1)), DataVsPred->GetYaxis()->FindBin(HCuts_I->GetBinContent(CutIndex+1)), NSigma);
//      DataVsPred->Fill(HCuts_Pt->GetBinContent(CutIndex+1), HCuts_I->GetBinContent(CutIndex+1), NSigma);
   }
   DataVsPred->SetMinimum(-3);
   DataVsPred->SetMaximum(3);
   Histos[0] = DataVsPred;   legend.push_back("Observed");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "PtCut", "ICut", 0,0, 0,0);
   //DrawLegend(Histos,legend,LegendTitle,"P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Prediction_PredVsObs");
   delete c1;

   PredMap->SetMinimum(1E-2);
   DataMap->SetMinimum(1E-2);
   PredMap->SetMaximum(std::max(PredMap->GetMaximum(),DataMap->GetMaximum()));
   DataMap->SetMaximum(std::max(PredMap->GetMaximum(),DataMap->GetMaximum()));
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   c1->SetLogz(true);
   Histos[0] = PredMap;   legend.push_back("Observed");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "PtCut", "ICut", 0,0, 0,0);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Prediction_Pred");
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   c1->SetLogz(true);
   Histos[0] = DataMap;   legend.push_back("Observed");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "PtCut", "ICut", 0,0, 0,0);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Prediction_Data");
   delete c1;



}

TH2D* GetCutIndexSliceFromTH3(TH3D* tmp, unsigned int CutIndex, string Name="zy"){
   tmp->GetXaxis()->SetRange(CutIndex+1,CutIndex+1);
   return (TH2D*)tmp->Project3D(Name.c_str());
}


TH1D* GetCutIndexSliceFromTH2(TH2D* tmp, unsigned int CutIndex, string Name="_py"){
   return tmp->ProjectionY(Name.c_str(),CutIndex+1,CutIndex+1);
}

void Make2DPlot_Core(string InputPattern, unsigned int CutIndex){
   TCanvas* c1;
   TLegend* leg;
 

   string Input = InputPattern + "Histos.root";
   string outpath = InputPattern;
   MakeDirectories(outpath);

   TFile* InputFile = new TFile(Input.c_str());
   TFile* InputFileData = new TFile((InputPattern + "Histos_Data.root").c_str());


   TH1D* Gluino300_Mass = GetCutIndexSliceFromTH2((TH2D*)GetObjectFromPath(InputFile, "Gluino300/Mass"    ), CutIndex, "G300Mass");
   TH2D* Gluino300_PIs  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino300/AS_PIs"  ), CutIndex, "G300PIs_zy");
   TH2D* Gluino300_PIm  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino300/AS_PIm"  ), CutIndex, "G300PIm_zy");
   TH2D* Gluino300_TOFIs= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino300/AS_TOFIs"), CutIndex, "G300TIs_zy");
   TH2D* Gluino300_TOFIm= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino300/AS_TOFIm"), CutIndex, "G300TIm_zy");
   TH1D* Gluino500_Mass = GetCutIndexSliceFromTH2((TH2D*)GetObjectFromPath(InputFile, "Gluino500/Mass"    ), CutIndex, "G500Mass");
   TH2D* Gluino500_PIs  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino500/AS_PIs"  ), CutIndex, "G500PIs_zy");
   TH2D* Gluino500_PIm  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino500/AS_PIm"  ), CutIndex, "G500PIm_zy");
   TH2D* Gluino500_TOFIs= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino500/AS_TOFIs"), CutIndex, "G500TIs_zy");
   TH2D* Gluino500_TOFIm= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino500/AS_TOFIm"), CutIndex, "G500TIm_zy");
   TH1D* Gluino800_Mass = GetCutIndexSliceFromTH2((TH2D*)GetObjectFromPath(InputFile, "Gluino800/Mass"    ), CutIndex, "G800Mass");
   TH2D* Gluino800_PIs  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino800/AS_PIs"  ), CutIndex, "G800PIs_zy");
   TH2D* Gluino800_PIm  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino800/AS_PIm"  ), CutIndex, "G800PIm_zy");
   TH2D* Gluino800_TOFIs= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino800/AS_TOFIs"), CutIndex, "G800TIs_zy");
   TH2D* Gluino800_TOFIm= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino800/AS_TOFIm"), CutIndex, "G800TIm_zy");
   TH2D* Data_PIs       = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFileData, "Data/AS_PIs"       ), CutIndex);
   TH2D* Data_PIm       = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFileData, "Data/AS_PIm"       ), CutIndex);
   TH2D* Data_TOFIs     = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFileData, "Data/AS_TOFIs"     ), CutIndex);
   TH2D* Data_TOFIm     = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFileData, "Data/AS_TOFIm"     ), CutIndex);
   TH2D* Data_PIm_075   = (TH2D*)Data_PIm->Clone();   Data_PIm_075->Reset(); 
   TH2D* Data_PIm_150   = (TH2D*)Data_PIm->Clone();   Data_PIm_150->Reset();
   TH2D* Data_PIm_300   = (TH2D*)Data_PIm->Clone();   Data_PIm_300->Reset();
   TH2D* Data_PIm_450   = (TH2D*)Data_PIm->Clone();   Data_PIm_450->Reset();
   TH2D* Data_PIm_All   = (TH2D*)Data_PIm->Clone();   Data_PIm_All->Reset();

   for(unsigned int i=0;i<(unsigned int)Data_PIm->GetNbinsX();i++){
   for(unsigned int j=0;j<(unsigned int)Data_PIm->GetNbinsY();j++){
      if(Data_PIm->GetBinContent(i,j)<=0)continue;
      double M = GetMass(Data_PIm->GetXaxis ()->GetBinCenter(i), Data_PIm->GetYaxis ()->GetBinCenter(j));
      if(isnan(M))continue;
      if     (M<100){ Data_PIm_075->SetBinContent(i,j, Data_PIm->GetBinContent(i,j) ); }
      else if(M<200){ Data_PIm_150->SetBinContent(i,j, Data_PIm->GetBinContent(i,j) ); }
      else if(M<300){ Data_PIm_300->SetBinContent(i,j, Data_PIm->GetBinContent(i,j) ); }
      else if(M<395){ Data_PIm_450->SetBinContent(i,j, Data_PIm->GetBinContent(i,j) ); }
      else          { Data_PIm_All->SetBinContent(i,j, Data_PIm->GetBinContent(i,j) ); }
   }}

   Gluino300_Mass = (TH1D*) Gluino300_Mass->Rebin(2);
   Gluino500_Mass = (TH1D*) Gluino500_Mass->Rebin(2);
   Gluino800_Mass = (TH1D*) Gluino800_Mass->Rebin(2);

   double Min = 1E-3;
   double Max = 1E4;

   char YAxisLegend[1024];
   sprintf(YAxisLegend,"#tracks / %2.0f GeV/c^{2}",Gluino300_Mass->GetXaxis()->GetBinWidth(1));


   c1 = new TCanvas("c1","c1", 600, 600);
   Gluino300_Mass->SetAxisRange(0,1250,"X");
   Gluino300_Mass->SetAxisRange(Min,Max,"Y");
   Gluino300_Mass->SetTitle("");
   Gluino300_Mass->SetStats(kFALSE);
   Gluino300_Mass->GetXaxis()->SetTitle("m (GeV/c^{2})");
   Gluino300_Mass->GetYaxis()->SetTitle(YAxisLegend);
   Gluino300_Mass->SetLineWidth(2);
   Gluino300_Mass->SetLineColor(Color[0]);
   Gluino300_Mass->SetMarkerColor(Color[0]);
   Gluino300_Mass->SetMarkerStyle(Marker[0]);
   Gluino300_Mass->Draw("HIST E1");
   Gluino500_Mass->Draw("HIST E1 same");
   Gluino500_Mass->SetLineColor(Color[1]);
   Gluino500_Mass->SetMarkerColor(Color[1]);
   Gluino500_Mass->SetMarkerStyle(Marker[1]);
   Gluino500_Mass->SetLineWidth(2);
   Gluino800_Mass->SetLineWidth(2);
   Gluino800_Mass->SetLineColor(Color[2]);
   Gluino800_Mass->SetMarkerColor(Color[2]);
   Gluino800_Mass->SetMarkerStyle(Marker[2]);
   Gluino800_Mass->Draw("HIST E1 same");
   c1->SetLogy(true);

   TLine* line300 = new TLine(300, Min, 300, Max);
   line300->SetLineWidth(2);
   line300->SetLineColor(Color[0]);
   line300->SetLineStyle(2);
   line300->Draw("same");

   TLine* line500 = new TLine(500, Min, 500, Max);
   line500->SetLineWidth(2);
   line500->SetLineColor(Color[1]);
   line500->SetLineStyle(2);
   line500->Draw("same");

   TLine* line800 = new TLine(800, Min, 800, Max);
   line800->SetLineWidth(2);
   line800->SetLineColor(Color[2]);
   line800->SetLineStyle(2);
   line800->Draw("same");

   leg = new TLegend(0.80,0.93,0.80 - 0.20,0.93 - 6*0.03);
   leg->SetHeader(LegendFromType(InputPattern).c_str());
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Gluino300_Mass, "Gluino300"   ,"P");
   leg->AddEntry(Gluino500_Mass, "Gluino500"   ,"P");
   leg->AddEntry(Gluino800_Mass, "Gluino800"   ,"P");
   leg->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Gluino_Mass");
   delete c1;


   c1 = new TCanvas("c1","c1", 600, 600);
   c1->SetLogz(true);
   Data_PIs->SetTitle("");
   Data_PIs->SetStats(kFALSE);
   Data_PIs->GetXaxis()->SetTitle("p (GeV/c)");
   Data_PIs->GetYaxis()->SetTitle(dEdxS_Legend.c_str());
   Data_PIs->SetAxisRange(0,1250,"X");
   Data_PIs->SetMarkerSize (0.2);
   Data_PIs->SetMarkerColor(Color[4]);
   Data_PIs->SetFillColor(Color[4]);
   Data_PIs->Draw("COLZ");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Data_PIs", true);
   delete c1;

   c1 = new TCanvas("c1","c1", 600, 600);
   c1->SetLogz(true);
   Data_PIm->SetTitle("");
   Data_PIm->SetStats(kFALSE);
   Data_PIm->GetXaxis()->SetTitle("p (GeV/c)");
   Data_PIm->GetYaxis()->SetTitle(dEdxM_Legend.c_str());
   Data_PIm->SetAxisRange(0,1250,"X");
   Data_PIm->SetAxisRange(0,15,"Y");
   Data_PIm->SetMarkerSize (0.2);
   Data_PIm->SetMarkerColor(Color[4]);
   Data_PIm->SetFillColor(Color[4]);
   Data_PIm->Draw("COLZ");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Data_PIm", true);
   delete c1;


   c1 = new TCanvas("c1","c1", 600, 600);
   c1->SetLogz(true);
   Data_TOFIs->SetTitle("");
   Data_TOFIs->SetStats(kFALSE);
   Data_TOFIs->GetXaxis()->SetTitle("1/#beta_{TOF}");
   Data_TOFIs->GetYaxis()->SetTitle(dEdxS_Legend.c_str());
   Data_TOFIs->SetAxisRange(0,1250,"X");
   Data_TOFIs->SetMarkerSize (0.2);
   Data_TOFIs->SetMarkerColor(Color[4]);
   Data_TOFIs->SetFillColor(Color[4]);
   Data_TOFIs->Draw("COLZ");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Data_TOFIs", true);
   delete c1;


   c1 = new TCanvas("c1","c1", 600, 600);
   c1->SetLogz(true);
   Data_TOFIm->SetTitle("");
   Data_TOFIm->SetStats(kFALSE);
   Data_TOFIm->GetXaxis()->SetTitle("1/#beta_{TOF}");
   Data_TOFIm->GetYaxis()->SetTitle(dEdxM_Legend.c_str());
   Data_TOFIm->SetAxisRange(0,15,"Y");
   Data_TOFIm->SetMarkerSize (0.2);
   Data_TOFIm->SetMarkerColor(Color[4]);
   Data_TOFIm->SetFillColor(Color[4]);
   Data_TOFIm->Draw("COLZ");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Data_TOFIm", true);
   delete c1;

   c1 = new TCanvas("c1","c1", 600, 600);
   Gluino800_PIs->SetTitle("");
   Gluino800_PIs->SetStats(kFALSE);
   Gluino800_PIs->GetXaxis()->SetTitle("p (GeV/c)");
   Gluino800_PIs->GetYaxis()->SetTitle(dEdxS_Legend.c_str());
   Gluino800_PIs->SetAxisRange(0,1250,"X");
   Gluino800_PIs->Scale(1/Gluino800_PIs->Integral());
   Gluino800_PIs->SetMarkerSize (0.2);
   Gluino800_PIs->SetMarkerColor(Color[2]);
   Gluino800_PIs->SetFillColor(Color[2]);
   Gluino800_PIs->Draw("BOX");
   Gluino500_PIs->Scale(1/Gluino500_PIs->Integral());
   Gluino500_PIs->SetMarkerSize (0.2);
   Gluino500_PIs->SetMarkerColor(Color[1]);
   Gluino500_PIs->SetFillColor(Color[1]);
   Gluino500_PIs->Draw("BOX same");
   Gluino300_PIs->Scale(1/Gluino300_PIs->Integral());
   Gluino300_PIs->SetMarkerSize (0.2);
   Gluino300_PIs->SetMarkerColor(Color[0]);
   Gluino300_PIs->SetFillColor(Color[0]);
   Gluino300_PIs->Draw("BOX same");

   leg = new TLegend(0.80,0.93,0.80 - 0.20,0.93 - 6*0.03);
   leg->SetHeader(LegendFromType(InputPattern).c_str());
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Gluino300_PIs, "Gluino300"   ,"F");
   leg->AddEntry(Gluino500_PIs, "Gluino500"   ,"F");
   leg->AddEntry(Gluino800_PIs, "Gluino800"   ,"F");
   leg->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Gluino_PIs", true);
   delete c1;

   c1 = new TCanvas("c1","c1", 600, 600);
   Gluino300_PIm->SetTitle("");
   Gluino300_PIm->SetStats(kFALSE);
   Gluino300_PIm->GetXaxis()->SetTitle("p (GeV/c)");
   Gluino300_PIm->GetYaxis()->SetTitle(dEdxM_Legend.c_str());
   Gluino300_PIm->SetAxisRange(0,1250,"X");
   Gluino300_PIm->SetAxisRange(0,15,"Y");
   Gluino300_PIm->Scale(1/Gluino300_PIm->Integral());
   Gluino300_PIm->SetMarkerSize (0.2);
   Gluino300_PIm->SetMarkerColor(Color[2]);
   Gluino300_PIm->SetFillColor(Color[2]);
   Gluino300_PIm->Draw("BOX");
   Gluino500_PIm->Scale(1/Gluino500_PIm->Integral());
   Gluino500_PIm->SetMarkerSize (0.2);
   Gluino500_PIm->SetMarkerColor(Color[1]);
   Gluino500_PIm->SetFillColor(Color[1]);
   Gluino500_PIm->Draw("BOX same");
   Gluino800_PIm->Scale(1/Gluino800_PIm->Integral());
   Gluino800_PIm->SetMarkerSize (0.2);
   Gluino800_PIm->SetMarkerColor(Color[0]);
   Gluino800_PIm->SetFillColor(Color[0]);
   Gluino800_PIm->Draw("BOX same");

   TF1* MassLine800 = GetMassLine(800, true);
   MassLine800->SetLineColor(kGray+3);
   MassLine800->SetLineWidth(2);
   MassLine800->Draw("same");
   TF1* MassLine500 = GetMassLine(500, true);
   MassLine500->SetLineColor(kBlue-7);
   MassLine500->SetLineWidth(2);
   MassLine500->Draw("same");
   TF1* MassLine300 = GetMassLine(300, true);
   MassLine300->SetLineColor(kRed-7);
   MassLine300->SetLineWidth(2);
   MassLine300->Draw("same");

   leg = new TLegend(0.80,0.93,0.80 - 0.20,0.93 - 6*0.03);
   leg->SetHeader(LegendFromType(InputPattern).c_str());
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Gluino300_PIm, "Gluino300"   ,"F");
   leg->AddEntry(Gluino500_PIm, "Gluino500"   ,"F");
   leg->AddEntry(Gluino800_PIm, "Gluino800"   ,"F");
   leg->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Gluino_PIm", true);
   delete c1;

   c1 = new TCanvas("c1","c1", 600, 600);
   Gluino800_TOFIs->SetTitle("");
   Gluino800_TOFIs->SetStats(kFALSE);
   Gluino800_TOFIs->GetXaxis()->SetTitle("1/#beta_{TOF}");
   Gluino800_TOFIs->GetYaxis()->SetTitle(dEdxS_Legend.c_str());
   Gluino800_TOFIs->SetAxisRange(0,1250,"X");
   Gluino800_TOFIs->Scale(1/Gluino800_TOFIs->Integral());
   Gluino800_TOFIs->SetMarkerSize (0.2);
   Gluino800_TOFIs->SetMarkerColor(Color[2]);
   Gluino800_TOFIs->SetFillColor(Color[2]);
   Gluino800_TOFIs->Draw("BOX");
   Gluino500_TOFIs->Scale(1/Gluino500_TOFIs->Integral());
   Gluino500_TOFIs->SetMarkerSize (0.2);
   Gluino500_TOFIs->SetMarkerColor(Color[1]);
   Gluino500_TOFIs->SetFillColor(Color[1]);
   Gluino500_TOFIs->Draw("BOX same");
   Gluino300_TOFIs->Scale(1/Gluino300_TOFIs->Integral());
   Gluino300_TOFIs->SetMarkerSize (0.2);
   Gluino300_TOFIs->SetMarkerColor(Color[0]);
   Gluino300_TOFIs->SetFillColor(Color[0]);
   Gluino300_TOFIs->Draw("BOX same");

   leg = new TLegend(0.80,0.93,0.80 - 0.20,0.93 - 6*0.03);
   leg->SetHeader(LegendFromType(InputPattern).c_str());
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Gluino300_TOFIs, "Gluino300"   ,"F");
   leg->AddEntry(Gluino500_TOFIs, "Gluino500"   ,"F");
   leg->AddEntry(Gluino800_TOFIs, "Gluino800"   ,"F");
   leg->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Gluino_TOFIs", true);
   delete c1;

   c1 = new TCanvas("c1","c1", 600, 600);
   Gluino800_TOFIm->SetTitle("");
   Gluino800_TOFIm->SetStats(kFALSE);
   Gluino800_TOFIm->GetXaxis()->SetTitle("1/#beta_{TOF}");
   Gluino800_TOFIm->GetYaxis()->SetTitle(dEdxM_Legend.c_str());
   Gluino800_TOFIm->SetAxisRange(0,1250,"X");
   Gluino800_TOFIm->SetAxisRange(0,15,"Y");
   Gluino800_TOFIm->Scale(1/Gluino800_TOFIm->Integral());
   Gluino800_TOFIm->SetMarkerSize (0.2);
   Gluino800_TOFIm->SetMarkerColor(Color[2]);
   Gluino800_TOFIm->SetFillColor(Color[2]);
   Gluino800_TOFIm->Draw("BOX");
   Gluino500_TOFIm->Scale(1/Gluino500_TOFIm->Integral());
   Gluino500_TOFIm->SetMarkerSize (0.2);
   Gluino500_TOFIm->SetMarkerColor(Color[1]);
   Gluino500_TOFIm->SetFillColor(Color[1]);
   Gluino500_TOFIm->Draw("BOX same");
   Gluino300_TOFIm->Scale(1/Gluino300_TOFIm->Integral());
   Gluino300_TOFIm->SetMarkerSize (0.2);
   Gluino300_TOFIm->SetMarkerColor(Color[0]);
   Gluino300_TOFIm->SetFillColor(Color[0]);
   Gluino300_TOFIm->Draw("BOX same");

   leg = new TLegend(0.80,0.93,0.80 - 0.20,0.93 - 6*0.03);
   leg->SetHeader(LegendFromType(InputPattern).c_str());
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Gluino300_TOFIm, "Gluino300"   ,"F");
   leg->AddEntry(Gluino500_TOFIm, "Gluino500"   ,"F");
   leg->AddEntry(Gluino800_TOFIm, "Gluino800"   ,"F");
   leg->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Gluino_TOFIm", true);
   delete c1;

   c1 = new TCanvas("c1","c1", 600, 600);
   Data_PIm_075->SetTitle("");
   Data_PIm_075->SetStats(kFALSE);
   Data_PIm_075->GetXaxis()->SetTitle("p (GeV/c)");
   Data_PIm_075->GetYaxis()->SetTitle(dEdxM_Legend.c_str());
   Data_PIm_075->SetAxisRange(0,15,"Y");
   Data_PIm_075->SetAxisRange(0,1250,"X");
   Data_PIm_075->SetMarkerSize (0.6);
   Data_PIm_075->SetMarkerColor(Color[4]);
   Data_PIm_075->SetMarkerStyle(Marker[4]);
   Data_PIm_075->SetFillColor(Color[4]);
   Data_PIm_075->Draw("");
   Data_PIm_150->SetMarkerSize (0.8);
   Data_PIm_150->SetMarkerColor(Color[3]);
   Data_PIm_150->SetMarkerStyle(Marker[3]);
   Data_PIm_150->SetFillColor(Color[3]);
   Data_PIm_150->Draw("same");
   Data_PIm_300->SetMarkerSize (1.0);
   Data_PIm_300->SetMarkerColor(Color[2]);
   Data_PIm_300->SetMarkerStyle(Marker[2]);
   Data_PIm_300->SetFillColor(Color[2]);
   Data_PIm_300->Draw("same");
   Data_PIm_450->SetMarkerSize (1.2);
   Data_PIm_450->SetMarkerColor(Color[1]);
   Data_PIm_450->SetMarkerStyle(Marker[1]);
   Data_PIm_450->SetFillColor(Color[1]);
   Data_PIm_450->Draw("same");
   Data_PIm_All->SetMarkerSize (1.4);
   Data_PIm_All->SetMarkerColor(Color[0]);
   Data_PIm_All->SetFillColor(Color[0]);
   Data_PIm_All->Draw("same");

   leg = new TLegend(0.80,0.93,0.80 - 0.30,0.93 - 6*0.03);
   leg->SetHeader(LegendFromType(InputPattern).c_str());
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Data_PIm_075, "M < 100 GeV","P");
   leg->AddEntry(Data_PIm_150, "100 < M < 200 GeV","P");
   leg->AddEntry(Data_PIm_300, "200 < M < 300 GeV","P");
   leg->AddEntry(Data_PIm_450, "300 < M < 400 GeV","P");
   leg->AddEntry(Data_PIm_All, "400 < M GeV"      ,"P");
   leg->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Data_PIm_Colored", true);
   delete c1;
}

void MassPrediction(string InputPattern, unsigned int CutIndex, string HistoSuffix)
{
   double Rescale, RMS;
//   GetPredictionRescale(InputPattern,Rescale, RMS, RecomputeRescale);
//   RMS = fabs(1.0-Rescale)/2.0;
   Rescale = 1.0;
   RMS     = 0.05;

   string outpath = InputPattern;
   MakeDirectories(outpath);

   TFile* InputFile;
   string Input;

   std::vector<string> legend;
   TCanvas* c1;

   char Buffer[2048];
   sprintf(Buffer,"%s/Histos_Data.root",InputPattern.c_str());
   InputFile = new TFile(Buffer);
   if(!InputFile || InputFile->IsZombie() || !InputFile->IsOpen() || InputFile->TestBit(TFile::kRecovered) )return;
   TH1D* Pred     = ((TH2D*)GetObjectFromPath(InputFile, string("Pred_") + HistoSuffix   ))->ProjectionY("TmpPredMass"    ,CutIndex+1,CutIndex+1,"o");
   TH1D* Data     = ((TH2D*)GetObjectFromPath(InputFile, string("Data/") + HistoSuffix   ))->ProjectionY("TmpDataMass"    ,CutIndex+1,CutIndex+1,"o");


      double D,P,Perr;
      D = Data->Integral( Data->GetXaxis()->FindBin(0.0),  Data->GetXaxis()->FindBin(75.0));
      P = Pred->Integral( Pred->GetXaxis()->FindBin(0.0),  Pred->GetXaxis()->FindBin(75.0));
      Perr = 0; for(int i=Pred->GetXaxis()->FindBin(0.0);i<Pred->GetXaxis()->FindBin(75.0);i++){ Perr += pow(Pred->GetBinError(i),2); }  Perr = sqrt(Perr);
      printf("%3.0f<M<%3.0f --> D=%9.3f P = %9.3f +- %6.3f(stat) +- %6.3f(syst) (=%6.3f)\n", 0.0,75.0, D, P, Perr, P*(2*RMS),sqrt(Perr*Perr + pow(P*(2*RMS),2)));


   for(double M=0;M<500;M+=100){
      double D,P,Perr;
      D = Data->Integral( Data->GetXaxis()->FindBin(M),  Data->GetXaxis()->FindBin(2000.0));  
      P = Pred->Integral( Pred->GetXaxis()->FindBin(M),  Pred->GetXaxis()->FindBin(2000.0));
      Perr = 0; for(int i=Pred->GetXaxis()->FindBin(M);i<Pred->GetXaxis()->FindBin(2000.0);i++){ Perr += pow(Pred->GetBinError(i),2); }  Perr = sqrt(Perr);
      printf("%3.0f<M<2000 --> D=%9.3f P = %9.3f +- %6.3f(stat) +- %6.3f(syst) (=%6.3f)\n", M, D, P, Perr, P*(2*RMS),sqrt(Perr*Perr + pow(P*(2*RMS),2)));
   }
//   for(int i=Pred->GetXaxis()->FindBin(0.0);i<Pred->GetXaxis()->FindBin(2000.0);i++){printf("MassBin=%6.2f  --> BinEntry=%6.2E +- %6.2E\n",Pred->GetXaxis()->GetBinCenter(i), Pred->GetBinContent(i), Pred->GetBinError(i));}


      printf("FullSpectrum --> D=%9.3f P = %9.3f +- %6.3f(stat) +- %6.3f(syst) (=%6.3f)\n", Data->Integral(), Pred->Integral(), 0.0, 0.0, 0.0 );
      printf("UnderFlow = %6.2f OverFlow = %6.2f\n", Pred->GetBinContent(0), Pred->GetBinContent(Pred->GetNbinsX()+1) );



   TH1D*  H_A            = (TH1D*)GetObjectFromPath(InputFile, "H_A");
   TH1D*  H_B            = (TH1D*)GetObjectFromPath(InputFile, "H_B");
   TH1D*  H_C            = (TH1D*)GetObjectFromPath(InputFile, "H_C");
   TH1D*  H_D            = (TH1D*)GetObjectFromPath(InputFile, "H_D");
   TH1D*  H_E            = (TH1D*)GetObjectFromPath(InputFile, "H_E");
   TH1D*  H_F            = (TH1D*)GetObjectFromPath(InputFile, "H_F");
   TH1D*  H_G            = (TH1D*)GetObjectFromPath(InputFile, "H_G");
   TH1D*  H_H            = (TH1D*)GetObjectFromPath(InputFile, "H_H");
   TH1D*  H_P            = (TH1D*)GetObjectFromPath(InputFile, "H_P");
   printf("OBSERVED  EVENTS = %6.2E\n",H_D->GetBinContent(CutIndex+1));
   printf("PREDICTED EVENTS = %6.2E+-%6.2E\n",H_P->GetBinContent(CutIndex+1), H_P->GetBinError(CutIndex+1));


   Pred->Rebin(2);
   Data->Rebin(2);

   double Max = 2.0 * std::max(Data->GetMaximum(), Pred->GetMaximum());
   double Min = 0.1 * std::min(0.01,Pred->GetMaximum());

   TLegend* leg;
   c1 = new TCanvas("c1","c1,",600,600);

   char YAxisLegend[1024];
   sprintf(YAxisLegend,"Tracks / %2.0f GeV/c^{2}",Data->GetXaxis()->GetBinWidth(1));

   TH1D* PredErr = (TH1D*) Pred->Clone("PredErr");
   for(unsigned int i=0;i<(unsigned int)Pred->GetNbinsX();i++){
      double error = sqrt(pow(PredErr->GetBinError(i),2) + pow(PredErr->GetBinContent(i)*2*RMS,2));
      PredErr->SetBinError(i,error);       
      if(PredErr->GetBinContent(i)<Min && i>5){for(unsigned int j=i+1;j<(unsigned int)PredErr->GetNbinsX();j++)PredErr->SetBinContent(j,0);}
   }
   PredErr->SetLineColor(8);
   PredErr->SetFillColor(8);
   PredErr->SetFillStyle(3001);
   PredErr->SetMarkerStyle(22);
   PredErr->SetMarkerColor(2);
   PredErr->SetMarkerSize(1.0);
   PredErr->GetXaxis()->SetNdivisions(505);
   PredErr->SetTitle("");
   PredErr->SetStats(kFALSE);
   PredErr->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
   PredErr->GetYaxis()->SetTitle(YAxisLegend);
   PredErr->GetYaxis()->SetTitleOffset(1.50);
   PredErr->SetMaximum(Max);
   PredErr->SetMinimum(Min);
   PredErr->SetAxisRange(0,1400,"X");
   PredErr->Draw("E5");

   Pred->SetMarkerStyle(22);
   Pred->SetMarkerColor(2);
   Pred->SetMarkerSize(1.5);
   Pred->SetLineColor(2);
   Pred->SetFillColor(0);
   Pred->Draw("same HIST P");

   Data->SetMarkerStyle(20);
   Data->SetMarkerColor(1);
   Data->SetMarkerSize(1.0);
   Data->SetLineColor(1);
   Data->SetFillColor(0);
   Data->Draw("E1 same");

   leg = new TLegend(0.79,0.93,0.40,0.68);
   leg->SetHeader(LegendFromType(InputPattern).c_str());
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   TH1D* PredLeg = (TH1D*) Pred->Clone("RescLeg");
   PredLeg->SetFillColor(PredErr->GetFillColor());
   PredLeg->SetFillStyle(PredErr->GetFillStyle());
   leg->AddEntry(PredLeg, "Data-based prediction"  ,"PF");
   leg->AddEntry(Data, "Data"        ,"P");
   leg->Draw();

   DrawPreliminary(IntegratedLuminosity);
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("Rescale_") + HistoSuffix);
   delete c1;

/*
  std::cout << "TESTA\n";


   c1 = new TCanvas("c1","c1,",600,600);

   sprintf(YAxisLegend,"Tracks / %2.0f GeV/c^{2}",Pred2->GetXaxis()->GetBinWidth(1));
  std::cout << "TESTB\n";

   TH1D* PredTOFErr = (TH1D*) PredTOF->Clone("RescTOFErr");
   for(unsigned int i=0;i<(unsigned int)PredTOF->GetNbinsX();i++){
      double error2 = pow(PredTOFErr->GetBinError(i),2);
      error2 += pow(PredTOF->GetBinContent(i)*2*RMS,2);
      PredTOFErr->SetBinError(i,sqrt(error2));       
      if(PredTOFErr->GetBinContent(i)<Min && i>5){for(unsigned int j=i+1;j<(unsigned int)PredTOF->GetNbinsX();j++)PredTOFErr->SetBinContent(j,0);}
   }
  std::cout << "TESTC\n";

   PredTOFErr->SetLineColor(9);
   PredTOFErr->SetFillColor(9);
   PredTOFErr->SetFillStyle(3001);
   PredTOFErr->SetMarkerStyle(22);
   PredTOFErr->SetMarkerColor(2);
   PredTOFErr->SetMarkerSize(1.0);
   PredTOFErr->GetXaxis()->SetNdivisions(505);
   PredTOFErr->SetTitle("");
   PredTOFErr->SetStats(kFALSE);
   PredTOFErr->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
   PredTOFErr->GetYaxis()->SetTitle(YAxisLegend);
   PredTOFErr->GetYaxis()->SetTitleOffset(1.50);
   PredTOFErr->SetMaximum(Max);
   PredTOFErr->SetMinimum(Min);
   PredTOFErr->SetAxisRange(0,1400,"X");
   PredTOFErr->Draw("E5");

  std::cout << "TESTD\n";


   PredTOF->SetMarkerStyle(23);
   PredTOF->SetMarkerColor(4);
   PredTOF->SetMarkerSize(1.5);
   PredTOF->SetLineColor(2);
   PredTOF->SetFillColor(0);
   PredTOF->Draw("same HIST P");

  std::cout << "TESTE\n";


   DataTOF1->SetMarkerStyle(20);
   DataTOF1->SetMarkerColor(1);
   DataTOF1->SetMarkerSize(1.5);
   DataTOF1->SetLineColor(1);
   DataTOF1->SetFillColor(0);
   DataTOF1->Draw("E1 same");

   leg = new TLegend(0.79,0.93,0.40,0.68);
   leg->SetHeader(LegendFromType(InputPattern).c_str());
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   TH1D* PredTOFLeg = (TH1D*) PredTOF->Clone("PredTOFLeg");
   PredTOFLeg->SetFillColor(PredTOFErr->GetFillColor());
   PredTOFLeg->SetFillStyle(PredTOFErr->GetFillStyle());
   leg->AddEntry(PredTOFLeg, "Data-based prediction ATLAS (TOF)"  ,"PF");
   leg->AddEntry(Data1, "Data (TOF)"        ,"P");
   leg->Draw();

   DrawPreliminary(IntegratedLuminosity);
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, "Rescale_MassTOF");
   delete c1;




   c1 = new TCanvas("c1","c1,",600,600);

   sprintf(YAxisLegend,"Tracks / %2.0f GeV/c^{2}",Pred2->GetXaxis()->GetBinWidth(1));
  std::cout << "TESTB\n";

   TH1D* PredCombErr = (TH1D*) PredComb->Clone("RescCombErr");
   for(unsigned int i=0;i<(unsigned int)PredComb->GetNbinsX();i++){
      double error2 = pow(PredCombErr->GetBinError(i),2);
      error2 += pow(PredComb->GetBinContent(i)*2*RMS,2);
      PredCombErr->SetBinError(i,sqrt(error2));       
      if(PredCombErr->GetBinContent(i)<Min && i>5){for(unsigned int j=i+1;j<(unsigned int)PredComb->GetNbinsX();j++)PredCombErr->SetBinContent(j,0);}
   }
  std::cout << "TESTC\n";

   PredCombErr->SetLineColor(9);
   PredCombErr->SetFillColor(9);
   PredCombErr->SetFillStyle(3001);
   PredCombErr->SetMarkerStyle(22);
   PredCombErr->SetMarkerColor(2);
   PredCombErr->SetMarkerSize(1.0);
   PredCombErr->GetXaxis()->SetNdivisions(505);
   PredCombErr->SetTitle("");
   PredCombErr->SetStats(kFALSE);
   PredCombErr->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
   PredCombErr->GetYaxis()->SetTitle(YAxisLegend);
   PredCombErr->GetYaxis()->SetTitleOffset(1.50);
   PredCombErr->SetMaximum(Max);
   PredCombErr->SetMinimum(Min);
   PredCombErr->SetAxisRange(0,1400,"X");
   PredCombErr->Draw("E5");

  std::cout << "TESTD\n";


   PredComb->SetMarkerStyle(23);
   PredComb->SetMarkerColor(4);
   PredComb->SetMarkerSize(1.5);
   PredComb->SetLineColor(2);
   PredComb->SetFillColor(0);
   PredComb->Draw("same HIST P");

  std::cout << "TESTE\n";


   DataComb->SetMarkerStyle(20);
   DataComb->SetMarkerColor(1);
   DataComb->SetMarkerSize(1.5);
   DataComb->SetLineColor(1);
   DataComb->SetFillColor(0);
   DataComb->Draw("E1 same");

   leg = new TLegend(0.79,0.93,0.40,0.68);
   leg->SetHeader(LegendFromType(InputPattern).c_str());
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   TH1D* PredCombLeg = (TH1D*) PredComb->Clone("PredCombLeg");
   PredCombLeg->SetFillColor(PredCombErr->GetFillColor());
   PredCombLeg->SetFillStyle(PredCombErr->GetFillStyle());
   leg->AddEntry(PredCombLeg, "Data-based prediction ATLAS (Combined)"  ,"PF");
   leg->AddEntry(Data1, "Data (TOF)"        ,"P");
   leg->Draw();

   DrawPreliminary(IntegratedLuminosity);
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, "Rescale_MassComb");
   delete c1;

*/

/*
   for(unsigned int s=0;s<signals.size();s++){
      if(!signals[s].MakePlot)continue;
      TH1D* SIGN = (TH1D*)GetObjectFromPath(InputFile1, string("Mass_") + signals[s].Name);
      SIGN->Rebin(10);

      c1 = new TCanvas("c1","c1,",600,600);

      double MaxSign = std::max(Max, SIGN->GetMaximum()*2);
      Resc1Err->SetMaximum(MaxSign);
      Resc1Err->SetMinimum(Min);
      Resc1Err->Draw("E5");
      SIGN->SetLineWidth(1);
      SIGN->SetLineColor(38);
      SIGN->SetFillColor(38);
      SIGN->Draw("same HIST");
      Resc1Err->Draw("same E5");
      Resc1->Draw("same HIST P");
      Data1->Draw("E1 same");


      leg = new TLegend(0.79,0.93,0.40,0.68);
      if(IsTrackerOnly){ 
         leg->SetHeader("Tracker - Only");
      }else{
         leg->SetHeader("Tracker + Muon");
      }
      leg->SetFillColor(0);
      leg->SetBorderSize(0);
      leg->AddEntry(RescLeg, "Data-based prediction"  ,"PF");
      leg->AddEntry(Data1, "Data"        ,"P");
      leg->AddEntry(SIGN,  (string("MC - ") + signals[s].Legend).c_str()        ,"F");
      leg->Draw();

      DrawPreliminary(IntegratedLuminosity);
      c1->SetLogy(true);
      SaveCanvas(c1, outpath, string("Rescale_Mass_") + signals[s].Name);

      delete c1;
   }








  TH1D* Ratio1       = (TH1D*)Pred1->Clone("Ratio1");
  TH1D* Ratio2       = (TH1D*)Resc1->Clone("Ratio2");
  TH1D* Ratio3       = (TH1D*)Resc1->Clone("Ratio3");
  TH1D* DataWithStat = (TH1D*)Data1->Clone("DataWithStat");
  Ratio1->Divide(DataWithStat);
  Ratio2->Divide(DataWithStat);
  Ratio3->Divide(MCTr1);

   c1 = new TCanvas("c1","c1,",600,600);
   Ratio2->SetAxisRange(0,1400,"X");
   Ratio2->SetAxisRange(0,2,"Y");
   Ratio2->SetTitle("");
   Ratio2->SetStats(kFALSE);
   Ratio2->GetXaxis()->SetTitle("m (GeV/c^{2})");
   Ratio2->GetYaxis()->SetTitle("Ratio");
   Ratio2->GetYaxis()->SetTitleOffset(1.50);
   Ratio2->SetMarkerStyle(21);
   Ratio2->SetMarkerColor(4);
   Ratio2->SetMarkerSize(1);
   Ratio2->SetLineColor(4);
   Ratio2->SetFillColor(0);
   Ratio2->Draw("E1");
  
   TBox* b = new TBox(0,1.0-2*RMS,1410,1.0+2*RMS);
   b->SetFillStyle(3003);
   b->SetFillColor(8);
   b->Draw("same");

   TLine* l = new TLine(0,1.0,1410,1.0);
   l->Draw("same");

   Ratio2->Draw("same E1");

   leg = new TLegend(0.79,0.93,0.40,0.75);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Ratio2, "Rescaled Prediction / Data"     ,"P");
   leg->Draw();


   DrawPreliminary(IntegratedLuminosity);  
   SaveCanvas(c1, outpath, "Rescale_Ratio");
   delete c1;

*/
   InputFile->Close();
}



int JobIdToIndex(string JobId){
   for(unsigned int s=0;s<signals.size();s++){
      if(signals[s].Name==JobId)return s;
   }return -1;
}




