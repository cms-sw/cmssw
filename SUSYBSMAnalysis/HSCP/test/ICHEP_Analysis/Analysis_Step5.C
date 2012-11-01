// Original Author:  Loic Quertenmont
#include "Analysis_Global.h"

#include "Analysis_Global.h"
#include "Analysis_CommonFunction.h"
#include "Analysis_PlotFunction.h"
#include "Analysis_PlotStructure.h"
#include "Analysis_Samples.h"
#include "tdrstyle.C"
#include "Math/QuantFuncMathCore.h"
#include "TMath.h"
#include "TGraphAsymmErrors.h"

using namespace std;

/////////////////////////// FUNCTION DECLARATION /////////////////////////////

void MassPrediction(string InputPattern, unsigned int CutIndex, string HistoSuffix="Mass", string Data="Data8TeV");
void PredictionAndControlPlot(string InputPattern, string Data, unsigned int CutIndex, unsigned int CutIndex_Flip);
void CutFlow(string InputPattern, unsigned int CutIndex=0);
void SelectionPlot (string InputPattern, unsigned int CutIndex);

void Make2DPlot_Core(string ResultPattern, unsigned int CutIndex);
void SignalMassPlot(string InputPattern, unsigned int CutIndex);
void GetSystematicOnPrediction(string InputPattern, string DataName="Data8TeV");
void MakeExpLimitpLot(string Input, string Output);
void CosmicBackgroundSystematic(string InputPattern, string DataType="8TeV");
void CheckPrediction(string InputPattern, string HistoSuffix="_Flip", string DataType="Data8TeV");
void CheckPredictionBin(string InputPattern, string HistoSuffix="_Flip", string DataType="Data8TeV", string bin="");
void CollisionBackgroundSystematicFromFlip(string InputPattern, string DataType="Data8TeV");

std::vector<stSample> samples;

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
   GetSampleDefinition(samples);


   string InputPattern;				unsigned int CutIndex;     unsigned int CutIndex_Flip;  unsigned int CutIndexTight;
   std::vector<string> Legends;                 std::vector<string> Inputs;

   InputPattern = "Results/Type0/";   CutIndex = 4; CutIndexTight = 84; //set of cuts from the array, 0 means no cut
   MassPrediction(InputPattern, CutIndex,      "Mass", "8TeV_Loose");
   MassPrediction(InputPattern, CutIndex,      "Mass", "7TeV_Loose");
   MassPrediction(InputPattern, CutIndexTight, "Mass", "8TeV_Tight");
   MassPrediction(InputPattern, CutIndexTight, "Mass", "7TeV_Tight");
   PredictionAndControlPlot(InputPattern, "Data7TeV", CutIndex, CutIndex_Flip);
   PredictionAndControlPlot(InputPattern, "Data8TeV", CutIndex, CutIndex_Flip);
   //CutFlow(InputPattern, CutIndex);
   SelectionPlot(InputPattern, CutIndex);

   InputPattern = "Results/Type2/";   CutIndex = 16; CutIndexTight = 905; CutIndex_Flip=16;
   MassPrediction(InputPattern, CutIndex,      "Mass", "8TeV_Loose");
   MassPrediction(InputPattern, CutIndex,      "Mass", "7TeV_Loose");
   MassPrediction(InputPattern, CutIndexTight, "Mass", "8TeV_Tight");
   MassPrediction(InputPattern, CutIndexTight, "Mass", "7TeV_Tight");
   MassPrediction(InputPattern, CutIndex_Flip, "Mass_Flip");
   PredictionAndControlPlot(InputPattern, "Data7TeV", CutIndex, CutIndex_Flip);
   PredictionAndControlPlot(InputPattern, "Data8TeV", CutIndex, CutIndex_Flip);
   CutFlow(InputPattern, CutIndex);
   SelectionPlot(InputPattern, CutIndex);
   GetSystematicOnPrediction(InputPattern, "Data7TeV");
   GetSystematicOnPrediction(InputPattern, "Data8TeV");
   CheckPrediction(InputPattern, "_Flip", "Data7TeV");
   CheckPrediction(InputPattern, "_Flip", "Data8TeV");

   InputPattern = "Results/Type3/";   CutIndex = 79; CutIndex_Flip=58;
   PredictionAndControlPlot(InputPattern, "Data7TeV", CutIndex, CutIndex_Flip);
   PredictionAndControlPlot(InputPattern, "Data8TeV", CutIndex, CutIndex_Flip);
   CutFlow(InputPattern, CutIndex);
   SelectionPlot(InputPattern, CutIndex);
   CosmicBackgroundSystematic(InputPattern, "8TeV");
   CosmicBackgroundSystematic(InputPattern, "7TeV");
   CheckPrediction(InputPattern, "", "Data8TeV");
   CheckPrediction(InputPattern, "_Flip", "Data8TeV");
   CheckPrediction(InputPattern, "", "Data7TeV");
   CheckPrediction(InputPattern, "_Flip", "Data7TeV");
   CollisionBackgroundSystematicFromFlip(InputPattern, "Data8TeV");

   /*
   CheckPredictionBin(InputPattern, "_Flip", "Data8TeV", "0");
   CheckPredictionBin(InputPattern, "_Flip", "Data8TeV", "1");
   CheckPredictionBin(InputPattern, "_Flip", "Data8TeV", "2");
   CheckPredictionBin(InputPattern, "_Flip", "Data8TeV", "3");
   CheckPredictionBin(InputPattern, "_Flip", "Data8TeV", "4");
   CheckPredictionBin(InputPattern, "_Flip", "Data8TeV", "5");
   */
   //InputPattern = "Results/Type5/";   CutIndex = 67; CutIndex_Flip=2;
   //SelectionPlot(InputPattern, CutIndex);
   //CutFlow(InputPattern);

   //   InputPattern = "Results/Type4/";   CutIndex = 21; CutIndex_Flip=21;
   //   CollisionBackgroundSystematicFromFlip(InputPattern, "Data7TeV");
   //   CollisionBackgroundSystematicFromFlip(InputPattern, "Data8TeV");

   InputPattern = "Results/Type5/";   CutIndex = 48; CutIndex_Flip=2;
   InitdEdx("dedxRAsmi");
   PredictionAndControlPlot(InputPattern, "Data7TeV", CutIndex, CutIndex_Flip);
   PredictionAndControlPlot(InputPattern, "Data8TeV", CutIndex, CutIndex_Flip);
   SelectionPlot(InputPattern, CutIndex);
   CutFlow(InputPattern);

     //This function has not yet been reviewed after july's update
//   MakeExpLimitpLot("Results_1toys_lp/dedxASmi/combined/Eta15/PtMin35/Type0/EXCLUSION/Stop200.info","tmp1.png");
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
// General code for validation plots and final paper plots production 


// Make the plot of the mass distibution: Observed, data-driven prediciton and signal expectation
void MassPrediction(string InputPattern, unsigned int CutIndex, string HistoSuffix, string DataName){
   if(DataName.find("7TeV")!=string::npos){SQRTS=7.0;}else{SQRTS=8.0;}
   bool IsTkOnly = (InputPattern.find("Type0",0)<std::string::npos);
   double SystError     = 0.05;

   TH1D *Pred8TeV=NULL, *Data8TeV=NULL, *Pred7TeV=NULL, *Data7TeV=NULL, *MCPred=NULL, *MC=NULL, *Signal=NULL;
   TFile* InputFile = new TFile((InputPattern + "/Histos.root").c_str());
   if(!InputFile || InputFile->IsZombie() || !InputFile->IsOpen() || InputFile->TestBit(TFile::kRecovered) )return;   

   string SName,SLeg;

   //README: Comments or uncomment lines below in order to decide what you want to see on your plot
   if(DataName.find("8TeV")!=string::npos){
                    SName="Gluino_7TeV_M600_f10";     SLeg="Gluino (M=600 GeV/#font[12]{c}^{2})";
      if(!IsTkOnly){SName="GMStau_7TeV_M156";         SLeg="Stau (M=156 GeV/#font[12]{c}^{2})";}

      Pred8TeV    = ((TH2D*)GetObjectFromPath(InputFile, string("Data8TeV/Pred_") + HistoSuffix   ))->ProjectionY("TmpPredMass"   ,CutIndex+1,CutIndex+1,"o");
      Data8TeV    = ((TH2D*)GetObjectFromPath(InputFile, string("Data8TeV/"     ) + HistoSuffix   ))->ProjectionY("TmpDataMass"   ,CutIndex+1,CutIndex+1,"o");
//    Pred7TeV    = ((TH2D*)GetObjectFromPath(InputFile, string("Data7TeV/Pred_") + HistoSuffix   ))->ProjectionY("TmpPred7TeVMass" ,CutIndex+1,CutIndex+1,"o");
//    Data7TeV    = ((TH2D*)GetObjectFromPath(InputFile, string("Data7TeV/"     ) + HistoSuffix   ))->ProjectionY("TmpData7TeVMass" ,CutIndex+1,CutIndex+1,"o");
      MCPred    = ((TH2D*)GetObjectFromPath(InputFile, string("MCTr_8TeV/Pred_"  ) + HistoSuffix   ))->ProjectionY("TmpMCPred"     ,CutIndex+1,CutIndex+1,"o");
      MC        = ((TH2D*)GetObjectFromPath(InputFile, string("MCTr_8TeV/"       ) + HistoSuffix   ))->ProjectionY("TmpMCMass"     ,CutIndex+1,CutIndex+1,"o");
      Signal    = ((TH2D*)GetObjectFromPath(InputFile, string(SName+"/"     ) + HistoSuffix   ))->ProjectionY("TmpSignalMass" ,CutIndex+1,CutIndex+1,"o");
   }else{
                    SName="Gluino_7TeV_M600_f10";     SLeg="Gluino (M=600 GeV/#font[12]{c}^{2})";
      if(!IsTkOnly){SName="GMStau_7TeV_M156";         SLeg="Stau (M=156 GeV/#font[12]{c}^{2})";}

      Pred8TeV    = ((TH2D*)GetObjectFromPath(InputFile, string("Data7TeV/Pred_") + HistoSuffix   ))->ProjectionY("TmpPredMass"   ,CutIndex+1,CutIndex+1,"o");
      Data8TeV    = ((TH2D*)GetObjectFromPath(InputFile, string("Data7TeV/"     ) + HistoSuffix   ))->ProjectionY("TmpDataMass"   ,CutIndex+1,CutIndex+1,"o");
//    Pred7TeV    = ((TH2D*)GetObjectFromPath(InputFile, string("Data7TeV/Pred_") + HistoSuffix   ))->ProjectionY("TmpPred7TeVMass" ,CutIndex+1,CutIndex+1,"o");
//    Data7TeV    = ((TH2D*)GetObjectFromPath(InputFile, string("Data7TeV/"     ) + HistoSuffix   ))->ProjectionY("TmpData7TeVMass" ,CutIndex+1,CutIndex+1,"o");
      MCPred    = ((TH2D*)GetObjectFromPath(InputFile, string("MCTr_7TeV/Pred_"  ) + HistoSuffix   ))->ProjectionY("TmpMCPred"     ,CutIndex+1,CutIndex+1,"o");
      MC        = ((TH2D*)GetObjectFromPath(InputFile, string("MCTr_7TeV/"       ) + HistoSuffix   ))->ProjectionY("TmpMCMass"     ,CutIndex+1,CutIndex+1,"o");
      Signal    = ((TH2D*)GetObjectFromPath(InputFile, string(SName+"/"     ) + HistoSuffix   ))->ProjectionY("TmpSignalMass" ,CutIndex+1,CutIndex+1,"o");
   }

   //rescale Data7TeV & MC samples to prediction of 2012 data
   if(Data8TeV && Pred8TeV){
      if(Data7TeV && Pred7TeV)Data7TeV->Scale(Pred8TeV->Integral()/Pred7TeV->Integral());
      if(Pred7TeV)          Pred7TeV->Scale(Pred8TeV->Integral()/Pred7TeV->Integral());
      if(MC)              MC    ->Scale(Pred8TeV->Integral()/MCPred->Integral());
      if(MCPred)          MCPred->Scale(Pred8TeV->Integral()/MCPred->Integral());
   }

   //compute integral for few mass window
   if(Data8TeV && Pred8TeV){
      for(double M=0;M<=1000;M+=100){
	if(M>400 && (int)M%200!=0)continue;
         double D = Data8TeV->Integral( Data8TeV->GetXaxis()->FindBin(M),  Data8TeV->GetXaxis()->FindBin(2000.0));
         double P = Pred8TeV->Integral( Pred8TeV->GetXaxis()->FindBin(M),  Pred8TeV->GetXaxis()->FindBin(2000.0));
         double Perr = 0; for(int i=Pred8TeV->GetXaxis()->FindBin(M);i<Pred8TeV->GetXaxis()->FindBin(2000.0);i++){ Perr += pow(Pred8TeV->GetBinError(i),2); }  Perr = sqrt(Perr);
         printf("%4.0f<M<2000 --> Obs=%9.3f Data-Pred = %9.3f +- %8.3f(syst+stat) %9.3f (syst) %9.3f (stat)\n", M, D, P, sqrt(Perr*Perr + pow(P*(2*SystError),2)), P*(2*SystError), Perr);
      }
   }

   //Rebin the histograms and find who is the highest
   double Max = 1.0; double Min=0.01;
   if(Data8TeV){Data8TeV->Rebin(4);  Max=std::max(Max, Data8TeV->GetMaximum());}
   if(Pred8TeV){Pred8TeV->Rebin(4);  Max=std::max(Max, Pred8TeV->GetMaximum());}
   if(Data7TeV){Data7TeV->Rebin(4);  Max=std::max(Max, Data7TeV->GetMaximum());}
   if(Pred7TeV){Pred7TeV->Rebin(4);  Max=std::max(Max, Pred7TeV->GetMaximum());}
   if(Signal){Signal->Rebin(4);  Max=std::max(Max, Signal->GetMaximum());}
   if(MC)    {MC    ->Rebin(4);  Max=std::max(Max, MC    ->GetMaximum());}
   if(MCPred){MCPred->Rebin(4);  Max=std::max(Max, MCPred->GetMaximum());}
   Max*=2.5;

   //compute error bands associated to the predictions
   TH1D *Pred8TeVErr=NULL, *Pred7TeVErr=NULL, *PredMCErr=NULL;
   if(Pred8TeV)Pred8TeVErr = (TH1D*) Pred8TeV->Clone("Pred8TeVErr");
   if(Pred7TeV)Pred7TeVErr = (TH1D*) Pred7TeV->Clone("Pred7TeVErr");
   if(MCPred)PredMCErr = (TH1D*) MCPred->Clone("PredMCErr");

   if(Pred8TeV){for(unsigned int i=0;i<(unsigned int)Pred8TeV->GetNbinsX();i++){
      double error = sqrt(pow(Pred8TeVErr->GetBinError(i),2) + pow(Pred8TeVErr->GetBinContent(i)*2*SystError,2));
      Pred8TeVErr->SetBinError(i,error);       
      if(Pred8TeVErr->GetBinContent(i)<Min && i>5){for(unsigned int j=i+1;j<(unsigned int)Pred8TeVErr->GetNbinsX();j++)Pred8TeVErr->SetBinContent(j,0);}
   }}

   if(Pred7TeV){for(unsigned int i=0;i<(unsigned int)Pred7TeV->GetNbinsX();i++){
      double error = sqrt(pow(Pred7TeVErr->GetBinError(i),2) + pow(Pred7TeVErr->GetBinContent(i)*2*SystError,2));
      Pred7TeVErr->SetBinError(i,error);
      if(Pred7TeVErr->GetBinContent(i)<Min && i>5){for(unsigned int j=i+1;j<(unsigned int)Pred7TeVErr->GetNbinsX();j++)Pred7TeVErr->SetBinContent(j,0);}      
   }}

   if(MCPred){for(unsigned int i=0;i<(unsigned int)MCPred->GetNbinsX();i++){
      double error = sqrt(pow(PredMCErr->GetBinError(i),2) + pow(PredMCErr->GetBinContent(i)*2*SystError,2));
      PredMCErr->SetBinError(i,error);
      if(PredMCErr->GetBinContent(i)<Min && i>5){for(unsigned int j=i+1;j<(unsigned int)PredMCErr->GetNbinsX();j++)PredMCErr->SetBinContent(j,0);}
   }}

   //Prepare the canvas for drawing and draw everything on it
   std::vector<string> legend;
   TLegend* leg;
   TCanvas* c1 = new TCanvas("c1","c1,",600,600);
   char YAxisLegend[1024]; sprintf(YAxisLegend,"Tracks / %2.0f GeV/#font[12]{c}^{2}",(Data8TeV!=NULL?Data8TeV:Data7TeV)->GetXaxis()->GetBinWidth(1));

   TH1D* frame = new TH1D("frame", "frame", 1,0,1400);
   frame->GetXaxis()->SetNdivisions(505);
   frame->SetTitle("");
   frame->SetStats(kFALSE);
   frame->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   frame->GetYaxis()->SetTitle(YAxisLegend);
   frame->GetYaxis()->SetTitleOffset(1.50);
   frame->SetMaximum(Max);
   frame->SetMinimum(Min);
   frame->SetAxisRange(0,1400,"X");
   frame->Draw("AXIS");

   if(Signal){   
      Signal->SetMarkerStyle(21);
      Signal->SetMarkerColor(8);
      Signal->SetMarkerSize(1.5);
      Signal->SetLineColor(1);
      Signal->SetFillColor(8);
      Signal->Draw("same HIST");
   }

   if(MCPred){
//      PredMCErr->SetLineStyle(0);
      PredMCErr->SetLineColor(9);
      PredMCErr->SetFillColor(9);
      PredMCErr->SetFillStyle(1001);
      PredMCErr->SetMarkerStyle(23);
      PredMCErr->SetMarkerColor(9);
      PredMCErr->SetMarkerSize(1.0);
      PredMCErr->Draw("same E5");

      MCPred->SetMarkerStyle(29);
      MCPred->SetMarkerColor(9);
      MCPred->SetMarkerSize(1.5);
      MCPred->SetLineColor(9);
      MCPred->SetFillColor(0);
      MCPred->Draw("same HIST P");
   }

   if(MC){
      MC->SetFillStyle(3002);
      MC->SetLineColor(22);
      MC->SetFillColor(11);
      MC->SetMarkerStyle(0);
      MC->Draw("same HIST E1");
   }

   if(Pred7TeV){
      Pred7TeVErr->SetLineColor(7);
      Pred7TeVErr->SetFillColor(7);
      Pred7TeVErr->SetFillStyle(1001);
      Pred7TeVErr->SetMarkerStyle(22);
      Pred7TeVErr->SetMarkerColor(7);
      Pred7TeVErr->SetMarkerSize(1.0);
      Pred7TeVErr->Draw("same E5");

      Pred7TeV->SetMarkerStyle(26);
      Pred7TeV->SetMarkerColor(2);
      Pred7TeV->SetMarkerSize(1.5);
      Pred7TeV->SetLineColor(2);
      Pred7TeV->SetFillColor(0);
      Pred7TeV->Draw("same HIST P");
   }

   if(Data7TeV){
      Data7TeV->SetBinContent(Data7TeV->GetNbinsX(), Data7TeV->GetBinContent(Data7TeV->GetNbinsX()) + Data7TeV->GetBinContent(Data7TeV->GetNbinsX()+1));
      Data7TeV->SetMarkerStyle(24);
      Data7TeV->SetMarkerColor(1);
      Data7TeV->SetMarkerSize(1.0);
      Data7TeV->SetLineColor(1);
      Data7TeV->SetFillColor(0);
      Data7TeV->Draw("E1 same");
   }

   if(Pred8TeV){
      Pred8TeVErr->SetLineColor(5);
      Pred8TeVErr->SetFillColor(5);
      Pred8TeVErr->SetFillStyle(1001);
      Pred8TeVErr->SetMarkerStyle(22);
      Pred8TeVErr->SetMarkerColor(5);
      Pred8TeVErr->SetMarkerSize(1.0);
      Pred8TeVErr->Draw("same E5");

      Pred8TeV->SetMarkerStyle(22);
      Pred8TeV->SetMarkerColor(2);
      Pred8TeV->SetMarkerSize(1.5);
      Pred8TeV->SetLineColor(2);
      Pred8TeV->SetFillColor(0);
      Pred8TeV->Draw("same HIST P");
   }

   if(Data8TeV){
      Data8TeV->SetBinContent(Data8TeV->GetNbinsX(), Data8TeV->GetBinContent(Data8TeV->GetNbinsX()) + Data8TeV->GetBinContent(Data8TeV->GetNbinsX()+1));
      Data8TeV->SetMarkerStyle(20);
      Data8TeV->SetMarkerColor(1);
      Data8TeV->SetMarkerSize(1.0);
      Data8TeV->SetLineColor(1);
      Data8TeV->SetFillColor(0);
      Data8TeV->Draw("E1 same");
   }

   //Fill the legend
   if(IsTkOnly) leg = new TLegend(0.82,0.93,0.25,0.66);
   else         leg = new TLegend(0.79,0.93,0.25,0.66);
//   leg->SetHeader(LegendFromType(InputPattern).c_str());
   leg->SetFillStyle(0);
   leg->SetBorderSize(0);
   if(Data8TeV){leg->AddEntry(Data8TeV, "Observed"        ,"P");}
   if(Pred8TeV){TH1D* PredLeg8TeV = (TH1D*)Pred8TeV->Clone("RescLeg12");
      PredLeg8TeV->SetFillColor(Pred8TeVErr->GetFillColor());
      PredLeg8TeV->SetFillStyle(Pred8TeVErr->GetFillStyle());
      leg->AddEntry(PredLeg8TeV, "Data8TeV-based SM prediction"  ,"PF");
   }
   if(Data7TeV){leg->AddEntry(Data7TeV, "Observed (2011)"     ,"P");}
   if(Pred7TeV){TH1D* PredLeg7TeV = (TH1D*)Pred7TeV->Clone("RescLeg11");
      PredLeg7TeV->SetFillColor(Pred7TeVErr->GetFillColor());
      PredLeg7TeV->SetFillStyle(Pred7TeVErr->GetFillStyle());
      leg->AddEntry(PredLeg7TeV, "Data7TeV-based SM prediction"  ,"PF");
   }
   if(MC    ){leg->AddEntry(MC, "Simulation"     ,"LF");}
   if(MCPred){TH1D* MCPredLeg = (TH1D*) MCPred->Clone("RescMCLeg");
      MCPredLeg->SetFillColor(PredMCErr->GetFillColor());
      MCPredLeg->SetFillStyle(PredMCErr->GetFillStyle());
      leg->AddEntry(MCPredLeg, "SM prediction (MC)"  ,"PF");
   }
   if(Signal)leg->AddEntry(Signal, SLeg.c_str()              ,"F");
   leg->Draw();

   //add CMS label and save
   DrawPreliminary(LegendFromType(InputPattern), SQRTS, IntegratedLuminosityFromE(SQRTS));
   c1->SetLogy(true);
   SaveCanvas(c1, InputPattern, string("Rescale_") + HistoSuffix + "_" + DataName);


   delete c1;
   InputFile->Close();
}

// make some control plots to show that ABCD method can be used
void PredictionAndControlPlot(string InputPattern, string Data, unsigned int CutIndex, unsigned int CutIndex_Flip){
   if(Data.find("7TeV")!=string::npos){SQRTS=7.0;}else{SQRTS=8.0;}

   TCanvas* c1;
   TObject** Histos = new TObject*[10];
   std::vector<string> legend;
   TypeMode = TypeFromPattern(InputPattern); 
   string LegendTitle = LegendFromType(InputPattern);;

   TFile* InputFile = new TFile((InputPattern + "/Histos.root").c_str());
   TH1D* CtrlPt_S1_Is         = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S1_Is" ); CtrlPt_S1_Is ->Rebin(5);
   TH1D* CtrlPt_S1_Im         = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S1_Im" ); CtrlPt_S1_Im ->Rebin(1);
   TH1D* CtrlPt_S1_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S1_TOF"); CtrlPt_S1_TOF->Rebin(1);
   TH1D* CtrlPt_S2_Is         = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S2_Is" ); CtrlPt_S2_Is ->Rebin(5);
   TH1D* CtrlPt_S2_Im         = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S2_Im" ); CtrlPt_S2_Im ->Rebin(1);
   TH1D* CtrlPt_S2_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S2_TOF"); CtrlPt_S2_TOF->Rebin(1);
   TH1D* CtrlPt_S3_Is         = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S3_Is" ); CtrlPt_S3_Is ->Rebin(5);
   TH1D* CtrlPt_S3_Im         = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S3_Im" ); CtrlPt_S3_Im ->Rebin(1);
   TH1D* CtrlPt_S3_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S3_TOF"); CtrlPt_S3_TOF->Rebin(1);
   TH1D* CtrlPt_S4_Is         = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S4_Is" ); CtrlPt_S4_Is ->Rebin(5);
   TH1D* CtrlPt_S4_Im         = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S4_Im" ); CtrlPt_S4_Im ->Rebin(1);
   TH1D* CtrlPt_S4_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S4_TOF"); CtrlPt_S4_TOF->Rebin(1);

   TH1D* CtrlIs_S1_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlIs_S1_TOF"); CtrlIs_S1_TOF->Rebin(1);
   TH1D* CtrlIs_S2_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlIs_S2_TOF"); CtrlIs_S2_TOF->Rebin(1);
   TH1D* CtrlIs_S3_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlIs_S3_TOF"); CtrlIs_S3_TOF->Rebin(1);
   TH1D* CtrlIs_S4_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlIs_S4_TOF"); CtrlIs_S4_TOF->Rebin(1);

   TH1D* CtrlIm_S1_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlIm_S1_TOF"); CtrlIm_S1_TOF->Rebin(1);
   TH1D* CtrlIm_S2_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlIm_S2_TOF"); CtrlIm_S2_TOF->Rebin(1);
   TH1D* CtrlIm_S3_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlIm_S3_TOF"); CtrlIm_S3_TOF->Rebin(1);
   TH1D* CtrlIm_S4_TOF        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlIm_S4_TOF"); CtrlIm_S4_TOF->Rebin(1);

   TH1D* CtrlPt_S1_TOF_Binned[MaxPredBins];
   TH1D* CtrlPt_S2_TOF_Binned[MaxPredBins];
   TH1D* CtrlPt_S3_TOF_Binned[MaxPredBins];
   TH1D* CtrlPt_S4_TOF_Binned[MaxPredBins];

   if(TypeMode==3) PredBins=6;
   for(int i=0; i<PredBins; i++) {
     char Suffix[1024];
     sprintf(Suffix,"_%i",i);
     string Bin=Suffix;

     CtrlPt_S1_TOF_Binned[i]        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S1_TOF_Binned"+Bin); CtrlPt_S1_TOF_Binned[i]->Rebin(1);
     CtrlPt_S2_TOF_Binned[i]        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S2_TOF_Binned"+Bin); CtrlPt_S2_TOF_Binned[i]->Rebin(1);
     CtrlPt_S3_TOF_Binned[i]        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S3_TOF_Binned"+Bin); CtrlPt_S3_TOF_Binned[i]->Rebin(1);
     CtrlPt_S4_TOF_Binned[i]        = (TH1D*)GetObjectFromPath(InputFile, Data+"/CtrlPt_S4_TOF_Binned"+Bin); CtrlPt_S4_TOF_Binned[i]->Rebin(1);
   }

/*
   std::vector<std::string> PtLimitsNames;
   if(TypeMode!=3) {
     PtLimitsNames.push_back(" 50<p_{T}< 60 GeV");
     PtLimitsNames.push_back(" 60<p_{T}< 80 GeV");
     PtLimitsNames.push_back(" 80<p_{T}<100 GeV");
     PtLimitsNames.push_back("100<p_{T}");
   }
   else {
     PtLimitsNames.push_back(" 80<p_{T}< 120 GeV");
     PtLimitsNames.push_back(" 1200<p_{T}< 170 GeV");
     PtLimitsNames.push_back(" 170<p_{T}<240 GeV");
     PtLimitsNames.push_back("240<p_{T}");
   }

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlPt_S1_Is->Integral()>0)CtrlPt_S1_Is->Scale(1/CtrlPt_S1_Is->Integral());
   if(CtrlPt_S2_Is->Integral()>0)CtrlPt_S2_Is->Scale(1/CtrlPt_S2_Is->Integral());
   if(CtrlPt_S3_Is->Integral()>0)CtrlPt_S3_Is->Scale(1/CtrlPt_S3_Is->Integral());
   if(CtrlPt_S4_Is->Integral()>0)CtrlPt_S4_Is->Scale(1/CtrlPt_S4_Is->Integral());
   Histos[0] = CtrlPt_S1_Is;                     legend.push_back(PtLimitsNames[0]);
   Histos[1] = CtrlPt_S2_Is;                     legend.push_back(PtLimitsNames[1]);
   Histos[2] = CtrlPt_S3_Is;                     legend.push_back(PtLimitsNames[2]);
   Histos[3] = CtrlPt_S4_Is;                     legend.push_back(PtLimitsNames[3]);
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend, "arbitrary units",TypeMode!=5?0:0.7,TypeMode!=5?0.5:1.0, 0,0);
   DrawLegend(Histos,legend,"", "P");
   c1->SetLogy(true);
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,InputPattern,string("Control_")+Data+"_Pt_IsSpectrum");
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlPt_S1_Im->Integral()>0)CtrlPt_S1_Im->Scale(1/CtrlPt_S1_Im->Integral());
   if(CtrlPt_S2_Im->Integral()>0)CtrlPt_S2_Im->Scale(1/CtrlPt_S2_Im->Integral());
   if(CtrlPt_S3_Im->Integral()>0)CtrlPt_S3_Im->Scale(1/CtrlPt_S3_Im->Integral());
   if(CtrlPt_S4_Im->Integral()>0)CtrlPt_S4_Im->Scale(1/CtrlPt_S4_Im->Integral());
   Histos[0] = CtrlPt_S1_Im;                     legend.push_back(PtLimitsNames[0]);
   Histos[1] = CtrlPt_S2_Im;                     legend.push_back(PtLimitsNames[1]);
   Histos[2] = CtrlPt_S3_Im;                     legend.push_back(PtLimitsNames[2]);
   Histos[3] = CtrlPt_S4_Im;                     legend.push_back(PtLimitsNames[3]);
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend, "arbitrary units", 3.0,5, 0,0);
   DrawLegend(Histos,legend,"","P");
   c1->SetLogy(true);
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,InputPattern,string("Control_")+Data+"_Pt_ImSpectrum");
   delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlPt_S1_TOF->Integral()>0)CtrlPt_S1_TOF->Scale(1/CtrlPt_S1_TOF->Integral());
   if(CtrlPt_S2_TOF->Integral()>0)CtrlPt_S2_TOF->Scale(1/CtrlPt_S2_TOF->Integral());
   if(CtrlPt_S3_TOF->Integral()>0)CtrlPt_S3_TOF->Scale(1/CtrlPt_S3_TOF->Integral());
   if(CtrlPt_S4_TOF->Integral()>0)CtrlPt_S4_TOF->Scale(1/CtrlPt_S4_TOF->Integral());
   Histos[0] = CtrlPt_S1_TOF;                    legend.push_back(PtLimitsNames[0]);
   Histos[1] = CtrlPt_S2_TOF;                    legend.push_back(PtLimitsNames[1]);
   Histos[2] = CtrlPt_S3_TOF;                    legend.push_back(PtLimitsNames[2]);
   Histos[3] = CtrlPt_S4_TOF;                    legend.push_back(PtLimitsNames[3]);
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "arbitrary units", 0,2, 0,0); 
   DrawLegend(Histos,legend, "" ,"P");
   c1->SetLogy(true);
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   if(TypeMode>=2)SaveCanvas(c1,InputPattern,string("Control_")+Data+"_Pt_TOFSpectrum");
   c1->SetLogy(false);
   if(TypeMode>=2)SaveCanvas(c1,InputPattern,string("Control_")+Data+"_Pt_TOFSpectrumNoLog");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlIs_S1_TOF->Integral()>0)CtrlIs_S1_TOF->Scale(1/CtrlIs_S1_TOF->Integral());
   if(CtrlIs_S2_TOF->Integral()>0)CtrlIs_S2_TOF->Scale(1/CtrlIs_S2_TOF->Integral());
   if(CtrlIs_S3_TOF->Integral()>0)CtrlIs_S3_TOF->Scale(1/CtrlIs_S3_TOF->Integral());
   if(CtrlIs_S4_TOF->Integral()>0)CtrlIs_S4_TOF->Scale(1/CtrlIs_S4_TOF->Integral());
   Histos[0] = CtrlIs_S1_TOF;                     legend.push_back("0.0<I_{as}<0.05");
   Histos[1] = CtrlIs_S2_TOF;                     legend.push_back("0.05<I_{as}<0.1");
   Histos[2] = CtrlIs_S3_TOF;                     legend.push_back("0.1<I_{as}<0.2");
   Histos[3] = CtrlIs_S4_TOF;                     legend.push_back("0.2<I_{as}");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "arbitrary units", 1,1.7, 0,0);
   DrawLegend(Histos,legend, "","P");
   c1->SetLogy(false);
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   if(TypeMode>=2)SaveCanvas(c1,InputPattern,string("Control_")+Data+"_Is_TOFSpectrum");

   c1->SetLogy(true);
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   DrawLegend(Histos,legend, "","P");
   if(TypeMode>=2)SaveCanvas(c1,InputPattern,string("Control_")+Data+"_Is_TOFSpectrumLog");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlIm_S1_TOF->Integral()>0)CtrlIm_S1_TOF->Scale(1/CtrlIm_S1_TOF->Integral());
   if(CtrlIm_S2_TOF->Integral()>0)CtrlIm_S2_TOF->Scale(1/CtrlIm_S2_TOF->Integral());
   if(CtrlIm_S3_TOF->Integral()>0)CtrlIm_S3_TOF->Scale(1/CtrlIm_S3_TOF->Integral());
   if(CtrlIm_S4_TOF->Integral()>0)CtrlIm_S4_TOF->Scale(1/CtrlIm_S4_TOF->Integral());
   Histos[0] = CtrlIm_S1_TOF;                     legend.push_back("3.5<I_{as}<3.8");
   Histos[1] = CtrlIm_S2_TOF;                     legend.push_back("3.8<I_{as}<4.1");
   Histos[2] = CtrlIm_S3_TOF;                     legend.push_back("4.1<I_{as}<4.4");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "arbitrary units", 1,1.7, 0,0);
   DrawLegend(Histos,legend,"","P");
   c1->SetLogy(false);
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   if(TypeMode>=2)SaveCanvas(c1,InputPattern,string("Control_")+Data+"_Im_TOFSpectrum");
   delete c1;

   for(int i=0; i<PredBins; i++) {
     char Suffix[1024];
     sprintf(Suffix,"_%i",i);
     string Bin=Suffix;

     c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
     if(CtrlPt_S1_TOF_Binned[i]->Integral()>0)CtrlPt_S1_TOF_Binned[i]->Scale(1/CtrlPt_S1_TOF_Binned[i]->Integral());
     if(CtrlPt_S2_TOF_Binned[i]->Integral()>0)CtrlPt_S2_TOF_Binned[i]->Scale(1/CtrlPt_S2_TOF_Binned[i]->Integral());
     if(CtrlPt_S3_TOF_Binned[i]->Integral()>0)CtrlPt_S3_TOF_Binned[i]->Scale(1/CtrlPt_S3_TOF_Binned[i]->Integral());
     if(CtrlPt_S4_TOF_Binned[i]->Integral()>0)CtrlPt_S4_TOF_Binned[i]->Scale(1/CtrlPt_S4_TOF_Binned[i]->Integral());
     Histos[0] = CtrlPt_S1_TOF_Binned[i];                    legend.push_back(PtLimitsNames[0]);
     Histos[1] = CtrlPt_S2_TOF_Binned[i];                    legend.push_back(PtLimitsNames[1]);
     Histos[2] = CtrlPt_S3_TOF_Binned[i];                    legend.push_back(PtLimitsNames[2]);
     Histos[3] = CtrlPt_S4_TOF_Binned[i];                    legend.push_back(PtLimitsNames[3]);
     DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "arbitrary units", 0,2, 0,0);
     DrawLegend(Histos,legend,LegendTitle,"P");
     c1->SetLogy(true);
     DrawPreliminary(SQRTS, IntegratedLuminosity);
     if(TypeMode>=2)SaveCanvas(c1,InputPattern,string("Control_")+Data+"_Pt_TOFSpectrum_Binned"+Bin);
     c1->SetLogy(false);
     if(TypeMode>=2)SaveCanvas(c1,InputPattern,string("Control_")+Data+"_Pt_TOFSpectrumNoLog_Binned"+Bin);
     delete c1;
   }

   if(TypeMode<3) {//These plots only made for analyses using mass distribution
   //Show P, I and TOF distribution in the signal region (observed and predicted)
   TH2D* Pred_P                = (TH2D*)GetObjectFromPath(InputFile, Data+"/Pred_P");
   TH2D* Pred_I                = (TH2D*)GetObjectFromPath(InputFile, Data+"/Pred_I");
   TH2D* Pred_TOF              = (TH2D*)GetObjectFromPath(InputFile, Data+"/Pred_TOF");
   TH2D* Data_I                = (TH2D*)GetObjectFromPath(InputFile, Data+"/RegionD_I");   
   TH2D* Data_P                = (TH2D*)GetObjectFromPath(InputFile, Data+"/RegionD_P");   
   TH2D* Data_TOF              = (TH2D*)GetObjectFromPath(InputFile, Data+"/RegionD_TOF"); 

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   c1->SetLogy(true);
   Histos[0] = (TH1D*)(Data_P->ProjectionY("PA",CutIndex+1,CutIndex+1,"o"));   legend.push_back("Observed");
   Histos[1] = (TH1D*)(Pred_P->ProjectionY("PB",CutIndex+1,CutIndex+1,"o"));   legend.push_back("Predicted");
   ((TH1D*)Histos[0])->Scale(1/std::max(((TH1D*)Histos[0])->Integral(),1.0));
   ((TH1D*)Histos[1])->Scale(1/std::max(((TH1D*)Histos[1])->Integral(),1.0));
   ((TH1D*)Histos[0])->Rebin(10);
   ((TH1D*)Histos[1])->Rebin(10);  
   DrawSuperposedHistos((TH1**)Histos, legend, "Hist E1",  "p (Gev/c)", "u.a.", 0,1500, 0,0);
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,InputPattern,string("Prediction_")+Data+"_PSpectrum");
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
   DrawSuperposedHistos((TH1**)Histos, legend, "Hist E1",  dEdxM_Legend, "u.a.", 0,6, 0,0);
   DrawLegend(Histos,legend,"", "P");
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,InputPattern,string("Prediction_")+Data+"_ISpectrum");
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
   DrawLegend(Histos,legend,"", "P");
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   if(TypeMode>=2)SaveCanvas(c1,InputPattern,string("Prediction_")+Data+"_TOFSpectrum");
   delete Histos[0]; delete Histos[1];
   delete c1;

   //Show P, I and TOF distribution in the region with TOF < 1(observed and predicted)
   TH2D* Pred_P_Flip                = (TH2D*)GetObjectFromPath(InputFile, Data+"/Pred_P_Flip");
   TH2D* Pred_I_Flip                 = (TH2D*)GetObjectFromPath(InputFile, Data+"/Pred_I_Flip");
   TH2D* Pred_TOF_Flip               = (TH2D*)GetObjectFromPath(InputFile, Data+"/Pred_TOF_Flip");
   TH2D* Data_I_Flip                 = (TH2D*)GetObjectFromPath(InputFile, Data+"/RegionD_I_Flip");   
   TH2D* Data_P_Flip                 = (TH2D*)GetObjectFromPath(InputFile, Data+"/RegionD_P_Flip");   
   TH2D* Data_TOF_Flip               = (TH2D*)GetObjectFromPath(InputFile, Data+"/RegionD_TOF_Flip"); 

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   c1->SetLogy(true);
   Histos[0] = (TH1D*)(Data_P_Flip ->ProjectionY("PA_Flip",CutIndex_Flip+1,CutIndex_Flip+1,"o"));   legend.push_back("Observed");
   Histos[1] = (TH1D*)(Pred_P_Flip ->ProjectionY("PB_Flip",CutIndex_Flip+1,CutIndex_Flip+1,"o"));   legend.push_back("Predicted");
   ((TH1D*)Histos[0])->Scale(1/std::max(((TH1D*)Histos[0])->Integral(),1.0));
   ((TH1D*)Histos[1])->Scale(1/std::max(((TH1D*)Histos[1])->Integral(),1.0));
   ((TH1D*)Histos[0])->Rebin(10);
   ((TH1D*)Histos[1])->Rebin(10);  
   DrawSuperposedHistos((TH1**)Histos, legend, "Hist E1",  "p (Gev/c)", "u.a.", 0,1500, 0,0);
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,InputPattern,string("Prediction_")+Data+"_PSpectrum_Flip");
   delete Histos[0]; delete Histos[1];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   c1->SetLogy(true);
   Histos[0] = (TH1D*)(Data_I_Flip ->ProjectionY("IA_Flip",CutIndex_Flip+1,CutIndex_Flip+1,"o"));   legend.push_back("Observed");
   Histos[1] = (TH1D*)(Pred_I_Flip ->ProjectionY("IB_Flip",CutIndex_Flip+1,CutIndex_Flip+1,"o"));   legend.push_back("Predicted");
   ((TH1D*)Histos[0])->Scale(1/std::max(((TH1D*)Histos[0])->Integral(),1.0));
   ((TH1D*)Histos[1])->Scale(1/std::max(((TH1D*)Histos[1])->Integral(),1.0));
   ((TH1D*)Histos[0])->Rebin(2); 
   ((TH1D*)Histos[1])->Rebin(2);
   DrawSuperposedHistos((TH1**)Histos, legend, "Hist E1",  dEdxM_Legend, "u.a.", 0,6, 0,0);
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,InputPattern,string("Prediction_")+Data+"_ISpectrum_Flip");
   delete Histos[0]; delete Histos[1];
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   c1->SetLogy(true);
   Histos[0] = (TH1D*)(Data_TOF_Flip ->ProjectionY("TA_Flip",CutIndex_Flip+1,CutIndex_Flip+1,"o"));   legend.push_back("Observed");
   Histos[1] = (TH1D*)(Pred_TOF_Flip ->ProjectionY("TB_Flip",CutIndex_Flip+1,CutIndex_Flip+1,"o"));   legend.push_back("Predicted");
   ((TH1D*)Histos[0])->Scale(1/std::max(((TH1D*)Histos[0])->Integral(),1.0));
   ((TH1D*)Histos[1])->Scale(1/std::max(((TH1D*)Histos[1])->Integral(),1.0));
   ((TH1D*)Histos[0])->Rebin(2); 
   ((TH1D*)Histos[1])->Rebin(2);
   DrawSuperposedHistos((TH1**)Histos, legend, "Hist E1",  "1/#beta", "u.a.", 0,0, 0,0);
   DrawLegend(Histos,legend, "" ,"P");
   DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   if(TypeMode>=2)SaveCanvas(c1,InputPattern,string("Prediction_")+Data+"_TOFSpectrum_Flip");
   delete Histos[0]; delete Histos[1];
   delete c1;
   }
*/
   if(TypeMode==5){
      TH1D* HCuts_Pt              = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
      TH1D* HCuts_I               = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");
      TH1D* HCuts_TOF             = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");
      TH1D* H_D                   = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_D");
      TH1D* H_P                   = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_P");

      std::map<double, TGraphErrors*> mapPred;
      std::map<double, TGraphErrors*> mapObs;

      double max = 1;
      double xmin=100; double xmax=0;
      for(int CutIndex_=1;CutIndex_<H_P->GetNbinsX();CutIndex_++){
         double PtCut = HCuts_Pt->GetBinContent(CutIndex_+1);
         int N = 0;for(int i=CutIndex_;i<H_P->GetNbinsX();i++){if(HCuts_Pt->GetBinContent(i+1)==PtCut){N++;}else{break;}}
         mapPred[PtCut] = new TGraphErrors(N);
         mapObs[PtCut] = new TGraphErrors(N);
         for(int i=0;i<N;i++){
           xmin = std::min(xmin, HCuts_I->GetBinContent(CutIndex_+i+1));
           xmax = std::max(xmax, HCuts_I->GetBinContent(CutIndex_+i+1));

           max = std::max(max, H_P->GetBinContent(CutIndex_+i+1));
           mapPred[PtCut]->SetPoint     (i, HCuts_I->GetBinContent(CutIndex_+i+1), H_P->GetBinContent(CutIndex_+i+1));
           mapPred[PtCut]->SetPointError(i, 0                                    , H_P->GetBinError  (CutIndex_+i+1));

           max = std::max(max, H_D->GetBinContent(CutIndex_+i+1));
           mapObs [PtCut]->SetPoint     (i, HCuts_I->GetBinContent(CutIndex_+i+1), H_D->GetBinContent(CutIndex_+i+1)); 
           mapObs [PtCut]->SetPointError(i, 0                                    , H_D->GetBinError  (CutIndex_+i+1));
         }
         CutIndex_+=N-1;
      }


      c1 = new TCanvas("c1","c1,",600,600);
      c1->SetLogy(true);

      TH1D* frame = new TH1D("frame", "frame", 1,std::max(xmin,0.05),xmax);
      frame->GetXaxis()->SetNdivisions(505);
      frame->SetTitle("");
      frame->SetStats(kFALSE);
      frame->GetXaxis()->SetTitle(dEdxS_Legend.c_str());
      frame->GetYaxis()->SetTitle("#Events");
      frame->GetYaxis()->SetTitleOffset(1.50);
      frame->SetMaximum(max);
      frame->SetMinimum(0.1);
      frame->Draw("AXIS");

      mapObs[75.0]->SetMarkerColor(2);
      mapObs[75.0]->SetMarkerStyle(20); 
      mapObs[75.0]->Draw("P");
      mapPred[75.0]->SetLineColor(2);
      mapPred[75.0]->SetLineWidth(2.0);
      mapPred[75.0]->Draw("C");

      mapObs[100.0]->SetMarkerColor(4);
      mapObs[100.0]->SetMarkerStyle(20);
      mapObs[100.0]->Draw("P");
      mapPred[100.0]->SetLineColor(4);  
      mapPred[100.0]->SetLineWidth(2.0);
      mapPred[100.0]->Draw("C");

      mapObs[125.0]->SetMarkerColor(8);
      mapObs[125.0]->SetMarkerStyle(20);
      mapObs[125.0]->Draw("P");
      mapPred[125.0]->SetLineColor(8);
      mapPred[125.0]->SetLineWidth(2.0);
      mapPred[125.0]->Draw("C");




      TLegend* LEG = new TLegend(0.45,0.65,0.65,0.90);
      LEG->SetFillColor(0);
      LEG->SetFillStyle(0);
      LEG->SetBorderSize(0);

      TH1D* obsLeg = (TH1D*)mapObs [75.0]->Clone("ObsLeg");
      obsLeg->SetMarkerColor(1);
      LEG->AddEntry(obsLeg, "Data"    ,"P");
      LEG->AddEntry(mapPred[75.0], "Pred (pT>75GeV)","L");
      LEG->AddEntry(mapPred[100.0], "Pred (pT>100GeV)","L");
      LEG->AddEntry(mapPred[125.0], "Pred (pT>125GeV)","L");
      LEG->Draw("same");


      DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
      SaveCanvas(c1,InputPattern,string("Prediction_")+Data+"_LQ_PredVsObs");
      delete c1;
   }



   //README: Draw a map of the prediction/observation for the various selection
   //TH1D* HCuts_Pt              = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
   //TH1D* HCuts_I               = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");
   //TH1D* HCuts_TOF             = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");
   //TH1D* H_D                   = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_D");
   //TH1D* H_P                   = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_P");
   //c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   //TH2D* DataVsPred = new TH2D("DataVsPred","DataVsPred",17,30,200,  8,0.05,0.5); 
   //TH2D* DataMap    = new TH2D("DataMap"   ,"DataMap"   ,17,30,200,  8,0.05,0.5);
   //TH2D* PredMap    = new TH2D("PredMap"   ,"PredMap"   ,17,30,200,  8,0.05,0.5);
   //for(unsigned int CutIndex_=0;CutIndex_<H_P->GetNbinsX();CutIndex_++){
   //   double P    = H_P->GetBinContent(CutIndex_+1);
   //   double D    = H_D->GetBinContent(CutIndex_+1);
   //   double Err  = sqrt( pow(H_P->GetBinError(CutIndex_+1),2) + std::max(D,1.0)   );
   //   double NSigma = (D-P)/Err;

   //   DataMap->SetBinContent(DataVsPred->GetXaxis()->FindBin(HCuts_Pt->GetBinContent(CutIndex_+1)), DataVsPred->GetYaxis()->FindBin(HCuts_I->GetBinContent(CutIndex_+1)), D);
   //   PredMap->SetBinContent(DataVsPred->GetXaxis()->FindBin(HCuts_Pt->GetBinContent(CutIndex_+1)), DataVsPred->GetYaxis()->FindBin(HCuts_I->GetBinContent(CutIndex_+1)), P);
   //   if(isnan(P) || P<=0)continue; //Is <=0 only when prediction failed or is not meaningful (i.e. WP=(0,0,0) )
   //   DataVsPred->SetBinContent(DataVsPred->GetXaxis()->FindBin(HCuts_Pt->GetBinContent(CutIndex_+1)), DataVsPred->GetYaxis()->FindBin(HCuts_I->GetBinContent(CutIndex_+1)), NSigma);
   //}
   //DataVsPred->SetMinimum(-3);
   //DataVsPred->SetMaximum(3);
   //Histos[0] = DataVsPred;   legend.push_back("Observed");
   //DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "PtCut", "ICut", 0,0, 0,0);
   //DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   //SaveCanvas(c1,InputPattern,string("Prediction_")+Data+"_PredVsObs");
   //delete c1;

   //PredMap->SetMinimum(1E-2);
   //DataMap->SetMinimum(1E-2);
   //PredMap->SetMaximum(std::max(PredMap->GetMaximum(),DataMap->GetMaximum()));
   //DataMap->SetMaximum(std::max(PredMap->GetMaximum(),DataMap->GetMaximum()));
   //c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   //c1->SetLogz(true);
   //Histos[0] = PredMap;   legend.push_back("Observed");
   //DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "PtCut", "ICut", 0,0, 0,0);
   //DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   //SaveCanvas(c1,InputPattern,string("Prediction_")+Data+"_Pred");
   //delete c1;

   //c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   //c1->SetLogz(true);
   //Histos[0] = DataMap;   legend.push_back("Observed");
   //DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "PtCut", "ICut", 0,0, 0,0);
   //DrawPreliminary(LegendTitle, SQRTS, IntegratedLuminosityFromE(SQRTS));
   //SaveCanvas(c1,InputPattern,string("Prediction_")+Data+"_Data");
   //delete c1;
   InputFile->Close();
}

// print the event flow table for all signal point and Data and MCTr
void CutFlow(string InputPattern, unsigned int CutIndex){

   TFile* InputFile = new TFile((InputPattern + "Histos.root").c_str());
   if(!InputFile)std::cout << "FileProblem\n";

   TH1D*  HCuts_Pt       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
   TH1D*  HCuts_I        = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");
   TH1D*  HCuts_TOF      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");

    char Buffer[1024]; sprintf(Buffer,"%s/CutFlow_%03i_Pt%03.0f_I%05.3f_TOF%04.3f.txt",InputPattern.c_str(),CutIndex,HCuts_Pt->GetBinContent(CutIndex+1),HCuts_I->GetBinContent(CutIndex+1),HCuts_TOF->GetBinContent(CutIndex+1));
    FILE* pFile = fopen(Buffer,"w");
    stPlots plots;

    if(stPlots_InitFromFile(InputFile, plots,"Data8TeV")){
       stPlots_Dump(plots, pFile, CutIndex);
       stPlots_Clear(&plots);}

    if(stPlots_InitFromFile(InputFile, plots,"Data7TeV")){
       stPlots_Dump(plots, pFile, CutIndex);
       stPlots_Clear(&plots); 
    }

    if(stPlots_InitFromFile(InputFile, plots,"MCTr_7TeV")){
       stPlots_Dump(plots, pFile, CutIndex);
       stPlots_Clear(&plots);
    }

    if(stPlots_InitFromFile(InputFile, plots,"MCTr_8TeV")){
      stPlots_Dump(plots, pFile, CutIndex);
      stPlots_Clear(&plots);
    }

    for(unsigned int s=0;s<samples.size();s++){
       if(samples[s].Type!=2 || !samples[s].MakePlot)continue;

       if(stPlots_InitFromFile(InputFile, plots, samples[s].Name)){
          stPlots_Dump(plots, pFile, CutIndex);       
          stPlots_Clear(&plots);
       }
    }

    fclose(pFile);
    InputFile->Close();
}

// make all plots of the preselection and selection variables as well as some plots showing 2D planes
void SelectionPlot(string InputPattern, unsigned int CutIndex){
    string LegendTitle = LegendFromType(InputPattern);;

    TFile* InputFile = new TFile((InputPattern + "Histos.root").c_str());
    stPlots Data8TeVPlots, Data7TeVPlots, MCTr8TeVPlots, MCTr7TeVPlots, Cosmic7TeVPlots, Cosmic8TeVPlots, SignPlots[samples.size()];

    TypeMode = TypeFromPattern(InputPattern);
    stPlots_InitFromFile(InputFile, Data8TeVPlots,"Data8TeV");
    stPlots_InitFromFile(InputFile, Data7TeVPlots,"Data7TeV");
    stPlots_InitFromFile(InputFile, MCTr8TeVPlots  ,"MCTr_8TeV");   
    stPlots_InitFromFile(InputFile, MCTr7TeVPlots  ,"MCTr_7TeV");
    if(TypeMode==3) {
      stPlots_InitFromFile(InputFile, Cosmic8TeVPlots,"Cosmic8TeV");
      //stPlots_InitFromFile(InputFile, Cosmic7TeVPlots,"Cosmic8TeV");
    }

    for(unsigned int s=0;s<samples.size();s++){
       if (samples[s].Name!="Gluino_7TeV_M300_f10" && samples[s].Name!="Gluino_7TeV_M600_f10" && samples[s].Name!="Gluino_7TeV_M800_f10" && samples[s].Name!="Gluino_8TeV_M300_f10" && samples[s].Name!="Gluino_8TeV_M600_f10" && samples[s].Name!="Gluino_8TeV_M800_f10" && samples[s].Name!="GMStau_7TeV_M247" && samples[s].Name!="GMStau_7TeV_M370" && samples[s].Name!="GMStau_7TeV_M494" && samples[s].Name!="GMStau_8TeV_M247" && samples[s].Name!="GMStau_8TeV_M370" && samples[s].Name!="GMStau_8TeV_M494" && samples[s].Name!="DY_7TeV_M100_Q1o3" &&  samples[s].Name!="DY_7TeV_M600_Q1o3" && samples[s].Name!="DY_7TeV_M100_Q2o3" &&  samples[s].Name!="DY_7TeV_M600_Q2o3" && samples[s].Name!="DY_8TeV_M100_Q1o3" &&  samples[s].Name!="DY_8TeV_M600_Q1o3" && samples[s].Name!="DY_8TeV_M100_Q2o3" &&  samples[s].Name!="DY_8TeV_M600_Q2o3") continue;
       if(!stPlots_InitFromFile(InputFile, SignPlots[s],samples[s].Name)){printf("Missing sample %s\n",samples[s].Name.c_str());continue;}
       //stPlots_Draw(SignPlots[s], InputPattern + "/Selection_" +  samples[s].Name, LegendTitle, CutIndex);
    }

    SQRTS=8; stPlots_Draw(Data8TeVPlots, InputPattern + "/Selection_Data8TeV", LegendTitle, CutIndex);
    SQRTS=7; stPlots_Draw(Data7TeVPlots, InputPattern + "/Selection_Data7TeV", LegendTitle, CutIndex);
    SQRTS=8; stPlots_Draw(MCTr8TeVPlots  , InputPattern + "/Selection_MCTr_8TeV"  , LegendTitle, CutIndex);
    SQRTS=7; stPlots_Draw(MCTr7TeVPlots  , InputPattern + "/Selection_MCTr_7TeV"  , LegendTitle, CutIndex);

    if(TypeMode==3) {
      stPlots_Draw(Cosmic8TeVPlots, InputPattern + "/Selection_Cosmic8TeV", LegendTitle, CutIndex);
      //stPlots_Draw(Cosmic11Plots, InputPattern + "/Selection_Cosmic11", LegendTitle, CutIndex);
    }

    stPlots_DrawComparison(InputPattern + "/Selection_Comp_Data"  , LegendTitle, CutIndex, &Data8TeVPlots, &Data7TeVPlots, &MCTr8TeVPlots, &MCTr7TeVPlots);

    if(TypeMode<=2){ SQRTS=7; stPlots_DrawComparison(InputPattern + "/Selection_Comp_7TeV_Gluino", LegendTitle, CutIndex, &Data7TeVPlots, &MCTr7TeVPlots,     &SignPlots[JobIdToIndex("Gluino_7TeV_M300_f10",samples)], &SignPlots[JobIdToIndex("Gluino_7TeV_M600_f10",samples)], &SignPlots[JobIdToIndex("Gluino_7TeV_M800_f10",samples)]);}
    if(TypeMode<=2){ SQRTS=8; stPlots_DrawComparison(InputPattern + "/Selection_Comp_8TeV_Gluino", LegendTitle, CutIndex, &Data8TeVPlots, &MCTr8TeVPlots,     &SignPlots[JobIdToIndex("Gluino_8TeV_M300_f10",samples)], &SignPlots[JobIdToIndex("Gluino_8TeV_M600_f10",samples)], &SignPlots[JobIdToIndex("Gluino_8TeV_M800_f10",samples)]);}
    if(TypeMode==3){ 
      //SQRTS=7; stPlots_DrawComparison(InputPattern + "/Selection_Comp_Cosmic_7TeV", LegendTitle, CutIndex, &Data7TeVPlots,&SignPlots[JobIdToIndex("Gluino_7TeV_M800_f10",samples)]);
      SQRTS=8; stPlots_DrawComparison(InputPattern + "/Selection_Comp_Cosmic_8TeV", LegendTitle, CutIndex, &Data8TeVPlots, &MCTr8TeVPlots, &Cosmic8TeVPlots, &SignPlots[JobIdToIndex("Gluino_8TeV_M800_f10",samples)]);
      //SQRTS=78; stPlots_DrawComparison(InputPattern + "/Selection_Comp_Cosmic_78TeV", LegendTitle, CutIndex, &Data8TeVPlots, &Data7TeVPlots, &Cosmic8TeVPlots, &SignPlots[JobIdToIndex("Gluino_7TeV_M800_f10",samples)], &SignPlots[JobIdToIndex("Gluino_8TeV_M800_f10",samples)]);
    }
    if(TypeMode==5){ SQRTS=7; stPlots_DrawComparison(InputPattern + "/Selection_Comp_7TeV_DY"    , LegendTitle, CutIndex, &Data7TeVPlots, &MCTr7TeVPlots,   &SignPlots[JobIdToIndex("DY_7TeV_M100_Q1o3",samples)], &SignPlots[JobIdToIndex("DY_7TeV_M100_Q2o3",samples)], &SignPlots[JobIdToIndex("DY_7TeV_M600_Q2o3",samples)]);}
    if(TypeMode==5){ SQRTS=8; stPlots_DrawComparison(InputPattern + "/Selection_Comp_8TeV_DY"    , LegendTitle, CutIndex, &Data8TeVPlots, &MCTr8TeVPlots,   &SignPlots[JobIdToIndex("DY_8TeV_M100_Q1o3",samples)], &SignPlots[JobIdToIndex("DY_8TeV_M100_Q2o3",samples)], &SignPlots[JobIdToIndex("DY_8TeV_M600_Q2o3",samples)]);}

    stPlots_Clear(&Data8TeVPlots);
    stPlots_Clear(&Data7TeVPlots);
    stPlots_Clear(&MCTr8TeVPlots);
    stPlots_Clear(&MCTr7TeVPlots);

    for(unsigned int s=0;s<samples.size();s++){
       if (samples[s].Name!="Gluino_7TeV_M300_f10" && samples[s].Name!="Gluino_7TeV_M600_f10" && samples[s].Name!="Gluino_7TeV_M800_f10" && "Gluino_8TeV_M300_f10" && samples[s].Name!="Gluino_8TeV_M600_f10" && samples[s].Name!="Gluino_8TeV_M800_f10" && samples[s].Name!="GMStau_7TeV_M247" && samples[s].Name!="GMStau_7TeV_M370" && samples[s].Name!="GMStau_7TeV_M494" && samples[s].Name!="GMStau_8TeV_M247" && samples[s].Name!="GMStau_8TeV_M370" && samples[s].Name!="GMStau_8TeV_M494" && samples[s].Name!="DY_7TeV_M100_Q1o3" &&  samples[s].Name!="DY_7TeV_M600_Q1o3" && samples[s].Name!="DY_7TeV_M100_Q2o3" &&  samples[s].Name!="DY_7TeV_M600_Q2o3" && samples[s].Name!="DY_8TeV_M100_Q1o3" &&  samples[s].Name!="DY_8TeV_M600_Q1o3" && samples[s].Name!="DY_8TeV_M100_Q2o3" &&  samples[s].Name!="DY_8TeV_M600_Q2o3") continue;
       if(!stPlots_InitFromFile(InputFile, SignPlots[s],samples[s].Name)) continue;
       stPlots_Clear(&SignPlots[s]);
    }
    InputFile->Close();
}


// Determine the systematic uncertainty by computing datadriven prediction from different paths (only works with 3D ABCD method)
void GetSystematicOnPrediction(string InputPattern, string DataName){
   if(DataName.find("7TeV")!=string::npos){SQRTS=7.0;}else{SQRTS=8.0;}

   TypeMode = TypeFromPattern(InputPattern); 
   if(TypeMode!=2)return;

   TFile* InputFile = new TFile((InputPattern + "Histos.root").c_str());
   TH1D*  HCuts_Pt       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
   TH1D*  HCuts_I        = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");
   TH1D*  HCuts_TOF      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");
   TH1D*  H_A            = (TH1D*)GetObjectFromPath(InputFile, DataName+"/H_A");
   TH1D*  H_B            = (TH1D*)GetObjectFromPath(InputFile, DataName+"/H_B");
   TH1D*  H_C            = (TH1D*)GetObjectFromPath(InputFile, DataName+"/H_C");
 //TH1D*  H_D            = (TH1D*)GetObjectFromPath(InputFile, DataName+"/H_D");
   TH1D*  H_E            = (TH1D*)GetObjectFromPath(InputFile, DataName+"/H_E");
   TH1D*  H_F            = (TH1D*)GetObjectFromPath(InputFile, DataName+"/H_F");
   TH1D*  H_G            = (TH1D*)GetObjectFromPath(InputFile, DataName+"/H_G");
   TH1D*  H_H            = (TH1D*)GetObjectFromPath(InputFile, DataName+"/H_H");
 //TH1D*  H_P            = (TH1D*)GetObjectFromPath(InputFile, DataName+"/H_P");

   int    ArrN[6];  ArrN[0] = 0; ArrN[1] = 0; ArrN[2] = 0;  ArrN[3] = 0;  ArrN[4] = 0; ArrN[5] = 0;
   double ArrPred[5][6][20];  double ArrErr[5][6][20];  int ArrPredN[5][6];  for(unsigned int i=0;i<5;i++){for(unsigned int j=0;j<6;j++){ArrPredN[i][j]=0;}}
 //double ArrMean [6][20];
   double ArrSigma[6][20];
   double ArrDist [6][20];
   double ArrSum  [6][20];
   double ArrSyst [6][20];
   double ArrStat [6][20];
   double ArrStatB[6][20];
   double ArrPt   [6][20];
   double ArrI    [6][20];
   double ArrT    [6][20];

   std::vector<int> Index;   std::vector<int> Plot;
   //variation on TOF cut 50, 0.05 1.05->1.2
   Index.push_back(16);      Plot.push_back(0);
   Index.push_back(17);      Plot.push_back(0);
   Index.push_back(18);      Plot.push_back(0);
   Index.push_back(19);      Plot.push_back(0);
   Index.push_back(20);      Plot.push_back(0);
   Index.push_back(21);      Plot.push_back(0);
   Index.push_back(22);      Plot.push_back(0);
   //variation on I cut 50, 0.05->0.225 1.05
   Index.push_back(16);      Plot.push_back(1);
   Index.push_back(30);      Plot.push_back(1);
   Index.push_back(44);      Plot.push_back(1);
   Index.push_back(58);      Plot.push_back(1);
   Index.push_back(72);      Plot.push_back(1);
   Index.push_back(86);      Plot.push_back(1);
   Index.push_back(100);     Plot.push_back(1);
   Index.push_back(114);     Plot.push_back(1);
   //variation on Pt cut 50->115 0.05 1.05
   Index.push_back(16);      Plot.push_back(2);
   Index.push_back(436);     Plot.push_back(2);
   Index.push_back(856);     Plot.push_back(2);
   Index.push_back(1276);    Plot.push_back(2);
   Index.push_back(1696);    Plot.push_back(2);
   Index.push_back(2116);    Plot.push_back(2);
   Index.push_back(2536);    Plot.push_back(2);
   Index.push_back(2746);    Plot.push_back(2);
   //variation on Pt cut 50->115 0.1 1.1 
   Index.push_back(46);      Plot.push_back(3);
   Index.push_back(466);     Plot.push_back(3);
   Index.push_back(886);     Plot.push_back(3);
   Index.push_back(1306);    Plot.push_back(3);
   Index.push_back(1726);    Plot.push_back(3);
   Index.push_back(2146);    Plot.push_back(3);
   Index.push_back(2566);    Plot.push_back(3);
   Index.push_back(2776);    Plot.push_back(3);
   //variation on Pt cut 50->115 0.15 1.05 
   Index.push_back(72);      Plot.push_back(4);
   Index.push_back(492);     Plot.push_back(4);
   Index.push_back(912);     Plot.push_back(4);
   Index.push_back(1332);    Plot.push_back(4);
   Index.push_back(1752);    Plot.push_back(4);
   Index.push_back(2172);    Plot.push_back(4);
   Index.push_back(2592);    Plot.push_back(4);
   Index.push_back(2802);    Plot.push_back(4);
   //Not used
   Index.push_back(82 + 4);  Plot.push_back(5);
   Index.push_back(154+ 4);  Plot.push_back(5);
   Index.push_back(226+ 4);  Plot.push_back(5);
   Index.push_back(298+ 4);  Plot.push_back(5);
   Index.push_back(370+ 4);  Plot.push_back(5);
   Index.push_back(442+ 4);  Plot.push_back(5);
   Index.push_back(514+ 4);  Plot.push_back(5);
   Index.push_back(586+ 4);  Plot.push_back(5);
   Index.push_back(658+ 4);  Plot.push_back(5);
   Index.push_back(730+ 4);  Plot.push_back(5);
   Index.push_back(802+ 4);  Plot.push_back(5);

   for(unsigned int i=0;i<Index.size();i++){      
      int CutIndex = Index[i];
      const double& A=H_A->GetBinContent(CutIndex+1);
      const double& B=H_B->GetBinContent(CutIndex+1);
      const double& C=H_C->GetBinContent(CutIndex+1);
    //const double& D=H_D->GetBinContent(CutIndex+1);
      const double& E=H_E->GetBinContent(CutIndex+1);
      const double& F=H_F->GetBinContent(CutIndex+1);
      const double& G=H_G->GetBinContent(CutIndex+1);
      const double& H=H_H->GetBinContent(CutIndex+1);

      double Pred[5];
      double Err [5]; 
      double N = 0;
      double Sigma = 0;
      double Mean = 0;

      for(unsigned int p=0;p<4;p++){
         Pred[p] = -1;
         Err [p] = -1;
         if(p==0){
             if(A<25 || F<25 || G<25 || E<25)continue;
             Pred[p] = (A*F*G)/(E*E);
             Err [p] =  Pred [p] * sqrt( 1/A + 1/F + 1/G + 4/E);
          }else if(p==1){
             if(A<25 || H<25 || E<25)continue;
             Pred[p] = ((A*H)/E);
             Err [p] =  Pred[p] * sqrt( 1/A+1/H+1/E );
          }else if (p==2){
             if(B<25 || G<25 || E<25)continue;
             Pred[p] = ((B*G)/E); 
             Err [p] =  Pred[p] * sqrt( 1/B+ 1/G+ 1/E );
          }else if (p==3){
             if(F<25 || C<25 || E<25)continue;
             Pred[p] = ((F*C)/E);
             Err [p] =  Pred[p] * sqrt( 1/F + 1/C + 1/E );
          }

          if(Pred[p]>=0){
             N++;
             Mean  += Pred[p]/pow(Err [p],2);
             Sigma += 1      /pow(Err [p],2);
          }         

          ArrPred [p][Plot[i]][ArrN[Plot[i]]] = Pred[p];
          ArrErr  [p][Plot[i]][ArrN[Plot[i]]] = Err [p];
          if(Pred[p]>=0)ArrPredN[p][Plot[i]]++;
      }

      Mean  = Mean/Sigma;
      Sigma = sqrt(Sigma);
        
      double Dist    = fabs(Pred[0] - Mean);
      double Sum=0, Stat=0, Syst=0, StatB=0;

      for(unsigned int p=0;p<4;p++){
         if(Pred[p]>=0){
            Sum   += pow(Pred[p]-Mean,2);
            Stat  += pow(Err [p],2);
            StatB += Err [p];
         }
      }
      Sum  = sqrt(Sum/(N-1));
      Stat = sqrt(Stat)/N;
      StatB= StatB/N;
      Syst = sqrt(Sum*Sum - Stat*Stat);
    //printf("pT>%6.2f I> %6.2f TOF>%6.2f : ", HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1));
    //printf("A =%6.2E, B=%6.2E, C=%6.2E, D=%6.2E E =%6.2E, F=%6.2E, G=%6.2E, H=%6.2E\n", A,B,C,D, E, F, G, H);

      printf("--> N = %1.0f Mean = %8.2E  Sigma=%8.2E  Dist=%8.2E  Sum=%8.2E  Stat=%8.2E  Syst=%8.2E\n", N, Mean, Sigma/Mean, Dist/Mean, Sum/Mean, Stat/Mean, Syst/Mean);
      if(N>0){
       //ArrMean   [Plot[i]][ArrN[Plot[i]]] = Mean;
         ArrSigma  [Plot[i]][ArrN[Plot[i]]] = Sigma/Mean;
         ArrDist   [Plot[i]][ArrN[Plot[i]]] = Dist/Mean;
         ArrSum    [Plot[i]][ArrN[Plot[i]]] = Sum/Mean;
         ArrSyst   [Plot[i]][ArrN[Plot[i]]] = Syst/Mean;
         ArrStat   [Plot[i]][ArrN[Plot[i]]] = Stat/Mean;
         ArrStatB  [Plot[i]][ArrN[Plot[i]]] = StatB/Mean;
         ArrPt     [Plot[i]][ArrN[Plot[i]]] = HCuts_Pt ->GetBinContent(CutIndex+1); ;
         ArrI      [Plot[i]][ArrN[Plot[i]]] = HCuts_I  ->GetBinContent(CutIndex+1); ;
         ArrT      [Plot[i]][ArrN[Plot[i]]] = HCuts_TOF->GetBinContent(CutIndex+1); ;
         ArrN[Plot[i]]++;
      }
   }

   TGraphErrors* graph_T0 = new TGraphErrors(ArrPredN[0][0],ArrT [0],ArrPred[0][0],0,ArrErr[0][0]);   graph_T0->SetLineColor(1);  graph_T0->SetMarkerColor(1);   graph_T0->SetMarkerStyle(20);
   TGraphErrors* graph_T1 = new TGraphErrors(ArrPredN[1][0],ArrT [0],ArrPred[1][0],0,ArrErr[1][0]);   graph_T1->SetLineColor(2);  graph_T1->SetMarkerColor(2);   graph_T1->SetMarkerStyle(21); 
   TGraphErrors* graph_T2 = new TGraphErrors(ArrPredN[2][0],ArrT [0],ArrPred[2][0],0,ArrErr[2][0]);   graph_T2->SetLineColor(4);  graph_T2->SetMarkerColor(4);   graph_T2->SetMarkerStyle(22);
   TGraphErrors* graph_T3 = new TGraphErrors(ArrPredN[3][0],ArrT [0],ArrPred[3][0],0,ArrErr[3][0]);   graph_T3->SetLineColor(8);  graph_T3->SetMarkerColor(8);   graph_T3->SetMarkerStyle(23);
   TGraphErrors* graph_I0 = new TGraphErrors(ArrPredN[0][1],ArrI [1],ArrPred[0][1],0,ArrErr[0][1]);   graph_I0->SetLineColor(1);  graph_I0->SetMarkerColor(1);   graph_I0->SetMarkerStyle(20);
   TGraphErrors* graph_I1 = new TGraphErrors(ArrPredN[1][1],ArrI [1],ArrPred[1][1],0,ArrErr[1][1]);   graph_I1->SetLineColor(2);  graph_I1->SetMarkerColor(2);   graph_I1->SetMarkerStyle(21);
   TGraphErrors* graph_I2 = new TGraphErrors(ArrPredN[2][1],ArrI [1],ArrPred[2][1],0,ArrErr[2][1]);   graph_I2->SetLineColor(4);  graph_I2->SetMarkerColor(4);   graph_I2->SetMarkerStyle(22);
   TGraphErrors* graph_I3 = new TGraphErrors(ArrPredN[3][1],ArrI [1],ArrPred[3][1],0,ArrErr[3][1]);   graph_I3->SetLineColor(8);  graph_I3->SetMarkerColor(8);   graph_I3->SetMarkerStyle(23);
   TGraphErrors* graph_P0 = new TGraphErrors(ArrPredN[0][2],ArrPt[2],ArrPred[0][2],0,ArrErr[0][2]);   graph_P0->SetLineColor(1);  graph_P0->SetMarkerColor(1);   graph_P0->SetMarkerStyle(20);
   TGraphErrors* graph_P1 = new TGraphErrors(ArrPredN[1][2],ArrPt[2],ArrPred[1][2],0,ArrErr[1][2]);   graph_P1->SetLineColor(2);  graph_P1->SetMarkerColor(2);   graph_P1->SetMarkerStyle(21);
   TGraphErrors* graph_P2 = new TGraphErrors(ArrPredN[2][2],ArrPt[2],ArrPred[2][2],0,ArrErr[2][2]);   graph_P2->SetLineColor(4);  graph_P2->SetMarkerColor(4);   graph_P2->SetMarkerStyle(22);
   TGraphErrors* graph_P3 = new TGraphErrors(ArrPredN[3][2],ArrPt[2],ArrPred[3][2],0,ArrErr[3][2]);   graph_P3->SetLineColor(8);  graph_P3->SetMarkerColor(8);   graph_P3->SetMarkerStyle(23);

   TLegend* LEG = NULL;
   LEG = new TLegend(0.50,0.65,0.80,0.90);
   LEG->SetFillColor(0); 
   LEG->SetBorderSize(0);
   LEG->AddEntry(graph_T0, "D=AFG/EE"    ,"LP");
   LEG->AddEntry(graph_T1, "D=AH/E"      ,"LP");
   LEG->AddEntry(graph_T2, "D=BG/E"      ,"LP");
   LEG->AddEntry(graph_T3, "D=FC/E"      ,"LP");

   TCanvas* c1;
   c1 = new TCanvas("c1", "c1",600,600);
   c1->SetLogy(true);
   TMultiGraph* MGTOF = new TMultiGraph();
   MGTOF->Add(graph_T0      ,"LP");   
   MGTOF->Add(graph_T1      ,"LP");
   MGTOF->Add(graph_T2      ,"LP");
   MGTOF->Add(graph_T3      ,"LP");
   MGTOF->Draw("A");
   MGTOF->SetTitle("");
   MGTOF->GetXaxis()->SetTitle("1/#beta cut");
   MGTOF->GetYaxis()->SetTitle("Number of expected backgrounds");
   MGTOF->GetYaxis()->SetTitleOffset(1.70);
   MGTOF->GetYaxis()->SetRangeUser(500,1E6);
   LEG->Draw();
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,InputPattern,string("Systematics_")+DataName+"_TOF_Value");
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGI = new TMultiGraph();
   c1->SetLogy(true);
   MGI->Add(graph_I0      ,"LP");
   MGI->Add(graph_I1      ,"LP");
   MGI->Add(graph_I2      ,"LP");
   MGI->Add(graph_I3      ,"LP");
   MGI->Draw("A");
   MGI->SetTitle("");
   MGI->GetXaxis()->SetTitle("I_{as} cut");
   MGI->GetYaxis()->SetTitle("Number of expected backgrounds");
   MGI->GetYaxis()->SetTitleOffset(1.70);
   MGI->GetYaxis()->SetRangeUser(500,1E6);
   LEG->Draw();
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,InputPattern,string("Systematics_")+DataName+"_I_Value");
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   c1->SetLogy(true);
   TMultiGraph* MGP = new TMultiGraph();
   MGP->Add(graph_P0      ,"LP");
   MGP->Add(graph_P1      ,"LP");
   MGP->Add(graph_P2      ,"LP");
   MGP->Add(graph_P3      ,"LP");
   MGP->Draw("A");
   MGP->SetTitle("");
   MGP->GetXaxis()->SetTitle("p_{T} cut");
   MGP->GetYaxis()->SetTitle("Number of expected backgrounds");
   MGP->GetYaxis()->SetTitleOffset(1.70);
   MGP->GetYaxis()->SetRangeUser(500,1E6);
   LEG->Draw();
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,InputPattern,string("Systematics_")+DataName+"_P_Value");
   delete c1;

   for(unsigned int p=0;p<3;p++){
      string Title; string Name;
      if(p==0){ Title = "1/#beta cut";  Name="TOF_";  }
      if(p==1){ Title = "dEdx cut";     Name="I_";    }
      if(p==2){ Title = "p_{T} cut";    Name="pT_";   }

      c1 = new TCanvas("c1","c1", 600, 600);
      TGraph* graph_s;
      if(p==0)graph_s = new TGraph(ArrN[p],ArrT [p],ArrSigma[p]);
      if(p==1)graph_s = new TGraph(ArrN[p],ArrI [p],ArrSigma[p]);
      if(p==2)graph_s = new TGraph(ArrN[p],ArrPt[p],ArrSigma[p]);
      graph_s->SetTitle("");
      graph_s->GetYaxis()->SetTitle("Prediction #sigma/#mu");
      graph_s->GetYaxis()->SetTitleOffset(1.70);
      graph_s->GetXaxis()->SetTitle(Title.c_str());
      graph_s->Draw("AC*");
      SaveCanvas(c1,InputPattern,string(string("Systematics_")+DataName+"_")+Name+"Sigma");
      delete c1;

      c1 = new TCanvas("c1","c1", 600, 600);
      TGraph* graph_d;
      if(p==0)graph_d = new TGraph(ArrN[p],ArrT [p],ArrDist[p]);
      if(p==1)graph_d = new TGraph(ArrN[p],ArrI [p],ArrDist[p]);
      if(p==2)graph_d = new TGraph(ArrN[p],ArrPt[p],ArrDist[p]);
      graph_d->SetTitle("");
      graph_d->GetYaxis()->SetTitle("Prediction Dist/#mu");
      graph_d->GetYaxis()->SetTitleOffset(1.70);
      graph_d->GetXaxis()->SetTitle(Title.c_str());
      graph_d->Draw("AC*");
      SaveCanvas(c1,InputPattern,string(string("Systematics_")+DataName+"_")+Name+"Dist");
      delete c1;

      c1 = new TCanvas("c1","c1", 600, 600);
      TGraph* graph_sum;
      if(p==0)graph_sum = new TGraph(ArrN[p],ArrT [p],ArrSum[p]);
      if(p==1)graph_sum = new TGraph(ArrN[p],ArrI [p],ArrSum[p]);
      if(p==2)graph_sum = new TGraph(ArrN[p+2],ArrPt[p+2],ArrSum[p+2]);
      graph_sum->SetTitle("");
      graph_sum->GetYaxis()->SetTitle("Prediction #sigma_{Stat+Syst}/#mu");
      graph_sum->GetYaxis()->SetTitleOffset(1.70);
      graph_sum->GetXaxis()->SetTitle(Title.c_str());
      graph_sum->Draw("AC*");
      graph_sum->GetYaxis()->SetRangeUser(0,0.5);

      if(p==2){
         TGraph* graph_sum2 = new TGraph(ArrN[p+1],ArrPt[p+1],ArrSum[p+1]);
         graph_sum2->SetLineColor(2);
         graph_sum2->SetMarkerColor(2);
         graph_sum2->Draw("C*");

         TGraph* graph_sum3 = new TGraph(ArrN[p+0],ArrPt[p+0],ArrSum[p+0]);
         graph_sum3->SetLineColor(4);
         graph_sum3->SetMarkerColor(4);
         graph_sum3->Draw("C*");

       //TGraph* graph_sum4 = new TGraph(ArrN[p+3],ArrPt[p+3],ArrSum[p+3]);
       //graph_sum4->SetLineColor(8);
       //graph_sum4->SetMarkerColor(8);
       //graph_sum4->Draw("C*");

         LEG = new TLegend(0.50,0.65,0.80,0.90);
         LEG->SetFillColor(0);
         LEG->SetBorderSize(0);
         LEG->AddEntry(graph_sum,  "I_{as}>0.15 & 1/#beta>1.05", "L");
         LEG->AddEntry(graph_sum2, "I_{as}>0.05 & 1/#beta>1.05", "L");
         LEG->AddEntry(graph_sum3, "I_{as}>0.10 & 1/#beta>1.10", "L");
         LEG->Draw();
      }
      SaveCanvas(c1,InputPattern,string(string("Systematics_")+DataName+"_")+Name+"Sum");
      delete c1;

      c1 = new TCanvas("c1","c1", 600, 600);
      TGraph* graph_stat;
      if(p==0)graph_stat = new TGraph(ArrN[p],ArrT [p],ArrStat[p]);
      if(p==1)graph_stat = new TGraph(ArrN[p],ArrI [p],ArrStat[p]);
      if(p==2)graph_stat = new TGraph(ArrN[p+2],ArrPt[p+2],ArrStat[p+2]);
      graph_stat->SetTitle("");
      graph_stat->GetYaxis()->SetTitle("Prediction #sigma_{Stat}/#mu");
      graph_stat->GetYaxis()->SetTitleOffset(1.70);
      graph_stat->GetXaxis()->SetTitle(Title.c_str());
      graph_stat->Draw("AC*");
      graph_stat->GetYaxis()->SetRangeUser(0,0.15);

      if(p==2){
         TGraph* graph_stat2 = new TGraph(ArrN[p+1],ArrPt[p+1],ArrStat[p+1]);
         graph_stat2->SetLineColor(2);
         graph_stat2->SetMarkerColor(2);
         graph_stat2->Draw("C*");

         TGraph* graph_stat3 = new TGraph(ArrN[p+0],ArrPt[p+0],ArrStat[p+0]);
         graph_stat3->SetLineColor(4);
         graph_stat3->SetMarkerColor(4);
         graph_stat3->Draw("C*");

       //TGraph* graph_stat4 = new TGraph(ArrN[p+3],ArrPt[p+3],ArrStat[p+3]);
       //graph_stat4->SetLineColor(8);
       //graph_stat4->SetMarkerColor(8);
       //graph_stat4->Draw("C*");

         LEG = new TLegend(0.50,0.65,0.80,0.90);
         LEG->SetFillColor(0);
         LEG->SetBorderSize(0);
         LEG->AddEntry(graph_stat,  "I_{as}>0.15 & 1/#beta>1.05", "L");
         LEG->AddEntry(graph_stat2, "I_{as}>0.05 & 1/#beta>1.05", "L");
         LEG->AddEntry(graph_stat3, "I_{as}>0.10 & 1/#beta>1.10", "L");
         LEG->Draw();
      }
      SaveCanvas(c1,InputPattern,string(string("Systematics_")+DataName+"_")+Name+"Stat");
      delete c1;

      c1 = new TCanvas("c1","c1", 600, 600);
      TGraph* graph_statB;
      if(p==0)graph_statB = new TGraph(ArrN[p],ArrT [p],ArrStat[p]);
      if(p==1)graph_statB = new TGraph(ArrN[p],ArrI [p],ArrStat[p]);
      if(p==2)graph_statB = new TGraph(ArrN[p+2],ArrPt[p+2],ArrStatB[p+2]);
      graph_statB->SetTitle("");
      graph_statB->GetYaxis()->SetTitle("Prediction #sigma_{Stat}/#mu");
      graph_statB->GetYaxis()->SetTitleOffset(1.70);
      graph_statB->GetXaxis()->SetTitle(Title.c_str());
      
      graph_statB->Draw("AC*");
      graph_statB->GetYaxis()->SetRangeUser(0,0.15);

      if(p==2){
         TGraph* graph_statB2 = new TGraph(ArrN[p+1],ArrPt[p+1],ArrStatB[p+1]);
         graph_statB2->SetLineColor(2);
         graph_statB2->SetMarkerColor(2);
         graph_statB2->Draw("C*");

         TGraph* graph_statB3 = new TGraph(ArrN[p+0],ArrPt[p+0],ArrStatB[p+0]);
         graph_statB3->SetLineColor(4);
         graph_statB3->SetMarkerColor(4);
         graph_statB3->Draw("C*");

       //TGraph* graph_statB4 = new TGraph(ArrN[p+3],ArrPt[p+3],ArrStat[p+3]);
       //graph_statB4->SetLineColor(8);
       //graph_statB4->SetMarkerColor(8);
       //graph_statB4->Draw("C*");

         LEG = new TLegend(0.50,0.65,0.80,0.90);
         LEG->SetFillColor(0);
         LEG->SetBorderSize(0);
         LEG->AddEntry(graph_statB,  "I_{as}>0.15 & 1/#beta>1.05", "L");
         LEG->AddEntry(graph_statB2, "I_{as}>0.05 & 1/#beta>1.05", "L");
         LEG->AddEntry(graph_statB3, "I_{as}>0.10 & 1/#beta>1.10", "L");
         LEG->Draw();
      }
      SaveCanvas(c1,InputPattern,string(string("Systematics_")+DataName+"_")+Name+"StatB");
      delete c1;

      c1 = new TCanvas("c1","c1", 600, 600);
      TGraph* graph_syst;
      if(p==0)graph_syst = new TGraph(ArrN[p],ArrT [p],ArrSyst[p]);
      if(p==1)graph_syst = new TGraph(ArrN[p],ArrI [p],ArrSyst[p]);
      if(p==2)graph_syst = new TGraph(ArrN[p+2],ArrPt[p+2],ArrSyst[p+2]);
      graph_syst->SetTitle("");
      graph_syst->GetYaxis()->SetTitle("Prediction #sigma_{Syst}/#mu");
      graph_syst->GetYaxis()->SetTitleOffset(1.70);
      graph_syst->GetXaxis()->SetTitle(Title.c_str());
      graph_syst->Draw("AC*");
      graph_syst->GetXaxis()->SetRangeUser(40,100);
      graph_syst->GetYaxis()->SetRangeUser(0,0.15);

      if(p==2){
         TGraph* graph_syst2 = new TGraph(ArrN[p+1],ArrPt[p+1],ArrSyst[p+1]);
         graph_syst2->SetLineColor(2);
         graph_syst2->SetMarkerColor(2);
         graph_syst2->Draw("C*");

         TGraph* graph_syst3 = new TGraph(ArrN[p+0],ArrPt[p+0],ArrSyst[p+0]);
         graph_syst3->SetLineColor(4);
         graph_syst3->SetMarkerColor(4);
         graph_syst3->Draw("C*");

       //TGraph* graph_syst4 = new TGraph(ArrN[p+3],ArrPt[p+3],ArrSyst[p+3]);
       //graph_syst4->SetLineColor(8);
       //graph_syst4->SetMarkerColor(8);
       //graph_syst4->Draw("C*");

         LEG = new TLegend(0.50,0.65,0.80,0.90);
         LEG->SetFillColor(0);
         LEG->SetBorderSize(0);
         LEG->AddEntry(graph_syst,  "I_{as}>0.15 & 1/#beta>1.05", "L");
         LEG->AddEntry(graph_syst2, "I_{as}>0.05 & 1/#beta>1.05", "L");
         LEG->AddEntry(graph_syst3, "I_{as}>0.10 & 1/#beta>1.10", "L");
         LEG->Draw();
      }
      SaveCanvas(c1,InputPattern,string(string("Systematics_")+DataName+"_")+Name+"Syst");
      delete c1;
    }
    InputFile->Close();
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////
// WARNING... ALL THE FUNCTIONS BELOW HAVE NOT BEEN REVIEWED SINCE JULY UPDATE... THEY ARE PROBABLY STILL WORKING FINE, BUT CAN NOT BE SURE TILL SOMEONE TRY
// PLEASE TAKE SOME TIME TO REVIEW THE FUNCTION BELOW IF YOU NEED TO USE THEM, AND MOVE THEM ABOVE THIS WARNING WHEN VALIDATED

void SignalMassPlot(string InputPattern, unsigned int CutIndex){

   string SavePath  = InputPattern + "MassPlots/";
   MakeDirectories(SavePath);

   string Input     = InputPattern + "Histos.root";
   TFile* InputFile = new TFile(Input.c_str());
   for(unsigned int s=0;s<samples.size();s++){
      if(samples[s].Type!=2 || !samples[s].MakePlot)continue;           
      TH1D* Mass = GetCutIndexSliceFromTH2((TH2D*)GetObjectFromPath(InputFile, samples[s].Name + "/Mass"    ), CutIndex, "SignalMass");
      Mass->Scale(1.0/Mass->Integral());

      char YAxisLegend[1024];
      sprintf(YAxisLegend,"#tracks / %2.0f GeV/c^{2}",Mass->GetXaxis()->GetBinWidth(1));


      TCanvas* c1 = new TCanvas("c1","c1", 600, 600);
      Mass->SetAxisRange(0,1250,"X");
//    Mass->SetAxisRange(Min,Max,"Y");
      Mass->SetTitle("");
//      Mass->SetStats(kFALSE);
      Mass->GetXaxis()->SetTitle("m (GeV/c^{2})");
      Mass->GetYaxis()->SetTitle(YAxisLegend);
      Mass->SetLineWidth(2);
      Mass->SetLineColor(Color[0]);
      Mass->SetMarkerColor(Color[0]);
      Mass->SetMarkerStyle(Marker[0]);
      Mass->Draw("HIST E1");
      c1->SetLogy(true);
      SaveCanvas(c1,SavePath,samples[s].Name);   
      delete c1;
   }
}



void Make2DPlot_Core(string InputPattern, unsigned int CutIndex){
   TCanvas* c1;
   TLegend* leg;

   string Input = InputPattern + "Histos.root";
   string outpath = InputPattern;
   MakeDirectories(outpath);

   TFile* InputFile = new TFile((InputPattern + "Histos.root").c_str());
   TH1D* Gluino300_Mass = GetCutIndexSliceFromTH2((TH2D*)GetObjectFromPath(InputFile, "Gluino300_f10/Mass"    ), CutIndex, "G300Mass");
   TH2D* Gluino300_PIs  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino300_f10/AS_PIs"  ), CutIndex, "G300PIs_zy");
   TH2D* Gluino300_PIm  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino300_f10/AS_PIm"  ), CutIndex, "G300PIm_zy");
   TH2D* Gluino300_TOFIs= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino300_f10/AS_TOFIs"), CutIndex, "G300TIs_zy");
   TH2D* Gluino300_TOFIm= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino300_f10/AS_TOFIm"), CutIndex, "G300TIm_zy");
   TH1D* Gluino500_Mass = GetCutIndexSliceFromTH2((TH2D*)GetObjectFromPath(InputFile, "Gluino500_f10/Mass"    ), CutIndex, "G500Mass");
   TH2D* Gluino500_PIs  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino500_f10/AS_PIs"  ), CutIndex, "G500PIs_zy");
   TH2D* Gluino500_PIm  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino500_f10/AS_PIm"  ), CutIndex, "G500PIm_zy");
   TH2D* Gluino500_TOFIs= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino500_f10/AS_TOFIs"), CutIndex, "G500TIs_zy");
   TH2D* Gluino500_TOFIm= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino500_f10/AS_TOFIm"), CutIndex, "G500TIm_zy");
   TH1D* Gluino800_Mass = GetCutIndexSliceFromTH2((TH2D*)GetObjectFromPath(InputFile, "Gluino800_f10/Mass"    ), CutIndex, "G800Mass");
   TH2D* Gluino800_PIs  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino800_f10/AS_PIs"  ), CutIndex, "G800PIs_zy");
   TH2D* Gluino800_PIm  = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino800_f10/AS_PIm"  ), CutIndex, "G800PIm_zy");
   TH2D* Gluino800_TOFIs= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino800_f10/AS_TOFIs"), CutIndex, "G800TIs_zy");
   TH2D* Gluino800_TOFIm= GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Gluino800_f10/AS_TOFIm"), CutIndex, "G800TIm_zy");
   TH2D* Data_PIs       = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Data/AS_PIs"       ), CutIndex);
   TH2D* Data_PIm       = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Data/AS_PIm"       ), CutIndex);
   TH2D* Data_TOFIs     = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Data/AS_TOFIs"     ), CutIndex);
   TH2D* Data_TOFIm     = GetCutIndexSliceFromTH3((TH3D*)GetObjectFromPath(InputFile, "Data/AS_TOFIm"     ), CutIndex);
   TH2D* Data_PIm_075   = (TH2D*)Data_PIm->Clone();   Data_PIm_075->Reset(); 
   TH2D* Data_PIm_150   = (TH2D*)Data_PIm->Clone();   Data_PIm_150->Reset();
   TH2D* Data_PIm_300   = (TH2D*)Data_PIm->Clone();   Data_PIm_300->Reset();
   TH2D* Data_PIm_450   = (TH2D*)Data_PIm->Clone();   Data_PIm_450->Reset();
   TH2D* Data_PIm_All   = (TH2D*)Data_PIm->Clone();   Data_PIm_All->Reset();

   for(unsigned int i=0;i<(unsigned int)Data_PIm->GetNbinsX();i++){
   for(unsigned int j=0;j<(unsigned int)Data_PIm->GetNbinsY();j++){
      if(Data_PIm->GetBinContent(i,j)<=0)continue;
      double M = GetMass(Data_PIm->GetXaxis ()->GetBinCenter(i), Data_PIm->GetYaxis ()->GetBinCenter(j), false);
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
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
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
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
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
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
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
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
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
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
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
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
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
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
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
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
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
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1, outpath, "Gluino_TOFIm", true);
   delete c1;

   c1 = new TCanvas("c1","c1", 600, 600);
   Data_PIm_075->SetTitle("");
   Data_PIm_075->SetStats(kFALSE);
   Data_PIm_075->GetXaxis()->SetTitle("p (GeV/c)");
   Data_PIm_075->GetYaxis()->SetTitle(dEdxM_Legend.c_str());
//   Data_PIm_075->SetAxisRange(0,15,"Y");
//   Data_PIm_075->SetAxisRange(0,1250,"X");
   Data_PIm_075->SetAxisRange(3,10,"Y");
   Data_PIm_075->SetAxisRange(0,2000,"X");
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

   for(double m=100;m<1000;m+=100){
      TF1* MassLine = GetMassLine(m, false);
      MassLine->SetLineColor(1);
      MassLine->SetLineWidth(1);
      MassLine->Draw("same");
   }

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
   DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1, outpath, "Data_PIm_Colored", true);
   delete c1;
}


void MakeExpLimitpLot(string Input, string Output){
   TH2D* ExpLimitPlot = new TH2D("ExpLimitPlot","ExpLimitPlot", 10,37.5,87.5,13,0.0875,0.4175);

   std::vector<double> VectPt;
   std::vector<double> VectI;
   std::vector<double> VectExpLim;
   FILE* pFile = fopen(Input.c_str(),"r");
   if(!pFile){
      printf("Not Found: %s\n",Input.c_str());
      return;
   }

   unsigned int Index;
   double Pt, I, TOF, MassMin, MassMax;
   double NData, NPred, NPredErr, SignalEff;
   double ExpLimit;
   char Model[256], Tmp[2048];
   while ( ! feof (pFile) ){
     fscanf(pFile,"%s Testing CutIndex= %d (Pt>%lf I>%lf TOF>%lf) %lf<M<%lf Ndata=%lE NPred=%lE+-%lE SignalEff=%lf --> %lE expected",Model,&Index,&Pt,&I,&TOF,&MassMin,&MassMax,&NData,&NPred,&NPredErr,&SignalEff,&ExpLimit);
     fgets(Tmp, 256 , pFile);
//     if(Pt<80 && I<0.38)printf("%s Testing CutIndex= %d (Pt>%f I>%f TOF>%f) %f<M<%f Ndata=%E NPred=%E+-%E SignalEff=%f --> %E expected %s",Model,Index,Pt,I,TOF,MassMin,MassMax,NData,NPred,NPredErr,SignalEff,ExpLimit, Tmp);
//     ExpLimitPlot->SetBinContent(PtMap[Pt],IsMap[I],ExpLimit);
     ExpLimitPlot->Fill(Pt,I,ExpLimit);
   }
   fclose(pFile);
  

   TCanvas* c1 = new TCanvas("c1","c1",600,600);
   c1->SetLogz(true);
   ExpLimitPlot->SetTitle("");
   ExpLimitPlot->SetStats(kFALSE);
   ExpLimitPlot->GetXaxis()->SetTitle("Pt Cut");
   ExpLimitPlot->GetYaxis()->SetTitle("I  Cut");
   ExpLimitPlot->GetXaxis()->SetTitleOffset(1.1);
   ExpLimitPlot->GetYaxis()->SetTitleOffset(1.70);
   ExpLimitPlot->GetXaxis()->SetNdivisions(505);
   ExpLimitPlot->GetYaxis()->SetNdivisions(505);
   ExpLimitPlot->SetMaximum(1E0);
   ExpLimitPlot->SetMinimum(1E-2);
   ExpLimitPlot->Draw("COLZ");
   c1->SaveAs(Output.c_str());
   delete c1;
   return;
}


void CosmicBackgroundSystematic(string InputPattern, string DataType){
   if(DataType.find("7TeV")!=string::npos){SQRTS=7.0;}else{SQRTS=8.0;}

  string SavePath  = InputPattern;
  MakeDirectories(SavePath);
  TCanvas* c1;
  TH1** Histos = new TH1*[10];
  std::vector<string> legend;

  string LegendTitle = LegendFromType(InputPattern);

  TFile* InputFile = new TFile((InputPattern + "Histos.root").c_str());

  TH1D*  HCuts_Pt       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
  TH1D*  HCuts_TOF      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");

  TH2D* H_D_DzSidebands = ((TH2D*)GetObjectFromPath(InputFile, "Data" + DataType + "/H_D_DzSidebands"));
  TH2D* H_D_DzSidebands_Cosmic = (TH2D*)GetObjectFromPath(InputFile, "Cosmic8TeV/H_D_DzSidebands");
  TH1D* H_D_Cosmic           = (TH1D*)GetObjectFromPath(InputFile, "Cosmic8TeV/H_D");
  //TH1D* H_D_Data             = (TH1D*)GetObjectFromPath(InputFile, "Data8TeV/H_D");

  std::vector<int> Index;   std::vector<int> Plot;
  for(int CutIndex=0; CutIndex<HCuts_Pt->GetNbinsX(); CutIndex++) {
    if(fabs(HCuts_TOF->GetBinContent(CutIndex+1)-1.25)<0.001) {Index.push_back(CutIndex); Plot.push_back(0);}
    if(fabs(HCuts_TOF->GetBinContent(CutIndex+1)-1.3)<0.001) {Index.push_back(CutIndex); Plot.push_back(1);}
    if(fabs(HCuts_TOF->GetBinContent(CutIndex+1)-1.35)<0.001) {Index.push_back(CutIndex); Plot.push_back(2);}
  }

  const int TimeRegions=3;
  TH1F *Pred[TimeRegions*DzRegions];
  TH1F *StatSyst[TimeRegions];
  TH1F *Stat[TimeRegions];
  TH1F *Syst[TimeRegions];

  string Preds[TimeRegions] = {"125", "130", "135"};
  string RegionNames[DzRegions]={"Region0","Region1","Region2","Region3","Region4", "Region5"};
  string LegendNames[DzRegions]={"dz < 20 cm","20 cm < dz < 30 cm","30 cm < dz < 50 cm","50 cm < dz < 70 cm","70 cm < dz < 120 cm", "dz > 120 cm"};

  for(int i=0; i<TimeRegions; i++) {
    StatSyst[i] = new TH1F(("StatSyst_TOF" + Preds[i]).c_str(), "StatSyst_TOF100", 9, 95, 365);
    Stat[i] = new TH1F(("Stat_TOF" + Preds[i]).c_str(), "Stat_TOF100", 9, 95, 365);
    Syst[i] = new TH1F(("Syst_TOF" + Preds[i]).c_str(), "Syst_TOF110", 9, 95, 365);

    for(int Region=0; Region<DzRegions; Region++) {
      string Name="Pred_TOF" + Preds[i] + "_"+ RegionNames[Region];
      Pred[i*DzRegions+Region] = new TH1F(Name.c_str(), Name.c_str(), 9, 95, 365);
    }
  }

  const double alpha = 1 - 0.6827;
  //cout << endl << endl;
  for(unsigned int i=0; i<Index.size(); i++) {
    int CutIndex=Index[i];
    double D_Cosmic = H_D_Cosmic->GetBinContent(CutIndex+1);
    double D_Cosmic_Var = pow(ROOT::Math::gamma_quantile_c(alpha/2,D_Cosmic+1,1) - D_Cosmic,2);

    int Bin=Pred[0]->FindBin(HCuts_Pt->GetBinContent(CutIndex+1));

    double Sigma = 0;
    double Mean = 0;
    int N=0;

    for(int Region=2; Region<DzRegions; Region++) {
      double D_Sideband = H_D_DzSidebands->GetBinContent(CutIndex+1, Region+1);
      double D_Sideband_Cosmic = H_D_DzSidebands_Cosmic->GetBinContent(CutIndex+1, Region+1);

      double NPred = D_Sideband * D_Cosmic / D_Sideband_Cosmic;

      double D_Sideband_Cosmic_Var = pow(ROOT::Math::gamma_quantile_c(alpha/2,D_Sideband_Cosmic+1,1) - D_Sideband_Cosmic,2);
      double D_Sideband_Var = pow(ROOT::Math::gamma_quantile_c(alpha/2,D_Sideband+1,1) - D_Sideband,2);

      double NPredErr = sqrt( (pow(D_Cosmic/D_Sideband_Cosmic,2)*D_Sideband_Var) + (pow(D_Sideband/D_Sideband_Cosmic,2)*D_Cosmic_Var) + (pow((D_Cosmic*(D_Sideband)/(D_Sideband_Cosmic*D_Sideband_Cosmic)),2)*D_Sideband_Cosmic_Var) );

      Pred[Plot[i]*DzRegions + Region]->SetBinContent(Bin, NPred);
      Pred[Plot[i]*DzRegions + Region]->SetBinError(Bin, NPredErr);
      Mean+=NPred/pow(NPredErr,2);
      Sigma+=1/pow(NPredErr,2);
      N++;
      if(fabs(HCuts_TOF->GetBinContent(CutIndex+1)-1.3)<0.001 && fabs(HCuts_Pt->GetBinContent(CutIndex+1)-230)<0.001) {
	//cout << endl << "D Sideband " << D_Sideband << " D_Cosmic " << D_Cosmic << " D_Sideband_Cosmic " << D_Sideband_Cosmic << endl;
	//cout << "For Dz region " << LegendNames[Region] << " NPred " << NPred << " +- " << NPredErr << endl;
      }
    }

    Mean  = Mean/Sigma;
    Sigma = sqrt(Sigma);

    double SUM=0, STAT=0, SYST=0;

    for(int p=0;p<DzRegions;p++){
      SUM   += pow(Pred[Plot[i]*DzRegions + p]->GetBinContent(Bin)-Mean,2);
      STAT  += pow(Pred[Plot[i]*DzRegions + p]->GetBinError(Bin),2);
    }
    
    SUM  = sqrt(SUM/(N-1));
    STAT = sqrt(STAT)/(N-1);
    SYST = sqrt(SUM*SUM - STAT*STAT);
    if(fabs(HCuts_TOF->GetBinContent(CutIndex+1)-1.3)<0.001 && fabs(HCuts_Pt->GetBinContent(CutIndex+1)-230)<0.001)
      //cout << "Mean " << Mean << " Sigma " << Sigma << " Stat " << STAT/Mean << " StatSyst " << SUM/Mean << "Syst " << SYST/Mean << endl;
    Stat[Plot[i]]->SetBinContent(Bin, STAT/Mean);
    StatSyst[Plot[i]]->SetBinContent(Bin, SUM/Mean);
    Syst[Plot[i]]->SetBinContent(Bin, SYST/Mean);
  }
    //cout << endl << endl;
  for(int i=0; i<TimeRegions; i++) {
    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    for(int Region=2; Region<DzRegions; Region++) {
      Histos[Region-2] = Pred[i*DzRegions + Region];         legend.push_back(LegendNames[Region]);
    }
    DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Pt Cut", "Predicted", 0,0, 0,0);
    DrawLegend((TObject**)Histos,legend,LegendTitle,"P",0.8, 0.9, 0.4, 0.05);
    c1->SetLogy(false);
    DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
    SaveCanvas(c1,SavePath,(DataType + "CosmicPrediction_TOF" + Preds[i]).c_str());
    delete c1;
  }  

  c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
  for(int i=0; i<TimeRegions; i++) {
    Histos[i] = StatSyst[i];         legend.push_back(Preds[i]);
  }
  DrawSuperposedHistos((TH1**)Histos, legend, "",  "Pt Cut", "Stat+Syst Rel. Error", 0,0, 0,1.4);
  DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
  c1->SetLogy(false);
  DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
  SaveCanvas(c1,SavePath,DataType +"CosmicStatSyst");
  delete c1;

  c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
  for(int i=0; i<TimeRegions; i++) {
    Histos[i] = Stat[i];         legend.push_back(Preds[i]);
  }
  DrawSuperposedHistos((TH1**)Histos, legend, "",  "Pt Cut", "Stat Rel. Error", 0,0, 0,1.4);
  DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
  c1->SetLogy(false);
  DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
  SaveCanvas(c1,SavePath,DataType +"CosmicStat");
  delete c1;

  c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
  for(int i=0; i<TimeRegions; i++) {
    Histos[i] = Syst[i];         legend.push_back(Preds[i]);
  }
  DrawSuperposedHistos((TH1**)Histos, legend, "",  "Pt Cut", "Syst Rel. Error", 0,0, 0,1.4);
  DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
  c1->SetLogy(false);
  DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
  SaveCanvas(c1,SavePath,DataType +"CosmicSyst");
  delete c1;
}  

void CheckPrediction(string InputPattern, string HistoSuffix, string DataType){
  TypeMode = TypeFromPattern(InputPattern);
  if(TypeMode==0)return;

  std::vector<string> legend;
  string LegendTitle = LegendFromType(InputPattern);
  string SavePath  = InputPattern;
  MakeDirectories(SavePath);
  TCanvas* c1;
  TH1** Histos = new TH1*[100];

  TFile* InputFile = new TFile((InputPattern + "Histos.root").c_str());
  TH1D*  HCuts_Pt       = (TH1D*)GetObjectFromPath(InputFile, string("HCuts_Pt") + HistoSuffix);
  TH1D*  HCuts_I        = (TH1D*)GetObjectFromPath(InputFile, string("HCuts_I") + HistoSuffix);
  TH1D*  HCuts_TOF      = (TH1D*)GetObjectFromPath(InputFile, string("HCuts_TOF") + HistoSuffix);

  TH1D*  H_D            = (TH1D*)GetObjectFromPath(InputFile, string(DataType+"/H_D") + HistoSuffix);
  TH1D*  H_P            = (TH1D*)GetObjectFromPath(InputFile, string(DataType+"/H_P") + HistoSuffix);

  std::vector<int> Index;   std::vector<int> Plot;
  std::vector<double> TOFCuts;
  double TOFCutMax=0, TOFCutMin=9999;

  map<std::pair<double, double>,int> CutMap;

  int countPlots=0;
  for(int CutIndex=1; CutIndex<HCuts_TOF->GetNbinsX(); CutIndex++) {
    TOFCuts.push_back(HCuts_TOF->GetBinContent(CutIndex+1));
    if(HCuts_TOF->GetBinContent(CutIndex+1)<TOFCutMin) TOFCutMin=HCuts_TOF->GetBinContent(CutIndex+1);
    if(HCuts_TOF->GetBinContent(CutIndex+1)>TOFCutMax) TOFCutMax=HCuts_TOF->GetBinContent(CutIndex+1);

    std::pair<double, double> key(HCuts_I->GetBinContent(CutIndex+1), HCuts_Pt->GetBinContent(CutIndex+1));
    //New combination of TOF and I cuts
    if(CutMap.find(key)==CutMap.end()) {
      CutMap[key]=countPlots;
      countPlots++;
    }
  }

  std::vector<TH1D*> Pred;
  std::vector<TH1D*> Data;
  std::vector<TH1D*> Ratio;

  for(int i=0; i<countPlots; i++) {
    char DataName[1024];
    sprintf(DataName,"Data_%i",i);
    TH1D* TempData = new TH1D(DataName, DataName, TOFCuts.size(), TOFCutMin, TOFCutMax);
    Data.push_back(TempData);
    char PredName[1024];
    sprintf(PredName,"Pred_%i",i);
    TH1D* TempPred = new TH1D(PredName, PredName, TOFCuts.size(), TOFCutMin, TOFCutMax);
    Pred.push_back(TempPred);
    char RatioName[1024];
    sprintf(RatioName,"Ratio_%i",i);
    TH1D* TempRatio = new TH1D(RatioName, RatioName, TOFCuts.size(), TOFCutMin, TOFCutMax);
    Ratio.push_back(TempRatio);
  }


  for(int CutIndex=1; CutIndex<HCuts_TOF->GetNbinsX(); CutIndex++) {

    std::pair<double, double> key(HCuts_I->GetBinContent(CutIndex+1), HCuts_Pt->GetBinContent(CutIndex+1));
    int plot = CutMap.find(key)->second;
    int bin = Data[plot]->FindBin(HCuts_TOF->GetBinContent(CutIndex+1));

    double D = H_D->GetBinContent(CutIndex+1);
    Data[plot]->SetBinContent(bin, D);

    double P = H_P->GetBinContent(CutIndex+1);
    double Perr = H_P->GetBinError(CutIndex+1);
    Pred[plot]->SetBinContent(bin, P);
    Pred[plot]->SetBinError(bin, Perr);

    Ratio[plot]->SetBinContent(bin, D/P);
    Ratio[plot]->SetBinError(bin, sqrt( D/(P*P) + pow(D*Perr/(P*P),2) ));
  }

  for(int i=0; i<countPlots; i++) {
    map<std::pair<double, double>,int>::iterator it;
    double ICut=-1, PtCut=-1;
    for ( it=CutMap.begin() ; it != CutMap.end(); it++ ) if((*it).second==i) {
      ICut = (*it).first.first;
      PtCut = (*it).first.second;
    }

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = Data[i];      legend.push_back("Obs");
    Histos[1] = Pred[i];    legend.push_back("Pred");
    DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta Cut", "Tracks", 0, 0, 1, 100000);
    DrawLegend((TObject**)Histos,legend,LegendTitle,"P", 0.5);
    c1->SetLogy(true);
    DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));

    char Title[1024];
    if(ICut>-1 && PtCut>-1) sprintf(Title,"Pred%s_I%0.2f_Pt%3.0f_",HistoSuffix.c_str(), ICut, PtCut);
    else if(PtCut>-1) sprintf(Title,"Pred%s_Pt%3.0f_",HistoSuffix.c_str(),PtCut);
    else if(ICut>-1) sprintf(Title,"Pred%s_I%0.2f_",HistoSuffix.c_str(),ICut);
    SaveCanvas(c1,SavePath,Title + DataType);
    delete c1;
    delete Histos[0]; delete Histos[1];
  }

  legend.clear();
  for(int i=0; i<countPlots; i++) {
    map<std::pair<double, double>,int>::iterator it;
    double ICut=-1, PtCut=-1;
    for ( it=CutMap.begin() ; it != CutMap.end(); it++ ) if((*it).second==i) {
      ICut = (*it).first.first;
      PtCut = (*it).first.second;
    }
    char LegendName[1024];
    if(ICut>-1 && PtCut>-1) sprintf(LegendName,"I>%0.2f Pt>%3.0f",ICut, PtCut);
    else if(PtCut>-1) sprintf(LegendName,"Pt>%3.0f",PtCut);
    else if(ICut>-1) sprintf(LegendName,"I>%0.2f",ICut);
    Histos[i] = Ratio[i];            legend.push_back(LegendName);
  }

  c1 = new TCanvas("c1","c1,",600,600);
  DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta Cut", "Data/MC", 0, 0, 0,0);
  DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
  c1->SetLogy(false);
  DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
  SaveCanvas(c1,SavePath,"Pred_Ratio_" + DataType + HistoSuffix);
  delete c1;
}


void CollisionBackgroundSystematicFromFlip(string InputPattern, string DataType){
   if(DataType.find("7TeV")!=string::npos){SQRTS=7.0;}else{SQRTS=8.0;}
  TypeMode = TypeFromPattern(InputPattern);

  string SavePath  = InputPattern;
  MakeDirectories(SavePath);
  TCanvas* c1;

  std::vector<string> legend;
  TGraphErrors** Graphs = new TGraphErrors*[10];

  string LegendTitle = LegendFromType(InputPattern);

  TFile* InputFile = new TFile((InputPattern + "Histos.root").c_str());

  TH1D*  HCuts_Pt       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
  TH1D*  HCuts_TOF      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");
  TH1D*  HCuts_I        = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");

  TH1D*  HCuts_Pt_Flip       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt_Flip");
  TH1D*  HCuts_TOF_Flip      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF_Flip");
  TH1D*  HCuts_I_Flip        = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I_Flip");

   TH1D*  H_A            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_A");
   TH1D*  H_B            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_B");
   TH1D*  H_C            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_C");
   //TH1D*  H_D            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_D");
   TH1D*  H_E            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_E");
   TH1D*  H_F            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_F");
   TH1D*  H_G            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_G");
   TH1D*  H_H            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_H");

   TH1D*  H_A_Flip            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_A_Flip");
   TH1D*  H_B_Flip            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_B_Flip");
   TH1D*  H_C_Flip            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_C_Flip");
   TH1D*  H_D_Flip            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_D_Flip");
   TH1D*  H_E_Flip            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_E_Flip");
   TH1D*  H_F_Flip            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_F_Flip");
   TH1D*  H_G_Flip            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_G_Flip");
   TH1D*  H_H_Flip            = (TH1D*)GetObjectFromPath(InputFile, DataType + "/H_H_Flip");

   TH1D*  H_B_Binned[MaxPredBins];
   TH1D*  H_F_Binned[MaxPredBins];
   TH1D*  H_H_Binned[MaxPredBins];

   TH1D*  H_B_Flip_Binned[MaxPredBins];
   TH1D*  H_D_Flip_Binned[MaxPredBins];
   TH1D*  H_F_Flip_Binned[MaxPredBins];
   TH1D*  H_H_Flip_Binned[MaxPredBins];

   if(TypeMode==3) {
     for(int i=0; i<MaxPredBins; i++) {
       char Suffix[1024];
       sprintf(Suffix,"_%i",i);
       string Bin=Suffix;

       H_B_Binned[i]    = (TH1D*)GetObjectFromPath(InputFile, (DataType + "/H_B_Binned" + Bin).c_str());
       H_F_Binned[i]    = (TH1D*)GetObjectFromPath(InputFile, (DataType + "/H_F_Binned" + Bin).c_str());
       H_H_Binned[i]    = (TH1D*)GetObjectFromPath(InputFile, (DataType + "/H_H_Binned" + Bin).c_str());

       H_B_Flip_Binned[i]    = (TH1D*)GetObjectFromPath(InputFile, (DataType + "/H_B_Binned_Flip" + Bin).c_str());
       H_D_Flip_Binned[i]    = (TH1D*)GetObjectFromPath(InputFile, (DataType + "/H_D_Binned_Flip" + Bin).c_str());
       H_F_Flip_Binned[i]    = (TH1D*)GetObjectFromPath(InputFile, (DataType + "/H_F_Binned_Flip" + Bin).c_str());
       H_H_Flip_Binned[i]    = (TH1D*)GetObjectFromPath(InputFile, (DataType + "/H_H_Binned_Flip" + Bin).c_str());
     }
   }

  std::vector<int> Index;   std::vector<int> Index_Flip;   std::vector<int> Plot;
  for(int CutIndex_Flip=0; CutIndex_Flip<HCuts_Pt_Flip->GetNbinsX(); CutIndex_Flip++) {
    if(fabs(HCuts_TOF_Flip->GetBinContent(CutIndex_Flip+1)-0.9)<0.001 || fabs(HCuts_TOF_Flip->GetBinContent(CutIndex_Flip+1)-0.8)<0.001) {
      for(int CutIndex=0; CutIndex<HCuts_Pt->GetNbinsX(); CutIndex++) {
	if(fabs(HCuts_TOF_Flip->GetBinContent(CutIndex_Flip+1)-(2-HCuts_TOF->GetBinContent(CutIndex+1)))<0.0001 &&
	   fabs(HCuts_Pt_Flip->GetBinContent(CutIndex_Flip+1)-HCuts_Pt->GetBinContent(CutIndex+1))<0.0001 &&
	   fabs(HCuts_I_Flip->GetBinContent(CutIndex_Flip+1)-HCuts_I->GetBinContent(CutIndex+1))<0.0001) {

	  if ( (TypeMode == 4) && (fabs(HCuts_I->GetBinContent(CutIndex+1)) > 0.36))	    {
	      continue;
	  }
	  else	    {
	    Index.push_back(CutIndex);
	    Index_Flip.push_back(CutIndex_Flip);
	    if(fabs(HCuts_TOF_Flip->GetBinContent(CutIndex_Flip+1)-0.9)<0.001) Plot.push_back(0);
	    if(fabs(HCuts_TOF_Flip->GetBinContent(CutIndex_Flip+1)-0.8)<0.001) Plot.push_back(1);
	  }
	}
      }
    }
  }
 

  const int TimeRegions=2;
  int NCuts = 20;
  if (TypeMode == 4)  NCuts = Index.size()/TimeRegions;

  double Pred[TimeRegions][3][NCuts];
  double PredErr[TimeRegions][3][NCuts];
  double StatSyst[TimeRegions][NCuts];
  double Stat[TimeRegions][NCuts];
  double Syst[TimeRegions][NCuts];
  double PtCut[TimeRegions][NCuts];
  double IasCut[TimeRegions][NCuts];
  double PredN[TimeRegions]={0};

  string Preds[TimeRegions] = {"110", "120"};
  string PredsLegend[TimeRegions] = {"1/#beta>1.1", "1/#beta>1.2"};
  string LegendNames[3]={"BH/F", "BH'/F'", "BD'/B'"};
  if (TypeMode == 4)      {      LegendNames[0] = "CH/G";      LegendNames[1] = "CH'/G'";      LegendNames[2] = "CD'/C'";    }

  string outfile = SavePath + "BkgUncertainty_"  + DataType.substr(4)  +".txt";
  ofstream fout(outfile.c_str());
  if ( ! fout.good() )    { 
    cout << "unable to create file " << outfile << endl;
    return;
  }

  char record[400];
  sprintf(record, "                                                 This is %5s data", DataType.substr(4).c_str());
  fout << record << endl << endl;

  sprintf(record, "     %-10s%-10s%-14s%-14s%-14s%-14s%-14s%-14s%-14s%-14s%-14s", "Ias", "1/beta", "Pred1", "Pred2", "Pred3", "DeltaPred1", "DeltaPred2", "DeltaPred3", "STAT", "STATSYST", "SYST");
  fout << record << endl ;
  
  for(unsigned int i=0; i<Index.size(); i++) {
    int CutIndex=Index[i];
    int Point=PredN[Plot[i]];

    double Sigma = 0;
    double Mean = 0;
    int N=0;

    double A = H_A->GetBinContent(Index[i]);
    double B = H_B->GetBinContent(Index[i]);
    double C = H_C->GetBinContent(Index[i]);
    //double D = H_D->GetBinContent(Index[i]);
    double E = H_E->GetBinContent(Index[i]);
    double F = H_F->GetBinContent(Index[i]);
    double G = H_G->GetBinContent(Index[i]);
    double H = H_H->GetBinContent(Index[i]);

    double A_Flip = H_A_Flip->GetBinContent(Index_Flip[i]);
    double B_Flip = H_B_Flip->GetBinContent(Index_Flip[i]);
    double C_Flip = H_C_Flip->GetBinContent(Index_Flip[i]);
    double D_Flip = H_D_Flip->GetBinContent(Index_Flip[i]);
    double E_Flip = H_E_Flip->GetBinContent(Index_Flip[i]);
    double F_Flip = H_F_Flip->GetBinContent(Index_Flip[i]);
    double G_Flip = H_G_Flip->GetBinContent(Index_Flip[i]);
    double H_Flip = H_H_Flip->GetBinContent(Index_Flip[i]);

    double NPred[3]={0};
    double NPredErr[3]={0};

    if(TypeMode==2) {
      NPred[0]    = (A*F*G)/(E*E);
      NPredErr[0] = sqrt( ((pow(F*G,2)* A + pow(A*G,2)*F + pow(A*F,2)*G)/pow(E,4)) + (pow((2*A*F*G)/pow(E,3),2)*E));

      NPred[1]    = (A*F_Flip*G_Flip)/(E_Flip*E_Flip);
      NPredErr[1] = sqrt( ((pow(A_Flip*G_Flip,2)* F_Flip + pow(A*F_Flip,2)*G_Flip)/pow(E_Flip,4)) + (pow((2*A_Flip*F*G_Flip)/pow(E_Flip,3),2)*E_Flip));

      NPred[2]    = (A*B_Flip*C_Flip)/(A_Flip*A_Flip);
      NPredErr[2] = sqrt( ((pow(A*C_Flip,2)* B_Flip + pow(A*B_Flip,2)*C_Flip)/pow(A_Flip,4)) + (pow((2*A*B_Flip*C_Flip)/pow(A_Flip,3),2)*A_Flip));
    }

    if(TypeMode==3) {
      for(int j=0; j<MaxPredBins; j++) {
	//double A_Binned = H_A_Binned[j]->GetBinContent(Index[i]);
	double B_Binned = H_B_Binned[j]->GetBinContent(Index[i]);
	//double C_Binned = H_C_Binned[j]->GetBinContent(Index[i]);
	//double D_Binned = H_D_Binned[j]->GetBinContent(Index[i]);
	//double E_Binned = H_E_Binned[j]->GetBinContent(Index[i]);
	double F_Binned = H_F_Binned[j]->GetBinContent(Index[i]);
	//double G_Binned = H_G_Binned[j]->GetBinContent(Index[i]);
	double H_Binned = H_H_Binned[j]->GetBinContent(Index[i]);

	//double A_Flip_Binned = H_A_Flip_Binned[j]->GetBinContent(Index_Flip[i]);
	double B_Flip_Binned = H_B_Flip_Binned[j]->GetBinContent(Index_Flip[i]);
	//double C_Flip_Binned = H_C_Flip_Binned[j]->GetBinContent(Index_Flip[i]);
	double D_Flip_Binned = H_D_Flip_Binned[j]->GetBinContent(Index_Flip[i]);
	//double E_Flip_Binned = H_E_Flip_Binned[j]->GetBinContent(Index_Flip[i]);
	double F_Flip_Binned = H_F_Flip_Binned[j]->GetBinContent(Index_Flip[i]);
	//double G_Flip_Binned = H_G_Flip_Binned[j]->GetBinContent(Index_Flip[i]);
	double H_Flip_Binned = H_H_Flip_Binned[j]->GetBinContent(Index_Flip[i]);

	NPred[0]+=H_Binned*B_Binned/F_Binned;
	NPredErr[0] += (pow(B_Binned/F_Binned,2)*H_Binned) + (pow((B_Binned*(H_Binned)/(F_Binned*F_Binned)),2)*F_Binned);

        NPred[1]+=H_Flip_Binned*B_Binned/F_Flip_Binned;
        NPredErr[1] += (pow(B_Binned/F_Flip_Binned,2)*H_Flip_Binned) + (pow((B_Binned*(H_Flip_Binned)/(F_Flip_Binned*F_Flip_Binned)),2)*F_Flip_Binned);

        NPred[2]+=D_Flip_Binned*B_Binned/B_Flip_Binned;
        NPredErr[2] += (pow(B_Binned/B_Flip_Binned,2)*D_Flip_Binned) + (pow((B_Binned*(D_Flip_Binned)/(B_Flip_Binned*B_Flip_Binned)),2)*B_Flip_Binned);
      }
    }

    if(TypeMode==4) {
      NPred[0]    = ((C*H)/G);
      NPredErr[0] = (pow(C/G,2)*H) + (pow((H*(C)/(G*G)),2)*G);
      
      NPred[1]    = ((C*H_Flip)/G_Flip);
      NPredErr[1] = (pow(C/G_Flip,2)*H_Flip) + (pow((H_Flip*(C)/(G_Flip*G_Flip)),2)*G_Flip);
      
      NPred[2]    = ((C*D_Flip)/C_Flip);
      NPredErr[2] = (pow(C/C_Flip,2)*D_Flip) + (pow((D_Flip*(C)/(C_Flip*C_Flip)),2)*C_Flip);
    }

    for(int Region=0; Region<3; Region++) {
      NPredErr[Region]=sqrt(NPredErr[Region]);
      Pred[Plot[i]][Region][Point] = NPred[Region];
      PredErr[Plot[i]][Region][Point] = NPredErr[Region];
      Mean+=NPred[Region]/pow(NPredErr[Region],2);
      Sigma+=1/pow(NPredErr[Region],2);
      N++;
    }

    Mean  = Mean/Sigma;
    Sigma = sqrt(Sigma);

    double SUM=0, STAT=0, SYST=0;

    for(int p=0;p<3;p++){
      SUM   += pow(Pred[Plot[i]][p][Point]-Mean,2);
      STAT  += pow(PredErr[Plot[i]][p][Point],2);
    }
  
    SUM  = sqrt(SUM/(N-1));
    STAT = sqrt(STAT/(N-1)); //HERE IT MUST BE N-1 !!!, IF YOU HAVE ARGUMENTS TO USE N, PLEASE ARGUE
    if(SUM*SUM > STAT*STAT) SYST = sqrt(SUM*SUM - STAT*STAT);
    else SYST=0;

    Stat[Plot[i]][Point]=STAT/Mean;
    StatSyst[Plot[i]][Point]=SUM/Mean;
    Syst[Plot[i]][Point]=SYST/Mean;

    PtCut[Plot[i]][Point]=HCuts_Pt->GetBinContent(CutIndex);
    IasCut[Plot[i]][Point]=HCuts_I->GetBinContent(CutIndex);
    PredN[Plot[i]]++;

    sprintf(record, "%-5i%-10.3f%-10.3f%-14.3f%-14.3f%-14.3f%-14.3f%-14.3f%-14.3f%-14.3f%-14.3f%-14.3f", i, HCuts_I->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), Pred[Plot[i]][0][Point], Pred[Plot[i]][1][Point], Pred[Plot[i]][2][Point], PredErr[Plot[i]][0][Point], PredErr[Plot[i]][1][Point], PredErr[Plot[i]][2][Point],Stat[Plot[i]][Point], StatSyst[Plot[i]][Point], Syst[Plot[i]][Point]);
    fout << record << endl ;

  }
  fout.close();


  TMultiGraph* PredGraphs;
  for(int i=0; i<TimeRegions; i++) { 
    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    PredGraphs = new TMultiGraph();
    for(int Region=0; Region<3; Region++) {
      if (TypeMode!=4)	{
	Graphs[Region] = new TGraphErrors(PredN[Plot[i]],PtCut[Plot[i]],Pred[Plot[i]][Region],0,PredErr[Plot[i]][Region]); legend.push_back(LegendNames[Region]);
      }
      else	{
	std::cout << PredN[Plot[i]] << "    "  << *IasCut[Plot[i]] << "    "  << *Pred[Plot[i]][Region] << "    "  << *PredErr[Plot[i]][Region]  << std::endl;
	Graphs[Region] = new TGraphErrors(PredN[Plot[1]],IasCut[Plot[i]],Pred[Plot[i]][Region],0,PredErr[Plot[i]][Region]); legend.push_back(LegendNames[Region]);
      }
      Graphs[Region]->SetLineColor(Color[Region]);  Graphs[Region]->SetMarkerColor(Color[Region]);   Graphs[Region]->SetMarkerStyle(GraphStyle[Region]);
      PredGraphs->Add(Graphs[Region],"LP");
    }

    PredGraphs->Draw("A");
    PredGraphs->SetTitle("");
    PredGraphs->GetXaxis()->SetTitle("P_T cut");
    if (TypeMode==4)      PredGraphs->GetXaxis()->SetTitle("I_{as} cut");
    PredGraphs->GetYaxis()->SetTitle("Number of expected backgrounds");
    PredGraphs->GetYaxis()->SetTitleOffset(1.70);
    PredGraphs->GetYaxis()->SetRangeUser(0,400);
    c1->SetLogy(0);

    if (TypeMode == 4)   
      {
        double yup = *Pred[Plot[0]][0];
	double ydown = 1.0000;
	for(int Region=0; Region<3; Region++) {
	double Predmin =  Pred[TimeRegions-1][Region][NCuts-1];
	if (Predmin < ydown) ydown = Predmin;
	}
        PredGraphs->GetYaxis()->SetRangeUser(ydown*0.4, yup*1.4);
	c1->SetLogy(true);
      }
    DrawLegend((TObject**)Graphs,legend,LegendTitle,"P",0.8, 0.9, 0.4, 0.05);
    DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
    SaveCanvas(c1,SavePath,(DataType + "CollisionPrediction_TOF" + Preds[i]).c_str());
    delete c1;
    delete PredGraphs;
  }


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    PredGraphs = new TMultiGraph();
    for(int i=0; i<TimeRegions; i++) {

      if (TypeMode!=4)	{
	Graphs[i] = new TGraphErrors(PredN[Plot[i]],PtCut[Plot[i]],StatSyst[Plot[i]],0,0); legend.push_back(PredsLegend[i]);
      }
      else 	{
	Graphs[i] = new TGraphErrors(PredN[Plot[i]],IasCut[Plot[i]],StatSyst[Plot[i]],0,0); legend.push_back(PredsLegend[i]);	
	}
      Graphs[i]->SetLineColor(Color[i]);  Graphs[i]->SetMarkerColor(Color[i]);   Graphs[i]->SetMarkerStyle(GraphStyle[i]);
      PredGraphs->Add(Graphs[i],"LP");


    }
    PredGraphs->Draw("A");
    PredGraphs->SetTitle("");
    PredGraphs->GetXaxis()->SetTitle("P_T cut");
    if (TypeMode==4)      PredGraphs->GetXaxis()->SetTitle("I_{as} cut");
    PredGraphs->GetYaxis()->SetTitle("Relative Statistical + Systematic Uncertainty");
    PredGraphs->GetYaxis()->SetTitleOffset(1.70);
    PredGraphs->GetYaxis()->SetRangeUser(0,1.);
    c1->SetLogy(0);
    DrawLegend((TObject**)Graphs,legend,LegendTitle,"P",0.8, 0.9, 0.4, 0.05);
    DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
    SaveCanvas(c1,SavePath,DataType + "CollisionStatSyst");
    delete c1;
    delete PredGraphs;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    PredGraphs = new TMultiGraph();
    for(int i=0; i<TimeRegions; i++) {
      if (TypeMode!=4)	{
	Graphs[i] = new TGraphErrors(PredN[Plot[i]],PtCut[Plot[i]],Stat[Plot[i]],0,0); legend.push_back(PredsLegend[i]);
      }
      else	{
	Graphs[i] = new TGraphErrors(PredN[Plot[i]],IasCut[Plot[i]],Stat[Plot[i]],0,0); legend.push_back(PredsLegend[i]);	
	}
      Graphs[i]->SetLineColor(Color[i]);  Graphs[i]->SetMarkerColor(Color[i]);   Graphs[i]->SetMarkerStyle(GraphStyle[i]);
      PredGraphs->Add(Graphs[i],"LP");

    }
    PredGraphs->Draw("A");
    PredGraphs->SetTitle("");
    PredGraphs->GetXaxis()->SetTitle("P_T cut");
    if (TypeMode==4)      PredGraphs->GetXaxis()->SetTitle("I_{as} cut");
    PredGraphs->GetYaxis()->SetTitle("Relative Statistical Uncertainty");
    PredGraphs->GetYaxis()->SetTitleOffset(1.70);
    PredGraphs->GetYaxis()->SetRangeUser(0,1.);
    c1->SetLogy(0);
    DrawLegend((TObject**)Graphs,legend,LegendTitle,"P",0.8, 0.9, 0.4, 0.05);
    DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
    SaveCanvas(c1,SavePath,DataType + "CollisionStat");
    delete c1;
    delete PredGraphs;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    PredGraphs = new TMultiGraph();
    for(int i=0; i<TimeRegions; i++) {
      if (TypeMode!=4)	{
	  Graphs[i] = new TGraphErrors(PredN[Plot[i]],PtCut[Plot[i]],Syst[Plot[i]],0,0); legend.push_back(PredsLegend[i]);	 
	}
      else 	{
	Graphs[i] = new TGraphErrors(PredN[Plot[i]],IasCut[Plot[i]],Syst[Plot[i]],0,0); legend.push_back(PredsLegend[i]);	 
	}
      Graphs[i]->SetLineColor(Color[i]);  Graphs[i]->SetMarkerColor(Color[i]);   Graphs[i]->SetMarkerStyle(GraphStyle[i]);
      PredGraphs->Add(Graphs[i],"LP");
    }
    PredGraphs->Draw("A");
    PredGraphs->SetTitle("");
    PredGraphs->GetXaxis()->SetTitle("P_T cut");
    if (TypeMode==4)      PredGraphs->GetXaxis()->SetTitle("I_{as} cut");
    PredGraphs->GetYaxis()->SetTitle("Relative Systemtic Uncertainty");
    PredGraphs->GetYaxis()->SetTitleOffset(1.70);
    PredGraphs->GetYaxis()->SetRangeUser(0,1.);
    c1->SetLogy(0);
    DrawLegend((TObject**)Graphs,legend,LegendTitle,"P",0.8, 0.9, 0.4, 0.05);
    DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
    SaveCanvas(c1,SavePath,DataType + "CollisionSyst");
    delete c1;
    delete PredGraphs;

  /*
  c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
  for(int i=0; i<TimeRegions; i++) {
    Histos[i] = Stat[i];         legend.push_back(PredsLegend[i]);
  }
  DrawSuperposedHistos((TH1**)Histos, legend, "",  "Pt Cut", "Stat Rel. Error", 0,0, 0,1.4);
  DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
  c1->SetLogy(false);
  DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
  SaveCanvas(c1,SavePath,DataType + "CollisionStat");
  delete c1;

  c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
  for(int i=0; i<TimeRegions; i++) {
    Histos[i] = Syst[i];         legend.push_back(PredsLegend[i]);
  }
  DrawSuperposedHistos((TH1**)Histos, legend, "",  "Pt Cut", "Syst Rel. Error", 0,0, 0,1.4);
  DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
  c1->SetLogy(false);
  DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
  SaveCanvas(c1,SavePath,DataType +"CollisionSyst");
  delete c1;
  */
}





void CheckPredictionBin(string InputPattern, string HistoSuffix, string DataType, string bin){
  TypeMode = TypeFromPattern(InputPattern);
  if(TypeMode==0)return;

  std::vector<string> legend;
  string LegendTitle = LegendFromType(InputPattern);
  string SavePath  = InputPattern;
  MakeDirectories(SavePath);
  TCanvas* c1;
  TH1** Histos = new TH1*[100];

  TFile* InputFile = new TFile((InputPattern + "Histos.root").c_str());
  TH1D*  HCuts_Pt       = (TH1D*)GetObjectFromPath(InputFile, string("HCuts_Pt") + HistoSuffix);
  TH1D*  HCuts_I        = (TH1D*)GetObjectFromPath(InputFile, string("HCuts_I") + HistoSuffix);
  TH1D*  HCuts_TOF      = (TH1D*)GetObjectFromPath(InputFile, string("HCuts_TOF") + HistoSuffix);
  cout << "Data histo " << string(DataType+"/H_D_Binned_" + HistoSuffix + "_" + bin) << " P Name " << string(DataType+"/H_P_Binned_" + HistoSuffix + "_" + bin) << endl;
  TH1D*  H_D            = (TH1D*)GetObjectFromPath(InputFile, string(DataType+"/H_D_Binned" + HistoSuffix + "_" + bin));
  TH1D*  H_P            = (TH1D*)GetObjectFromPath(InputFile, string(DataType+"/H_P_Binned" + HistoSuffix + "_" + bin));

  std::vector<int> Index;   std::vector<int> Plot;
  std::vector<double> TOFCuts;
  double TOFCutMax=0, TOFCutMin=9999;

  map<std::pair<double, double>,int> CutMap;

  int countPlots=0;
  for(int CutIndex=1; CutIndex<HCuts_TOF->GetNbinsX(); CutIndex++) {
    TOFCuts.push_back(HCuts_TOF->GetBinContent(CutIndex+1));
    if(HCuts_TOF->GetBinContent(CutIndex+1)<TOFCutMin) TOFCutMin=HCuts_TOF->GetBinContent(CutIndex+1);
    if(HCuts_TOF->GetBinContent(CutIndex+1)>TOFCutMax) TOFCutMax=HCuts_TOF->GetBinContent(CutIndex+1);

    std::pair<double, double> key(HCuts_I->GetBinContent(CutIndex+1), HCuts_Pt->GetBinContent(CutIndex+1));
    //New combination of TOF and I cuts
    if(CutMap.find(key)==CutMap.end()) {
      CutMap[key]=countPlots;
      countPlots++;
    }
  }

  std::vector<TH1D*> Pred;
  std::vector<TH1D*> Data;
  std::vector<TH1D*> Ratio;

  for(int i=0; i<countPlots; i++) {
    char DataName[1024];
    sprintf(DataName,"Data_%i",i);
    TH1D* TempData = new TH1D(DataName, DataName, TOFCuts.size(), TOFCutMin, TOFCutMax);
    Data.push_back(TempData);
    char PredName[1024];
    sprintf(PredName,"Pred_%i",i);
    TH1D* TempPred = new TH1D(PredName, PredName, TOFCuts.size(), TOFCutMin, TOFCutMax);
    Pred.push_back(TempPred);
    char RatioName[1024];
    sprintf(RatioName,"Ratio_%i",i);
    TH1D* TempRatio = new TH1D(RatioName, RatioName, TOFCuts.size(), TOFCutMin, TOFCutMax);
    Ratio.push_back(TempRatio);
  }


  for(int CutIndex=1; CutIndex<HCuts_TOF->GetNbinsX(); CutIndex++) {

    std::pair<double, double> key(HCuts_I->GetBinContent(CutIndex+1), HCuts_Pt->GetBinContent(CutIndex+1));
    int plot = CutMap.find(key)->second;
    int histo_bin = Data[plot]->FindBin(HCuts_TOF->GetBinContent(CutIndex+1));

    double D = H_D->GetBinContent(CutIndex+1);
    Data[plot]->SetBinContent(histo_bin, D);

    double P = H_P->GetBinContent(CutIndex+1);
    double Perr = H_P->GetBinError(CutIndex+1);
    Pred[plot]->SetBinContent(histo_bin, P);
    Pred[plot]->SetBinError(histo_bin, Perr);

    Ratio[plot]->SetBinContent(histo_bin, D/P);
    Ratio[plot]->SetBinError(histo_bin, sqrt( D/(P*P) + pow(D*Perr/(P*P),2) ));
  }

  for(int i=0; i<countPlots; i++) {
    map<std::pair<double, double>,int>::iterator it;
    double ICut=-1, PtCut=-1;
    for ( it=CutMap.begin() ; it != CutMap.end(); it++ ) if((*it).second==i) {
      ICut = (*it).first.first;
      PtCut = (*it).first.second;
    }

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = Data[i];      legend.push_back("Obs");
    Histos[1] = Pred[i];    legend.push_back("Pred");
    DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta Cut", "Tracks", 0, 0, 0,0);
    DrawLegend((TObject**)Histos,legend,LegendTitle,"P", 0.5);
    c1->SetLogy(true);
    DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));

    char Title[1024];
    if(ICut>-1 && PtCut>-1) sprintf(Title,"Pred%s_I%0.2f_Pt%3.0f_",HistoSuffix.c_str(), ICut, PtCut);
    else if(PtCut>-1) sprintf(Title,"Pred%s_Pt%3.0f_",HistoSuffix.c_str(),PtCut);
    else if(ICut>-1) sprintf(Title,"Pred%s_I%0.2f_",HistoSuffix.c_str(),ICut);
    SaveCanvas(c1,SavePath,Title + DataType + "_Binned_" + bin);
    delete c1;
    delete Histos[0]; delete Histos[1];
  }

  legend.clear();
  for(int i=0; i<countPlots; i++) {
    map<std::pair<double, double>,int>::iterator it;
    double ICut=-1, PtCut=-1;
    for ( it=CutMap.begin() ; it != CutMap.end(); it++ ) if((*it).second==i) {
      ICut = (*it).first.first;
      PtCut = (*it).first.second;
    }
    char LegendName[1024];
    if(ICut>-1 && PtCut>-1) sprintf(LegendName,"I>%0.2f Pt>%3.0f",ICut, PtCut);
    else if(PtCut>-1) sprintf(LegendName,"Pt>%3.0f",PtCut);
    else if(ICut>-1) sprintf(LegendName,"I>%0.2f",ICut);
    Histos[i] = Ratio[i];            legend.push_back(LegendName);
  }

  c1 = new TCanvas("c1","c1,",600,600);
  DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta Cut", "Data/MC", 0, 0, 0,0);
  DrawLegend((TObject**)Histos,legend,LegendTitle,"P");
  c1->SetLogy(false);
  DrawPreliminary(SQRTS, IntegratedLuminosityFromE(SQRTS));
  SaveCanvas(c1,SavePath,"Pred_Ratio_" + DataType + HistoSuffix);
  delete c1;
}
