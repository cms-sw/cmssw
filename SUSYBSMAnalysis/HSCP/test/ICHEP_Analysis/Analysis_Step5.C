
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
#include "TMultiGraph.h"
#include "TPaveText.h"
#include "tdrstyle.C"


#include "Analysis_CommonFunction.h"
#include "Analysis_Global.h"
#include "Analysis_PlotFunction.h"
#include "Analysis_PlotStructure.h"
#include "Analysis_Samples.h"

using namespace std;

/////////////////////////// FUNCTION DECLARATION /////////////////////////////

void CutFlow(string InputPattern, unsigned int CutIndex=0);
void SelectionPlot (string InputPattern, unsigned int CutIndex, unsigned int GluinoCutIndex);
void MassPrediction(string InputPattern, unsigned int CutIndex, string HistoSuffix="Mass");
void PredictionAndControlPlot(string InputPattern, unsigned int CutIndex);
void Make2DPlot_Core(string ResultPattern, unsigned int CutIndex);
void SignalMassPlot(string InputPattern, unsigned int CutIndex);
void GetSystematicOnPrediction(string InputPattern);
int JobIdToIndex(string JobId);
void MassPredictionTight(string InputPattern, unsigned int CutIndex, string HistoSuffix="Mass");
void MakeExpLimitpLot(string Input, string Output);

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

   string InputDir;				unsigned int CutIndex;
   std::vector<string> Legends;                 std::vector<string> Inputs;
   int GluinoCutIndex;

//   MakeExpLimitpLot("Results_1toys_lp/dedxASmi/combined/Eta15/PtMin35/Type0/EXCLUSION/Stop200.info","tmp1.png");

   InputDir = "Results/dedxASmi/combined/Eta15/PtMin50/Type0/";   CutIndex = 4; //on set of cuts from the array, 0 means no cut
   MassPrediction(InputDir, CutIndex, "Mass");
   PredictionAndControlPlot(InputDir, CutIndex);
   //CutFlow(InputDir);
   SelectionPlot(InputDir, CutIndex, 0);
   
   InputDir = "Results/dedxASmi/combined/Eta15/PtMin50/Type2/";   CutIndex = 16;
   MassPrediction(InputDir, CutIndex, "Mass");
   //CutFlow(InputDir);
   SelectionPlot(InputDir, CutIndex, 0);
   GetSystematicOnPrediction(InputDir);
   PredictionAndControlPlot(InputDir, CutIndex);

// #don't thin this is still used, but keep there just in case
//   InputDir = "Results/dedxASmi/combined/Eta15/PtMin45/Type0/";   CutIndex = 11;/*65;*//*39;*/  MassPredictionTight(InputDir, CutIndex, "Mass");
//   CutIndex=50;
//   GluinoCutIndex=11;   
//   SelectionPlot(InputDir, CutIndex, GluinoCutIndex);   

//   InputDir = "Results/dedxASmi/combined/Eta15/PtMin45/Type2/";   CutIndex = 275;/*211;*//*167;95;*/  MassPredictionTight(InputDir, CutIndex, "Mass");
//   GluinoCutIndex=845;
//   SelectionPlot(InputDir, CutIndex, GluinoCutIndex);
   return;
}



TH2D* GetCutIndexSliceFromTH3(TH3D* tmp, unsigned int CutIndex, string Name="zy"){
   tmp->GetXaxis()->SetRange(CutIndex+1,CutIndex+1);
   return (TH2D*)tmp->Project3D(Name.c_str());
}


TH1D* GetCutIndexSliceFromTH2(TH2D* tmp, unsigned int CutIndex, string Name="_py"){
   return tmp->ProjectionY(Name.c_str(),CutIndex+1,CutIndex+1);
}


//////////////////////////////////////////////////     CREATE PLOTS OF SELECTION

void GetSystematicOnPrediction(string InputPattern){
   string Input     = InputPattern + "Histos_Data.root";
   TFile* InputFile = new TFile(Input.c_str());
   string SavePath  = InputPattern + "Systematic/";

   MakeDirectories(SavePath);  

   TH1D*  HCuts_Pt       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
   TH1D*  HCuts_I        = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");
   TH1D*  HCuts_TOF      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");
   TH1D*  H_A            = (TH1D*)GetObjectFromPath(InputFile, "H_A");
   TH1D*  H_B            = (TH1D*)GetObjectFromPath(InputFile, "H_B");
   TH1D*  H_C            = (TH1D*)GetObjectFromPath(InputFile, "H_C");
   TH1D*  H_D            = (TH1D*)GetObjectFromPath(InputFile, "H_D");
   TH1D*  H_E            = (TH1D*)GetObjectFromPath(InputFile, "H_E");
   TH1D*  H_F            = (TH1D*)GetObjectFromPath(InputFile, "H_F");
   TH1D*  H_G            = (TH1D*)GetObjectFromPath(InputFile, "H_G");
   TH1D*  H_H            = (TH1D*)GetObjectFromPath(InputFile, "H_H");
   TH1D*  H_P            = (TH1D*)GetObjectFromPath(InputFile, "H_P");

   int    ArrN[6];  ArrN[0] = 0; ArrN[1] = 0; ArrN[2] = 0;  ArrN[3] = 0;  ArrN[4] = 0; ArrN[5] = 0;
   double ArrPred[5][6][20];  double ArrErr[5][6][20];  int ArrPredN[5][6];  for(unsigned int i=0;i<5;i++){for(unsigned int j=0;j<6;j++){ArrPredN[i][j]=0;}}
   double ArrMean[6][20];
   double ArrSigma[6][20];
   double ArrDist[6][20];
   double ArrMaxDist[6][20];
   double ArrSum[6][20];
   double ArrSyst[6][20];
   double ArrStat[6][20];
   double ArrStatB[6][20];
   double ArrPt[6][20];
   double ArrI[6][20];
   double ArrT[6][20];


   std::vector<int> Index;   std::vector<int> Plot;
//vary respective to TOF cut 50, 0.05 1.05->1.2
   Index.push_back(16);      Plot.push_back(0);
   Index.push_back(17);      Plot.push_back(0);
   Index.push_back(18);      Plot.push_back(0);
   Index.push_back(19);      Plot.push_back(0);
   Index.push_back(20);      Plot.push_back(0);
   Index.push_back(21);      Plot.push_back(0);
   Index.push_back(22);      Plot.push_back(0);
//vary respective to I cut 50, 0.05->0.225 1.05
   Index.push_back(16);      Plot.push_back(1);
   Index.push_back(30);      Plot.push_back(1);
   Index.push_back(44);      Plot.push_back(1);
   Index.push_back(58);      Plot.push_back(1);
   Index.push_back(72);      Plot.push_back(1);
   Index.push_back(86);      Plot.push_back(1);
   Index.push_back(100);      Plot.push_back(1);
   Index.push_back(114);      Plot.push_back(1);
//vary respective to Pt cut 50->115 0.05 1.05
   Index.push_back(16);      Plot.push_back(2);
   Index.push_back(436);     Plot.push_back(2);
   Index.push_back(856);     Plot.push_back(2);
   Index.push_back(1276);     Plot.push_back(2);
   Index.push_back(1696);     Plot.push_back(2);
   Index.push_back(2116);     Plot.push_back(2);
   Index.push_back(2536);    Plot.push_back(2);
   Index.push_back(2746);    Plot.push_back(2);
//vary respective to Pt cut 50->115 0.1 1.1 
   Index.push_back(46);      Plot.push_back(3);
   Index.push_back(466);     Plot.push_back(3);
   Index.push_back(886);     Plot.push_back(3);
   Index.push_back(1306);     Plot.push_back(3);
   Index.push_back(1726);     Plot.push_back(3);
   Index.push_back(2146);     Plot.push_back(3);
   Index.push_back(2566);    Plot.push_back(3);
   Index.push_back(2776);    Plot.push_back(3);
//vary respective to Pt cut 50->115 0.15 1.05 
   Index.push_back(72);      Plot.push_back(4);
   Index.push_back(492);     Plot.push_back(4);
   Index.push_back(912);     Plot.push_back(4);
   Index.push_back(1332);     Plot.push_back(4);
   Index.push_back(1752);     Plot.push_back(4);
   Index.push_back(2172);     Plot.push_back(4);
   Index.push_back(2592);    Plot.push_back(4);
   Index.push_back(2802);    Plot.push_back(4);
   //Not used
   Index.push_back(82 + 4);     Plot.push_back(5);
   Index.push_back(154+ 4);     Plot.push_back(5);
   Index.push_back(226+ 4);     Plot.push_back(5);
   Index.push_back(298+ 4);     Plot.push_back(5);
   Index.push_back(370+ 4);     Plot.push_back(5);
   Index.push_back(442+ 4);     Plot.push_back(5);
   Index.push_back(514+ 4);     Plot.push_back(5);
   Index.push_back(586+ 4);     Plot.push_back(5);
   Index.push_back(658+ 4);     Plot.push_back(5);
   Index.push_back(730+ 4);     Plot.push_back(5);
   Index.push_back(802+ 4);     Plot.push_back(5);


     for(unsigned int i=0;i<Index.size();i++){
      int CutIndex = Index[i];

      const double& A=H_A->GetBinContent(CutIndex+1);
      const double& B=H_B->GetBinContent(CutIndex+1);
      const double& C=H_C->GetBinContent(CutIndex+1);
      const double& D=H_D->GetBinContent(CutIndex+1);
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

//     printf("pT>%6.2f I> %6.2f TOF>%6.2f : ", HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1));
//     printf("A =%6.2E, B=%6.2E, C=%6.2E, D=%6.2E E =%6.2E, F=%6.2E, G=%6.2E, H=%6.2E\n", A,B,C,D, E, F, G, H);

//     for(unsigned int p=0;p<4;p++){printf("Method %i --> P =%6.2E+-%6.2E\n", p,Pred[p], Err [p]);}
     printf("--> N = %1.0f Mean = %8.2E  Sigma=%8.2E  Dist=%8.2E  Sum=%8.2E  Stat=%8.2E  Syst=%8.2E\n", N, Mean, Sigma/Mean, Dist/Mean, Sum/Mean, Stat/Mean, Syst/Mean);
      if(N>0){
      ArrMean   [Plot[i]][ArrN[Plot[i]]] = Mean;
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

   TLegend* LEG = new TLegend(0.50,0.65,0.80,0.90);
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
   MGTOF->GetYaxis()->SetRangeUser(10,1E6);
   LEG->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"TOF_Value");
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
   MGI->GetYaxis()->SetRangeUser(10,1E6);
   LEG->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"I_Value");
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
   MGP->GetYaxis()->SetRangeUser(10,1E6);
   LEG->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"P_Value");
   delete c1;





    for(unsigned int p=0;p<3;p++){
      string Title; string Name;
      if(p==0){ Title = "1/#beta cut";  Name="TOF_";  }
      if(p==1){ Title = "dEdx cut";     Name="I_";    }
      if(p==2){ Title = "p_{T} cut";    Name="pT_";    }


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
      SaveCanvas(c1,SavePath,Name+"Sigma");
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
      SaveCanvas(c1,SavePath,Name+"Dist");
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
/*
         TGraph* graph_sum4 = new TGraph(ArrN[p+3],ArrPt[p+3],ArrSum[p+3]);
         graph_sum4->SetLineColor(8);
         graph_sum4->SetMarkerColor(8);
         graph_sum4->Draw("C*");
*/

          TLegend* LEG = new TLegend(0.50,0.65,0.80,0.90);
          LEG->SetFillColor(0);
          LEG->SetBorderSize(0);
          LEG->AddEntry(graph_sum,  "I_{as}>0.15 & 1/#beta>1.05", "L");
          LEG->AddEntry(graph_sum2, "I_{as}>0.05 & 1/#beta>1.05", "L");
          LEG->AddEntry(graph_sum3, "I_{as}>0.10 & 1/#beta>1.10", "L");
          LEG->Draw();
      }
      SaveCanvas(c1,SavePath,Name+"Sum");
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
      graph_stat->GetYaxis()->SetRangeUser(0,0.25);

      if(p==2){
         TGraph* graph_stat2 = new TGraph(ArrN[p+1],ArrPt[p+1],ArrStat[p+1]);
         graph_stat2->SetLineColor(2);
         graph_stat2->SetMarkerColor(2);
         graph_stat2->Draw("C*");

         TGraph* graph_stat3 = new TGraph(ArrN[p+0],ArrPt[p+0],ArrStat[p+0]);
         graph_stat3->SetLineColor(4);
         graph_stat3->SetMarkerColor(4);
         graph_stat3->Draw("C*");
/*
         TGraph* graph_stat4 = new TGraph(ArrN[p+3],ArrPt[p+3],ArrStat[p+3]);
         graph_stat4->SetLineColor(8);
         graph_stat4->SetMarkerColor(8);
         graph_stat4->Draw("C*");
*/

          TLegend* LEG = new TLegend(0.50,0.65,0.80,0.90);
          LEG->SetFillColor(0);
          LEG->SetBorderSize(0);
          LEG->AddEntry(graph_stat,  "I_{as}>0.15 & 1/#beta>1.05", "L");
          LEG->AddEntry(graph_stat2, "I_{as}>0.05 & 1/#beta>1.05", "L");
          LEG->AddEntry(graph_stat3, "I_{as}>0.10 & 1/#beta>1.10", "L");
          LEG->Draw();
      }
      SaveCanvas(c1,SavePath,Name+"Stat");
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
      graph_statB->GetYaxis()->SetRangeUser(0,0.25);

      if(p==2){
         TGraph* graph_statB2 = new TGraph(ArrN[p+1],ArrPt[p+1],ArrStatB[p+1]);
         graph_statB2->SetLineColor(2);
         graph_statB2->SetMarkerColor(2);
         graph_statB2->Draw("C*");

         TGraph* graph_statB3 = new TGraph(ArrN[p+0],ArrPt[p+0],ArrStatB[p+0]);
         graph_statB3->SetLineColor(4);
         graph_statB3->SetMarkerColor(4);
         graph_statB3->Draw("C*");
/*
         TGraph* graph_statB4 = new TGraph(ArrN[p+3],ArrPt[p+3],ArrStat[p+3]);
         graph_statB4->SetLineColor(8);
         graph_statB4->SetMarkerColor(8);
         graph_statB4->Draw("C*");
*/

          TLegend* LEG = new TLegend(0.50,0.65,0.80,0.90);
          LEG->SetFillColor(0);
          LEG->SetBorderSize(0);
          LEG->AddEntry(graph_statB,  "I_{as}>0.15 & 1/#beta>1.05", "L");
          LEG->AddEntry(graph_statB2, "I_{as}>0.05 & 1/#beta>1.05", "L");
          LEG->AddEntry(graph_statB3, "I_{as}>0.10 & 1/#beta>1.10", "L");
          LEG->Draw();
      }
      SaveCanvas(c1,SavePath,Name+"StatB");
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
      graph_syst->GetYaxis()->SetRangeUser(0,0.25);

      if(p==2){
         TGraph* graph_syst2 = new TGraph(ArrN[p+1],ArrPt[p+1],ArrSyst[p+1]);
         graph_syst2->SetLineColor(2);
         graph_syst2->SetMarkerColor(2);
         graph_syst2->Draw("C*");

         TGraph* graph_syst3 = new TGraph(ArrN[p+0],ArrPt[p+0],ArrSyst[p+0]);
         graph_syst3->SetLineColor(4);
         graph_syst3->SetMarkerColor(4);
         graph_syst3->Draw("C*");
/*
         TGraph* graph_syst4 = new TGraph(ArrN[p+3],ArrPt[p+3],ArrSyst[p+3]);
         graph_syst4->SetLineColor(8);
         graph_syst4->SetMarkerColor(8);
         graph_syst4->Draw("C*");
*/

          TLegend* LEG = new TLegend(0.50,0.65,0.80,0.90);
          LEG->SetFillColor(0);
          LEG->SetBorderSize(0);
          LEG->AddEntry(graph_syst,  "I_{as}>0.15 & 1/#beta>1.05", "L");
          LEG->AddEntry(graph_syst2, "I_{as}>0.05 & 1/#beta>1.05", "L");
          LEG->AddEntry(graph_syst3, "I_{as}>0.10 & 1/#beta>1.10", "L");
          LEG->Draw();
      }


      SaveCanvas(c1,SavePath,Name+"Syst");
      delete c1;


    }
}


void CutFlow(string InputPattern, unsigned int CutIndex){
   string Input     = InputPattern + "Histos.root";
   string SavePath  = InputPattern + "/CutFlow/";
   MakeDirectories(SavePath);

  // TFile* InputFile = new TFile(Input.c_str());  //signal
   TFile* InputFileData = new TFile((InputPattern + "Histos_Data.root").c_str());
   TFile* InputFileData11 = new TFile((InputPattern + "Histos_Data11.root").c_str());
   TFile* InputFileMC   = new TFile((InputPattern + "Histos_MC.root").c_str());
   if(!InputFileMC)std::cout << "FileProblem\n";

   TH1D*  HCuts_Pt       = (TH1D*)GetObjectFromPath(InputFileData, "HCuts_Pt");
   TH1D*  HCuts_I        = (TH1D*)GetObjectFromPath(InputFileData, "HCuts_I");
   TH1D*  HCuts_TOF      = (TH1D*)GetObjectFromPath(InputFileData, "HCuts_TOF");

    char Buffer[1024]; sprintf(Buffer,"%s/CutFlow_%03i_Pt%03.0f_I%05.3f_TOF%04.3f.txt",SavePath.c_str(),CutIndex,HCuts_Pt->GetBinContent(CutIndex+1),HCuts_I->GetBinContent(CutIndex+1),HCuts_TOF->GetBinContent(CutIndex+1));
    FILE* pFile = fopen(Buffer,"w");
    stPlots DataPlots;
    stPlots_InitFromFile(InputFileData, DataPlots,"Data", InputFileData);
    stPlots_Dump(DataPlots, pFile, CutIndex);
    stPlots_Clear(DataPlots);

    stPlots DataPlots11;
    stPlots_InitFromFile(InputFileData11, DataPlots11,"Data", InputFileData11); DataPlots11.Name="Data11";    
    stPlots_Dump(DataPlots11, pFile, CutIndex);
    stPlots_Clear(DataPlots11); 


    stPlots MCTrPlots;
    stPlots_InitFromFile(InputFileMC, MCTrPlots,"MCTr", InputFileMC);
    stPlots_Dump(MCTrPlots, pFile, CutIndex);
    stPlots_Clear(MCTrPlots);
   /* 
    for(unsigned int s=0;s<signals.size();s++){
       if(!signals[s].MakePlot)continue;
       stPlots SignPlots;
       stPlots_InitFromFile(InputFile, SignPlots,signals[s].Name, InputFile);
//       stPlots_Dump(SignPlots, pFile, CutIndex);       
       stPlots_Clear(SignPlots);
    }
     */
    fclose(pFile);
}

void SignalMassPlot(string InputPattern, unsigned int CutIndex){

   string SavePath  = InputPattern + "MassPlots/";
   MakeDirectories(SavePath);

   string Input     = InputPattern + "Histos.root";
   TFile* InputFile = new TFile(Input.c_str());
   for(unsigned int s=0;s<signals.size();s++){
      TH1D* Mass = GetCutIndexSliceFromTH2((TH2D*)GetObjectFromPath(InputFile, signals[s].Name + "/Mass"    ), CutIndex, "SignalMass");
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
      SaveCanvas(c1,SavePath,signals[s].Name);   
      delete c1;
   }
}



void SelectionPlot(string InputPattern, unsigned int CutIndex, unsigned int GluinoCutIndex){

   string LegendTitle = LegendFromType(InputPattern);;

   string Input     = InputPattern + "Histos.root";
   string SavePath  = InputPattern;
   MakeDirectories(SavePath);

   TFile* InputFile = new TFile(Input.c_str()); //signal
   TFile* InputFileData = new TFile((InputPattern + "Histos_Data.root").c_str());
   TFile* InputFileData11 = new TFile((InputPattern + "Histos_Data11.root").c_str());
   TFile* InputFileMC   = new TFile((InputPattern + "Histos_MC.root").c_str());

   //string Dir2011 = "/uscms_data/d2/farrell3/WorkArea/CMSSW_4_2_8_patch7/src/SUSYBSMAnalysis/HSCP/test/ICHEP_Analysis/Results/dedxASmi/combined/Eta15/PtMin50/Type2/";
   //TFile* InputFileData11 = new TFile((Dir2011 + "Histos_Data.root").c_str());
   //TFile* InputFileMC   = new TFile((Dir2011 + "Histos_MC.root").c_str());


   stPlots DataPlots, DataPlots11, MCTrPlots, SignPlots[signals.size()];
   stPlots_InitFromFile(InputFileData, DataPlots,"Data", InputFileData);DataPlots.Name="Data12";
   stPlots_InitFromFile(InputFileData11, DataPlots11,"Data", InputFileData11);DataPlots11.Name ="Data11";
   stPlots_InitFromFile(InputFileMC, MCTrPlots,"MCTr", InputFileMC);

   for(unsigned int s=0;s<signals.size();s++){
     if (signals[s].Name!="Gluino300" && signals[s].Name!="Gluino600" && signals[s].Name!="Gluino800" && signals[s].Name!="GMStau247" && signals[s].Name!="GMStau370" && signals[s].Name!="GMStau494") continue;
     stPlots_InitFromFile(InputFile, SignPlots[s],signals[s].Name, InputFile);
     if(!signals[s].MakePlot)continue;
//      stPlots_Draw(SignPlots[s], SavePath + "/Selection_" +  signals[s].Name, LegendTitle, CutIndex);
   }

   stPlots_Draw(DataPlots, SavePath + "/Selection_Data12", LegendTitle, CutIndex);
   stPlots_Draw(DataPlots11, SavePath + "/Selection_Data11", LegendTitle, CutIndex);
   stPlots_Draw(MCTrPlots, SavePath + "/Selection_MCTr", LegendTitle, CutIndex);

//   stPlots_Draw(SignPlots[SID_GL600 ], SavePath + "/Selection_" +  signals[SID_GL600 ].Name, LegendTitle);
//   stPlots_Draw(SignPlots[SID_GL600N], SavePath + "/Selection_" +  signals[SID_GL600N].Name, LegendTitle);
//   stPlots_Draw(SignPlots[SID_ST300 ], SavePath + "/Selection_" +  signals[SID_ST300 ].Name, LegendTitle);
//   stPlots_Draw(SignPlots[SID_ST300N], SavePath + "/Selection_" +  signals[SID_ST300N].Name, LegendTitle);
//   stPlots_Draw(SignPlots[SID_GS126 ], SavePath + "/Selection_" +  signals[SID_GS126 ].Name, LegendTitle);


   stPlots_DrawComparison(SavePath + "/Selection_Comp_Data" , LegendTitle, CutIndex, &DataPlots, &DataPlots11, &MCTrPlots);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_Gluino" , LegendTitle, GluinoCutIndex, &DataPlots, &MCTrPlots, &SignPlots[0], &SignPlots[3], &SignPlots[5]);


   /*
  //stPlots_DrawComparison(SavePath + "/Selection_Comp_DCStau" , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_D12K182 ], &SignPlots[SID_D12K595 ], &SignPlots[SID_D12K700 ]);
//   stPlots_DrawComparison(SavePath + "/Selection_Comp_Stop"   , LegendTitle, CutIndex, &DataPlots, &SignPlots[24]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_GMStau" , LegendTitle, CutIndex, &DataPlots, &MCTrPlots, &SignPlots[40], &SignPlots[42], &SignPlots[44]);
   return;

   stPlots_DrawComparison(SavePath + "/Selection_Comp_Gluino" , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_GL300 ], &SignPlots[SID_GL500 ], &SignPlots[SID_GL900 ]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_GluinoN", LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_GL300N], &SignPlots[SID_GL500N], &SignPlots[SID_GL900N]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_Stop"   , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_ST200 ], &SignPlots[SID_ST500 ], &SignPlots[SID_ST800 ]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_StopN"  , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_ST200N], &SignPlots[SID_ST500N], &SignPlots[SID_ST800N]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_GMStau" , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_GS126 ], &SignPlots[SID_GS247 ], &SignPlots[SID_GS308 ]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_PPStau" , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_PS126 ], &SignPlots[SID_PS247 ], &SignPlots[SID_PS308 ]);
   stPlots_DrawComparison(SavePath + "/Selection_Comp_DCStau" , LegendTitle, CutIndex, &DataPlots, &SignPlots[SID_D08K121 ], &SignPlots[SID_D08K242 ], &SignPlots[SID_D08K302 ]);
*/
   stPlots_Clear(DataPlots);
   stPlots_Clear(DataPlots11);
   stPlots_Clear(MCTrPlots);
/*   for(unsigned int s=0;s<signals.size();s++){
      if(!signals[s].MakePlot)continue;
      stPlots_Clear(SignPlots[s]);
   }
*/
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
   TH2D* Data_I                = (TH2D*)GetObjectFromPath(InputFile, "RegionD_I");   
   TH2D* Data_P                = (TH2D*)GetObjectFromPath(InputFile, "RegionD_P");   
   TH2D* Data_TOF              = (TH2D*)GetObjectFromPath(InputFile, "RegionD_TOF"); 

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

   TH1D* CtrlIm_S1_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlIm_S1_TOF"); CtrlIm_S1_TOF->Rebin(1);
   TH1D* CtrlIm_S2_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlIm_S2_TOF"); CtrlIm_S2_TOF->Rebin(1);
   TH1D* CtrlIm_S3_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlIm_S3_TOF"); CtrlIm_S3_TOF->Rebin(1);
   TH1D* CtrlIm_S4_TOF        = (TH1D*)GetObjectFromPath(InputFile, "CtrlIm_S4_TOF"); CtrlIm_S4_TOF->Rebin(1);

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlPt_S1_Is->Integral()>0)CtrlPt_S1_Is->Scale(1/CtrlPt_S1_Is->Integral());
   if(CtrlPt_S2_Is->Integral()>0)CtrlPt_S2_Is->Scale(1/CtrlPt_S2_Is->Integral());
   if(CtrlPt_S3_Is->Integral()>0)CtrlPt_S3_Is->Scale(1/CtrlPt_S3_Is->Integral());
   if(CtrlPt_S4_Is->Integral()>0)CtrlPt_S4_Is->Scale(1/CtrlPt_S4_Is->Integral());
   Histos[0] = CtrlPt_S1_Is;                     legend.push_back(" 50<p_{T}< 60 GeV");
   Histos[1] = CtrlPt_S2_Is;                     legend.push_back(" 60<p_{T}< 80 GeV");
   Histos[2] = CtrlPt_S3_Is;                     legend.push_back(" 80<p_{T}<100 GeV");
   Histos[3] = CtrlPt_S4_Is;                     legend.push_back("100<p_{T}");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxS_Legend, "arbitrary units", 0,0.5, 0,0);
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
   Histos[0] = CtrlPt_S1_Im;                     legend.push_back(" 50<p_{T}< 60 GeV");
   Histos[1] = CtrlPt_S2_Im;                     legend.push_back(" 60<p_{T}< 80 GeV");
   Histos[2] = CtrlPt_S3_Im;                     legend.push_back(" 80<p_{T}<100 GeV");
   Histos[3] = CtrlPt_S4_Im;                     legend.push_back("100<p_{T}");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxM_Legend, "arbitrary units", 3.0,5, 0,0);
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
   Histos[0] = CtrlPt_S1_TOF;                    legend.push_back(" 50<p_{T}< 60 GeV");
   Histos[1] = CtrlPt_S2_TOF;                    legend.push_back(" 60<p_{T}< 80 GeV");
   Histos[2] = CtrlPt_S3_TOF;                    legend.push_back(" 80<p_{T}<100 GeV");
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
   Histos[0] = CtrlIs_S1_TOF;                     legend.push_back("0.0<I_{as}<0.05");
   Histos[1] = CtrlIs_S2_TOF;                     legend.push_back("0.05<I_{as}<0.1");
   Histos[2] = CtrlIs_S3_TOF;                     legend.push_back("0.1<I_{as}<0.2");
   Histos[3] = CtrlIs_S4_TOF;                     legend.push_back("0.2<I_{as}");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "arbitrary units", 1,1.7, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(false);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"ControlIs_TOFSpectrum");

   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"ControlIs_TOFSpectrumLog");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlIm_S1_TOF->Integral()>0)CtrlIm_S1_TOF->Scale(1/CtrlIm_S1_TOF->Integral());
   if(CtrlIm_S2_TOF->Integral()>0)CtrlIm_S2_TOF->Scale(1/CtrlIm_S2_TOF->Integral());
   if(CtrlIm_S3_TOF->Integral()>0)CtrlIm_S3_TOF->Scale(1/CtrlIm_S3_TOF->Integral());
   if(CtrlIm_S4_TOF->Integral()>0)CtrlIm_S4_TOF->Scale(1/CtrlIm_S4_TOF->Integral());
   Histos[0] = CtrlIm_S1_TOF;                     legend.push_back("3.5<I_{as}<3.8");
   Histos[1] = CtrlIm_S2_TOF;                     legend.push_back("3.8<I_{as}<4.1");
   Histos[2] = CtrlIm_S3_TOF;                     legend.push_back("4.1<I_{as}<4.4");
   //Histos[3] = CtrlIm_S4_TOF;                     legend.push_back("0.3<I_{as}");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "1/#beta", "arbitrary units", 1,1.7, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(false);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"ControlIm_TOFSpectrum");
   delete c1;

/*
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

*/

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
      TF1* MassLine = GetMassLine(m);
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
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Data_PIm_Colored", true);
   delete c1;
}

void MassPrediction(string InputPattern, unsigned int CutIndex, string HistoSuffix)
{
   bool IsTkOnly = (InputPattern.find("Type0",0)<std::string::npos);


   double Rescale, RMS;
//   GetPredictionRescale(InputPattern,Rescale, RMS, RecomputeRescale);
//   RMS = fabs(1.0-Rescale)/2.0;
   Rescale = 1.0;
   RMS     = 0.05;

   string outpath = InputPattern;
   MakeDirectories(outpath);



   TFile* InputFile_Data;
   string Input;

   std::vector<string> legend;
   TCanvas* c1;

   char Buffer[2048];
   sprintf(Buffer,"%s/Histos_Data.root",InputPattern.c_str());

   InputFile_Data = new TFile(Buffer);

   if(!InputFile_Data || InputFile_Data->IsZombie() || !InputFile_Data->IsOpen() || InputFile_Data->TestBit(TFile::kRecovered) )return;
   TH1D* Pred     = ((TH2D*)GetObjectFromPath(InputFile_Data, string("Pred_") + HistoSuffix   ))->ProjectionY("TmpPredMass"    ,CutIndex+1,CutIndex+1,"o");
   TH1D* Data     = ((TH2D*)GetObjectFromPath(InputFile_Data, string("Data/") + HistoSuffix   ))->ProjectionY("TmpDataMass"    ,CutIndex+1,CutIndex+1,"o");
   /*
   TFile* InputFile_Data11 = new TFile((InputPattern+"/Histos_Data11.root").c_str());
   if(!InputFile_Data11 || InputFile_Data11->IsZombie() || !InputFile_Data11->IsOpen() || InputFile_Data11->TestBit(TFile::kRecovered) ){printf("problem with file %s\n", (InputPattern+"/Histos_Data11.root").c_str());return;}
   TH1D* Pred11     = ((TH2D*)GetObjectFromPath(InputFile_Data11, string("Pred_") + HistoSuffix   ))->ProjectionY("TmpPred11Mass"    ,CutIndex+1,CutIndex+1,"o");
   TH1D* Data11     = ((TH2D*)GetObjectFromPath(InputFile_Data11, string("Data/") + HistoSuffix   ))->ProjectionY("TmpData11Mass"    ,CutIndex+1,CutIndex+1,"o");

   TFile* InputFile = new TFile((InputPattern+"/Histos.root").c_str());
   TH1D* Gluino600   = ((TH2D*)GetObjectFromPath(InputFile, string("Gluino600/") + HistoSuffix   ))->ProjectionY("TmpG600Mass"    ,CutIndex+1,CutIndex+1,"o");
   TH1D* GMStau156   = ((TH2D*)GetObjectFromPath(InputFile, string("GMStau156/") + HistoSuffix   ))->ProjectionY("TmpS156Mass"    ,CutIndex+1,CutIndex+1,"o");

   TFile* InputFile_MC = new TFile((InputPattern+"/Histos_MC.root").c_str());
   TH1D* MC     = ((TH2D*)GetObjectFromPath(InputFile_MC, string("MCTr/") + HistoSuffix   ))->ProjectionY("TmpMCMass"    ,CutIndex+1,CutIndex+1,"o");
   TH1D* MCPred = ((TH2D*)GetObjectFromPath(InputFile_MC, string("Pred_") + HistoSuffix   ))->ProjectionY("TmpMCPred"    ,CutIndex+1,CutIndex+1,"o");
   */
   TH1D*  H_A            = (TH1D*)GetObjectFromPath(InputFile_Data, "H_A");
   TH1D*  H_B            = (TH1D*)GetObjectFromPath(InputFile_Data, "H_B");
   TH1D*  H_C            = (TH1D*)GetObjectFromPath(InputFile_Data, "H_C");
   TH1D*  H_D            = (TH1D*)GetObjectFromPath(InputFile_Data, "H_D");
   TH1D*  H_E            = (TH1D*)GetObjectFromPath(InputFile_Data, "H_E");
   TH1D*  H_F            = (TH1D*)GetObjectFromPath(InputFile_Data, "H_F");
   TH1D*  H_G            = (TH1D*)GetObjectFromPath(InputFile_Data, "H_G");
   TH1D*  H_H            = (TH1D*)GetObjectFromPath(InputFile_Data, "H_H");
   TH1D*  H_P            = (TH1D*)GetObjectFromPath(InputFile_Data, "H_P");
   printf("OBSERVED  EVENTS = %6.2E\n",H_D->GetBinContent(CutIndex+1));
   printf("PREDICTED EVENTS = %6.2E+-%6.2E\n",H_P->GetBinContent(CutIndex+1), H_P->GetBinError(CutIndex+1));


   //MCPred->Scale(H_P->GetBinContent(CutIndex+1)/MC->Integral());
   //MC    ->Scale(H_P->GetBinContent(CutIndex+1)/MC->Integral());
   /*
   MC    ->Scale(H_P->GetBinContent(CutIndex+1)/MCPred->Integral());
   MCPred->Scale(H_P->GetBinContent(CutIndex+1)/MCPred->Integral());

   //rescale 2011 data to 2012 predicted yield 
   Data11->Scale(H_P->GetBinContent(CutIndex+1)/Pred11->Integral());
   Pred11->Scale(H_P->GetBinContent(CutIndex+1)/Pred11->Integral());
   */


   for(double M=0;M<=1000;M+=200){
      double D,P,Perr;
      D = Data->Integral( Data->GetXaxis()->FindBin(M),  Data->GetXaxis()->FindBin(2000.0));
      P = Pred->Integral( Pred->GetXaxis()->FindBin(M),  Pred->GetXaxis()->FindBin(2000.0));
      Perr = 0; for(int i=Pred->GetXaxis()->FindBin(M);i<Pred->GetXaxis()->FindBin(2000.0);i++){ Perr += pow(Pred->GetBinError(i),2); }  Perr = sqrt(Perr);
      double MD,MDerr, MP,MPerr;
      /*
      MD = MC->Integral( MC->GetXaxis()->FindBin(M),  MC->GetXaxis()->FindBin(2000.0));
      MP = MCPred->Integral( MCPred->GetXaxis()->FindBin(M),  MCPred->GetXaxis()->FindBin(2000.0));
      MDerr = 0; for(int i=MC->GetXaxis()->FindBin(M);i<MC->GetXaxis()->FindBin(2000.0);i++){ MDerr += pow(MC->GetBinError(i),2); }  MDerr = sqrt(MDerr);
      MPerr = 0; for(int i=MCPred->GetXaxis()->FindBin(M);i<MCPred->GetXaxis()->FindBin(2000.0);i++){MPerr += pow(MCPred->GetBinError(i),2); }  MPerr = sqrt(MPerr);
      printf("%4.0f<M<2000 --> Obs=%9.3f Data-Pred = %9.3f +- %8.3f(syst+stat) %9.3f (syst) %9.3f (stat) MC=%9.3f+-%8.3f   MC-Pred = %8.3f +- %9.3f (syst+stat) %9.3f (syst) %9.3f (stat)\n", M, D, P, sqrt(Perr*Perr + pow(P*(2*RMS),2)), P*(2*RMS), Perr, MD, MDerr, MP, sqrt(MPerr*MPerr + pow(MP*(2*RMS),2)), MP*(2*RMS), MPerr );
      */
   }
   printf("FullSpectrum --> D=%9.3f P = %9.3f +- %6.3f(stat) +- %6.3f(syst) (=%6.3f)\n", Data->Integral(), Pred->Integral(), 0.0, 0.0, 0.0 );
   printf("UnderFlow = %6.2f OverFlow = %6.2f\n", Data->GetBinContent(0), Data->GetBinContent(Data->GetNbinsX()+1) );
   printf("UnderFlow = %6.2f OverFlow = %6.2f\n", Pred->GetBinContent(0), Pred->GetBinContent(Pred->GetNbinsX()+1) );

   Pred->Rebin(2);
   Data->Rebin(2);
   /*
   TH1D* Signal = Gluino600;
   if(!IsTkOnly)Signal = GMStau156;
   Signal->Rebin(2);
   MC->Rebin(2);
   MCPred->Rebin(2);


   Pred11->Rebin(2);
   Data11->Rebin(2);
   */


   double Max = 2.0 * std::max(Data->GetMaximum(), Pred->GetMaximum());
   //double Max = 2.0 * std::max(std::max(Data->GetMaximum(), Pred->GetMaximum()), Signal->GetMaximum());
   double Min = 0.01;// 0.1 * std::min(0.01,Pred->GetMaximum());

   TLegend* leg;
   c1 = new TCanvas("c1","c1,",600,600);

   char YAxisLegend[1024];
   sprintf(YAxisLegend,"Tracks / %2.0f GeV/#font[12]{c}^{2}",Data->GetXaxis()->GetBinWidth(1));

   TH1D* PredErr = (TH1D*) Pred->Clone("PredErr");
   //TH1D* PredErr11 = (TH1D*) Pred->Clone("PredErr11");
   //TH1D* MCPredErr = (TH1D*) MCPred->Clone("MCPredErr");
   for(unsigned int i=0;i<(unsigned int)Pred->GetNbinsX();i++){
      double error = sqrt(pow(PredErr->GetBinError(i),2) + pow(PredErr->GetBinContent(i)*2*RMS,2));
      PredErr->SetBinError(i,error);       
      if(PredErr->GetBinContent(i)<Min && i>5){for(unsigned int j=i+1;j<(unsigned int)PredErr->GetNbinsX();j++)PredErr->SetBinContent(j,0);}


      //error = sqrt(pow(PredErr11->GetBinError(i),2) + pow(PredErr11->GetBinContent(i)*2*RMS,2));
      //PredErr11->SetBinError(i,error);
      //if(PredErr11->GetBinContent(i)<Min && i>5){for(unsigned int j=i+1;j<(unsigned int)PredErr11->GetNbinsX();j++)PredErr11->SetBinContent(j,0);}


      //error = sqrt(pow(MCPredErr->GetBinError(i),2) + pow(MCPredErr->GetBinContent(i)*2*RMS,2));
      //MCPredErr->SetBinError(i,error);
      //if(MCPredErr->GetBinContent(i)<Min && i>5){for(unsigned int j=i+1;j<(unsigned int)MCPredErr->GetNbinsX();j++)MCPredErr->SetBinContent(j,0);}
   }
   PredErr->SetLineColor(5);
   PredErr->SetFillColor(5);
   PredErr->SetFillStyle(1001);
   PredErr->SetMarkerStyle(22);
   PredErr->SetMarkerColor(5);
   PredErr->SetMarkerSize(1.0);
   PredErr->GetXaxis()->SetNdivisions(505);
   PredErr->SetTitle("");
   PredErr->SetStats(kFALSE);
   PredErr->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   PredErr->GetYaxis()->SetTitle(YAxisLegend);
   PredErr->GetYaxis()->SetTitleOffset(1.50);
   PredErr->SetMaximum(Max);
   PredErr->SetMinimum(Min);
   PredErr->SetAxisRange(0,1400,"X");
   PredErr->Draw("AXIS");
   /*
   Signal->SetMarkerStyle(21);
   Signal->SetMarkerColor(5);
   Signal->SetMarkerSize(1.5);
   Signal->SetLineColor(3);
   Signal->SetFillColor(3);
   //Signal->Draw("same HIST");

   MCPredErr->SetLineColor(7);
   MCPredErr->SetFillColor(7);
   MCPredErr->SetFillStyle(1001);
   MCPredErr->SetMarkerStyle(23);
   MCPredErr->SetMarkerColor(7);
   MCPredErr->SetMarkerSize(1.0);
   //MCPredErr->Draw("same E5");

   MCPred->SetMarkerStyle(23);
   MCPred->SetMarkerColor(4);
   MCPred->SetMarkerSize(1.5);
   MCPred->SetLineColor(4);
   MCPred->SetFillColor(0);
   //MCPred->Draw("same HIST P");

   //MC->SetFillStyle(3002);
   //MC->SetLineColor(22);
   //MC->SetFillColor(11);
   //MC->SetMarkerStyle(0);
   //MC->Draw("same HIST E1");

   PredErr11->SetLineColor(7);
   PredErr11->SetFillColor(7);
   PredErr11->SetFillStyle(1001);
   PredErr11->SetMarkerStyle(23);
   PredErr11->SetMarkerColor(7);
   PredErr11->SetMarkerSize(1.0);
   //PredErr11->Draw("same E5");

   Pred11->SetMarkerStyle(23);
   Pred11->SetMarkerColor(4);
   Pred11->SetMarkerSize(1.5);
   Pred11->SetLineColor(4);
   Pred11->SetFillColor(0);
   //Pred11->Draw("same HIST P");

   Data11->SetFillStyle(3002);
   Data11->SetLineColor(22);
   Data11->SetFillColor(11);
   Data11->SetMarkerStyle(0);
   //Data11->Draw("same HIST E1");
   */



   PredErr->Draw("same E5");

   Pred->SetMarkerStyle(22);
   Pred->SetMarkerColor(2);
   Pred->SetMarkerSize(1.5);
   Pred->SetLineColor(2);
   Pred->SetFillColor(0);
   Pred->Draw("same HIST P");

   Data->SetBinContent(Data->GetNbinsX(), Data->GetBinContent(Data->GetNbinsX()) + Data->GetBinContent(Data->GetNbinsX()+1));
   Data->SetMarkerStyle(20);
   Data->SetMarkerColor(1);
   Data->SetMarkerSize(1.0);
   Data->SetLineColor(1);
   Data->SetFillColor(0);
   Data->Draw("E1 same");





   //leg = new TLegend(0.69,0.93,0.40,0.68);
   if(IsTkOnly) leg = new TLegend(0.82,0.93,0.25,0.66);
   else leg = new TLegend(0.79,0.93,0.25,0.66);
   leg->SetHeader(LegendFromType(InputPattern).c_str());
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   TH1D* PredLeg = (TH1D*) Pred->Clone("RescLeg");
   PredLeg->SetFillColor(PredErr->GetFillColor());
   PredLeg->SetFillStyle(PredErr->GetFillStyle());
   leg->AddEntry(Data, "Observed"        ,"P");
   leg->AddEntry(PredLeg, "Data-based SM prediction"  ,"PF");
   //leg->AddEntry(MC, "Simulation"  ,"LF");
//   TH1D* MCPredLeg = (TH1D*) MCPred->Clone("RescMCLeg");
//   MCPredLeg->SetFillColor(MCPredErr->GetFillColor());
//   MCPredLeg->SetFillStyle(MCPredErr->GetFillStyle());
//   leg->AddEntry(MCPredLeg, "SM prediction (MC)"  ,"PF");

   //leg->AddEntry(Data11, "Data11"  ,"LF");
   //TH1D* Pred11Leg = (TH1D*) Pred11->Clone("RescPred11Leg");
   //Pred11Leg->SetFillColor(PredErr11->GetFillColor());
   //Pred11Leg->SetFillStyle(PredErr11->GetFillStyle());
   //leg->AddEntry(Pred11Leg, "Data-based SM prediction11"  ,"PF");
   //if(IsTkOnly)leg->AddEntry(Signal, "MC - Gluino (M=600 GeV/#font[12]{c}^{2})"        ,"F");
   //else        leg->AddEntry(Signal, "MC - Stau (M=156 GeV/#font[12]{c}^{2})"        ,"F");
   leg->Draw();

   DrawPreliminary(IntegratedLuminosity);
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("Rescale_") + HistoSuffix);
   delete c1;
   //InputFile->Close();
}

void MassPredictionTight(string InputPattern, unsigned int CutIndex, string HistoSuffix)
{
   bool IsTkOnly = (InputPattern.find("Type0",0)<std::string::npos);


   double Rescale, RMS;
   Rescale = 1.0;
   RMS     = 0.15;

   string outpath = InputPattern;
   MakeDirectories(outpath);

   TFile* InputFile_Data;
   string Input;

   std::vector<string> legend;
   TCanvas* c1;

   char Buffer[2048];
   sprintf(Buffer,"%s/Histos_Data.root",InputPattern.c_str());
   InputFile_Data = new TFile(Buffer);
   if(!InputFile_Data || InputFile_Data->IsZombie() || !InputFile_Data->IsOpen() || InputFile_Data->TestBit(TFile::kRecovered) )return;
   TH1D* Pred     = ((TH2D*)GetObjectFromPath(InputFile_Data, string("Pred_") + HistoSuffix   ))->ProjectionY("TmpPredMass"    ,CutIndex+1,CutIndex+1,"o");
   TH1D* Data     = ((TH2D*)GetObjectFromPath(InputFile_Data, string("Data/") + HistoSuffix   ))->ProjectionY("TmpDataMass"    ,CutIndex+1,CutIndex+1,"o");

   TFile* InputFile = new TFile((InputPattern+"/Histos.root").c_str());
   TH1D* GMStau247   = ((TH2D*)GetObjectFromPath(InputFile, string("GMStau247/") + HistoSuffix   ))->ProjectionY("TmpS247Mass"    ,CutIndex+1,CutIndex+1,"o");
   TH1D* GMStau156   = ((TH2D*)GetObjectFromPath(InputFile, string("GMStau156/") + HistoSuffix   ))->ProjectionY("TmpS156Mass"    ,CutIndex+1,CutIndex+1,"o");

   TH1D* Gluino800   = ((TH2D*)GetObjectFromPath(InputFile, string("Gluino800/") + HistoSuffix   ))->ProjectionY("TmpG800Mass"    ,CutIndex+1,CutIndex+1,"o");


   Pred->Rebin(4);
   Data->Rebin(4);
   TH1D* Signal = GMStau156;
   if(!IsTkOnly)Signal = GMStau247;
   Signal->Rebin(4);
   Gluino800->Rebin(4);

   double Max = 10.0 * std::max(std::max(Data->GetMaximum(), Pred->GetMaximum()), std::max(Signal->GetMaximum(), Gluino800->GetMaximum()));
   double Min = 0.01;// 0.1 * std::min(0.01,Pred->GetMaximum());
   double maxRange=1200;

   TLegend* leg;
   c1 = new TCanvas("c1","c1,",600,600);

   char YAxisLegend[1024];
   sprintf(YAxisLegend,"Tracks / %2.0f GeV/c^{2}",Data->GetXaxis()->GetBinWidth(1));


   double predOverFlow=0;
   for (int i=Pred->GetNbinsX(); i>0; i--) {
     if(Pred->GetBinLowEdge(i)>maxRange) predOverFlow+=Pred->GetBinContent(i);
     else {Pred->SetBinContent(i,predOverFlow+Pred->GetBinContent(i)); i=-1;}
   }

   double dataOverFlow=0;
   for (int i=Data->GetNbinsX(); i=0; i--) {
     if(Data->GetBinLowEdge(i)>maxRange) dataOverFlow+=Data->GetBinContent(i);
     else {
       Data->SetBinContent(i,dataOverFlow+Data->GetBinContent(i));
       Data->SetBinError(i,sqrt(dataOverFlow+Data->GetBinContent(i)));
       i=-1;
     }
   }


   TH1D* PredErr = (TH1D*) Pred->Clone("PredErr");
   for(unsigned int i=0;i<(unsigned int)Pred->GetNbinsX();i++){
      double error = sqrt(pow(PredErr->GetBinError(i+1),2) + pow(PredErr->GetBinContent(i+1)*2*RMS,2));
      PredErr->SetBinError(i+1,error);       
      if((PredErr->GetBinContent(i+1)<Min && i>5) || Pred->GetBinLowEdge(i+2)>maxRange){
	for(unsigned int j=i+1;j<(unsigned int)PredErr->GetNbinsX();j++)PredErr->SetBinContent(j+1,0);
	i=(unsigned int)Pred->GetNbinsX()+1;
      }
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
   PredErr->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   PredErr->GetYaxis()->SetTitle(YAxisLegend);
   PredErr->GetYaxis()->SetTitleOffset(1.50);
   PredErr->SetMaximum(Max);
   PredErr->SetMinimum(Min);
   PredErr->SetAxisRange(0,maxRange,"X");
   PredErr->Draw("AXIS");

   Gluino800->SetMarkerStyle(21);
   Gluino800->SetMarkerColor(46);
   Gluino800->SetMarkerSize(1.5);
   Gluino800->SetLineColor(46);
   Gluino800->SetFillColor(46);
   Gluino800->Draw("same HIST");

   Signal->SetMarkerStyle(21);
   Signal->SetMarkerColor(4);
   Signal->SetMarkerSize(1.5);
   Signal->SetLineColor(4);
   Signal->SetFillColor(38);
   Signal->Draw("same HIST");


   PredErr->Draw("same E5");

   Pred->SetMarkerStyle(22);
   Pred->SetMarkerColor(2);
   Pred->SetMarkerSize(1.5);
   Pred->SetLineColor(2);
   Pred->SetFillColor(0);
   Pred->Draw("same HIST P");

   Data->SetBinContent(Data->GetNbinsX(), Data->GetBinContent(Data->GetNbinsX()) + Data->GetBinContent(Data->GetNbinsX()+1));
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
   leg->AddEntry(Data, "Data"        ,"P");
   leg->AddEntry(PredLeg, "Data-based prediction"  ,"PF");
   if(IsTkOnly)leg->AddEntry(Signal, "MC - Stau (M=156 GeV/#font[12]{c}^{2})"        ,"F");
   else        leg->AddEntry(Signal, "MC - Stau (M=247 GeV/#font[12]{c}^{2})"        ,"F");
   leg->AddEntry(Gluino800, "MC - Gluino (M=800 GeV/#font[12]{c}^{2})"        ,"F");
   leg->Draw();

   DrawPreliminary(IntegratedLuminosity);
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("RescaleTight_") + HistoSuffix);
   delete c1;

   InputFile->Close();
}



int JobIdToIndex(string JobId){
   for(unsigned int s=0;s<signals.size();s++){
      if(signals[s].Name==JobId)return s;
   }return -1;
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
