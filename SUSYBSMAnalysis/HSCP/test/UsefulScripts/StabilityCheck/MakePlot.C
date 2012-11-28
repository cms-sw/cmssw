
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

std::map<unsigned int, double> RunToIntLumi;

bool LoadLumiToRun()
{
   float TotalIntLuminosity = 0;

   FILE* pFile = fopen("out.txt","r");
   if(!pFile){
      printf("Not Found: %s\n","out.txt");
      return false;
   }

   unsigned int Run; float IntLumi;
   unsigned int DeliveredLs; double DeliveredLumi;
   char Line[2048], Tmp1[2048], Tmp2[2048], Tmp3[2048];
   while ( ! feof (pFile) ){
     fscanf(pFile,"%s\n",Line);
     //printf("%s\n",Line);
     for(unsigned int i=0;Line[i]!='\0';i++){if(Line[i]==',')Line[i]=' ';} 
     sscanf(Line,"%d:%d %s %s %s %f\n",&Run,Tmp1,&Tmp1,Tmp2,Tmp3,&IntLumi);
     TotalIntLuminosity+= IntLumi/1000000.0;
     printf("%6i --> %f/pb   (%s | %s | %s)\n",Run,TotalIntLuminosity,Tmp1,Tmp2,Tmp3);
     RunToIntLumi[Run] = TotalIntLuminosity;
   }
   fclose(pFile);
   return true;
}


TGraph* ConvertFromRunToIntLumi(TProfile* Object, const char* DrawOption, string YLabel, double YRange_Min=2.8, double YRange_Max=3.5){
   TGraphErrors* graph = new TGraphErrors(Object->GetXaxis()->GetNbins());
   for(unsigned int i=1;i<Object->GetXaxis()->GetNbins()+1;i++){
      int RunNumber;
      sscanf(Object->GetXaxis()->GetBinLabel(i),"%d",&RunNumber);
      graph->SetPoint(i-1, RunToIntLumi[RunNumber], Object->GetBinContent(i));
      graph->SetPointError(i-1, 0.0*RunToIntLumi[RunNumber], Object->GetBinError(i));
   }
   graph->Draw(DrawOption);
   graph->SetTitle("");
   graph->GetYaxis()->SetTitle(Object->GetYaxis()->GetTitle());
   graph->GetYaxis()->SetTitleOffset(1.10);
   graph->GetXaxis()->SetTitle("Int. Luminosity (/pb)");
   graph->GetYaxis()->SetTitle(YLabel.c_str());
   graph->SetMarkerColor(Object->GetMarkerColor());
   graph->SetMarkerStyle(Object->GetMarkerStyle());
   graph->GetXaxis()->SetNdivisions(510);
   if(YRange_Min!=YRange_Max)graph->GetYaxis()->SetRangeUser(YRange_Min,YRange_Max);
   return graph;
}


void DrawLines(int color, double ymin=2.8, double ymax=3.5){
   unsigned int runs[] = {192701, 194552, 198301, 199878, 201820, 202914, 206037};
//    unsigned int runs[] = {193093, 194619, 198272, 199960};
   for(unsigned int i=0;i<sizeof(runs)/sizeof(unsigned int);i++){
      //find the closest processed run
      int closestRun = 0;
      for(std::map<unsigned int, double>::iterator it = RunToIntLumi.begin(); it!=RunToIntLumi.end();it++){
         if(it->first>runs[i] && abs(it->first-runs[i])<abs(closestRun-runs[i]))closestRun = it->first;
      }

      printf("Draw line for run %i at %f\n",closestRun, RunToIntLumi[closestRun]);
      TLine* line = new TLine( RunToIntLumi[closestRun], ymin, RunToIntLumi[closestRun], ymax);
      line->SetLineColor(color);
      line->SetLineWidth(3);
      line->Draw("same");
   }
}


void MakedEdxPlot()
{
   setTDRStyle();
   gStyle->SetPadTopMargin   (0.06);
   gStyle->SetPadBottomMargin(0.15);
   gStyle->SetPadRightMargin (0.03);
   gStyle->SetPadLeftMargin  (0.09);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505);

   TCanvas* c1;
   TObject** Histos = new TObject*[10];
   std::vector<string> legend;

   TFile* InputFile = new TFile("pictures/Histos.root");

   TProfile* SingleMu_PtProf           = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterPtProf");      
   TProfile* SingleMu_dEdxProf         = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxProf");   
   TProfile* SingleMu_dEdxMProf        = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMProf");
   TProfile* SingleMu_dEdxMSProf       = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMSProf");
   TProfile* SingleMu_dEdxMPProf       = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMPProf");
   TProfile* SingleMu_dEdxMSCProf      = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMSCProf");
   TProfile* SingleMu_dEdxMPCProf      = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMPCProf");
   TProfile* SingleMu_dEdxMSFProf      = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMSFProf");
   TProfile* SingleMu_dEdxMPFProf      = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMPFProf");

   TProfile* SingleMu_NVertProf        = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterNVertProf");

   SingleMu_NVertProf->LabelsDeflate("X");
   SingleMu_NVertProf->LabelsOption("av","X");

/*
   TFile* InputFileLumi166380 = new TFile("pictures/HistosLumi166380.root");
   TFile* InputFileLumi166512 = new TFile("pictures/HistosLumi166512.root");
   TFile* InputFileLumi167807 = new TFile("pictures/HistosLumi167807.root");
   TFile* InputFileLumi167898 = new TFile("pictures/HistosLumi167898.root");

   TProfile* SingleMu_dEdxMProfLumi166380         = (TProfile*)GetObjectFromPath(InputFileLumi166380, "HSCPHLTTriggerMuFilterdEdxMProf");
   TProfile* SingleMu_dEdxMProfLumi166512         = (TProfile*)GetObjectFromPath(InputFileLumi166512, "HSCPHLTTriggerMuFilterdEdxMProf");
   TProfile* SingleMu_dEdxMProfLumi167807         = (TProfile*)GetObjectFromPath(InputFileLumi167807, "HSCPHLTTriggerMuFilterdEdxMProf");
   TProfile* SingleMu_dEdxMProfLumi167898         = (TProfile*)GetObjectFromPath(InputFileLumi167898, "HSCPHLTTriggerMuFilterdEdxMProf");
*/

   if(LoadLumiToRun()){
      TLegend* leg;

      c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
      TGraph* graph =  ConvertFromRunToIntLumi(SingleMu_dEdxMProf  , "A*", "I_{h} (MeV/cm)");
      TGraph* graphS = ConvertFromRunToIntLumi(SingleMu_dEdxMSProf, "*" , "I_{h} (MeV/cm)");
      TGraph* graphP = ConvertFromRunToIntLumi(SingleMu_dEdxMPProf, "*" , "I_{h} (MeV/cm)");
      graphS->SetMarkerColor(2);    graphS->SetMarkerStyle(26);
      graphP->SetMarkerColor(4);    graphP->SetMarkerStyle(32);


      TF1* myfunc = new TF1("Fitgraph" ,"pol1",250,5000);  graph ->Fit(myfunc ,"QN","",250,5000); myfunc ->SetLineWidth(2); myfunc ->SetLineColor(graph ->GetMarkerColor()); myfunc ->Draw("same");
      TF1* myfuncS= new TF1("FitgraphS","pol1",250,5000);  graphS->Fit(myfuncS,"QN","",250,5000); myfuncS->SetLineWidth(2); myfuncS->SetLineColor(graphS->GetMarkerColor()); myfuncS->Draw("same");
      TF1* myfuncP= new TF1("FitgraphP","pol1",250,5000);  graphP->Fit(myfuncP,"QN","",250,5000); myfuncP->SetLineWidth(2); myfuncP->SetLineColor(graphP->GetMarkerColor()); myfuncP->Draw("same");
      printf("%25s --> Chi2/ndf = %6.2f --> a=%6.2E+-%6.2E   b=%6.2E+-%6.2E\n","dE/dx (Strip+Pixel)", myfunc ->GetChisquare()/ myfunc ->GetNDF(), myfunc ->GetParameter(0),myfunc ->GetParError(0),myfunc ->GetParameter(1),myfunc ->GetParError(1));
      printf("%25s --> Chi2/ndf = %6.2f --> a=%6.2E+-%6.2E   b=%6.2E+-%6.2E\n","dE/dx (Strip)"      , myfuncS->GetChisquare()/ myfuncS->GetNDF(), myfuncS->GetParameter(0),myfuncS->GetParError(0),myfuncS->GetParameter(1),myfuncS->GetParError(1));
      printf("%25s --> Chi2/ndf = %6.2f --> a=%6.2E+-%6.2E   b=%6.2E+-%6.2E\n","dE/dx (Pixel)"      , myfuncP->GetChisquare()/ myfuncP->GetNDF(), myfuncP->GetParameter(0),myfuncP->GetParError(0),myfuncP->GetParameter(1),myfuncP->GetParError(1));
      leg = new TLegend(0.79,0.92,0.79-0.20,0.92 - 3*0.05);     leg->SetFillColor(0);     leg->SetBorderSize(0);
      leg->AddEntry(graph, "dE/dx (Strip+Pixel)" ,"P");
      leg->AddEntry(graphS, "dE/dx (Strip)" ,"P");
      leg->AddEntry(graphP, "dE/dx (Pixel)" ,"P");
      leg->Draw();
      SaveCanvas(c1,"pictures/","GraphdEdx_Profile_dEdxM");
      delete c1;  delete leg;

      c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
      TGraph* graphSC = ConvertFromRunToIntLumi(SingleMu_dEdxMSCProf, "A*", "I_{h} (MeV/cm)");
      TGraph* graphSF = ConvertFromRunToIntLumi(SingleMu_dEdxMSFProf, "*" , "I_{h} (MeV/cm)");
      graphSC->SetMarkerColor(2);    graphSC->SetMarkerStyle(26);
      graphSF->SetMarkerColor(4);    graphSF->SetMarkerStyle(32);
      TF1* myfuncSC= new TF1("FitgraphSC","pol1",250,5000);  graphSC->Fit(myfuncSC,"QN","",250,5000); myfuncSC->SetLineWidth(2); myfuncSC->SetLineColor(graphSC->GetMarkerColor()); myfuncSC->Draw("same");
      TF1* myfuncSF= new TF1("FitgraphSF","pol1",250,5000);  graphSF->Fit(myfuncSF,"QN","",250,5000); myfuncSF->SetLineWidth(2); myfuncSF->SetLineColor(graphSF->GetMarkerColor()); myfuncSF->Draw("same");
      printf("%25s --> Chi2/ndf = %6.2f --> a=%6.2E+-%6.2E   b=%6.2E+-%6.2E\n","dE/dx (Strip) |eta|<0.5", myfuncSC->GetChisquare()/ myfuncSC->GetNDF(), myfuncSC->GetParameter(0),myfuncSC->GetParError(0),myfuncSC->GetParameter(1),myfuncSC->GetParError(1));
      printf("%25s --> Chi2/ndf = %6.2f --> a=%6.2E+-%6.2E   b=%6.2E+-%6.2E\n","dE/dx (Strip) |eta|>1.5", myfuncSF->GetChisquare()/ myfuncSF->GetNDF(), myfuncSF->GetParameter(0),myfuncSF->GetParError(0),myfuncSF->GetParameter(1),myfuncSF->GetParError(1));
      leg = new TLegend(0.79,0.92,0.79-0.20,0.92 - 3*0.05);     leg->SetFillColor(0);     leg->SetBorderSize(0);
      leg->AddEntry(graphSC, "dE/dx (Strip) |#eta|<0.5" ,"P");
      leg->AddEntry(graphSF, "dE/dx (Strip) |#eta|>1.5"  ,"P");
      leg->Draw();
      DrawLines(1);
      SaveCanvas(c1,"pictures/","GraphdEdx_Profile_dEdxMS");
      delete c1; delete leg;

      c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
      TGraph* graphPC = ConvertFromRunToIntLumi(SingleMu_dEdxMPCProf, "A*", "I_{h} (MeV/cm)");
      TGraph* graphPF = ConvertFromRunToIntLumi(SingleMu_dEdxMPFProf, "*" , "I_{h} (MeV/cm)");
      graphPC->SetMarkerColor(2);    graphPC->SetMarkerStyle(26);
      graphPF->SetMarkerColor(4);    graphPF->SetMarkerStyle(32);
      TF1* myfuncPC= new TF1("FitgraphPC","pol1",250,5000);  graphPC->Fit(myfuncPC,"QN","",250,5000); myfuncPC->SetLineWidth(2); myfuncPC->SetLineColor(graphPC->GetMarkerColor()); myfuncPC->Draw("same");
      TF1* myfuncPF= new TF1("FitgraphPF","pol1",250,5000);  graphPF->Fit(myfuncPF,"QN","",250,5000); myfuncPF->SetLineWidth(2); myfuncPF->SetLineColor(graphPF->GetMarkerColor()); myfuncPF->Draw("same");
      printf("%25s --> Chi2/ndf = %6.2f --> a=%6.2E+-%6.2E   b=%6.2E+-%6.2E\n","dE/dx (Pixel) |eta|<0.5", myfuncPC->GetChisquare()/ myfuncPC->GetNDF(), myfuncPC->GetParameter(0),myfuncPC->GetParError(0),myfuncPC->GetParameter(1),myfuncPC->GetParError(1));
      printf("%25s --> Chi2/ndf = %6.2f --> a=%6.2E+-%6.2E   b=%6.2E+-%6.2E\n","dE/dx (Pixel) |eta|>1.5", myfuncPF->GetChisquare()/ myfuncPF->GetNDF(), myfuncPF->GetParameter(0),myfuncPF->GetParError(0),myfuncPF->GetParameter(1),myfuncPF->GetParError(1));
      leg = new TLegend(0.79,0.92,0.79-0.20,0.92 - 3*0.05);     leg->SetFillColor(0);     leg->SetBorderSize(0);
      leg->AddEntry(graphPC, "dE/dx (Pixel) |#eta|<0.5" ,"P");
      leg->AddEntry(graphPF, "dE/dx (Pixel) |#eta|>1.5" ,"P");
      leg->Draw();
      SaveCanvas(c1,"pictures/","GraphdEdx_Profile_dEdxMP");
      delete c1; delete leg;




      c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
      TGraph* graphNV = ConvertFromRunToIntLumi(SingleMu_NVertProf, "A*" , "<#Reco Vertices>",0,0);
      SaveCanvas(c1,"pictures/","GraphdEdx_Profile_Vert");
      delete c1;

      c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
      TGraph* graphpT = ConvertFromRunToIntLumi(SingleMu_PtProf, "A*" , "<p_{T}> (GeV/c)",0,0);
      SaveCanvas(c1,"pictures/","GraphdEdx_Profile_pT");
      delete c1;


   }else{
      printf("TEST TEST TEST\n");
   }


   for(unsigned int i=0;i<SingleMu_PtProf->GetXaxis()->GetNbins();i++){
      if((i+11)%12==0)continue;
      SingleMu_PtProf->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxProf->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxMProf->GetXaxis()->SetBinLabel(i,"");
      SingleMu_NVertProf->GetXaxis()->SetBinLabel(i,"");
   }  


   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = SingleMu_NVertProf;                 legend.push_back("SingleMu40");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "<#Reco Vertices>", 0,0, 0,0);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","dEdx_Profile_NVert");
   delete c1;

 

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = SingleMu_PtProf;                    legend.push_back("SingleMu40");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "p_{T} (GeV/c)", 0,0, 0,0);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","dEdx_Profile_Pt");
   delete c1;


   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = SingleMu_dEdxProf;                  legend.push_back("SingleMu40");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{as}", 0,0, 0.02,0.06);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","dEdx_Profile_dEdx");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = SingleMu_dEdxMProf;                  legend.push_back("SingleMu40");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h}", 0,0, 3.2,3.4);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","dEdx_Profile_dEdxM");
   delete c1;

/*
   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = SingleMu_dEdxMProfLumi166380;       legend.push_back("SingleMu40 - Run166380");
   Histos[1] = SingleMu_dEdxMProfLumi166512;       legend.push_back("SingleMu40 - Run166512");
   Histos[2] = SingleMu_dEdxMProfLumi167807;       legend.push_back("SingleMu40 - Run167807");
   Histos[3] = SingleMu_dEdxMProfLumi167898;       legend.push_back("SingleMu40 - Run167898");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Lumi", "I_{h}", 0,0, 3.2,3.4);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","dEdx_Profile_dEdxMRun");
   delete c1;
*/


}





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
   TProfile* SingleMu_PtProf           = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterPtProf");
   TProfile* PFMet_PtProf              = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterPtProf");

   TProfile* Any_dEdxProf              = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxProf");
   TProfile* SingleMu_dEdxProf         = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxProf");
   TProfile* PFMet_dEdxProf            = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterdEdxProf");

   TProfile* Any_dEdxMProf             = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxMProf");
   TProfile* SingleMu_dEdxMProf        = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMProf");
   TProfile* PFMet_dEdxMProf           = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterdEdxMProf");

   TProfile* Any_dEdxMSProf             = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxMSProf");
   TProfile* SingleMu_dEdxMSProf        = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMSProf");
   TProfile* PFMet_dEdxMSProf           = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterdEdxMSProf");

   TProfile* Any_dEdxMPProf             = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxMPProf");
   TProfile* SingleMu_dEdxMPProf        = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMPProf");
   TProfile* PFMet_dEdxMPProf           = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterdEdxMPProf");

   TProfile* Any_dEdxMSCProf             = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxMSCProf");
   TProfile* SingleMu_dEdxMSCProf        = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMSCProf");
   TProfile* PFMet_dEdxMSCProf           = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterdEdxMSCProf");

   TProfile* Any_dEdxMPCProf             = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxMPCProf");
   TProfile* SingleMu_dEdxMPCProf        = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMPCProf");
   TProfile* PFMet_dEdxMPCProf           = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterdEdxMPCProf");

   TProfile* Any_dEdxMSFProf             = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxMSFProf");
   TProfile* SingleMu_dEdxMSFProf        = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMSFProf");
   TProfile* PFMet_dEdxMSFProf           = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterdEdxMSFProf");

   TProfile* Any_dEdxMPFProf             = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxMPFProf");
   TProfile* SingleMu_dEdxMPFProf        = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterdEdxMPFProf");
   TProfile* PFMet_dEdxMPFProf           = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterdEdxMPFProf");

   TProfile* Any_TOFProf               = (TProfile*)GetObjectFromPath(InputFile, "AnyTOFProf");
   TProfile* SingleMu_TOFProf          = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterTOFProf");
   TProfile* PFMet_TOFProf             = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterTOFProf");

   TProfile* Any_TOFDTProf             = (TProfile*)GetObjectFromPath(InputFile, "AnyTOFDTProf");
   TProfile* SingleMu_TOFDTProf        = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterTOFDTProf");
   TProfile* PFMet_TOFDTProf           = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterTOFDTProf");

   TProfile* Any_TOFCSCProf            = (TProfile*)GetObjectFromPath(InputFile, "AnyTOFCSCProf");
   TProfile* SingleMu_TOFCSCProf       = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterTOFCSCProf");
   TProfile* PFMet_TOFCSCProf          = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterTOFCSCProf");

   TProfile* Any_VertexProf               = (TProfile*)GetObjectFromPath(InputFile, "AnyVertexProf");
   TProfile* SingleMu_VertexProf          = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterVertexProf");
   TProfile* PFMet_VertexProf             = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterVertexProf");

   TProfile* Any_VertexDTProf             = (TProfile*)GetObjectFromPath(InputFile, "AnyVertexDTProf");
   TProfile* SingleMu_VertexDTProf        = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterVertexDTProf");
   TProfile* PFMet_VertexDTProf           = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterVertexDTProf");

   TProfile* Any_VertexCSCProf            = (TProfile*)GetObjectFromPath(InputFile, "AnyVertexCSCProf");
   TProfile* SingleMu_VertexCSCProf       = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterVertexCSCProf");
   TProfile* PFMet_VertexCSCProf          = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterVertexCSCProf");

   TProfile* Any_HdEdx                 = (TProfile*)GetObjectFromPath(InputFile, "AnyHdEdx");
   TProfile* SingleMu_HdEdx          = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterHdEdx");
   TProfile* PFMet_HdEdx             = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterHdEdx");

   TProfile* Any_HPt                 = (TProfile*)GetObjectFromPath(InputFile, "AnyHPt");
   TProfile* SingleMu_HPt          = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterHPt");
   TProfile* PFMet_HPt             = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterHPt");

   TProfile* Any_HTOF                 = (TProfile*)GetObjectFromPath(InputFile, "AnyHTOF");
   TProfile* SingleMu_HTOF          = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMuFilterHTOF");
   TProfile* PFMet_HTOF             = (TProfile*)GetObjectFromPath(InputFile, "HSCPHLTTriggerMetDeDxFilterHTOF");

   for(unsigned int i=0;i<SingleMu_PtProf->GetXaxis()->GetNbins();i++){
      if((i+11)%12==0)continue;
      Any_PtProf         ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_PtProf    ->GetXaxis()->SetBinLabel(i,"");
      PFMet_PtProf       ->GetXaxis()->SetBinLabel(i,"");

      Any_dEdxProf       ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxProf  ->GetXaxis()->SetBinLabel(i,"");
      PFMet_dEdxProf     ->GetXaxis()->SetBinLabel(i,"");

      Any_dEdxMProf      ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxMProf ->GetXaxis()->SetBinLabel(i,"");
      PFMet_dEdxMProf    ->GetXaxis()->SetBinLabel(i,"");

      Any_dEdxMSProf      ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxMSProf ->GetXaxis()->SetBinLabel(i,"");
      PFMet_dEdxMSProf    ->GetXaxis()->SetBinLabel(i,"");

      Any_dEdxMPProf      ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxMPProf ->GetXaxis()->SetBinLabel(i,"");
      PFMet_dEdxMPProf    ->GetXaxis()->SetBinLabel(i,"");

      Any_dEdxMSCProf     ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxMSCProf->GetXaxis()->SetBinLabel(i,"");
      PFMet_dEdxMSCProf   ->GetXaxis()->SetBinLabel(i,"");

      Any_dEdxMPCProf     ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxMPCProf->GetXaxis()->SetBinLabel(i,"");
      PFMet_dEdxMPCProf   ->GetXaxis()->SetBinLabel(i,"");

      Any_dEdxMSFProf     ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxMSFProf->GetXaxis()->SetBinLabel(i,"");
      PFMet_dEdxMSFProf   ->GetXaxis()->SetBinLabel(i,"");

      Any_dEdxMPFProf     ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxMPFProf->GetXaxis()->SetBinLabel(i,"");
      PFMet_dEdxMPFProf   ->GetXaxis()->SetBinLabel(i,"");

      Any_TOFProf        ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_TOFProf   ->GetXaxis()->SetBinLabel(i,"");
      PFMet_TOFProf      ->GetXaxis()->SetBinLabel(i,"");

      Any_TOFDTProf      ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_TOFDTProf ->GetXaxis()->SetBinLabel(i,"");
      PFMet_TOFDTProf    ->GetXaxis()->SetBinLabel(i,"");

      Any_TOFCSCProf     ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_TOFCSCProf->GetXaxis()->SetBinLabel(i,"");
      PFMet_TOFCSCProf   ->GetXaxis()->SetBinLabel(i,"");

      Any_VertexProf        ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_VertexProf   ->GetXaxis()->SetBinLabel(i,"");
      PFMet_VertexProf      ->GetXaxis()->SetBinLabel(i,"");

      Any_VertexDTProf      ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_VertexDTProf ->GetXaxis()->SetBinLabel(i,"");
      PFMet_VertexDTProf    ->GetXaxis()->SetBinLabel(i,"");

      Any_VertexCSCProf     ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_VertexCSCProf->GetXaxis()->SetBinLabel(i,"");
      PFMet_VertexCSCProf   ->GetXaxis()->SetBinLabel(i,"");

      Any_HdEdx          ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_HdEdx     ->GetXaxis()->SetBinLabel(i,"");
      PFMet_HdEdx        ->GetXaxis()->SetBinLabel(i,"");

      Any_HPt            ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_HPt       ->GetXaxis()->SetBinLabel(i,"");
      PFMet_HPt          ->GetXaxis()->SetBinLabel(i,"");

      Any_HTOF           ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_HTOF      ->GetXaxis()->SetBinLabel(i,"");
      PFMet_HTOF         ->GetXaxis()->SetBinLabel(i,"");
   }  



   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_PtProf;                         legend.push_back("Any");
   //Histos[1] = SingleMu_PtProf;                    legend.push_back("SingleMu40");
   //Histos[2] = PFMet_PtProf;                       legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "p_{T} (GeV/c)", 0,0, 0,150);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_Pt");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxProf;                       legend.push_back("Any");
   //Histos[1] = SingleMu_dEdxProf;                  legend.push_back("SingleMu40");
   //Histos[2] = PFMet_dEdxProf;                     legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{as}", 0,0, 0.02,0.05);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdx");
   delete c1;


   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxMProf;                      legend.push_back("Any");
   //Histos[1] = SingleMu_dEdxMProf;                 legend.push_back("SingleMu40");
   //Histos[2] = PFMet_dEdxMProf;                    legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h}", 0,0, 3.1,3.6);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdxM");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxMSProf;                      legend.push_back("Any");
   //Histos[1] = SingleMu_dEdxMSProf;                 legend.push_back("SingleMu40");
   //Histos[2] = PFMet_dEdxMSProf;                    legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h} S", 0,0, 3.1,3.6);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdxMS");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxMPProf;                      legend.push_back("Any");
   //Histos[1] = SingleMu_dEdxMPProf;                 legend.push_back("SingleMu40");
   //Histos[2] = PFMet_dEdxMPProf;                    legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h} P", 0,0, 3.1,3.6);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdxMP");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxMSCProf;                      legend.push_back("Any");
   //Histos[1] = SingleMu_dEdxMSCProf;                 legend.push_back("SingleMu40");
   //Histos[2] = PFMet_dEdxMSCProf;                    legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h} SC", 0,0, 3.1,3.6);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdxMSC");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxMPCProf;                      legend.push_back("Any");
   //Histos[1] = SingleMu_dEdxMPCProf;                 legend.push_back("SingleMu40");
   //Histos[2] = PFMet_dEdxMPCProf;                    legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h} PC", 0,0, 3.1,3.6);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdxMPC");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxMSFProf;                      legend.push_back("Any");
   //Histos[1] = SingleMu_dEdxMSFProf;                 legend.push_back("SingleMu40");
   //Histos[2] = PFMet_dEdxMSFProf;                    legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h} SF", 0,0, 3.1,3.6);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdxMSF");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxMPFProf;                      legend.push_back("Any");
   //Histos[1] = SingleMu_dEdxMPFProf;                 legend.push_back("SingleMu40");
   //Histos[2] = PFMet_dEdxMPFProf;                    legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h} PF", 0,0, 3.1,3.6);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdxMPF");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_TOFProf;                        legend.push_back("Any");
   //Histos[1] = SingleMu_TOFProf;                   legend.push_back("SingleMu40");
   //Histos[2] = PFMet_TOFProf;                      legend.push_back("PFMHT150");
   
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "1/#beta_{TOF}", 0,0, 0.85,1.15);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_TOF");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_TOFDTProf;                        legend.push_back("Any");
   //Histos[1] = SingleMu_TOFDTProf;                   legend.push_back("SingleMu40");
   //Histos[2] = PFMet_TOFDTProf;                      legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "1/#beta_{TOF_DT}", 0,0, 0.85,1.15);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_TOFDT");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_TOFCSCProf;                        legend.push_back("Any");
   //Histos[1] = SingleMu_TOFCSCProf;                   legend.push_back("SingleMu40");
   //Histos[2] = PFMet_TOFCSCProf;                      legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "1/#beta_{TOF_CSC}", 0,0, 0.85,1.15);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_TOFCSC");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_VertexProf;                        legend.push_back("Any");
   //Histos[1] = SingleMu_VertexProf;                   legend.push_back("SingleMu40");
   //Histos[2] = PFMet_VertexProf;                      legend.push_back("PFMHT150");
   
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "Vertex time [ns]", 0,0, -4,4);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_Vertex");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_VertexDTProf;                        legend.push_back("Any");
   //Histos[1] = SingleMu_VertexDTProf;                   legend.push_back("SingleMu40");
   //Histos[2] = PFMet_VertexDTProf;                      legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "Vertex Time DT [ns]", 0,0, -4,4);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_VertexDT");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_VertexCSCProf;                        legend.push_back("Any");
   //Histos[1] = SingleMu_VertexCSCProf;                   legend.push_back("SingleMu40");
   //Histos[2] = PFMet_VertexCSCProf;                      legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "Vertex Time CSC [ns]", 0,0, -4,4);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_VertexCSC");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   //Histos[0] = Any_HdEdx;                        legend.push_back("Any");
   Histos[0] = Any_HdEdx;                        legend.push_back("I_{as} > 0.15");
   //Histos[1] = SingleMu_HdEdx;                   legend.push_back("SingleMu40");
   //Histos[2] = PFMet_HdEdx;                      legend.push_back("PFMHT150");
   
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h} ROT", 0,0, 0,0.05);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_ROT_dEdx");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   //Histos[0] = Any_HPt;                        legend.push_back("Any");
   Histos[0] = Any_HPt;                        legend.push_back("p_{T} > 60 GeV/c");
   //Histos[1] = SingleMu_HPt;                   legend.push_back("SingleMu40");
   //Histos[2] = PFMet_HPt;                      legend.push_back("PFMHT150");
   
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "Pt ROT", 0,0, 0.15,0.5);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_ROT_Pt");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   //Histos[0] = Any_HTOF;                        legend.push_back("Any");
   Histos[0] = Any_HTOF;                        legend.push_back("1/#beta > 1.1");
   //Histos[1] = SingleMu_HTOF;                   legend.push_back("SingleMu40");
   //Histos[2] = PFMet_HTOF;                      legend.push_back("PFMHT150");
   
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "1/#beta_{TOF} ROT", 0,0, 0,0.2);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   //DrawLegend(Histos,legend,"","P");
   DrawPreliminary(SQRTS,IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_ROT_TOF");
   delete c1;

   MakedEdxPlot();
}


