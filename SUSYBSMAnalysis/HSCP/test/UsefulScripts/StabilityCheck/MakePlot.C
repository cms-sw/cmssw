
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
//     printf("%s\n",Line);
     for(unsigned int i=0;Line[i]!='\0';i++){if(Line[i]==',')Line[i]=' ';} 
     sscanf(Line,"%d %s %s %s %f\n",&Run,Tmp1,Tmp2,Tmp3,&IntLumi);
     TotalIntLuminosity+= IntLumi/1000000.0;
//     printf("%6i --> %f/pb   (%s | %s | %s)\n",Run,TotalIntLuminosity,Tmp1,Tmp2,Tmp3);
     RunToIntLumi[Run] = TotalIntLuminosity;
   }
   fclose(pFile);
   return true;
}



TGraph* ConvertFromRunToIntLumi(TProfile* Object, const char* DrawOption, string YLabel){
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
   graph->GetYaxis()->SetRangeUser(3.0,3.6);
   return graph;
}

void MakedEdxPlot()
{
   TCanvas* c1;
   TObject** Histos = new TObject*[10];
   std::vector<string> legend;

   TFile* InputFile = new TFile("pictures/Histos.root");

   TProfile* SingleMu_PtProf           = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMuPtProf");      
   TProfile* SingleMu_dEdxProf         = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxProf");   
   TProfile* SingleMu_dEdxMProf        = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxMProf");
   TProfile* SingleMu_dEdxMSProf       = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxMSProf");
   TProfile* SingleMu_dEdxMPProf       = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxMPProf");
   TProfile* SingleMu_dEdxMSCProf      = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxMSCProf");
   TProfile* SingleMu_dEdxMPCProf      = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxMPCProf");
   TProfile* SingleMu_dEdxMSFProf      = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxMSFProf");
   TProfile* SingleMu_dEdxMPFProf      = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxMPFProf");

   TProfile* SingleMu_NVertProf        = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMuNVertProf");

   SingleMu_NVertProf->LabelsDeflate("X");
   SingleMu_NVertProf->LabelsOption("av","X");

/*
   TFile* InputFileLumi166380 = new TFile("pictures/HistosLumi166380.root");
   TFile* InputFileLumi166512 = new TFile("pictures/HistosLumi166512.root");
   TFile* InputFileLumi167807 = new TFile("pictures/HistosLumi167807.root");
   TFile* InputFileLumi167898 = new TFile("pictures/HistosLumi167898.root");

   TProfile* SingleMu_dEdxMProfLumi166380         = (TProfile*)GetObjectFromPath(InputFileLumi166380, "HscpPathSingleMudEdxMProf");
   TProfile* SingleMu_dEdxMProfLumi166512         = (TProfile*)GetObjectFromPath(InputFileLumi166512, "HscpPathSingleMudEdxMProf");
   TProfile* SingleMu_dEdxMProfLumi167807         = (TProfile*)GetObjectFromPath(InputFileLumi167807, "HscpPathSingleMudEdxMProf");
   TProfile* SingleMu_dEdxMProfLumi167898         = (TProfile*)GetObjectFromPath(InputFileLumi167898, "HscpPathSingleMudEdxMProf");
*/

   if(LoadLumiToRun()){
      c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
      TGraph* graph = ConvertFromRunToIntLumi(SingleMu_dEdxMProf  , "A*", "I_{h} (MeV/cm)");
      TGraph* graphS = ConvertFromRunToIntLumi(SingleMu_dEdxMSProf, "*" , "I_{h} (MeV/cm)");
      TGraph* graphP = ConvertFromRunToIntLumi(SingleMu_dEdxMPProf, "*" , "I_{h} (MeV/cm)");
      graphS->SetMarkerColor(2);    graphS->SetMarkerStyle(26);
      graphP->SetMarkerColor(4);    graphP->SetMarkerStyle(32);
      SaveCanvas(c1,"pictures/","GraphdEdx_Profile_dEdxM");
      delete c1;

      c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
      TGraph* graphSC = ConvertFromRunToIntLumi(SingleMu_dEdxMSCProf, "A*", "I_{h} (MeV/cm)");
      TGraph* graphSF = ConvertFromRunToIntLumi(SingleMu_dEdxMSFProf, "*" , "I_{h} (MeV/cm)");
      graphSC->SetMarkerColor(2);    graphSC->SetMarkerStyle(26);
      graphSF->SetMarkerColor(4);    graphSF->SetMarkerStyle(32);
      SaveCanvas(c1,"pictures/","GraphdEdx_Profile_dEdxMS");
      delete c1;

      c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
      TGraph* graphPC = ConvertFromRunToIntLumi(SingleMu_dEdxMPCProf, "A*", "I_{h} (MeV/cm)");
      TGraph* graphPF = ConvertFromRunToIntLumi(SingleMu_dEdxMPFProf, "*" , "I_{h} (MeV/cm)");
      graphPC->SetMarkerColor(2);    graphPC->SetMarkerStyle(26);
      graphPF->SetMarkerColor(4);    graphPF->SetMarkerStyle(32);
      SaveCanvas(c1,"pictures/","GraphdEdx_Profile_dEdxMP");
      delete c1;
   }


   for(unsigned int i=0;i<SingleMu_PtProf->GetXaxis()->GetNbins();i++){
      if((i+3)%4==0)continue;
      SingleMu_PtProf->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxProf->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxMProf->GetXaxis()->SetBinLabel(i,"");
      SingleMu_NVertProf->GetXaxis()->SetBinLabel(i,"");
   }  


   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = SingleMu_NVertProf;                 legend.push_back("SingleMu30");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "<#Reco Vertices>", 0,0, 0,0);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","dEdx_Profile_NVert");
   delete c1;

 

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = SingleMu_PtProf;                    legend.push_back("SingleMu30");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "p_{T} (GeV/c)", 0,0, 0,0);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","dEdx_Profile_Pt");
   delete c1;


   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = SingleMu_dEdxProf;                  legend.push_back("SingleMu30");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{as}", 0,0, 0.02,0.06);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","dEdx_Profile_dEdx");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = SingleMu_dEdxMProf;                  legend.push_back("SingleMu30");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h}", 0,0, 3.2,3.4);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","dEdx_Profile_dEdxM");
   delete c1;



/*
   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = SingleMu_dEdxMProfLumi166380;       legend.push_back("SingleMu30 - Run166380");
   Histos[1] = SingleMu_dEdxMProfLumi166512;       legend.push_back("SingleMu30 - Run166512");
   Histos[2] = SingleMu_dEdxMProfLumi167807;       legend.push_back("SingleMu30 - Run167807");
   Histos[3] = SingleMu_dEdxMProfLumi167898;       legend.push_back("SingleMu30 - Run167898");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Lumi", "I_{h}", 0,0, 3.2,3.4);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
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
   TProfile* SingleMu_PtProf           = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMuPtProf");
   TProfile* PFMet_PtProf              = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetPtProf");

   TProfile* Any_dEdxProf              = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxProf");
   TProfile* SingleMu_dEdxProf         = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxProf");
   TProfile* PFMet_dEdxProf            = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetdEdxProf");

   TProfile* Any_dEdxMProf             = (TProfile*)GetObjectFromPath(InputFile, "AnydEdxMProf");
   TProfile* SingleMu_dEdxMProf        = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMudEdxMProf");
   TProfile* PFMet_dEdxMProf           = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetdEdxMProf");

   TProfile* Any_TOFProf               = (TProfile*)GetObjectFromPath(InputFile, "AnyTOFProf");
   TProfile* SingleMu_TOFProf          = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMuTOFProf");
   TProfile* PFMet_TOFProf             = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetTOFProf");

   TProfile* Any_TOFDTProf             = (TProfile*)GetObjectFromPath(InputFile, "AnyTOFDTProf");
   TProfile* SingleMu_TOFDTProf        = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMuTOFDTProf");
   TProfile* PFMet_TOFDTProf           = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetTOFDTProf");

   TProfile* Any_TOFCSCProf            = (TProfile*)GetObjectFromPath(InputFile, "AnyTOFCSCProf");
   TProfile* SingleMu_TOFCSCProf       = (TProfile*)GetObjectFromPath(InputFile, "HscpPathSingleMuTOFCSCProf");
   TProfile* PFMet_TOFCSCProf          = (TProfile*)GetObjectFromPath(InputFile, "HscpPathPFMetTOFCSCProf");


   for(unsigned int i=0;i<SingleMu_PtProf->GetXaxis()->GetNbins();i++){
      if((i+3)%4==0)continue;
      Any_PtProf         ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_PtProf    ->GetXaxis()->SetBinLabel(i,"");
      PFMet_PtProf       ->GetXaxis()->SetBinLabel(i,"");

      Any_dEdxProf       ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxProf  ->GetXaxis()->SetBinLabel(i,"");
      PFMet_dEdxProf     ->GetXaxis()->SetBinLabel(i,"");

      Any_dEdxMProf      ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_dEdxMProf ->GetXaxis()->SetBinLabel(i,"");
      PFMet_dEdxMProf    ->GetXaxis()->SetBinLabel(i,"");

      Any_TOFProf        ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_TOFProf   ->GetXaxis()->SetBinLabel(i,"");
      PFMet_TOFProf      ->GetXaxis()->SetBinLabel(i,"");

      Any_TOFDTProf      ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_TOFDTProf ->GetXaxis()->SetBinLabel(i,"");
      PFMet_TOFDTProf    ->GetXaxis()->SetBinLabel(i,"");

      Any_TOFCSCProf     ->GetXaxis()->SetBinLabel(i,"");
      SingleMu_TOFCSCProf->GetXaxis()->SetBinLabel(i,"");
      PFMet_TOFCSCProf   ->GetXaxis()->SetBinLabel(i,"");
   }  



   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_PtProf;                         legend.push_back("Any");
   Histos[1] = SingleMu_PtProf;                    legend.push_back("SingleMu30");
   Histos[2] = PFMet_PtProf;                       legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "p_{T} (GeV/c)", 0,0, 0,150);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_Pt");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxProf;                       legend.push_back("Any");
   Histos[1] = SingleMu_dEdxProf;                  legend.push_back("SingleMu30");
   Histos[2] = PFMet_dEdxProf;                     legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{as}", 0,0, 0.02,0.05);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdx");
   delete c1;


   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_dEdxMProf;                      legend.push_back("Any");
   Histos[1] = SingleMu_dEdxMProf;                 legend.push_back("SingleMu30");
   Histos[2] = PFMet_dEdxMProf;                    legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "I_{h}", 0,0, 3.2,3.45);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_dEdxM");
   delete c1;



   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_TOFProf;                        legend.push_back("Any");
   Histos[1] = SingleMu_TOFProf;                   legend.push_back("SingleMu30");
   Histos[2] = PFMet_TOFProf;                      legend.push_back("PFMHT150");
   
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "1/#beta_{TOF}", 0,0, 1,1.1);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_TOF");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_TOFDTProf;                        legend.push_back("Any");
   Histos[1] = SingleMu_TOFDTProf;                   legend.push_back("SingleMu30");
   Histos[2] = PFMet_TOFDTProf;                      legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "1/#beta_{TOF_DT}", 0,0, 1,1.1);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_TOFDT");
   delete c1;

   c1 = new TCanvas("c1","c1,",1200,600);          legend.clear();
   Histos[0] = Any_TOFCSCProf;                        legend.push_back("Any");
   Histos[1] = SingleMu_TOFCSCProf;                   legend.push_back("SingleMu30");
   Histos[2] = PFMet_TOFCSCProf;                      legend.push_back("PFMHT150");

   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "", "1/#beta_{TOF_CSC}", 0,0, 1,1.1);
   for(unsigned int i=0;i<legend.size();i++){((TProfile*)Histos[i])->SetMarkerSize(0.5);           ((TProfile*)Histos[i])->GetYaxis()->SetTitleOffset(0.9);}
   DrawLegend(Histos,legend,"","P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"pictures/","Summary_Profile_TOFCSC");
   delete c1;


   MakedEdxPlot();
}


