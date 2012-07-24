// Original Author:  Loic Quertenmont

#define ANALYSIS2011 //TEMPORARY THING TO HAVE THE 2011 X SECTION


#include "Analysis_Global.h"
#include "Analysis_CommonFunction.h"
#include "Analysis_PlotFunction.h"
#include "Analysis_Samples.h"
#include "tdrstyle.C"
#include "roostats_cl95.C"
#include "nSigma.C"

using namespace std;


struct stAllInfo{
   double Mass;
   double MassMean;
   double MassSigma;
   double MassCut;
   double XSec_Th;
   double XSec_Err;
   double XSec_Exp;
   double XSec_ExpUp;
   double XSec_ExpDown;
   double XSec_Exp2Up;
   double XSec_Exp2Down;
   double XSec_Obs;
   double Eff;
   double Eff_SYSTP;
   double Eff_SYSTI;
   double Eff_SYSTM;
   double Eff_SYSTT;
   double Eff_SYSTPU;
   double Significance;
   double Index;
   double WP_Pt;
   double WP_I;
   double WP_TOF;
   float  NData;
   float  NPred;
   float  NPredErr;
   float  NSign;

   stAllInfo(string path=""){
     Mass=-1; XSec_Th=-1; XSec_Err=-1; XSec_Exp=-1; XSec_ExpUp=-1;XSec_ExpDown=-1;XSec_Exp2Up=-1;XSec_Exp2Down=-1; XSec_Obs=-1; Eff=-1; Eff_SYSTP=-1; Eff_SYSTI=-1;  Eff_SYSTM=-1; Eff_SYSTT=-1; Eff_SYSTPU=-1;
      if(path=="")return;
      FILE* pFile = fopen(path.c_str(),"r");
      if(!pFile){printf("Can't open %s\n",path.c_str()); return;}
      fscanf(pFile,"Mass         : %lf\n",&Mass);
      fscanf(pFile,"MassMean     : %lf\n",&MassMean);
      fscanf(pFile,"MassSigma    : %lf\n",&MassSigma);
      fscanf(pFile,"MassCut      : %lf\n",&MassCut);
      fscanf(pFile,"Index        : %lf\n",&Index);
      fscanf(pFile,"WP_Pt        : %lf\n",&WP_Pt);
      fscanf(pFile,"WP_I         : %lf\n",&WP_I);
      fscanf(pFile,"WP_TOF       : %lf\n",&WP_TOF);
      fscanf(pFile,"Eff          : %lf\n",&Eff);
      fscanf(pFile,"Eff_SystP    : %lf\n",&Eff_SYSTP);
      fscanf(pFile,"Eff_SystI    : %lf\n",&Eff_SYSTI);
      fscanf(pFile,"Eff_SystM    : %lf\n",&Eff_SYSTM);
      fscanf(pFile,"Eff_SystT    : %lf\n",&Eff_SYSTT);
      fscanf(pFile,"Eff_SystPU   : %lf\n",&Eff_SYSTPU);
      fscanf(pFile,"Signif       : %lf\n",&Significance);
      fscanf(pFile,"XSec_Th      : %lf\n",&XSec_Th);
      fscanf(pFile,"XSec_Exp     : %lf\n",&XSec_Exp);
      fscanf(pFile,"XSec_ExpUp   : %lf\n",&XSec_ExpUp);
      fscanf(pFile,"XSec_ExpDown : %lf\n",&XSec_ExpDown);
      fscanf(pFile,"XSec_Exp2Up  : %lf\n",&XSec_Exp2Up);
      fscanf(pFile,"XSec_Exp2Down: %lf\n",&XSec_Exp2Down);
      fscanf(pFile,"XSec_Obs     : %lf\n",&XSec_Obs);
      fscanf(pFile,"NData        : %E\n" ,&NData);
      fscanf(pFile,"NPred        : %E\n" ,&NPred);
      fscanf(pFile,"NPredErr     : %E\n" ,&NPredErr);
      fscanf(pFile,"NSign        : %E\n" ,&NSign);
      fclose(pFile);
   }
};

double PlotMinScale = 0.0005;
double PlotMaxScale = 3;

stAllInfo Exclusion(string pattern, string modelName, string signal);
double GetSignalMeanHSCPPerEvent(string InputPattern, unsigned int CutIndex, double MinRange, double MaxRange);


TGraph* MakePlot(FILE* pFile, FILE* talkFile, string InputPattern, string ModelName, int XSectionType, std::vector<stSample>& modelSamples);
void CheckSignalUncertainty(FILE* pFile, FILE* talkFile, string InputPattern);
void DrawModelLimitWithBand(string InputPattern);

void DrawRatioBands(string InputPattern);

double MinRange = 0;
double MaxRange = 1999;

char Buffer[2048];

int    CurrentSampleIndex;
string InputPath;
string OutputPath;

TH1D* MassSign      = NULL;
TH1D* MassMCTr      = NULL;
TH1D* MassData      = NULL;
TH1D* MassPred      = NULL;
TH1D* MassSignPDF   = NULL;
TH1D* MassPredPDF   = NULL;
double FitParam[10];
TF1* Stau_MMC_Fit   = NULL;
TF1* Stop_MMC_Fit   = NULL;
TF1* MGStop_MMC_Fit = NULL;
TF1* Gluino_MMC_Fit = NULL;
TF1* Stau_SMC_Fit   = NULL;
TF1* Stop_SMC_Fit   = NULL;
TF1* MGStop_SMC_Fit = NULL;
TF1* Gluino_SMC_Fit = NULL;

std::vector<stSample> samples;
std::vector<std::string> modelVector;
std::map<std::string, std::vector<stSample> > modelMap;

double RescaleFactor;
double RescaleError;
int Mode=0;

void Analysis_Step6(string MODE="COMPILE", string InputPattern="", string modelName="", string signal=""){
   setTDRStyle();
   gStyle->SetPadTopMargin   (0.06);
   gStyle->SetPadBottomMargin(0.10);
   gStyle->SetPadRightMargin (0.18);
   gStyle->SetPadLeftMargin  (0.12);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505,"X");
   gStyle->SetNdivisions(550,"Y");

   if(MODE=="COMPILE")return;

   if(MODE=="ANALYSE"){
      Exclusion(InputPattern, modelName, signal);
      return;
   }
   
   string MuPattern  = "Results/dedxASmi/combined/Eta15/PtMin45/Type2/";
   string TkPattern  = "Results/dedxASmi/combined/Eta15/PtMin45/Type0/";

   string outpath = string("Results/EXCLUSION/");
   MakeDirectories(outpath);


   //determine the list of models that are considered
   GetSampleDefinition(samples);
   for(unsigned int s=0;s<samples.size();s++){
    if(samples[s].Type!=2)continue;
    modelMap[samples[s].ModelName()].push_back(samples[s]);   
    if(modelMap[samples[s].ModelName()].size()==1)modelVector.push_back(samples[s].ModelName());
   }

   //based on the modelMap
   DrawRatioBands(TkPattern); 
   DrawRatioBands(MuPattern);

   //draw the cross section limit for all model
   DrawModelLimitWithBand(TkPattern);
   DrawModelLimitWithBand(MuPattern);


   //make plots of the observed limit for all signal model (and mass point) and save the result in a latex table
   TCanvas* c1;
   FILE* pFile    = fopen((string("Analysis_Step6_Result") + ".txt").c_str(),"w");
   FILE* talkFile = fopen((outpath + "TalkPlots" + ".txt").c_str(),"w");

   fprintf(pFile   , "\\documentclass{article}\n");
   fprintf(pFile   , "\\begin{document}\n\n");
   fprintf(pFile   , "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");

   fprintf(talkFile, "\\documentclass{article}\n");
   fprintf(talkFile, "\\usepackage{rotating}\n");
   fprintf(talkFile, "\\begin{document}\n\n");
   fprintf(talkFile, "\\begin{tiny}\n\n");
   fprintf(talkFile, "\\begin{sidewaystable}\n   \\centering\n      \\begin{tabular}{|l|cccccccc|}\n      \\hline\n");
   fprintf(talkFile,"Sample & Mass(GeV) & Pt(GeV) & $I_{as}$ & TOF & Mass Cut (GeV) & N pred & N observed & Eff & Signif \\\\\n");
   fprintf(talkFile, "\\hline\n");

   TGraph** TkGraphs  = new TGraph*[modelVector.size()];
   for(unsigned int k=0; k<modelVector.size(); k++){
      TkGraphs[k] = MakePlot(pFile,talkFile,TkPattern,modelVector[k], 2, modelMap[modelVector[k]]);
   }
   fprintf(pFile   ,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(pFile   , "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");

   fprintf(talkFile,"      \\end{tabular}\n\\end{sidewaystable}\n\n");
   fprintf(talkFile, "\\begin{sidewaystable}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
   fprintf(talkFile,"Sample & Mass(GeV) & Pt(GeV) & $I_{as}$ & $#beta^{-1]$ & Mass Cut (GeV) & N pred & N observed & Eff \\\\\n");
   fprintf(talkFile, "\\hline\n");

   TGraph** MuGraphs = new TGraph*[modelVector.size()];
   for(unsigned int k=0; k<modelVector.size(); k++){
      bool isNeutral = false;if(modelVector[k].find("GluinoN")!=string::npos || modelVector[k].find("StopN")!=string::npos)isNeutral = true;
      if(isNeutral) continue;//skip charged suppressed models
      MuGraphs[k] = MakePlot(pFile,talkFile,MuPattern,modelVector[k], 2, modelMap[modelVector[k]]);
   }

   fprintf(pFile   ,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(pFile   ,"\\end{document}\n\n");

   fprintf(talkFile,"      \\end{tabular}\n\\end{sidewaystable}\n\n");
   fprintf(talkFile,"\\end{document}\n\n");


   //print a table with all uncertainty on signal efficiency
   CheckSignalUncertainty(pFile,talkFile,TkPattern);
   CheckSignalUncertainty(pFile,talkFile,MuPattern);

   //Get Theoretical xsection and error bands
   TGraph** ThXSec    = new TGraph*[modelVector.size()];
   TCutG ** ThXSecErr = new TCutG* [modelVector.size()];
   for(unsigned int k=0; k<modelVector.size(); k++){
     if(modelVector[k].find("Gluino")!=string::npos){
        ThXSec   [k] = new TGraph(sizeof(THXSEC7TeV_Gluino_Mass)/sizeof(double),THXSEC7TeV_Gluino_Mass,THXSEC7TeV_Gluino_Cen);
        ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr",sizeof(THXSEC7TeV_Gluino_Mass)/sizeof(double),THXSEC7TeV_Gluino_Mass,THXSEC7TeV_Gluino_Low,THXSEC7TeV_Gluino_High, PlotMinScale, PlotMaxScale);
      }else if(modelVector[k].find("Stop"  )!=string::npos){
         ThXSec   [k] = new TGraph(sizeof(THXSEC7TeV_Stop_Mass)/sizeof(double),THXSEC7TeV_Stop_Mass,THXSEC7TeV_Stop_Cen);
         ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr",sizeof(THXSEC7TeV_Stop_Mass)/sizeof(double),THXSEC7TeV_Stop_Mass,THXSEC7TeV_Stop_Low,THXSEC7TeV_Stop_High, PlotMinScale, PlotMaxScale);
      }else if(modelVector[k].find("GMStau"  )!=string::npos){
         ThXSec   [k] = MakePlot(NULL, NULL, TkPattern,modelVector[k], 0, modelMap[modelVector[k]]); 
         ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr", sizeof(THXSEC7TeV_GMStau_Mass)/sizeof(double),THXSEC7TeV_GMStau_Mass,THXSEC7TeV_GMStau_Low,THXSEC7TeV_GMStau_High, PlotMinScale, PlotMaxScale); 
      }else if(modelVector[k].find("PPStau"  )!=string::npos){
         ThXSec   [k] = MakePlot(NULL, NULL, TkPattern,modelVector[k], 0, modelMap[modelVector[k]]);   
         ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr", sizeof(THXSEC7TeV_PPStau_Mass)/sizeof(double),THXSEC7TeV_PPStau_Mass,THXSEC7TeV_PPStau_Low,THXSEC7TeV_PPStau_High, PlotMinScale, PlotMaxScale); 
      }else{
         ThXSec   [k] = MakePlot(NULL, NULL, TkPattern,modelVector[k], 0, modelMap[modelVector[k]]);
         double* XSecErrLow  = new double[ThXSec[k]->GetN()];
         double* XSecErrHigh = new double[ThXSec[k]->GetN()];
         //assume 15% error on xsection
         for(int i=0;i<ThXSec[k]->GetN();i++){ XSecErrLow[i] = ThXSec[k]->GetY()[i]*0.85; XSecErrHigh[i] = ThXSec[k]->GetY()[i]*1.15; }
         ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr", ThXSec[k]->GetN(),ThXSec[k]->GetX(),XSecErrLow,XSecErrHigh, PlotMinScale, PlotMaxScale); 
      }
   }

   //Print the excluded mass range
   fprintf(pFile,"\n\n\n-----------------------\n Mass range excluded   \n-------------------------\n");
   fprintf(pFile,"-----------------------\n0%% TK-ONLY       \n-------------------------\n");
   for(unsigned int k=0; k<modelVector.size(); k++){
      fprintf(pFile,"%20s --> Excluded mass below %8.3fGeV\n", modelVector[k].c_str(), FindIntersectionBetweenTwoGraphs(TkGraphs[k],  ThXSec[k], TkGraphs[k]->GetX()[0], TkGraphs[k]->GetX()[TkGraphs[k]->GetN()-1], 1, 0.00));
   }
   fprintf(pFile,"-----------------------\n0%% MU+TOF        \n-------------------------\n");
   for(unsigned int k=0; k<modelVector.size(); k++){
      bool isNeutral = false;if(modelVector[k].find("GluinoN")!=string::npos || modelVector[k].find("StopN")!=string::npos)isNeutral = true;
      if(isNeutral) continue;//skip charged suppressed models
      fprintf(pFile,"%20s --> Excluded mass below %8.3fGeV\n", modelVector[k].c_str(), FindIntersectionBetweenTwoGraphs(MuGraphs[k],  ThXSec[k], MuGraphs[k]->GetX()[0], MuGraphs[k]->GetX()[MuGraphs[k]->GetN()-1], 1, 0.00));
   }
   fclose(pFile);

   //Make the final plot with all curves in it
   // I don't like much this part because it is dependent of what is in Analysis_Samples.h in an hardcoded way   
   std::map<string, TGraph*> TkGraphMap;
   std::map<string, TGraph*> MuGraphMap;
   std::map<string, TGraph*> ThGraphMap;
   std::map<string, TCutG* > ThErrorMap;
   for(unsigned int k=0; k<modelVector.size(); k++){
      TkGraphMap[modelVector[k]] = TkGraphs [k];
      MuGraphMap[modelVector[k]] = MuGraphs [k];
      ThGraphMap[modelVector[k]] = ThXSec   [k];
      ThErrorMap[modelVector[k]] = ThXSecErr[k];
   }

   ThGraphMap["Gluino_f10"   ]->SetLineColor(4);  ThGraphMap["Gluino_f10"   ]->SetMarkerColor(4);   ThGraphMap["Gluino_f10"   ]->SetLineWidth(1);   ThGraphMap["Gluino_f10"   ]->SetLineStyle(1);  ThGraphMap["Gluino_f10"   ]->SetMarkerStyle(1);
   MuGraphMap["Gluino_f10"   ]->SetLineColor(4);  MuGraphMap["Gluino_f10"   ]->SetMarkerColor(4);   MuGraphMap["Gluino_f10"   ]->SetLineWidth(2);   MuGraphMap["Gluino_f10"   ]->SetLineStyle(1);  MuGraphMap["Gluino_f10"   ]->SetMarkerStyle(22);
   MuGraphMap["Gluino_f50"   ]->SetLineColor(4);  MuGraphMap["Gluino_f50"   ]->SetMarkerColor(4);   MuGraphMap["Gluino_f50"   ]->SetLineWidth(2);   MuGraphMap["Gluino_f50"   ]->SetLineStyle(1);  MuGraphMap["Gluino_f50"   ]->SetMarkerStyle(23);
   TkGraphMap["Gluino_f10"   ]->SetLineColor(4);  TkGraphMap["Gluino_f10"   ]->SetMarkerColor(4);   TkGraphMap["Gluino_f10"   ]->SetLineWidth(2);   TkGraphMap["Gluino_f10"   ]->SetLineStyle(1);  TkGraphMap["Gluino_f10"   ]->SetMarkerStyle(22);
   TkGraphMap["Gluino_f50"   ]->SetLineColor(4);  TkGraphMap["Gluino_f50"   ]->SetMarkerColor(4);   TkGraphMap["Gluino_f50"   ]->SetLineWidth(2);   TkGraphMap["Gluino_f50"   ]->SetLineStyle(1);  TkGraphMap["Gluino_f50"   ]->SetMarkerStyle(23);
   TkGraphMap["GluinoN_f10"  ]->SetLineColor(4);  TkGraphMap["GluinoN_f10"  ]->SetMarkerColor(4);   TkGraphMap["GluinoN_f10"  ]->SetLineWidth(2);   TkGraphMap["GluinoN_f10"  ]->SetLineStyle(1);  TkGraphMap["GluinoN_f10"  ]->SetMarkerStyle(26);
   ThGraphMap["Stop"         ]->SetLineColor(2);  ThGraphMap["Stop"         ]->SetMarkerColor(2);   ThGraphMap["Stop"         ]->SetLineWidth(1);   ThGraphMap["Stop"         ]->SetLineStyle(2);  ThGraphMap["Stop"         ]->SetMarkerStyle(1);
   MuGraphMap["Stop"         ]->SetLineColor(2);  MuGraphMap["Stop"         ]->SetMarkerColor(2);   MuGraphMap["Stop"         ]->SetLineWidth(2);   MuGraphMap["Stop"         ]->SetLineStyle(1);  MuGraphMap["Stop"         ]->SetMarkerStyle(21);
   TkGraphMap["Stop"         ]->SetLineColor(2);  TkGraphMap["Stop"         ]->SetMarkerColor(2);   TkGraphMap["Stop"         ]->SetLineWidth(2);   TkGraphMap["Stop"         ]->SetLineStyle(1);  TkGraphMap["Stop"         ]->SetMarkerStyle(21);
   TkGraphMap["StopN"        ]->SetLineColor(2);  TkGraphMap["StopN"        ]->SetMarkerColor(2);   TkGraphMap["StopN"        ]->SetLineWidth(2);   TkGraphMap["StopN"        ]->SetLineStyle(1);  TkGraphMap["StopN"        ]->SetMarkerStyle(25);
   ThGraphMap["GMStau"       ]->SetLineColor(1);  ThGraphMap["GMStau"       ]->SetMarkerColor(1);   ThGraphMap["GMStau"       ]->SetLineWidth(1);   ThGraphMap["GMStau"       ]->SetLineStyle(3);  ThGraphMap["GMStau"       ]->SetMarkerStyle(1);
   ThGraphMap["PPStau"       ]->SetLineColor(6);  ThGraphMap["PPStau"       ]->SetMarkerColor(6);   ThGraphMap["PPStau"       ]->SetLineWidth(1);   ThGraphMap["PPStau"       ]->SetLineStyle(4);  ThGraphMap["PPStau"       ]->SetMarkerStyle(1);
   ThGraphMap["DCRho08HyperK"]->SetLineColor(4);  ThGraphMap["DCRho08HyperK"]->SetMarkerColor(4);   ThGraphMap["DCRho08HyperK"]->SetLineWidth(1);   ThGraphMap["DCRho08HyperK"]->SetLineStyle(3);  ThGraphMap["DCRho08HyperK"]->SetMarkerStyle(1);
   ThGraphMap["DCRho12HyperK"]->SetLineColor(2);  ThGraphMap["DCRho12HyperK"]->SetMarkerColor(2);   ThGraphMap["DCRho12HyperK"]->SetLineWidth(1);   ThGraphMap["DCRho12HyperK"]->SetLineStyle(2);  ThGraphMap["DCRho12HyperK"]->SetMarkerStyle(1);
   ThGraphMap["DCRho16HyperK"]->SetLineColor(1);  ThGraphMap["DCRho16HyperK"]->SetMarkerColor(1);   ThGraphMap["DCRho16HyperK"]->SetLineWidth(1);   ThGraphMap["DCRho16HyperK"]->SetLineStyle(1);  ThGraphMap["DCRho16HyperK"]->SetMarkerStyle(1);
   MuGraphMap["GMStau"       ]->SetLineColor(1);  MuGraphMap["GMStau"       ]->SetMarkerColor(1);   MuGraphMap["GMStau"       ]->SetLineWidth(2);   MuGraphMap["GMStau"       ]->SetLineStyle(1);  MuGraphMap["GMStau"       ]->SetMarkerStyle(23);
   MuGraphMap["PPStau"       ]->SetLineColor(6);  MuGraphMap["PPStau"       ]->SetMarkerColor(6);   MuGraphMap["PPStau"       ]->SetLineWidth(2);   MuGraphMap["PPStau"       ]->SetLineStyle(1);  MuGraphMap["PPStau"       ]->SetMarkerStyle(23);
   MuGraphMap["DCRho08HyperK"]->SetLineColor(4);  MuGraphMap["DCRho08HyperK"]->SetMarkerColor(4);   MuGraphMap["DCRho08HyperK"]->SetLineWidth(2);   MuGraphMap["DCRho08HyperK"]->SetLineStyle(1);  MuGraphMap["DCRho08HyperK"]->SetMarkerStyle(22);
   MuGraphMap["DCRho12HyperK"]->SetLineColor(2);  MuGraphMap["DCRho12HyperK"]->SetMarkerColor(2);   MuGraphMap["DCRho12HyperK"]->SetLineWidth(2);   MuGraphMap["DCRho12HyperK"]->SetLineStyle(1);  MuGraphMap["DCRho12HyperK"]->SetMarkerStyle(23);
   MuGraphMap["DCRho16HyperK"]->SetLineColor(1);  MuGraphMap["DCRho16HyperK"]->SetMarkerColor(1);   MuGraphMap["DCRho16HyperK"]->SetLineWidth(2);   MuGraphMap["DCRho16HyperK"]->SetLineStyle(1);  MuGraphMap["DCRho16HyperK"]->SetMarkerStyle(26);
   TkGraphMap["GMStau"       ]->SetLineColor(1);  TkGraphMap["GMStau"       ]->SetMarkerColor(1);   TkGraphMap["GMStau"       ]->SetLineWidth(2);   TkGraphMap["GMStau"       ]->SetLineStyle(1);  TkGraphMap["GMStau"       ]->SetMarkerStyle(20);
   TkGraphMap["PPStau"       ]->SetLineColor(6);  TkGraphMap["PPStau"       ]->SetMarkerColor(6);   TkGraphMap["PPStau"       ]->SetLineWidth(2);   TkGraphMap["PPStau"       ]->SetLineStyle(1);  TkGraphMap["PPStau"       ]->SetMarkerStyle(20);
   TkGraphMap["DCRho08HyperK"]->SetLineColor(4);  TkGraphMap["DCRho08HyperK"]->SetMarkerColor(4);   TkGraphMap["DCRho08HyperK"]->SetLineWidth(2);   TkGraphMap["DCRho08HyperK"]->SetLineStyle(1);  TkGraphMap["DCRho08HyperK"]->SetMarkerStyle(22);
   TkGraphMap["DCRho12HyperK"]->SetLineColor(2);  TkGraphMap["DCRho12HyperK"]->SetMarkerColor(2);   TkGraphMap["DCRho12HyperK"]->SetLineWidth(2);   TkGraphMap["DCRho12HyperK"]->SetLineStyle(1);  TkGraphMap["DCRho12HyperK"]->SetMarkerStyle(23);
   TkGraphMap["DCRho16HyperK"]->SetLineColor(1);  TkGraphMap["DCRho16HyperK"]->SetMarkerColor(1);   TkGraphMap["DCRho16HyperK"]->SetLineWidth(2);   TkGraphMap["DCRho16HyperK"]->SetLineStyle(1);  TkGraphMap["DCRho16HyperK"]->SetMarkerStyle(26);


   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGMu = new TMultiGraph();
   MGMu->Add(ThGraphMap["Gluino_f10" ]      ,"L");
   MGMu->Add(ThGraphMap["Stop"       ]      ,"L");
   MGMu->Add(ThGraphMap["GMStau"     ]      ,"L");
   MGMu->Add(ThGraphMap["PPStau"     ]      ,"L");
   MGMu->Add(MuGraphMap["Gluino_f10" ]      ,"LP");
   MGMu->Add(MuGraphMap["Gluino_f50" ]      ,"LP");
   MGMu->Add(MuGraphMap["Stop"       ]      ,"LP");
   MGMu->Add(MuGraphMap["GMStau"     ]      ,"LP");
   MGMu->Add(MuGraphMap["PPStau"     ]      ,"LP");
   MGMu->Draw("A");
   ThErrorMap["Gluino_f10"]->Draw("f");
   ThErrorMap["Stop"      ]->Draw("f");
   ThErrorMap["GMStau"    ]->Draw("f");
   ThErrorMap["PPStau"    ]->Draw("f");
   MGMu->Draw("same");
   MGMu->SetTitle("");
   MGMu->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGMu->GetYaxis()->SetTitle("#sigma (pb)");
   MGMu->GetYaxis()->SetTitleOffset(1.70);
   MGMu->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   
   DrawPreliminary(IntegratedLuminosity);
   TLegend* LEGMu = new TLegend(0.45,0.65,0.65,0.90);   
   LEGMu->SetHeader("Tracker + TOF");
   LEGMu->SetFillColor(0); 
   LEGMu->SetFillStyle(0);
   LEGMu->SetBorderSize(0);
   LEGMu->AddEntry(MuGraphMap["Gluino_f50"] , "gluino; 50% #tilde{g}g"    ,"LP");
   LEGMu->AddEntry(MuGraphMap["Gluino_f10"] , "gluino; 10% #tilde{g}g"    ,"LP");
   LEGMu->AddEntry(MuGraphMap["Stop"      ] , "stop"                      ,"LP");
   LEGMu->AddEntry(MuGraphMap["PPStau"    ] , "Pair Prod. stau"           ,"LP");
   LEGMu->AddEntry(MuGraphMap["GMStau"    ] , "GMSB stau"                 ,"LP");

   TLegend* LEGTh = new TLegend(0.15,0.7,0.48,0.9);
   LEGTh->SetHeader("Theoretical Prediction");
   LEGTh->SetFillColor(0);
   LEGTh->SetFillStyle(0);
   LEGTh->SetBorderSize(0);
   TGraph* GlThLeg = (TGraph*) ThGraphMap["Gluino_f10"]->Clone("GluinoThLeg");
   GlThLeg->SetFillColor(ThErrorMap["Gluino_f10"]->GetFillColor());
   LEGTh->AddEntry(GlThLeg, "gluino (NLO+NLL)" ,"LF");
   TGraph* StThLeg = (TGraph*) ThGraphMap["Stop"      ]->Clone("StopThLeg");
   StThLeg->SetFillColor(ThErrorMap["Gluino_f10"]->GetFillColor());
   LEGTh->AddEntry(StThLeg   ,"stop   (NLO+NLL)" ,"LF");

   TGraph* PPStauThLeg = (TGraph*) ThGraphMap["PPStau"        ]->Clone("PPStauThLeg");
   PPStauThLeg->SetFillColor(ThErrorMap["Gluino_f10"]->GetFillColor());
   LEGTh->AddEntry(PPStauThLeg   ,"Pair Prod. stau   (NLO)" ,"LF");
   TGraph* StauThLeg = (TGraph*) ThGraphMap["GMStau"        ]->Clone("StauThLeg");
   StauThLeg->SetFillColor(ThErrorMap["Gluino_f10"]->GetFillColor());
   LEGTh->AddEntry(StauThLeg   ,"GMSB stau   (NLO)" ,"LF");

   LEGTh->Draw();
   LEGMu->Draw();
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("MuExclusionLog"));
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGTk = new TMultiGraph();
   MGTk->Add(ThGraphMap["Gluino_f10" ]     ,"L");
   MGTk->Add(ThGraphMap["Stop"       ]     ,"L");
   MGTk->Add(ThGraphMap["GMStau"     ]     ,"L");
   MGTk->Add(ThGraphMap["PPStau"     ]     ,"L");
   MGTk->Add(TkGraphMap["Gluino_f10" ]     ,"LP");
   MGTk->Add(TkGraphMap["Gluino_f50" ]     ,"LP");
   MGTk->Add(TkGraphMap["GluinoN_f10"]     ,"LP");
   MGTk->Add(TkGraphMap["Stop"       ]     ,"LP");
   MGTk->Add(TkGraphMap["StopN"      ]     ,"LP");
   MGTk->Add(TkGraphMap["GMStau"     ]     ,"LP");
   MGTk->Add(TkGraphMap["PPStau"     ]     ,"LP");
   MGTk->Draw("A");
   ThErrorMap["Gluino_f10"]->Draw("f");
   ThErrorMap["Stop"      ]->Draw("f");
   ThErrorMap["GMStau"    ]->Draw("f");
   ThErrorMap["PPStau"    ]->Draw("f");
   MGTk->Draw("same");
   MGTk->SetTitle("");
   MGTk->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGTk->GetYaxis()->SetTitle("#sigma (pb)");
   MGTk->GetYaxis()->SetTitleOffset(1.70);
   MGTk->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   
   DrawPreliminary(IntegratedLuminosity);
   
   TLegend* LEGTk = new TLegend(0.45,0.58,0.795,0.9);
   LEGTk->SetHeader("Tracker - Only");
   LEGTk->SetFillColor(0); 
   LEGTk->SetFillStyle(0);
   LEGTk->SetBorderSize(0);
   LEGTk->AddEntry(TkGraphMap["Gluino_f50" ], "gluino; 50% #tilde{g}g"            ,"LP");
   LEGTk->AddEntry(TkGraphMap["Gluino_f10" ], "gluino; 10% #tilde{g}g"            ,"LP");
   LEGTk->AddEntry(TkGraphMap["GluinoN_f10"], "gluino; 10% #tilde{g}g; ch. suppr.","LP");
   LEGTk->AddEntry(TkGraphMap["Stop"       ], "stop"                              ,"LP");
   LEGTk->AddEntry(TkGraphMap["StopN"      ], "stop; ch. suppr."                  ,"LP");
   LEGTk->AddEntry(TkGraphMap["PPStau"     ], "Pair Prod. stau"                   ,"LP");
   LEGTk->AddEntry(TkGraphMap["GMStau"     ], "GMSB stau"                         ,"LP");
   LEGTh->Draw();
   LEGTk->Draw();
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("TkExclusionLog"));
   delete c1;

    c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGDCMu = new TMultiGraph();
   MGDCMu->Add(ThGraphMap["DCRho08HyperK"]      ,"L");
   MGDCMu->Add(MuGraphMap["DCRho08HyperK"]      ,"LP");
   MGDCMu->Add(ThGraphMap["DCRho12HyperK"]      ,"L");
   MGDCMu->Add(MuGraphMap["DCRho12HyperK"]      ,"LP");
   MGDCMu->Add(ThGraphMap["DCRho16HyperK"]      ,"L");
   MGDCMu->Add(MuGraphMap["DCRho16HyperK"]      ,"LP");
   MGDCMu->Draw("A");
   MGDCMu->Draw("same");
   MGDCMu->SetTitle("");
   MGDCMu->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGDCMu->GetYaxis()->SetTitle("#sigma (pb)");
   MGDCMu->GetYaxis()->SetTitleOffset(1.70);
   MGDCMu->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   DrawPreliminary(IntegratedLuminosity);
   
   TLegend* LEGDCMu = new TLegend(0.50,0.65,0.80,0.9);
   LEGDCMu->SetHeader("Tracker + TOF");
   LEGDCMu->SetFillColor(0); 
   LEGDCMu->SetFillStyle(0);
   LEGDCMu->SetBorderSize(0);
   LEGDCMu->AddEntry(MuGraphMap["DCRho08HyperK"]   , "Hyper-K, #tilde{#rho} = 0.8 TeV"       ,"LP");
   LEGDCMu->AddEntry(MuGraphMap["DCRho12HyperK"]   , "Hyper-K, #tilde{#rho} = 1.2 TeV"       ,"LP");
   LEGDCMu->AddEntry(MuGraphMap["DCRho16HyperK"]   , "Hyper-K, #tilde{#rho} = 1.6 TeV"       ,"LP");

   TLegend* LEGDCTh = new TLegend(0.15,0.7,0.49,0.9);
   LEGDCTh->SetHeader("Theoretical Prediction");
   LEGDCTh->SetFillColor(0);
   LEGDCTh->SetFillStyle(0);
   LEGDCTh->SetBorderSize(0);
   TGraph* DCRho08HyperKThLeg = (TGraph*) ThGraphMap["DCRho08HyperK"]->Clone("DCRho08HyperKThLeg");
   DCRho08HyperKThLeg->SetFillColor(ThErrorMap["Gluino_f10"]->GetFillColor());
   LEGDCTh->AddEntry(DCRho08HyperKThLeg   ,"Hyper-K, #tilde{#rho} = 0.8 TeV   (LO)" ,"L");
   TGraph* DCRho12HyperKThLeg = (TGraph*) ThGraphMap["DCRho12HyperK"]->Clone("DCRho12HyperKThLeg");
   DCRho12HyperKThLeg->SetFillColor(ThErrorMap["Gluino_f10"]->GetFillColor());
   LEGDCTh->AddEntry(DCRho12HyperKThLeg   ,"Hyper-K, #tilde{#rho} = 1.2 TeV   (LO)" ,"L");
   TGraph* DCRho16HyperKThLeg = (TGraph*) ThGraphMap["DCRho16HyperK"]->Clone("DCRho16HyperKThLeg");
   DCRho16HyperKThLeg->SetFillColor(ThErrorMap["Gluino_f10"]->GetFillColor());
   LEGDCTh->AddEntry(DCRho16HyperKThLeg   ,"Hyper-K, #tilde{#rho} = 1.6 TeV   (LO)" ,"L");
   LEGDCTh->Draw();
   LEGDCMu->Draw();
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("MuDCExclusionLog"));
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGDCTk = new TMultiGraph();
   MGDCTk->Add(ThGraphMap["DCRho08HyperK"]      ,"L");
   MGDCTk->Add(TkGraphMap["DCRho08HyperK"]      ,"LP");
   MGDCTk->Add(ThGraphMap["DCRho12HyperK"]      ,"L");
   MGDCTk->Add(TkGraphMap["DCRho12HyperK"]      ,"LP");
   MGDCTk->Add(ThGraphMap["DCRho16HyperK"]      ,"L");
   MGDCTk->Add(TkGraphMap["DCRho16HyperK"]      ,"LP");
   MGDCTk->Draw("A");
   MGDCTk->Draw("same");
   MGDCTk->SetTitle("");
   MGDCTk->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGDCTk->GetYaxis()->SetTitle("#sigma (pb)");
   MGDCTk->GetYaxis()->SetTitleOffset(1.70);
   MGDCTk->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   DrawPreliminary(IntegratedLuminosity);

   TLegend* LEGDCTk = new TLegend(0.50,0.65,0.80,0.90);
   LEGDCTk->SetHeader("Tracker - Only");
   LEGDCTk->SetFillColor(0); 
   LEGDCTk->SetFillStyle(0);
   LEGDCTk->SetBorderSize(0);
   LEGDCTk->AddEntry(TkGraphMap["DCRho08HyperK"]   , "Hyper-K, #tilde{#rho} = 0.8 TeV"       ,"LP");
   LEGDCTk->AddEntry(TkGraphMap["DCRho12HyperK"]   , "Hyper-K, #tilde{#rho} = 1.2 TeV"       ,"LP");
   LEGDCTk->AddEntry(TkGraphMap["DCRho16HyperK"]   , "Hyper-K, #tilde{#rho} = 1.6 TeV"       ,"LP");
   LEGDCTk->Draw();
   LEGDCTh->Draw();

   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("TkDCExclusionLog"));
   delete c1;

   return; 
}


void CheckSignalUncertainty(FILE* pFile, FILE* talkFile, string InputPattern){   
   bool IsTkOnly = (InputPattern.find("Type0",0)<std::string::npos);
   if(IsTkOnly){
      fprintf(pFile   ,          "%20s    Eff   --> PScale |  DeDxScale | PUScale | TotalUncertainty     \n","Model");
      fprintf(talkFile, "\\hline\n%20s &  Eff     & PScale &  DeDxScale & PUScale & TotalUncertainty \\\\\n","Model");
   }else {
      fprintf(pFile,             "%20s   Eff   --> PScale |  DeDxScale | PUScale | TOFScale | TotalUncertainty     \n","Model");
      fprintf(talkFile, "\\hline\n%20s &  Eff    & PScale &  DeDxScale & PUScale & TOFScale & TotalUncertainty \\\\\n","Model");
   }

   for(unsigned int s=0;s<samples.size();s++){
      if(samples[s].Type!=2)continue;
      bool IsNeutral = (samples[s].ModelName().find("N")!=std::string::npos);
      if(!IsTkOnly && IsNeutral)continue;
      stAllInfo tmp(InputPattern+"/EXCLUSION" + "/"+samples[s].Name+".txt");

      double P       = tmp.Eff - tmp.Eff_SYSTP;
      double I       = tmp.Eff - tmp.Eff_SYSTI;
      double PU      = tmp.Eff - tmp.Eff_SYSTPU;
      double T       = tmp.Eff - tmp.Eff_SYSTT;
      double Ptemp=max(P, 0.0), Itemp=max(I, 0.0), PUtemp=max(PU, 0.0), Ttemp=max(T, 0.0);
      if  (IsTkOnly)fprintf(pFile, "%20s   %7.3f --> %7.3f  |  %7.3f  | %7.3f  | %7.3f\n"        ,samples[s].Name.c_str(), tmp.Eff, P/tmp.Eff, I/tmp.Eff, PU/tmp.Eff           , sqrt(Ptemp*Ptemp + Itemp*Itemp + PUtemp*PUtemp + Ttemp*Ttemp)/tmp.Eff);        
      else          fprintf(pFile, "%20s   %7.3f --> %7.3f  |  %7.3f  | %7.3f  | %7.3f | %7.3f\n",samples[s].Name.c_str(), tmp.Eff, P/tmp.Eff, I/tmp.Eff, PU/tmp.Eff, T/tmp.Eff, sqrt(Ptemp*Ptemp + Itemp*Itemp + PUtemp*PUtemp + Ttemp*Ttemp)/tmp.Eff);

      if(IsTkOnly)fprintf(talkFile, "\\hline\n%20s &  %7.1f\\%% & %7.1f\\%%  &  %7.1f\\%%  & %7.1f\\%%  & %7.1f\\%%             \\\\\n",samples[s].Name.c_str(), 100.*tmp.Eff, 100.*P/tmp.Eff, 100.*I/tmp.Eff, 100.*PU/tmp.Eff, 100.*sqrt(Ptemp*Ptemp + Itemp*Itemp + PUtemp*PUtemp + Ttemp*Ttemp)/tmp.Eff);	
      else        fprintf(talkFile, "\\hline\n%20s &  %7.1f\\%% & %7.1f\\%%  &  %7.1f\\%%  & %7.1f\\%%  & %7.1f\\%% & %7.1f\\%% \\\\\n",samples[s].Name.c_str(), 100.*tmp.Eff, 100.*P/tmp.Eff, 100.*I/tmp.Eff, 100.*PU/tmp.Eff, 100.*T/tmp.Eff, 100.*sqrt(Ptemp*Ptemp + Itemp*Itemp + PUtemp*PUtemp + Ttemp*Ttemp)/tmp.Eff);
   }
}


TGraph* MakePlot(FILE* pFile, FILE* talkFile, string InputPattern, string ModelName, int XSectionType, std::vector<stSample>& modelSamples){
   unsigned int N   = modelSamples.size();
   double* Mass     = new double   [modelSamples.size()];
   double* XSecTh   = new double   [modelSamples.size()];
   double* XSecObs  = new double   [modelSamples.size()];
   double* XSecExp  = new double   [modelSamples.size()];
   stAllInfo* Infos = new stAllInfo[modelSamples.size()];
   for(unsigned int i=0;i<modelSamples.size();i++){
      Infos       [i]=stAllInfo(InputPattern+"EXCLUSION/" + modelSamples[i].Name +".txt");
      Mass        [i]=Infos[i].Mass;
      XSecTh      [i]=Infos[i].XSec_Th;
      XSecObs     [i]=Infos[i].XSec_Obs;
      XSecExp     [i]=Infos[i].XSec_Exp;
   }
   
   if(XSectionType>0){
      //for(unsigned int i=0;i<N;i++)printf("%-18s %5.0f --> Pt>%+6.1f & I>%+5.3f & TOF>%+4.3f & M>%3.0f--> NData=%2.0f  NPred=%6.1E+-%6.1E  NSign=%6.1E (Eff=%3.2f) Local Significance %3.2f\n",ModelName.c_str(),Infos[i].Mass,Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].WP_TOF,Infos[i].MassCut, Infos[i].NData, Infos[i].NPred, Infos[i].NPredErr, Infos[i].NSign, Infos[i].Eff, Infos[i].Significance);

      for(unsigned int i=0;i<N;i++){
        if(Infos[i].WP_TOF==-1){fprintf(pFile,"%-20s & %4.0f & %6.0f & %5.3f & / & %4.0f & %6.3f $\\pm$ %6.3f & %2.0f & %4.3f & %6.1E & %6.1E & %6.1E & %3.2f \\\\\n", ModelName.c_str(), Infos[i].Mass,  Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].MassCut, Infos[i].NPred, Infos[i].NPredErr, Infos[i].NData, Infos[i].Eff, Infos[i].XSec_Th,Infos[i].XSec_Exp, Infos[i].XSec_Obs, Infos[i].Significance);
        }else{                  fprintf(pFile,"%-20s & %4.0f & %6.0f & %5.3f & %4.3f & %4.0f & %6.3f $\\pm$ %6.3f & %2.0f & %4.3f & %6.1E & %6.1E & %6.1E & %3.2f \\\\\n", ModelName.c_str(), Infos[i].Mass,  Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].WP_TOF,Infos[i].MassCut, Infos[i].NPred, Infos[i].NPredErr, Infos[i].NData, Infos[i].Eff, Infos[i].XSec_Th,Infos[i].XSec_Exp, Infos[i].XSec_Obs, Infos[i].Significance);
        }
        bool IsNeutral = (ModelName.find("N",0)<std::string::npos);
        if(Infos[i].WP_TOF==-1 && (ModelName=="GMSB Stau" || (int)Infos[i].Mass%200==0)) {
          fprintf(talkFile,"%-20s & %4.0f & %6.0f & %5.3f & / & %4.0f & %6.3f $\\pm$ %6.3f & %2.0f & %4.3f & %3.2f \\\\\n", ModelName.c_str(), Infos[i].Mass,  Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].MassCut, Infos[i].NPred, Infos[i].NPredErr, Infos[i].NData, Infos[i].Eff, Infos[i].Significance);
          fprintf(talkFile, "\\hline\n");
        }
        if(Infos[i].WP_TOF!=-1 && !IsNeutral) {
          fprintf(talkFile,"%-20s & %4.0f & %6.0f & %5.3f & %4.3f & %4.0f & %6.3f $\\pm$ %6.3f & %2.0f & %4.3f %3.2f \\\\\n", ModelName.c_str(), Infos[i].Mass,  Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].WP_TOF,Infos[i].MassCut, Infos[i].NPred, Infos[i].NPredErr, Infos[i].NData, Infos[i].Eff, Infos[i].Significance);
          fprintf(talkFile, "\\hline\n");
        }
      }
   }
   
   TGraph* graph = NULL;
   if(XSectionType==0)graph = new TGraph(N,Mass,XSecTh);
   if(XSectionType==1)graph = new TGraph(N,Mass,XSecExp);
   if(XSectionType==2)graph = new TGraph(N,Mass,XSecObs);
   graph->SetTitle("");
   graph->GetYaxis()->SetTitle("CrossSection ( pb )");
   graph->GetYaxis()->SetTitleOffset(1.70);
   return graph;
}


stAllInfo Exclusion(string pattern, string modelName, string signal){
   GetSampleDefinition(samples);
   CurrentSampleIndex        = JobIdToIndex(signal,samples); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return stAllInfo();  } 

   stAllInfo toReturn;
   toReturn.Mass          = samples[JobIdToIndex(signal,samples)].Mass;
   toReturn.MassMean      = 0;
   toReturn.MassSigma     = 0;
   toReturn.MassCut       = 0;
   toReturn.Index         = 0;
   toReturn.WP_Pt         = 0;
   toReturn.WP_I          = 0;
   toReturn.WP_TOF        = 0;
   toReturn.XSec_Th       = samples[JobIdToIndex(signal,samples)].XSec;
   toReturn.XSec_Err      = samples[JobIdToIndex(signal,samples)].XSec * 0.15;
   toReturn.XSec_Exp      = 1E50;
   toReturn.XSec_ExpUp    = 1E50;
   toReturn.XSec_ExpDown  = 1E50;
   toReturn.XSec_Exp2Up   = 1E50;
   toReturn.XSec_Exp2Down = 1E50;
   toReturn.XSec_Obs      = 1E50;
   toReturn.Eff           = 0;
   toReturn.Eff_SYSTP     = 0;
   toReturn.Eff_SYSTI     = 0;
   toReturn.Eff_SYSTM     = 0;
   toReturn.Eff_SYSTT     = 0;
   toReturn.NData         = 0;
   toReturn.NPred         = 0;
   toReturn.NPredErr      = 0;
   toReturn.NSign         = 0;

   double RescaleFactor = 1.0;
   double RescaleError  = 0.1;

   double MaxSOverB=-1; 
   int MaxSOverBIndex=-1;

   string outpath = pattern + "/EXCLUSION/";
   MakeDirectories(outpath);

   FILE* pFile = fopen((outpath+"/"+modelName+".info").c_str(),"w");
   if(!pFile)printf("Can't open file : %s\n",(outpath+"/"+modelName+".info").c_str());

   TFile* InputFile     = new TFile((pattern + "Histos.root").c_str());
   TH1D*  HCuts_Pt      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
   TH1D*  HCuts_I       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");
   TH1D*  HCuts_TOF     = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");
   TH1D*  H_A           = (TH1D*)GetObjectFromPath(InputFile, "Data11/H_A");
   TH1D*  H_B           = (TH1D*)GetObjectFromPath(InputFile, "Data11/H_B");
   TH1D*  H_C           = (TH1D*)GetObjectFromPath(InputFile, "Data11/H_C");
 //TH1D*  H_D           = (TH1D*)GetObjectFromPath(InputFile, "Data11/H_D");
   TH1D*  H_E           = (TH1D*)GetObjectFromPath(InputFile, "Data11/H_E");
   TH1D*  H_F           = (TH1D*)GetObjectFromPath(InputFile, "Data11/H_F");
   TH1D*  H_G           = (TH1D*)GetObjectFromPath(InputFile, "Data11/H_G");
 //TH1D*  H_H           = (TH1D*)GetObjectFromPath(InputFile, "Data11/H_H");
   TH1D*  H_P           = (TH1D*)GetObjectFromPath(InputFile, "Data11/H_P");
   TH2D*  MassData      = (TH2D*)GetObjectFromPath(InputFile, "Data11/Mass");
   TH2D*  MassPred      = (TH2D*)GetObjectFromPath(InputFile, "Data11/Pred_Mass");
   TH2D*  MassSign      = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass" );
   TH2D*  MassSignP     = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass_SystP");
   TH2D*  MassSignI     = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass_SystI");
   TH2D*  MassSignM     = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass_SystM");
   TH2D*  MassSignT     = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass_SystT");
   TH2D*  MassSignPU    = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass_SystPU" );
   TH1D* TotalE         = (TH1D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/TotalE" );
   TH1D* TotalEPU       = (TH1D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/TotalEPU" );
   double norm  =samples[CurrentSampleIndex].XSec*IntegratedLuminosity/TotalE  ->Integral();
   double normPU=samples[CurrentSampleIndex].XSec*IntegratedLuminosity/TotalEPU->Integral();
   fprintf(pFile,"NORM = %f\n",norm);

   TH1D *MassSignProj, *MassSignPProj, *MassSignIProj, *MassSignMProj, *MassSignTProj, *MassSignPUProj;
   ///##############################################################################"
   MassSignProj = MassSign->ProjectionY("MassSignProj0",1,1);
   double Mean  = MassSignProj->GetMean();
   double Width = MassSignProj->GetRMS();
   MinRange = std::max(0.0, Mean-2*Width);
   MinRange = MassSignProj->GetXaxis()->GetBinLowEdge(MassSignProj->GetXaxis()->FindBin(MinRange)); //Round to a bin value to avoid counting prpoblem due to the binning. 
   delete MassSignProj;
   ///##############################################################################"

   //Going to first loop and find the cut with the min S over sqrt(B) because this is quick and normally gives a cut with a reach near the minimum
   stAllInfo CutInfo[MassData->GetNbinsX()];
   for(int CutIndex=0;CutIndex<MassData->GetNbinsX();CutIndex++) CutInfo[CutIndex]=toReturn;

   for(int CutIndex=0;CutIndex<MassData->GetNbinsX();CutIndex++){
      if(HCuts_Pt ->GetBinContent(CutIndex+1) < 45 ) continue;  // Be sure the pT cut is high enough to get some statistic for both ABCD and mass shape
      //make sure we have a reliable prediction of the number of events      
      if(H_E->GetBinContent(CutIndex+1) >0 && (H_A->GetBinContent(CutIndex+1)<25 || H_F->GetBinContent(CutIndex+1)<25 || H_G->GetBinContent(CutIndex+1)<25))continue;  //Skip events where Prediction (AFG/EE) is not reliable
      if(H_E->GetBinContent(CutIndex+1)==0 && (H_C->GetBinContent(CutIndex+1)<25 || H_B->GetBinContent(CutIndex+1)<25))continue;  //Skip events where Prediction (CB/A) is not reliable

      //make sure we have a reliable prediction of the shape 
      double N_P = H_P->GetBinContent(CutIndex+1);
      if(H_E->GetBinContent(CutIndex+1) >0 && (H_A->GetBinContent(CutIndex+1)<0.25*N_P || H_F->GetBinContent(CutIndex+1)<0.25*N_P || H_G->GetBinContent(CutIndex+1)<0.25*N_P))continue;  //Skip events where Mass Prediction is not reliable
      if(H_E->GetBinContent(CutIndex+1)==0 && (H_C->GetBinContent(CutIndex+1)<0.25*N_P || H_B->GetBinContent(CutIndex+1)<0.25*N_P))continue;  //Skip events where Mass Prediction is not reliable

      double signalsMeanHSCPPerEvent = GetSignalMeanHSCPPerEvent(pattern,CutIndex, MinRange, MaxRange);
      TH1D* MassDataProj = MassData  ->ProjectionY("MassDataProj"  ,CutIndex+1,CutIndex+1);
      TH1D* MassPredProj = MassPred  ->ProjectionY("MassPredProj"  ,CutIndex+1,CutIndex+1);
      MassSignProj       = MassSign  ->ProjectionY("MassSignProj0" ,CutIndex+1,CutIndex+1); MassSignProj  ->Scale(norm);
      MassSignPProj      = MassSignP ->ProjectionY("MassSignProP0" ,CutIndex+1,CutIndex+1); MassSignPProj ->Scale(norm);
      MassSignIProj      = MassSignI ->ProjectionY("MassSignProI0" ,CutIndex+1,CutIndex+1); MassSignIProj ->Scale(norm);
      MassSignMProj      = MassSignM ->ProjectionY("MassSignProM0" ,CutIndex+1,CutIndex+1); MassSignMProj ->Scale(norm);
      MassSignTProj      = MassSignT ->ProjectionY("MassSignProT0" ,CutIndex+1,CutIndex+1); MassSignTProj ->Scale(norm);
      MassSignPUProj     = MassSignPU->ProjectionY("MassSignProPU0",CutIndex+1,CutIndex+1); MassSignPUProj->Scale(normPU);

      double NData       = MassDataProj->Integral(MassDataProj->GetXaxis()->FindBin(MinRange), MassDataProj->GetXaxis()->FindBin(MaxRange));
      double NPred       = MassPredProj->Integral(MassPredProj->GetXaxis()->FindBin(MinRange), MassPredProj->GetXaxis()->FindBin(MaxRange));
      double NPredErr    = pow(NPred*RescaleError,2);
      for(int i=MassPredProj->GetXaxis()->FindBin(MinRange); i<=MassPredProj->GetXaxis()->FindBin(MaxRange) ;i++){NPredErr+=pow(MassPredProj->GetBinError(i),2);}NPredErr=sqrt(NPredErr);

      if(isnan((float)NPred))continue;
      if(NPred<=0){continue;} //Is <=0 only when prediction failed or is not meaningful (i.e. WP=(0,0,0) )
//    if(NPred<1E-4){continue;} //This will never be the selection which gives the best expected limit (cutting too much on signal) --> Slowdown computation for nothing...
      if(NPred>1000){continue;}  //When NPred is too big, expected limits just take an infinite time! 

      double Eff       = 0;
      double EffP      = 0;
      double EffI      = 0;
      double EffM      = 0;
      double EffT      = 0;
      double EffPU     = 0;
      CurrentSampleIndex        = JobIdToIndex(signal,samples); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return toReturn;  } 

      double INTERN_ESign       = MassSignProj->Integral(MassSignProj            ->GetXaxis()->FindBin(MinRange), MassSignProj      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      ; 
      double INTERN_Eff         = INTERN_ESign       / (samples[CurrentSampleIndex].XSec*IntegratedLuminosity);
      Eff                       = INTERN_Eff;
      //fprintf(pFile  ,"%10s: INTERN_ESign=%6.2E   INTERN_Eff=%6.E   XSec=%6.2E   Lumi=%6.2E  || pred=%f\n",signal.c_str(),INTERN_ESign,INTERN_Eff,samples[CurrentSampleIndex].XSec, IntegratedLuminosity, MassPredProj->Integral());fflush(stdout);

      double INTERN_ESignP      = MassSignPProj->Integral(MassSignPProj            ->GetXaxis()->FindBin(MinRange), MassSignPProj      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      ;
      double INTERN_EffP        = INTERN_ESignP      / (samples[CurrentSampleIndex].XSec*IntegratedLuminosity);
      EffP                      = INTERN_EffP;

      double INTERN_ESignI      = MassSignIProj->Integral(MassSignIProj            ->GetXaxis()->FindBin(MinRange), MassSignIProj      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      ;
      double INTERN_EffI        = INTERN_ESignI      / (samples[CurrentSampleIndex].XSec*IntegratedLuminosity);
      EffI                      = INTERN_EffI;

      double INTERN_ESignM      = MassSignMProj->Integral(MassSignMProj            ->GetXaxis()->FindBin(MinRange), MassSignMProj      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      ;
      double INTERN_EffM        = INTERN_ESignM      / (samples[CurrentSampleIndex].XSec*IntegratedLuminosity);
      EffM                      = INTERN_EffM;

      double INTERN_ESignT      = MassSignTProj->Integral(MassSignTProj            ->GetXaxis()->FindBin(MinRange), MassSignTProj      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      ;
      double INTERN_EffT        = INTERN_ESignT      / (samples[CurrentSampleIndex].XSec*IntegratedLuminosity);
      EffT                      = INTERN_EffT;

      double INTERN_ESignPU      = MassSignPUProj->Integral(MassSignPUProj            ->GetXaxis()->FindBin(MinRange), MassSignPUProj      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      ;
      double INTERN_EffPU        = INTERN_ESignPU      / (samples[CurrentSampleIndex].XSec*IntegratedLuminosity);
      EffPU                      = INTERN_EffPU;

      if(Eff==0)continue;
      NPred*=RescaleFactor;

     //fprintf(pFile ,"CutIndex=%4i MeanHSCPPerEvents = %6.2f  NTracks = %6.3f\n",CutIndex,signalsMeanHSCPPerEvent, MassSignProj->Integral(), );

      fprintf(pFile  ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.3f TOF>%6.3f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E+-%6.3E SignalEff=%6.3f\n",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,NData,NPred, NPredErr,Eff);fflush(stdout);
      fprintf(stdout ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.3f TOF>%6.3f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E+-%6.3E SignalEff=%6.3f\n",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,NData,NPred, NPredErr,Eff);fflush(stdout);

      if(Eff/sqrt(max(0.1, NPred))>MaxSOverB) {MaxSOverB=Eff/sqrt(max(0.1, NPred)); MaxSOverBIndex=CutIndex;}

     toReturn.MassMean  = Mean;
     toReturn.MassSigma = Width;
     toReturn.MassCut   = MinRange;
     toReturn.Index     = CutIndex;
     toReturn.WP_Pt     = HCuts_Pt ->GetBinContent(CutIndex+1);
     toReturn.WP_I      = HCuts_I  ->GetBinContent(CutIndex+1);
     toReturn.WP_TOF    = HCuts_TOF->GetBinContent(CutIndex+1);
     toReturn.XSec_Th   = samples[JobIdToIndex(signal,samples)].XSec;
     toReturn.XSec_Err  = samples[JobIdToIndex(signal,samples)].XSec * 0.15;
     toReturn.Eff       = Eff;
     toReturn.Eff_SYSTP = EffP;
     toReturn.Eff_SYSTI = EffI;
     toReturn.Eff_SYSTM = EffM;
     toReturn.Eff_SYSTT = EffT;
     toReturn.Eff_SYSTPU= EffPU;
     toReturn.NData     = NData;
     toReturn.NPred     = NPred;
     toReturn.NPredErr  = NPredErr;
     toReturn.NSign     = Eff*(samples[CurrentSampleIndex].XSec*IntegratedLuminosity);

     CutInfo[CutIndex]=toReturn;
   }
   fclose(pFile);   

   //Find reach for point with best S Over sqrt(B) first.
   double NPredSB=CutInfo[MaxSOverBIndex].NPred;
   double NPredErrSB=CutInfo[MaxSOverBIndex].NPredErr;
   double EffSB=CutInfo[MaxSOverBIndex].Eff;

   double FiveSigma=1E50;
   for (int n_obs=5; n_obs<1000; n_obs++) {
     if(nSigma(NPredSB, n_obs, NPredErrSB/NPredSB)>=5) {
       FiveSigma=n_obs;
       break;
     }
   }

   double MinReach=(FiveSigma-NPredSB)/(EffSB*IntegratedLuminosity);
   toReturn=CutInfo[MaxSOverBIndex]; // In case this point does give the best reach avoids rounding errors

   for(int CutIndex=0;CutIndex<MassData->GetNbinsX();CutIndex++){
     double NPred=CutInfo[CutIndex].NPred;
     double NPredErr=CutInfo[CutIndex].NPredErr;
     double Eff=CutInfo[CutIndex].Eff;
     if(Eff==0) continue;  //Eliminate points where prediction could not be made
     double FiveSigma=1E50;
     for (int n_obs=5; n_obs<1000; n_obs++) {
       if(n_obs<(NPred+3*sqrt(NPred))) continue;    //5 sigma implies more than 5 times sqrt(B) excess so can cut these points, put it at 3 to be safe
       double thisReach=(n_obs-NPred)/(Eff*IntegratedLuminosity);
       if(thisReach>=MinReach) break;    // This selection point will not give the optimum reach so move on
       if(nSigma(NPred, n_obs, NPredErr/NPred)>=5) {
	 FiveSigma=n_obs;
	 break;
       }
     }

     double Reach=(FiveSigma-NPred)/(Eff*IntegratedLuminosity);
     if(Reach>MinReach) continue;
     MinReach=Reach;
     toReturn=CutInfo[CutIndex];
   }

   LimitResult CLMResults;
   double signalUncertainty=0.07;
   double NPred=toReturn.NPred;
   double NPredErr=toReturn.NPredErr;
   double Eff=toReturn.Eff;
   double NData=toReturn.NData;

   CLMResults =  roostats_limit(IntegratedLuminosity, IntegratedLuminosity*0.022, Eff, Eff*signalUncertainty,NPred, NPredErr, NData, false, 1, "cls", "", 12345);
   toReturn.XSec_Exp      = CLMResults.GetExpectedLimit();
   toReturn.XSec_ExpUp    = CLMResults.GetOneSigmaHighRange();
   toReturn.XSec_ExpDown  = CLMResults.GetOneSigmaLowRange();
   toReturn.XSec_Exp2Up   = CLMResults.GetTwoSigmaHighRange();
   toReturn.XSec_Exp2Down = CLMResults.GetTwoSigmaLowRange();
   toReturn.XSec_Obs      = CLMResults.GetObservedLimit();
   toReturn.Significance = nSigma(NPred, NData, NPredErr/NPred);

   FILE* pFile2 = fopen((outpath+"/"+modelName+".txt").c_str(),"w");
   if(!pFile2)printf("Can't open file : %s\n",(outpath+"/"+modelName+".txt").c_str());
   fprintf(pFile2,"Mass         : %f\n",samples[JobIdToIndex(signal,samples)].Mass);
   fprintf(pFile2,"MassMean     : %f\n",toReturn.MassMean);
   fprintf(pFile2,"MassSigma    : %f\n",toReturn.MassSigma);
   fprintf(pFile2,"MassCut      : %f\n",toReturn.MassCut);
   fprintf(pFile2,"Index        : %f\n",toReturn.Index);
   fprintf(pFile2,"WP_Pt        : %f\n",toReturn.WP_Pt);
   fprintf(pFile2,"WP_I         : %f\n",toReturn.WP_I);
   fprintf(pFile2,"WP_TOF       : %f\n",toReturn.WP_TOF);
   fprintf(pFile2,"Eff          : %f\n",toReturn.Eff);
   fprintf(pFile2,"Eff_SystP    : %f\n",toReturn.Eff_SYSTP);
   fprintf(pFile2,"Eff_SystI    : %f\n",toReturn.Eff_SYSTI);
   fprintf(pFile2,"Eff_SystM    : %f\n",toReturn.Eff_SYSTM);
   fprintf(pFile2,"Eff_SystT    : %f\n",toReturn.Eff_SYSTT);
   fprintf(pFile2,"Eff_SystPU   : %f\n",toReturn.Eff_SYSTPU);
   fprintf(pFile2,"Signif       : %f\n",toReturn.Significance);
   fprintf(pFile2,"XSec_Th      : %f\n",toReturn.XSec_Th);
   fprintf(pFile2,"XSec_Exp     : %f\n",toReturn.XSec_Exp);
   fprintf(pFile2,"XSec_ExpUp   : %f\n",toReturn.XSec_ExpUp);
   fprintf(pFile2,"XSec_ExpDown : %f\n",toReturn.XSec_ExpDown);
   fprintf(pFile2,"XSec_Exp2Up  : %f\n",toReturn.XSec_Exp2Up);
   fprintf(pFile2,"XSec_Exp2Down: %f\n",toReturn.XSec_Exp2Down);
   fprintf(pFile2,"XSec_Obs     : %f\n",toReturn.XSec_Obs);     
   fprintf(pFile2,"NData        : %+6.2E\n",toReturn.NData);
   fprintf(pFile2,"NPred        : %+6.2E\n",toReturn.NPred);
   fprintf(pFile2,"NPredErr     : %+6.2E\n",toReturn.NPredErr);
   fprintf(pFile2,"NSign        : %+6.2E\n",toReturn.NSign);
   fprintf(pFile2, "%f+-%f  %f+-%f %f+-%f %f\n",IntegratedLuminosity, IntegratedLuminosity*0.022, Eff, Eff*signalUncertainty,NPred, NPredErr, NData);
   fclose(pFile2); 
   return toReturn;
}

double GetSignalMeanHSCPPerEvent(string InputPattern, unsigned int CutIndex, double MinRange, double MaxRange){
   TFile* InputFile     = new TFile((InputPattern + "Histos.root").c_str());

   TH2D*  Mass                     = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name          + "/Mass");
   TH2D*  MaxEventMass             = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name          + "/MaxEventMass");
   TH1D*  NTracksPassingSelection  = Mass->ProjectionY("NTracksPassingSelection",CutIndex+1,CutIndex+1);
   TH1D*  NEventsPassingSelection  = MaxEventMass->ProjectionY("NEventsPassingSelection",CutIndex+1,CutIndex+1);

   double NTracks       = NTracksPassingSelection->Integral(NTracksPassingSelection->GetXaxis()->FindBin(MinRange), NTracksPassingSelection->GetXaxis()->FindBin(MaxRange));
   double NEvents       = NEventsPassingSelection->Integral(NEventsPassingSelection->GetXaxis()->FindBin(MinRange), NEventsPassingSelection->GetXaxis()->FindBin(MaxRange));
   double toReturn      = (float)std::max(1.0,NTracks/ NEvents);
   delete Mass;
   delete MaxEventMass;
   delete NTracksPassingSelection;
   delete NEventsPassingSelection;

   delete InputFile;
   return toReturn;
}



void DrawModelLimitWithBand(string InputPattern){
   bool IsTkOnly = (InputPattern.find("Type0",0)<std::string::npos);
   string prefix = "Mu";    if(IsTkOnly) prefix ="Tk";

   for(unsigned int k=0; k<modelVector.size(); k++){
      bool isNeutral = false;if(modelVector[k].find("GluinoN")!=string::npos || modelVector[k].find("StopN")!=string::npos)isNeutral = true;
      if(!IsTkOnly && isNeutral) continue;
      unsigned int N = modelMap[modelVector[k]].size();
      stAllInfo Infos;double Mass[N], XSecTh[N], XSecExp[N],XSecObs[N], XSecExpUp[N],XSecExpDown[N],XSecExp2Up[N],XSecExp2Down[N];
      for(unsigned int i=0;i<N;i++){
         Infos = stAllInfo(InputPattern+"EXCLUSION/" + modelMap[modelVector[k]][i].Name +".txt");
         Mass        [i]=Infos.Mass;
         XSecTh      [i]=Infos.XSec_Th;
         XSecObs     [i]=Infos.XSec_Obs;
         XSecExp     [i]=Infos.XSec_Exp;
         XSecExpUp   [i]=Infos.XSec_ExpUp;
         XSecExpDown [i]=Infos.XSec_ExpDown;
         XSecExp2Up  [i]=Infos.XSec_Exp2Up;
         XSecExp2Down[i]=Infos.XSec_Exp2Down;
      }

      TGraph* graphtheory  = new TGraph(N,Mass,XSecTh);
      TGraph* graphobs     = new TGraph(N,Mass,XSecObs);
      TGraph* graphexp     = new TGraph(N,Mass,XSecExp);
      TCutG*  ExpErr       = GetErrorBand("ExpErr"      ,N,Mass,XSecExpDown ,XSecExpUp , PlotMinScale, PlotMaxScale);
      TCutG*  Exp2SigmaErr = GetErrorBand("Exp2SigmaErr",N,Mass,XSecExp2Down,XSecExp2Up, PlotMinScale, PlotMaxScale);

      graphtheory->SetLineStyle(3);
      graphtheory->SetFillColor(kBlue);
      graphexp->SetLineStyle(4); 
      graphexp->SetLineColor(kRed);
      graphexp->SetMarkerStyle(); 
      graphexp->SetMarkerSize(0.); 
      Exp2SigmaErr->SetFillColor(kYellow);
      Exp2SigmaErr->SetLineColor(kWhite);
      ExpErr->SetFillColor(kGreen);
      ExpErr->SetLineColor(kWhite);
      graphobs->SetLineColor(kBlack);
      graphobs->SetLineWidth(2);
      graphobs->SetMarkerColor(kBlack);
      graphobs->SetMarkerStyle(23);

      TCanvas* c1 = new TCanvas("c1", "c1",600,600);
      TMultiGraph* MG = new TMultiGraph();
      MG->Add(graphexp      ,"LP");
      MG->Add(graphobs      ,"LP");
      MG->Add(graphtheory   ,"L");
      MG->Draw("A");
      Exp2SigmaErr->Draw("f");
      ExpErr      ->Draw("f");
      MG          ->Draw("same");
      MG->SetTitle("");
      MG->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
      MG->GetYaxis()->SetTitle("#sigma (pb)");
      MG->GetYaxis()->SetTitleOffset(1.70);
      MG->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
      DrawPreliminary(IntegratedLuminosity);
      
      TLegend* LEG = new TLegend(0.40,0.65,0.8,0.90);
      string headerstr;
      headerstr = "95% CL Limits (Tracker + TOF)";
      if(IsTkOnly) headerstr = "95% CL Limits (Tracker - Only)";
      LEG->SetHeader(headerstr.c_str());
      LEG->SetFillColor(0); 
      LEG->SetBorderSize(0);
      LEG->AddEntry(graphtheory,  modelMap[modelVector[k]][0].ModelLegend().c_str() ,"L");
      LEG->AddEntry(graphexp    , "Expected"             ,"L");
      LEG->AddEntry(ExpErr      , "Expected #pm 1#sigma" ,"F");
      LEG->AddEntry(Exp2SigmaErr, "Expected #pm 2#sigma ","F");
      LEG->AddEntry(graphobs    , "Observed"             ,"LP");
      LEG->Draw();
      c1->SetLogy(true);

      if(IsTkOnly)SaveCanvas(c1,"Results/EXCLUSION/", string("Tk"+ modelVector[k] + "ExclusionLog"));
      else        SaveCanvas(c1,"Results/EXCLUSION/", string("Mu"+ modelVector[k] + "ExclusionLog"));
      delete c1;
   }
}

// This code make the Expected Limit error band divided by expected limit plot for all signal models
// I don't like much this function... I started to rewrite it, but more work is still needed to improve it.
// I don't think two loops are needed, neither all these arrays...
void DrawRatioBands(string InputPattern)
{
   bool IsTkOnly = (InputPattern.find("Type0",0)<std::string::npos);
   string prefix = "Mu";    if(IsTkOnly) prefix ="Tk";

   TCanvas* c1            = new TCanvas("c1", "c1",600,800);
   TGraph** graphAtheory  = new TGraph*[modelVector.size()];
   TGraph** graphAobs     = new TGraph*[modelVector.size()];
   TGraph** graphAexp     = new TGraph*[modelVector.size()];
   TCutG**  ExpAErr       = new TCutG* [modelVector.size()];
   TCutG**  Exp2SigmaAErr = new TCutG* [modelVector.size()];
   TPad** padA            = new TPad*  [modelVector.size()];
   double step, top;

   top= 1.0/(modelVector.size()+2);
   step=(1.0-2.*top)/(modelVector.size());

   for(unsigned int k=0; k<modelVector.size(); k++){
      bool isNeutral = false;if(modelVector[k].find("GluinoN")!=string::npos || modelVector[k].find("StopN")!=string::npos)isNeutral = true;
      if(!IsTkOnly && isNeutral) continue;
      TPad* pad;
      if(k<(modelVector.size()-1)){
         pad = new TPad(Form("pad%i",k),Form("ExpErr%i",k),0.1,1-top-(k+1)*step,0.9,1-top-step*k);//lower left x, y, topright x, y
         pad->SetBottomMargin(0.);
      }else {
         pad = new TPad(Form("pad%i",k),Form("ExpErr%i",k),0.1,0.0,0.9,1-top-step*(k));//lower left x, y, topright x, y
         pad->SetBottomMargin(top/(step+top));
      }
      pad->SetLeftMargin(0.1);
      pad->SetRightMargin(0.);
      pad->SetTopMargin(0.);
      padA[k] = pad;  
      padA[k]->Draw();
   }

   for(unsigned int k=0; k<modelVector.size(); k++){
      bool isNeutral = false;if(modelVector[k].find("GluinoN")!=string::npos || modelVector[k].find("StopN")!=string::npos)isNeutral = true;
      if(!IsTkOnly && isNeutral) continue;

      TMultiGraph* MG = new TMultiGraph();
      unsigned int N = modelMap[modelVector[k]].size();
      stAllInfo Infos;double Mass[N], XSecTh[N], XSecExp[N],XSecObs[N], XSecExpUp[N],XSecExpDown[N],XSecExp2Up[N],XSecExp2Down[N];
      for(unsigned int i=0;i<N;i++){
         Infos = stAllInfo(InputPattern+"EXCLUSION/" + modelMap[modelVector[k]][i].Name +".txt");
         Mass        [i]=Infos.Mass;
         XSecTh      [i]=Infos.XSec_Th;
         XSecObs     [i]=Infos.XSec_Obs     /Infos.XSec_Exp;
         XSecExp     [i]=Infos.XSec_Exp     /Infos.XSec_Exp;
         XSecExpUp   [i]=Infos.XSec_ExpUp   /Infos.XSec_Exp;
         XSecExpDown [i]=Infos.XSec_ExpDown /Infos.XSec_Exp;
         XSecExp2Up  [i]=Infos.XSec_Exp2Up  /Infos.XSec_Exp;
         XSecExp2Down[i]=Infos.XSec_Exp2Down/Infos.XSec_Exp;
      }

      TGraph* graphtheory  = new TGraph(N,Mass,XSecTh);
      TGraph* graphobs     = new TGraph(N,Mass,XSecObs);
      TGraph* graphexp     = new TGraph(N,Mass,XSecExp);
      TCutG*  ExpErr       = GetErrorBand(Form("ExpErr%i",k)      ,N,Mass,XSecExpDown ,XSecExpUp,  0.0, 3.0);
      TCutG*  Exp2SigmaErr = GetErrorBand(Form("Exp2SigmaErr%i",k),N,Mass,XSecExp2Down,XSecExp2Up, 0.0, 3.0);

      graphAtheory [k] = graphtheory;      
      graphAobs    [k] = graphobs;
      graphAexp    [k] = graphexp;
      ExpAErr      [k] = ExpErr;

      Exp2SigmaAErr[k] = Exp2SigmaErr;
      graphAtheory [k]->SetLineStyle(3);
      graphAexp    [k]->SetLineStyle(4); 
      graphAexp    [k]->SetLineColor(kRed);
      graphAexp    [k]->SetMarkerStyle(); 
      graphAexp    [k]->SetMarkerSize(0.); 
      Exp2SigmaAErr[k]->SetFillColor(kYellow);
      Exp2SigmaAErr[k]->SetLineColor(kWhite);
      ExpAErr      [k]->SetFillColor(kGreen);
      ExpAErr      [k]->SetLineColor(kWhite);
      graphAobs    [k]->SetLineColor(kBlack);
      graphAobs    [k]->SetLineWidth(2);
      graphAobs    [k]->SetMarkerColor(kBlack);
      graphAobs    [k]->SetMarkerStyle(23);
      padA[k]->cd();

      int masst[2] = {0,1250};
      int xsect[2] = {2, 1};
      TGraph* graph = new TGraph(2,masst,xsect); //fake graph to set xaxis right
      graph->SetMarkerSize(0.);
      MG->Add(graph      ,"P");
      MG->Add(graphAobs[k]      ,"LP");
      MG->Draw("A");
      if(k==0){
	 TLegend* LEG;
	 LEG = new TLegend(0.13,0.01,0.32,0.99);
         string headerstr;
         headerstr = "Tracker + TOF";
         if(IsTkOnly) headerstr = "Tracker - Only";
         LEG->SetHeader(headerstr.c_str());
         LEG->SetFillColor(0); 
         LEG->SetBorderSize(0);
         LEG->AddEntry(ExpAErr[0], "Expected #pm 1#sigma","F");
         LEG->SetMargin(0.1);
         LEG->Draw();
      }  

      if(k==1){
         TLegend* LEG;
         LEG = new TLegend(0.13,0.01,0.32,0.99);
	 string headerstr;
	 LEG->SetFillColor(0);
	 LEG->SetBorderSize(0);
	 LEG->AddEntry(Exp2SigmaAErr[0], "Expected #pm 2#sigma","F");
	 LEG->AddEntry(graphAobs[0],"Observed" ,"LP");
	 LEG->SetMargin(0.1);
	 LEG->Draw();
      }

      Exp2SigmaAErr[k]->Draw("f");
      ExpAErr[k]  ->Draw("f");
      MG->Draw("same");
      MG->SetTitle("");
      if(k==modelVector.size()-1) {
         MG->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
         MG->GetXaxis()->SetTitleSize(0.2);
         MG->GetXaxis()->SetLabelSize(0.2);
      }

      TPaveText *pt;
      if(IsTkOnly) {
      if(k!=modelVector.size()-1) pt = new TPaveText(0.45, 0.6, 0.95, 0.87,"LBNDC");
      else pt = new TPaveText(0.45, 0.82, 0.95, 0.935,"LBNDC");
      }
      else {
	if(k!=modelVector.size()-1) pt = new TPaveText(0.55, 0.6, 0.95, 0.87,"LBNDC");
	else pt = new TPaveText(0.55, 0.82, 0.95, 0.935,"LBNDC");
      }

      pt->SetBorderSize(0);
      pt->SetLineWidth(0);
      pt->SetFillColor(kWhite);
      TText *text = pt->AddText(modelMap[modelVector[k]][0].ModelLegend().c_str()); 
      text ->SetTextAlign(12);
      text ->SetTextSize(0.3);
      if(k==modelVector.size()-1) text ->SetTextSize(0.5*text ->GetTextSize());
      pt->Draw();
      
      MG->GetXaxis()->SetRangeUser(0,1250);    
      MG->GetXaxis()->SetNdivisions(506,"Z");

      MG->GetYaxis()->SetRangeUser(0.001,2.99);
      MG->GetYaxis()->SetNdivisions(303, "Z");
      MG->GetYaxis()->SetLabelSize(0.3);
      if(k==modelVector.size()-1){
	MG->GetYaxis()->SetLabelSize(0.15);
      }
   }
   c1->cd();
   DrawPreliminary(IntegratedLuminosity);

   TPaveText *pt = new TPaveText(0.1, 0., 0.15, 0.7,"NDC");
   string tmp = "95% CL Limits (Relative to Expected Limit)";
   TText *text = pt->AddText(tmp.c_str()); 
   text ->SetTextAlign(12);
   text ->SetTextAngle(90);
   text ->SetTextSize(0.04);
   pt->SetBorderSize(0);
   pt->SetFillColor(0);
   pt->Draw();

   if(IsTkOnly) SaveCanvas(c1,"Results/EXCLUSION/", string("TkLimitsRatio"));
   else         SaveCanvas(c1,"Results/EXCLUSION/", string("MuLimitsRatio"));

   delete c1;
}
