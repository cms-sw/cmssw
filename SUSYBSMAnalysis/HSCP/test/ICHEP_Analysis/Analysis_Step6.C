// Original Author:  Loic Quertenmont

#include "Analysis_Global.h"
#include "Analysis_CommonFunction.h"
#include "Analysis_PlotFunction.h"
#include "Analysis_Samples.h"
#include "tdrstyle.C"

using namespace std;

class stAllInfo{
   public:
   double Mass, MassMean, MassSigma, MassCut;
   double XSec_Th, XSec_Err, XSec_Exp, XSec_ExpUp, XSec_ExpDown, XSec_Exp2Up, XSec_Exp2Down, XSec_Obs;
   double  Eff, Eff_SYSTP, Eff_SYSTI, Eff_SYSTM, Eff_SYSTT, Eff_SYSTPU;
   double Significance;
   double Index, WP_Pt, WP_I, WP_TOF;
   float  NData, NPred, NPredErr, NSign;
   double LInt;

   stAllInfo(string path=""){
      //Default Values
      Mass          = 0;      MassMean      = 0;      MassSigma     = 0;      MassCut       = 0;
      Index         = 0;      WP_Pt         = 0;      WP_I          = 0;      WP_TOF        = 0;
      XSec_Th       = 0;      XSec_Err      = 0;      XSec_Exp      = 1E50;   XSec_ExpUp    = 1E50;   XSec_ExpDown  = 1E50;    XSec_Exp2Up   = 1E50;    XSec_Exp2Down = 1E50;    XSec_Obs    = 1E50;
      Eff           = 0;      Eff_SYSTP     = 0;      Eff_SYSTI     = 0;      Eff_SYSTM     = 0;      Eff_SYSTT     = 0;
      NData         = 0;      NPred         = 0;      NPredErr      = 0;      NSign         = 0;      LInt          = 0;
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
      fscanf(pFile,"LInt         : %lf\n",&LInt);
      fclose(pFile);
   }

   void Save(string path=""){
      FILE* pFile = fopen(path.c_str(),"w");
      if(!pFile)printf("Can't open file : %s\n",path.c_str());
      fprintf(pFile,"Mass         : %f\n",Mass);
      fprintf(pFile,"MassMean     : %f\n",MassMean);
      fprintf(pFile,"MassSigma    : %f\n",MassSigma);
      fprintf(pFile,"MassCut      : %f\n",MassCut);
      fprintf(pFile,"Index        : %f\n",Index);
      fprintf(pFile,"WP_Pt        : %f\n",WP_Pt);
      fprintf(pFile,"WP_I         : %f\n",WP_I);
      fprintf(pFile,"WP_TOF       : %f\n",WP_TOF);
      fprintf(pFile,"Eff          : %f\n",Eff);
      fprintf(pFile,"Eff_SystP    : %f\n",Eff_SYSTP);
      fprintf(pFile,"Eff_SystI    : %f\n",Eff_SYSTI);
      fprintf(pFile,"Eff_SystM    : %f\n",Eff_SYSTM);
      fprintf(pFile,"Eff_SystT    : %f\n",Eff_SYSTT);
      fprintf(pFile,"Eff_SystPU   : %f\n",Eff_SYSTPU);
      fprintf(pFile,"Signif       : %f\n",Significance);
      fprintf(pFile,"XSec_Th      : %f\n",XSec_Th);
      fprintf(pFile,"XSec_Exp     : %f\n",XSec_Exp);
      fprintf(pFile,"XSec_ExpUp   : %f\n",XSec_ExpUp);
      fprintf(pFile,"XSec_ExpDown : %f\n",XSec_ExpDown);
      fprintf(pFile,"XSec_Exp2Up  : %f\n",XSec_Exp2Up);
      fprintf(pFile,"XSec_Exp2Down: %f\n",XSec_Exp2Down);
      fprintf(pFile,"XSec_Obs     : %f\n",XSec_Obs);     
      fprintf(pFile,"NData        : %+6.2E\n",NData);
      fprintf(pFile,"NPred        : %+6.2E\n",NPred);
      fprintf(pFile,"NPredErr     : %+6.2E\n",NPredErr);
      fprintf(pFile,"NSign        : %+6.2E\n",NSign);
      fprintf(pFile,"LInt         : %f\n",LInt);
      fclose(pFile); 
   }
};

//Background prediction rescale and uncertainty
double RescaleFactor = 1.0;
double RescaleError  = 0.1;

//final Plot y-axis range
double PlotMinScale = 0.0005;
double PlotMaxScale = 3;

void Optimize(string InputPattern, string Data, string signal, bool shape);
double GetSignalMeanHSCPPerEvent(string InputPattern, unsigned int CutIndex, double MinRange, double MaxRange);
TGraph* MakePlot(FILE* pFile, FILE* talkFile, string InputPattern, string ModelName, int XSectionType, std::vector<stSample>& modelSamples, double& LInt);
void CheckSignalUncertainty(FILE* pFile, FILE* talkFile, string InputPattern);
void DrawModelLimitWithBand(string InputPattern);
void DrawRatioBands(string InputPattern);


void makeDataCard(string outpath, string rootPath, string ChannelName, string SignalName, double Obs, double Pred, double PredRelErr, double Sign, bool Shape);
void saveHistoForLimit(TH1* histo, string Name, string Id);
void saveVariationHistoForLimit(TH1* histo, TH1* vardown, string Name, string variationName);
void testShapeBasedAnalysis(string InputPattern, string signal);
bool runCombine(bool Significance, string& InputPattern, string& signal, unsigned int CutIndex, bool Shape, bool Temporary, stAllInfo& result, TH2D* MassData, TH2D* MassPred, TH2D* MassSign, TH2D* MassSignP, TH2D* MassSignI, TH2D* MassSignM, TH2D* MassSignT, TH2D* MassSignPU);

double MinRange = 0;
double MaxRange = 1999;
int    CurrentSampleIndex;

std::vector<stSample> samples;
std::vector<std::string> modelVector;
std::map<std::string, std::vector<stSample> > modelMap;

string SHAPESTRING="";

void Analysis_Step6(string MODE="COMPILE", string InputPattern="", string signal=""){
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

   string Data = "Data11";
   if(Data=="Data11"){SQRTS=7.0;}else{SQRTS=8.0;}


   //determine the list of models that are considered
   GetSampleDefinition(samples);
   for(unsigned int s=0;s<samples.size();s++){
    if(samples[s].Type!=2)continue;
    modelMap[samples[s].ModelName()].push_back(samples[s]);   
    if(modelMap[samples[s].ModelName()].size()==1)modelVector.push_back(samples[s].ModelName());
   }

   if(MODE.find("SHAPE")!=string::npos){SHAPESTRING="SHAPE";}else{SHAPESTRING="";}
   if(MODE.find("ANALYSE")!=string::npos){
      if(SHAPESTRING==""){  Optimize(InputPattern, Data, signal, false);
      }else{                Optimize(InputPattern, Data, signal, false);//testShapeBasedAnalysis(InputPattern,signal);
      }
      return;
   }
   
   string MuPattern  = "Results/dedxASmi/combined/Eta15/PtMin45/Type2/";
   string TkPattern  = "Results/dedxASmi/combined/Eta15/PtMin45/Type0/";




   string outpath = string("Results/"+SHAPESTRING+"EXCLUSION/");
   MakeDirectories(outpath);


   //based on the modelMap
   DrawRatioBands(TkPattern); 
   DrawRatioBands(MuPattern);

   //draw the cross section limit for all model
   DrawModelLimitWithBand(TkPattern);
   DrawModelLimitWithBand(MuPattern);


   //make plots of the observed limit for all signal model (and mass point) and save the result in a latex table
   TCanvas* c1;
   double LInt;
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
      TkGraphs[k] = MakePlot(pFile,talkFile,TkPattern,modelVector[k], 2, modelMap[modelVector[k]], LInt);
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
      MuGraphs[k] = MakePlot(pFile,talkFile,MuPattern,modelVector[k], 2, modelMap[modelVector[k]], LInt);
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
         ThXSec   [k] = MakePlot(NULL, NULL, TkPattern,modelVector[k], 0, modelMap[modelVector[k]], LInt); 
         ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr", sizeof(THXSEC7TeV_GMStau_Mass)/sizeof(double),THXSEC7TeV_GMStau_Mass,THXSEC7TeV_GMStau_Low,THXSEC7TeV_GMStau_High, PlotMinScale, PlotMaxScale); 
      }else if(modelVector[k].find("PPStau"  )!=string::npos){
         ThXSec   [k] = MakePlot(NULL, NULL, TkPattern,modelVector[k], 0, modelMap[modelVector[k]], LInt);   
         ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr", sizeof(THXSEC7TeV_PPStau_Mass)/sizeof(double),THXSEC7TeV_PPStau_Mass,THXSEC7TeV_PPStau_Low,THXSEC7TeV_PPStau_High, PlotMinScale, PlotMaxScale); 
      }else{
         ThXSec   [k] = MakePlot(NULL, NULL, TkPattern,modelVector[k], 0, modelMap[modelVector[k]], LInt);
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
   
   DrawPreliminary(SQRTS, LInt);
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
   
   DrawPreliminary(SQRTS, LInt);
   
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
   DrawPreliminary(SQRTS, LInt);
   
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
   DrawPreliminary(SQRTS, LInt);

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
      stAllInfo tmp(InputPattern+"/"+SHAPESTRING+"EXCLUSION" + "/"+samples[s].Name+".txt");

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


TGraph* MakePlot(FILE* pFile, FILE* talkFile, string InputPattern, string ModelName, int XSectionType, std::vector<stSample>& modelSamples, double& LInt){
   unsigned int N   = modelSamples.size();
   double* Mass     = new double   [modelSamples.size()];
   double* XSecTh   = new double   [modelSamples.size()];
   double* XSecObs  = new double   [modelSamples.size()];
   double* XSecExp  = new double   [modelSamples.size()];
   stAllInfo* Infos = new stAllInfo[modelSamples.size()];
   for(unsigned int i=0;i<modelSamples.size();i++){
      Infos       [i]=stAllInfo(InputPattern+""+SHAPESTRING+"EXCLUSION/" + modelSamples[i].Name +".txt");
      Mass        [i]=Infos[i].Mass;
      XSecTh      [i]=Infos[i].XSec_Th;
      XSecObs     [i]=Infos[i].XSec_Obs;
      XSecExp     [i]=Infos[i].XSec_Exp;
      LInt           =Infos[i].LInt;
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

double GetSignalMeanHSCPPerEvent(string InputPattern, unsigned int CutIndex, double MinRange_, double MaxRange_){
   TFile* InputFile     = new TFile((InputPattern + "Histos.root").c_str());

   TH2D*  Mass                     = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name          + "/Mass");
   TH2D*  MaxEventMass             = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name          + "/MaxEventMass");
   TH1D*  NTracksPassingSelection  = Mass->ProjectionY("NTracksPassingSelection",CutIndex+1,CutIndex+1);
   TH1D*  NEventsPassingSelection  = MaxEventMass->ProjectionY("NEventsPassingSelection",CutIndex+1,CutIndex+1);

   double NTracks       = NTracksPassingSelection->Integral(NTracksPassingSelection->GetXaxis()->FindBin(MinRange_), NTracksPassingSelection->GetXaxis()->FindBin(MaxRange_));
   double NEvents       = NEventsPassingSelection->Integral(NEventsPassingSelection->GetXaxis()->FindBin(MinRange_), NEventsPassingSelection->GetXaxis()->FindBin(MaxRange_));
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

   double LInt = 0;
   for(unsigned int k=0; k<modelVector.size(); k++){
      bool isNeutral = false;if(modelVector[k].find("GluinoN")!=string::npos || modelVector[k].find("StopN")!=string::npos)isNeutral = true;
      if(!IsTkOnly && isNeutral) continue;
      unsigned int N = modelMap[modelVector[k]].size();
      stAllInfo Infos;double Mass[N], XSecTh[N], XSecExp[N],XSecObs[N], XSecExpUp[N],XSecExpDown[N],XSecExp2Up[N],XSecExp2Down[N];
      for(unsigned int i=0;i<N;i++){
         Infos = stAllInfo(InputPattern+""+SHAPESTRING+"EXCLUSION/" + modelMap[modelVector[k]][i].Name +".txt");
         Mass        [i]=Infos.Mass;
         XSecTh      [i]=Infos.XSec_Th;
         XSecObs     [i]=Infos.XSec_Obs;
         XSecExp     [i]=Infos.XSec_Exp;
         XSecExpUp   [i]=Infos.XSec_ExpUp;
         XSecExpDown [i]=Infos.XSec_ExpDown;
         XSecExp2Up  [i]=Infos.XSec_Exp2Up;
         XSecExp2Down[i]=Infos.XSec_Exp2Down;
         LInt           =Infos.LInt;
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
      DrawPreliminary(SQRTS, LInt);
      
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

      if(IsTkOnly)SaveCanvas(c1,"Results/"+SHAPESTRING+"EXCLUSION/", string("Tk"+ modelVector[k] + "ExclusionLog"));
      else        SaveCanvas(c1,"Results/"+SHAPESTRING+"EXCLUSION/", string("Mu"+ modelVector[k] + "ExclusionLog"));
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
   double LInt = 0;

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
         Infos = stAllInfo(InputPattern+""+SHAPESTRING+"EXCLUSION/" + modelMap[modelVector[k]][i].Name +".txt");
         Mass        [i]=Infos.Mass;
         XSecTh      [i]=Infos.XSec_Th;
         XSecObs     [i]=Infos.XSec_Obs     /Infos.XSec_Exp;
         XSecExp     [i]=Infos.XSec_Exp     /Infos.XSec_Exp;
         XSecExpUp   [i]=Infos.XSec_ExpUp   /Infos.XSec_Exp;
         XSecExpDown [i]=Infos.XSec_ExpDown /Infos.XSec_Exp;
         XSecExp2Up  [i]=Infos.XSec_Exp2Up  /Infos.XSec_Exp;
         XSecExp2Down[i]=Infos.XSec_Exp2Down/Infos.XSec_Exp;
         LInt           =Infos.LInt;
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
   DrawPreliminary(SQRTS, LInt);

   TPaveText *pt = new TPaveText(0.1, 0., 0.15, 0.7,"NDC");
   string tmp = "95% CL Limits (Relative to Expected Limit)";
   TText *text = pt->AddText(tmp.c_str()); 
   text ->SetTextAlign(12);
   text ->SetTextAngle(90);
   text ->SetTextSize(0.04);
   pt->SetBorderSize(0);
   pt->SetFillColor(0);
   pt->Draw();

   if(IsTkOnly) SaveCanvas(c1,"Results/"+SHAPESTRING+"EXCLUSION/", string("TkLimitsRatio"));
   else         SaveCanvas(c1,"Results/"+SHAPESTRING+"EXCLUSION/", string("MuLimitsRatio"));
   delete c1;
}

//will run on all possible selection and try to identify which is the best one for this sample
void Optimize(string InputPattern, string Data, string signal, bool shape){
   printf("Optimize selection for %s in %s\n",signal.c_str(), InputPattern.c_str());fflush(stdout);

   //Identify the signal sample
   GetSampleDefinition(samples);
   CurrentSampleIndex        = JobIdToIndex(signal,samples); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return;  } 

   //Load all input histograms
   TFile*InputFile     = new TFile((InputPattern + "Histos.root").c_str());
   TH1D* HCuts_Pt      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
   TH1D* HCuts_I       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");
   TH1D* HCuts_TOF     = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");
   TH1D* H_Lumi        = (TH1D*)GetObjectFromPath(InputFile, Data+"/IntLumi");
   TH1D* H_A           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_A");
   TH1D* H_B           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_B");
   TH1D* H_C           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_C");
 //TH1D* H_D           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_D");
   TH1D* H_E           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_E");
   TH1D* H_F           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_F");
   TH1D* H_G           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_G");
 //TH1D* H_H           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_H");
   TH1D* H_P           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_P");
   TH2D* MassData      = (TH2D*)GetObjectFromPath(InputFile, Data+"/Mass");
   TH2D* MassPred      = (TH2D*)GetObjectFromPath(InputFile, Data+"/Pred_Mass");
   TH2D* MassSign      = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass" );
   TH2D* MassSignP     = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass_SystP");
   TH2D* MassSignI     = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass_SystI");
   TH2D* MassSignM     = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass_SystM");
   TH2D* MassSignT     = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass_SystT");
   TH2D* MassSignPU    = (TH2D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/Mass_SystPU" );
   TH1D* TotalE        = (TH1D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/TotalE" );
   TH1D* TotalEPU      = (TH1D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/TotalEPU" );

   //normalise the signal samples to XSection * IntLuminosity
   double LInt  = H_Lumi->GetBinContent(1);
   double norm  = samples[CurrentSampleIndex].XSec*LInt/TotalE  ->Integral(); //normalize the samples to the actual lumi used for limits
   double normPU= samples[CurrentSampleIndex].XSec*LInt/TotalEPU->Integral();
   MassSign  ->Scale(norm);
   MassSignP ->Scale(norm);
   MassSignI ->Scale(norm);
   MassSignM ->Scale(norm);
   MassSignT ->Scale(norm);
   MassSignPU->Scale(normPU);

   //Compute mass range for the cut&count search
   double Mean=-1,Width=-1;
   if(!shape){
      TH1D* tmpMassSignProj = MassSign->ProjectionY("MassSignProj0",1,1);
      Mean  = tmpMassSignProj->GetMean();
      Width = tmpMassSignProj->GetRMS();
      MinRange = std::max(0.0, Mean-2*Width);
      MinRange = tmpMassSignProj->GetXaxis()->GetBinLowEdge(tmpMassSignProj->GetXaxis()->FindBin(MinRange)); //Round to a bin value to avoid counting prpoblem due to the binning. 
      delete tmpMassSignProj;
   }else{
      MinRange = 0;
   }

   //prepare output directory and log file
   string outpath = InputPattern + "/"+SHAPESTRING+"EXCLUSION/";
   MakeDirectories(outpath);
   FILE* pFile = fopen((outpath+"/"+signal+".info").c_str(),"w");
   if(!pFile)printf("Can't open file : %s\n",(outpath+"/"+signal+".info").c_str());

   stAllInfo result;
   stAllInfo toReturn;
   //loop on all possible selections and determine which is the best one
   for(int CutIndex=0;CutIndex<MassData->GetNbinsX();CutIndex++){
      //if(CutIndex>25)break;

      //make sure the pT cut is high enough to get some statistic for both ABCD and mass shape
      if(HCuts_Pt ->GetBinContent(CutIndex+1) < 50 ) continue;  

      //make sure we have a reliable prediction of the number of events      
      if(H_E->GetBinContent(CutIndex+1) >0 && (H_A->GetBinContent(CutIndex+1)<25 || H_F->GetBinContent(CutIndex+1)<25 || H_G->GetBinContent(CutIndex+1)<25))continue;  //Skip events where Prediction (AFG/EE) is not reliable
      if(H_E->GetBinContent(CutIndex+1)==0 && (H_C->GetBinContent(CutIndex+1)<25 || H_B->GetBinContent(CutIndex+1)<25))continue;  //Skip events where Prediction (CB/A) is not reliable

      //make sure we have a reliable prediction of the shape 
      double N_P = H_P->GetBinContent(CutIndex+1);
      if(H_E->GetBinContent(CutIndex+1) >0 && (H_A->GetBinContent(CutIndex+1)<0.25*N_P || H_F->GetBinContent(CutIndex+1)<0.25*N_P || H_G->GetBinContent(CutIndex+1)<0.25*N_P))continue;  //Skip events where Mass Prediction is not reliable
      if(H_E->GetBinContent(CutIndex+1)==0 && (H_C->GetBinContent(CutIndex+1)<0.25*N_P || H_B->GetBinContent(CutIndex+1)<0.25*N_P))continue;  //Skip events where Mass Prediction is not reliable

      //prepare outputs result structure
      result = toReturn;
      result.MassMean  = Mean;
      result.MassSigma = Width;
      result.MassCut   = MinRange;
      result.Mass      = samples[JobIdToIndex(signal,samples)].Mass;
      result.XSec_Th   = samples[JobIdToIndex(signal,samples)].XSec;
      result.XSec_Err  = samples[JobIdToIndex(signal,samples)].XSec * 0.15;
      result.Index     = CutIndex;
      result.WP_Pt     = HCuts_Pt ->GetBinContent(CutIndex+1);
      result.WP_I      = HCuts_I  ->GetBinContent(CutIndex+1);
      result.WP_TOF    = HCuts_TOF->GetBinContent(CutIndex+1);
      result.LInt      = LInt;

      //compute the limit for this point and check it run sucessfully (some point may be skipped because of lack of statistics or other reasons)
      //best expected limit
      if(!runCombine(false, InputPattern, signal, CutIndex, shape, true, result, MassData, MassPred, MassSign, MassSignP, MassSignI, MassSignM, MassSignT, MassSignPU))continue;

      //best significance
      //if(!runCombine(true, InputPattern, signal, CutIndex, shape, true, result, MassData, MassPred, MassSign, MassSignP, MassSignI, MassSignM, MassSignT, MassSignPU))continue;


      //repport the result for this point in the log file
      fprintf(pFile  ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.3f TOF>%6.3f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E+-%6.3E SignalEff=%6.3f ExpLimit=%6.3E (%6.3E) Signif=%6.3E",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,result.NData,result.NPred, result.NPredErr,result.Eff,result.XSec_Exp, result.XSec_Obs, result.Significance);fflush(stdout);
      fprintf(stdout ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.3f TOF>%6.3f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E+-%6.3E SignalEff=%6.3f ExpLimit=%6.3E (%6.3E) Signif=%6.3E",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,result.NData,result.NPred, result.NPredErr,result.Eff,result.XSec_Exp, result.XSec_Obs, result.Significance);fflush(stdout);
      if(result.XSec_Exp<toReturn.XSec_Exp){
//      if(result.Significance>toReturn.Significance){
         toReturn=result;
         fprintf(pFile  ," BestSelection\n");fflush(stdout);
         fprintf(stdout ," BestSelection\n");fflush(stdout);
      }else{
         fprintf(pFile  ,"\n");fflush(stdout);
         fprintf(stdout ,"\n");fflush(stdout);
      }
   }//end of selection cut loop
   fclose(pFile);   
 
   //recompute the limit for the final point and save the output in the final directory (also save some plots for the shape based analysis)
   runCombine(false, InputPattern, signal, toReturn.Index, shape, false, toReturn, MassData, MassPred, MassSign, MassSignP, MassSignI, MassSignM, MassSignT, MassSignPU);
  
   //all done, save the result to file
   toReturn.Save(outpath+"/"+signal+".txt");
}

// produce the Higgs combine stat tool datacard
void makeDataCard(string outpath, string rootPath, string ChannelName, string SignalName, double Obs, double Pred, double PredRelErr, double Sign, bool Shape){
   FILE* pFile = fopen(outpath.c_str(), "w");
   fprintf(pFile, "imax 1\n");
   fprintf(pFile, "jmax *\n");
   fprintf(pFile, "kmax *\n");
   if(Shape){
   fprintf(pFile, "-------------------------------\n");
   fprintf(pFile, "shapes * * %s $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC\n",rootPath.c_str());
   }
   fprintf(pFile, "-------------------------------\n");
   fprintf(pFile, "bin %s\n",ChannelName.c_str());
   fprintf(pFile, "Observation %f\n",Obs);
   fprintf(pFile, "-------------------------------\n");
   fprintf(pFile, "bin      %s %s\n",ChannelName.c_str(), ChannelName.c_str());
   fprintf(pFile, "process  %s pred\n",SignalName.c_str());
   fprintf(pFile, "process  0 1\n");
   fprintf(pFile, "rate    %f %f\n",Sign,Pred);
   fprintf(pFile, "-------------------------------\n");
   fprintf(pFile, "%5s    %6s 1.022     1.0  \n","Lumi" , "lnN");
   fprintf(pFile, "%5s    %6s -         %5.3f\n","systP", "lnN", PredRelErr);
   fprintf(pFile, "%5s    %6s 1.07      -    \n","systS", "lnN");
   if(Shape){
   fprintf(pFile, "%5s    %6s 1.000     -    \n","statS", "shapeN2");
   fprintf(pFile, "%5s    %6s -         1    \n","statP", "shapeN2");
   fprintf(pFile, "%5s    %6s 1.000     -    \n","mom"  , "shapeN2");
   fprintf(pFile, "%5s    %6s 1.000     -    \n","ias"  , "shapeN2");
   fprintf(pFile, "%5s    %6s 1.000     -    \n","ih"   , "shapeN2");
   fprintf(pFile, "%5s    %6s 1.000     -    \n","tof"  , "shapeN2");
   fprintf(pFile, "%5s    %6s 1.000     -    \n","pu"   , "shapeN2");
   }
   fclose(pFile);
}

//save histogram in root file (and it's statistical variation if it's not observed date histogram)
void saveHistoForLimit(TH1* histo, string Name, string Id){            
      histo   ->Write( Name                   .c_str());
      if(Name=="data_obs")return;
      
      TH1* statup   = (TH1*)histo->Clone((Name+"_stat"+Id+"Up").c_str());
      TH1* statdown = (TH1*)histo->Clone((Name+"_stat"+Id+"Down").c_str());       
      for(int ibin=1; ibin<=statup->GetXaxis()->GetNbins(); ibin++){
         statup  ->SetBinContent(ibin,statup  ->GetBinContent(ibin) + histo->GetBinError(ibin));
         statdown->SetBinContent(ibin,statdown->GetBinContent(ibin) - histo->GetBinError(ibin));
      }
      statup  ->Write((Name+"_stat"+Id+"Up"  ).c_str());
      statdown->Write((Name+"_stat"+Id+"Down").c_str());      

      delete statup;
      delete statdown;
}

//save histogram with variation in root file  (save it as symetrical up and down variation)
void saveVariationHistoForLimit(TH1* histo, TH1* vardown, string Name, string variationName){
      TH1* varup   = (TH1*)histo ->Clone((Name+"_"+variationName+"Up"  ).c_str());
      varup->Add(vardown,-1); //varup=x
      varup->Add(histo,1);
      varup  ->Write((Name+"_"+variationName+"Up"  ).c_str());
      vardown->Write((Name+"_"+variationName+"Down").c_str());
}

//function for debugging only, should be remove soon
//this function just compute the result of the shape based analysis using the optimal point from the cut&count analysis
void testShapeBasedAnalysis(string InputPattern, string signal){
   CurrentSampleIndex        = JobIdToIndex(signal, samples); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return;  }
   int s = CurrentSampleIndex;

   //Get Optimal cut from cut&count optimization
   stAllInfo result =  stAllInfo(InputPattern+"/EXCLUSION/"+signal+".txt");

   //load all intput histograms
   TFile* InputFile  = new TFile((InputPattern+"/Histos.root").c_str());
   TH2D*  MassData   = (TH2D*)GetObjectFromPath(InputFile, "Data11/Mass");
   TH2D*  MassPred   = (TH2D*)GetObjectFromPath(InputFile, "Data11/Pred_Mass");
   TH2D*  MassSign   = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass");
   TH2D*  MassSignP  = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass_SystP");
   TH2D*  MassSignI  = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass_SystI");
   TH2D*  MassSignM  = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass_SystM");
   TH2D*  MassSignT  = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass_SystT");
   TH2D*  MassSignPU = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass_SystPU");

   //compute shape based limits and save it's output
   runCombine(false, InputPattern, signal, result.Index, true, false, result, MassData, MassPred, MassSign, MassSignP, MassSignI, MassSignM, MassSignT, MassSignPU);

   //all done, save the results to file
   result.Save(InputPattern+"/"+SHAPESTRING+"EXCLUSION/"+signal+".txt");
}

//run the higgs combine stat tool
bool runCombine(bool Significance, string& InputPattern, string& signal, unsigned int CutIndex, bool Shape, bool Temporary, stAllInfo& result, TH2D* MassData, TH2D* MassPred, TH2D* MassSign, TH2D* MassSignP, TH2D* MassSignI, TH2D* MassSignM, TH2D* MassSignT, TH2D* MassSignPU){
   //make the projection of all the 2D input histogram to get the shape for this single point
   double signalsMeanHSCPPerEvent = GetSignalMeanHSCPPerEvent(InputPattern,CutIndex, MinRange, MaxRange);
   TH1D* MassDataProj       = MassData  ->ProjectionY("MassDataProj"  ,CutIndex+1,CutIndex+1);
   TH1D* MassPredProj       = MassPred  ->ProjectionY("MassPredProj"  ,CutIndex+1,CutIndex+1);
   TH1D* MassSignProj       = MassSign  ->ProjectionY("MassSignProj"  ,CutIndex+1,CutIndex+1);
   TH1D* MassSignProjP      = MassSignP ->ProjectionY("MassSignProP"  ,CutIndex+1,CutIndex+1);
   TH1D* MassSignProjI      = MassSignI ->ProjectionY("MassSignProI"  ,CutIndex+1,CutIndex+1);
   TH1D* MassSignProjM      = MassSignM ->ProjectionY("MassSignProM"  ,CutIndex+1,CutIndex+1);
   TH1D* MassSignProjT      = MassSignT ->ProjectionY("MassSignProT"  ,CutIndex+1,CutIndex+1);
   TH1D* MassSignProjPU     = MassSignPU->ProjectionY("MassSignProPU" ,CutIndex+1,CutIndex+1);

   //count events in the allowed range (infinite for shape based and restricted for cut&count)
   double NData       = MassDataProj->Integral(MassDataProj->GetXaxis()->FindBin(MinRange), MassDataProj->GetXaxis()->FindBin(MaxRange));
   double NPred       = MassPredProj->Integral(MassPredProj->GetXaxis()->FindBin(MinRange), MassPredProj->GetXaxis()->FindBin(MaxRange));
   double NPredErr    = pow(NPred*RescaleError,2);
   for(int i=MassPredProj->GetXaxis()->FindBin(MinRange); i<=MassPredProj->GetXaxis()->FindBin(MaxRange) ;i++){NPredErr+=pow(MassPredProj->GetBinError(i),2);}NPredErr=sqrt(NPredErr);
   double NSign       = (MassSignProj  ->Integral(MassSignProj  ->GetXaxis()->FindBin(MinRange), MassSignProj  ->GetXaxis()->FindBin(MaxRange))) / signalsMeanHSCPPerEvent;

   //skip pathological selection point
   if(isnan((float)NPred))return false;
   if(NPred<=0){return false;} //Is <=0 only when prediction failed or is not meaningful (i.e. WP=(0,0,0) )
   if(!Shape && NPred>1000){return false;}  //When NPred is too big, expected limits just take an infinite time! 

   //compute all efficiencies (not really needed anymore, but it's nice to look at these numbers afterward)
   double Eff         = (MassSignProj  ->Integral(MassSignProj  ->GetXaxis()->FindBin(MinRange), MassSignProj  ->GetXaxis()->FindBin(MaxRange))) / (result.XSec_Th*result.LInt*signalsMeanHSCPPerEvent);
   double EffP        = (MassSignProjP ->Integral(MassSignProjP ->GetXaxis()->FindBin(MinRange), MassSignProjP ->GetXaxis()->FindBin(MaxRange))) / (result.XSec_Th*result.LInt*signalsMeanHSCPPerEvent);
   double EffI        = (MassSignProjI ->Integral(MassSignProjI ->GetXaxis()->FindBin(MinRange), MassSignProjI ->GetXaxis()->FindBin(MaxRange))) / (result.XSec_Th*result.LInt*signalsMeanHSCPPerEvent);
   double EffM        = (MassSignProjM ->Integral(MassSignProjM ->GetXaxis()->FindBin(MinRange), MassSignProjM ->GetXaxis()->FindBin(MaxRange))) / (result.XSec_Th*result.LInt*signalsMeanHSCPPerEvent);
   double EffT        = (MassSignProjT ->Integral(MassSignProjT ->GetXaxis()->FindBin(MinRange), MassSignProjT ->GetXaxis()->FindBin(MaxRange))) / (result.XSec_Th*result.LInt*signalsMeanHSCPPerEvent);
   double EffPU       = (MassSignProjPU->Integral(MassSignProjPU->GetXaxis()->FindBin(MinRange), MassSignProjPU->GetXaxis()->FindBin(MaxRange))) / (result.XSec_Th*result.LInt*signalsMeanHSCPPerEvent);
   if(Eff==0)return false;

   //save these info to the result structure
   result.Eff       = Eff;
   result.Eff_SYSTP = EffP;
   result.Eff_SYSTI = EffI;
   result.Eff_SYSTM = EffM;
   result.Eff_SYSTT = EffT;
   result.Eff_SYSTPU= EffPU;
   result.NData     = NData;
   result.NPred     = NPred;
   result.NPredErr  = NPredErr;
   result.NSign     = NSign;
   NSign/=(result.XSec_Th*1000.0); //normalize xsection to 1fb

   //for shape based analysis we need to save all histograms into a root file
   char CutIndexStr[255];sprintf(CutIndexStr, "CutIndex%03.0f",result.Index);
   if(Shape){
      //prepare the histograms and variation
      //scale to 1fb xsection and to observed events instead of observed tracks
      MassSignProj  ->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));
      MassSignProjP ->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));
      MassSignProjI ->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));
      MassSignProjM ->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));
      MassSignProjT ->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));
      MassSignProjPU->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));

      //make histo that will contains the shapes for limit
      string shapeFilePath = "/tmp/shape_"+signal+".root";
      TFile* out = new TFile(shapeFilePath.c_str(),"RECREATE");   
      out->cd();
      out->mkdir(CutIndexStr);
      out->cd(CutIndexStr);

      //save histo into the file   
      saveHistoForLimit(MassDataProj, "data_obs","");
      saveHistoForLimit(MassPredProj, "pred", "P");
      saveHistoForLimit(MassSignProj, signal, "S");
      saveVariationHistoForLimit(MassSignProj, MassSignProjP , signal, "mom");
      saveVariationHistoForLimit(MassSignProj, MassSignProjI , signal, "ias");
      saveVariationHistoForLimit(MassSignProj, MassSignProjM , signal, "ih");
      saveVariationHistoForLimit(MassSignProj, MassSignProjT , signal, "tof");
      saveVariationHistoForLimit(MassSignProj, MassSignProjPU, signal, "pu");

      //close the output file
      out->Close();
   }

   //build the combine datacard, the same code is used both for cut&count and shape base
   string datacardPath = "/tmp/shape_"+signal+".dat";
   makeDataCard(datacardPath,string("shape_")+signal+".root", CutIndexStr,signal, NData, NPred, 1.0+(Shape?RescaleError:NPredErr/NPred), NSign, Shape);

   char massStr[255]; sprintf(massStr,"%.0f",result.Mass);
   char rangeStr[255];sprintf(rangeStr," --rMin %f --rMax %f ", 0.0f, 2*(3*sqrt(NPred)/NSign) );
   if(!Significance)printf("%f/%f --> %s\n",result.NSign,result.NPred,rangeStr);


   //prepare and run the script that will run the external "combine" tool from the Higgs group
   string CodeToExecute;
   CodeToExecute += "cd /tmp/;";
   if(Significance){
      CodeToExecute += "combine -M ProfileLikelihood -n " + signal + " -m " + massStr + " --significance -t 1000 --expectSignal=1 " + " shape_" + signal+".dat &> shape_" + signal + ".log;";
   }else{
      CodeToExecute += "combine -M Asymptotic        -n " + signal + " -m " + massStr + rangeStr + " shape_" + signal+".dat &> shape_" + signal + ".log;";
   }
   CodeToExecute += "cd $OLDPWD;";
   CodeToExecute+="cp /tmp/shape_" + signal + ".* " + InputPattern+"/"+SHAPESTRING+"EXCLUSION/." + ";";
   system(CodeToExecute.c_str());


   if(Significance){
      char line[4096];
      FILE* sFile = fopen((string("/tmp/shape_")+signal + ".log").c_str(), "r");
      if(!sFile)std::cout<<"FILE NOT OPEN:"<< (string("/tmp/shape_")+signal + ".log").c_str() << endl;
      int LineIndex=0; int GarbageI;      
      while(fgets(line, 4096, sFile)){LineIndex++;
        //printf("%i --> | %s\n",LineIndex,line);
        if(LineIndex!=7)continue;
        sscanf(line,"median expected limit: r < %lf @ 95%%CL (%i toyMC)",&(result.Significance),&GarbageI); 
      }     
   }else{
      //if all went well, the combine tool created a new file containing the result of the limit in the form of a TTree
      //we can open this TTree and access the values for the expected limit, uncertainty bands, and observed limits.
      TFile* file = NULL;
//      if(Significance){
//         file = TFile::Open((string("/tmp/")+"higgsCombine"+signal+".ProfileLikelihood.mH"+massStr+".123456.root").c_str());
//      }else{
         file = TFile::Open((string("/tmp/")+"higgsCombine"+signal+".Asymptotic.mH"+massStr+".root").c_str());
//      }
      if(!file || file->IsZombie())return false;
      TTree* tree = (TTree*)file->Get("limit");
      if(!tree)return false;
      double Tmass, Tlimit, TlimitErr; float TquantExp;
      tree->GetBranch("mh"              )->SetAddress(&Tmass    );
      tree->GetBranch("limit"           )->SetAddress(&Tlimit   );
      tree->GetBranch("limit"           )->SetAddress(&Tlimit   );
      tree->GetBranch("limitErr"        )->SetAddress(&TlimitErr);
      tree->GetBranch("quantileExpected")->SetAddress(&TquantExp);
      for(int ientry=0;ientry<tree->GetEntriesFast();ientry++){
        tree->GetEntry(ientry);
              if(TquantExp==0.025f){ result.XSec_Exp2Down = Tlimit/1000.0;
        }else if(TquantExp==0.160f){ result.XSec_ExpDown  = Tlimit/1000.0;
        }else if(TquantExp==0.500f){ result.XSec_Exp      = Tlimit/1000.0;
        }else if(TquantExp==0.840f){ result.XSec_ExpUp    = Tlimit/1000.0;
        }else if(TquantExp==0.975f){ result.XSec_Exp2Up   = Tlimit/1000.0;
        }else if(TquantExp==-1    ){ result.XSec_Obs      = Tlimit/1000.0;
        }else{printf("Quantil %f unused by the analysis --> check the code\n", TquantExp);
        }
      }
      file->Close();
   }

   //makePlots (only for shape based analysis)
   if(Shape && !Temporary){ 
      TCanvas* c1 = new TCanvas("c1", "c1",1200,600);
      c1->Divide(2,1);
      (c1->cd(1))->SetLogy(true);

      double Max = 2.0 * std::max(std::max(MassDataProj->GetMaximum(), MassPredProj->GetMaximum()), MassSignProj->GetMaximum());
      double Min = 0.01;
      MassSignProj->SetStats(kFALSE);
      MassSignProj->SetMaximum(Max);
      MassSignProj->SetMinimum(Min);
      MassSignProj->GetXaxis()->SetRangeUser(0,1400);
      MassSignProj->GetXaxis()->SetNdivisions(505,"X");
      MassSignProj->SetMarkerStyle(21);
      MassSignProj->SetMarkerColor(kBlue-9);
      MassSignProj->SetMarkerSize(1.5);
      MassSignProj->SetLineColor(kBlue-9);
      MassSignProj->SetFillColor(kBlue-9);
      MassSignProj->Draw("HIST");

      MassPredProj->SetBinContent(MassPredProj->GetNbinsX(), MassPredProj->GetBinContent(MassPredProj->GetNbinsX()) + MassPredProj->GetBinContent(MassPredProj->GetNbinsX()+1));
      MassPredProj->SetMarkerStyle(22);
      MassPredProj->SetMarkerColor(2);
      MassPredProj->SetMarkerSize(1.5);
      MassPredProj->SetLineColor(2);
      MassPredProj->SetFillColor(8);
      MassPredProj->SetFillStyle(3001);
      MassPredProj->Draw("same E3");

      MassDataProj->SetBinContent(MassDataProj->GetNbinsX(), MassDataProj->GetBinContent(MassDataProj->GetNbinsX()) + MassDataProj->GetBinContent(MassDataProj->GetNbinsX()+1));
      MassDataProj->SetMarkerStyle(20);
      MassDataProj->SetMarkerColor(1);
      MassDataProj->SetMarkerSize(1.0);
      MassDataProj->SetLineColor(1);
      MassDataProj->SetFillColor(0);
      MassDataProj->Draw("same E1");

      TLegend* leg = new TLegend(0.45,0.75,0.85,0.93);
      leg->SetHeader(NULL);
      leg->SetFillColor(0);
      leg->SetFillStyle(0);
      leg->SetBorderSize(0);
      leg->AddEntry(MassDataProj,"Data", "P");
      leg->AddEntry(MassPredProj,"Prediction", "FP");
      leg->AddEntry(MassSignProj,signal.c_str(), "F");
      leg->Draw();

      (c1->cd(2))->SetLogy(true);

      TH1* MassSignProjRatio  = (TH1D*)MassSignProj ->Clone(signal.c_str() );  MassSignProjRatio ->SetLineColor(1); MassSignProjRatio ->SetMarkerColor(1); MassSignProjRatio ->SetMarkerStyle(0);
      TH1* MassSignProjPRatio = (TH1D*)MassSignProjP->Clone("mom");            MassSignProjPRatio->SetLineColor(2); MassSignProjPRatio->SetMarkerColor(2); MassSignProjPRatio->SetMarkerStyle(20);
      TH1* MassSignProjIRatio = (TH1D*)MassSignProjI->Clone("Ias");            MassSignProjIRatio->SetLineColor(4); MassSignProjIRatio->SetMarkerColor(4); MassSignProjIRatio->SetMarkerStyle(21);
      TH1* MassSignProjMRatio = (TH1D*)MassSignProjM->Clone("Ih");             MassSignProjMRatio->SetLineColor(3); MassSignProjMRatio->SetMarkerColor(3); MassSignProjMRatio->SetMarkerStyle(22);
      TH1* MassSignProjTRatio = (TH1D*)MassSignProjT->Clone("TOF");            MassSignProjTRatio->SetLineColor(8); MassSignProjTRatio->SetMarkerColor(8); MassSignProjTRatio->SetMarkerStyle(23);
      TH1* MassSignProjLRatio = (TH1D*)MassSignProjPU->Clone("pu");            MassSignProjLRatio->SetLineColor(6); MassSignProjLRatio->SetMarkerColor(6); MassSignProjLRatio->SetMarkerStyle(33);

      MassSignProjRatio->SetStats(kFALSE);
      MassSignProjRatio->SetFillColor(0);
      MassSignProjRatio->SetLineWidth(2);
      MassSignProjRatio->GetXaxis()->SetRangeUser(0,1400);
      MassSignProjRatio->GetXaxis()->SetNdivisions(505,"X");
      MassSignProjRatio->SetMaximum(Max);
      MassSignProjRatio->SetMinimum(Min);
      MassSignProjRatio->Draw("hist");
      MassSignProjPRatio->Draw("same E1");
      MassSignProjIRatio->Draw("same E1");
      MassSignProjMRatio->Draw("same E1");
      MassSignProjTRatio->Draw("same E1");
      MassSignProjLRatio->Draw("same E1");

      TLegend* leg2 = new TLegend(0.45,0.65,0.85,0.93);
      leg2->SetHeader("Variations");
      leg2->SetFillColor(0);
      leg2->SetFillStyle(0);
      leg2->SetBorderSize(0);
      leg2->AddEntry(MassSignProjRatio,signal.c_str(), "L");
      leg2->AddEntry(MassSignProjPRatio,"momentum", "P");
      leg2->AddEntry(MassSignProjIRatio,"Ias", "P");
      leg2->AddEntry(MassSignProjMRatio,"Ih", "P");
      leg2->AddEntry(MassSignProjTRatio,"tof", "P");
      leg2->AddEntry(MassSignProjLRatio,"pu", "P");
      leg2->Draw();

      SaveCanvas(c1, InputPattern+"/"+SHAPESTRING+"EXCLUSION/shape", signal, true);
      delete leg2; delete leg; delete c1;
   }

   //all done, clean everything and return true
   delete MassDataProj; delete MassPredProj; delete MassSignProj; delete MassSignProjP; delete MassSignProjI; delete MassSignProjM; delete MassSignProjT; delete MassSignProjPU;
   return true;
}



