// Original Author:  Loic Quertenmont

#include "Analysis_Global.h"
#include "Analysis_CommonFunction.h"
#include "Analysis_PlotFunction.h"
#include "Analysis_Samples.h"
#include "tdrstyle.C"

#include "TGraphAsymmErrors.h"

using namespace std;

class stAllInfo{
   public:
   double Mass, MassMean, MassSigma, MassCut;
   double XSec_Th, XSec_Err, XSec_Exp, XSec_ExpUp, XSec_ExpDown, XSec_Exp2Up, XSec_Exp2Down, XSec_Obs;
   double  Eff, Eff_SYSTP, Eff_SYSTI, Eff_SYSTM, Eff_SYSTT, Eff_SYSTPU;
   double Significance; double XSec_5Sigma;
   double Index, WP_Pt, WP_I, WP_TOF;
   float  NData, NPred, NPredErr, NSign;
   double LInt;

   stAllInfo(string path=""){
      //Default Values
      Mass          = 0;      MassMean      = 0;      MassSigma     = 0;      MassCut       = 0;
      Index         = 0;      WP_Pt         = 0;      WP_I          = 0;      WP_TOF        = 0;
      XSec_Th       = 0;      XSec_Err      = 0;      XSec_Exp      = 1E50;   XSec_ExpUp    = 1E50;   XSec_ExpDown  = 1E50;    XSec_Exp2Up   = 1E50;    XSec_Exp2Down = 1E50;    XSec_Obs    = 1E50;
      Significance  = 1E50;   XSec_5Sigma   = 1E50;
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
      fscanf(pFile,"XSec_5Sigma  : %lf\n",&XSec_5Sigma);
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
      fprintf(pFile,"XSec_5Sigma  : %f\n",XSec_5Sigma);
      fclose(pFile); 
   }
};


string EXCLUSIONDIR = "EXCLUSION";

//Background prediction rescale and uncertainty
double RescaleFactor = 1.0;
double RescaleError  = 0.1;

//final Plot y-axis range
double PlotMinScale = 0.0001;
double PlotMaxScale = 3;

void Optimize(string InputPattern, string Data, string signal, bool shape, bool cutFromFile);
double GetSignalMeanHSCPPerEvent(string InputPattern, unsigned int CutIndex, double MinRange, double MaxRange);
TGraph* MakePlot(FILE* pFile, FILE* talkFile, string InputPattern, string ModelName, int XSectionType, std::vector<stSample>& modelSamples, double& LInt);
TGraph* CheckSignalUncertainty(FILE* pFile, FILE* talkFile, string InputPattern, string ModelName, std::vector<stSample>& modelSample);
void DrawModelLimitWithBand(string InputPattern);
void DrawRatioBands(string InputPattern);
void printSummary(FILE* pFile, FILE* talkFile, string InputPattern, string ModelName, std::vector<stSample>& modelSamples);


void makeDataCard(string outpath, string rootPath, string ChannelName, string SignalName, double Obs, double Pred, double PredRelErr, double Sign, bool Shape);
void saveHistoForLimit(TH1* histo, string Name, string Id);
void saveVariationHistoForLimit(TH1* histo, TH1* vardown, string Name, string variationName);
void testShapeBasedAnalysis(string InputPattern, string signal);
double computeSignificance(string datacard, bool expected, string& signal, string massStr, float Strength);
bool runCombine(bool fastOptimization, bool getXsection, bool getSignificance, string& InputPattern, string& signal, unsigned int CutIndex, bool Shape, bool Temporary, stAllInfo& result, TH1* MassData, TH1* MassPred, TH1* MassSign, TH1* MassSignP, TH1* MassSignI, TH1* MassSignM, TH1* MassSignT, TH1* MassSignPU);
bool Combine(string InputPattern, string signal7, string signal8);
bool useSample(int TypeMode, string sample);

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

   string Data;
   if(MODE.find("SHAPE")!=string::npos){SHAPESTRING="SHAPE";}else{SHAPESTRING="";}
   if(MODE.find("COMPUTELIMIT")!=string::npos || MODE.find("OPTIMIZE")!=string::npos){
      if(signal.find("7TeV")!=string::npos){Data = "Data7TeV"; SQRTS=7.0; EXCLUSIONDIR+="7TeV"; }
      if(signal.find("8TeV")!=string::npos){Data = "Data8TeV"; SQRTS=8.0; EXCLUSIONDIR+="8TeV"; }
      printf("EXCLUSIONDIR = %s\nData = %s\n",EXCLUSIONDIR.c_str(), Data.c_str());  

      if(MODE.find("COMPUTELIMIT")!=string::npos){Optimize(InputPattern, Data, signal, SHAPESTRING!="", true);      return;}
      if(MODE.find("OPTIMIZE")!=string::npos){    Optimize(InputPattern, Data, signal, SHAPESTRING!="", false);     return;} //testShapeBasedAnalysis(InputPattern,signal);  //use the second part if you want to run shape based analyssi on optimal point form c&c      
   }

   if(MODE.find("COMBINE")!=string::npos){
      printf("COMBINE!!!\n");

      string signal7TeV = signal; if(signal7TeV.find("_8TeV")!=string::npos) signal7TeV = signal7TeV.replace(signal7TeV.find("_8TeV"),5, "_7TeV");
      string signal8TeV = signal; if(signal8TeV.find("_7TeV")!=string::npos) signal8TeV = signal8TeV.replace(signal8TeV.find("_7TeV"),5, "_8TeV");

      string EXCLUSIONDIR_SAVE = EXCLUSIONDIR;
      //2011 Limits
      Data = "Data7TeV"; SQRTS=7.0; EXCLUSIONDIR=EXCLUSIONDIR_SAVE+"7TeV";
      Optimize(InputPattern, Data, signal7TeV, SHAPESTRING!="", true);

      //2012 Limits
      Data = "Data8TeV"; SQRTS=8.0; EXCLUSIONDIR=EXCLUSIONDIR_SAVE+"8TeV";
      Optimize(InputPattern, Data, signal8TeV, SHAPESTRING!="", true);

      //Combined Limits
      EXCLUSIONDIR=EXCLUSIONDIR_SAVE+"COMB";  SQRTS=78.0;
      Combine(InputPattern, signal7TeV, signal8TeV);
      return;
   }

   if(MODE.find("7TeV")!=string::npos){Data = "Data7TeV"; SQRTS=7.0; EXCLUSIONDIR+="7TeV"; }
   if(MODE.find("8TeV")!=string::npos){Data = "Data8TeV"; SQRTS=8.0; EXCLUSIONDIR+="8TeV"; }
   printf("EXCLUSIONDIR = %s\nData = %s\n",EXCLUSIONDIR.c_str(), Data.c_str());  

   
   string TkPattern  = "Results/Type0/";
   string MuPattern  = "Results/Type2/";
   string MOPattern  = "Results/Type3/";
   string HQPattern  = "Results/Type4/";
   string LQPattern  = "Results/Type5/";

   bool Combine = (MODE.find("COMB")!=string::npos);
   if(Combine){EXCLUSIONDIR+="COMB"; SQRTS=78.0;}

   string outpath = string("Results/"+SHAPESTRING+EXCLUSIONDIR+"/");
   MakeDirectories(outpath);

   //determine the list of models that are considered
   GetSampleDefinition(samples);

   if(SQRTS!=78.0) keepOnlySamplesAt7and8TeVX(samples, SQRTS);

   for(unsigned int s=0;s<samples.size();s++){
    if(samples[s].Type!=2)continue;
    //printf("Name-->Model >>  %30s --> %s\n",samples[s].Name.c_str(), samples[s].ModelName().c_str());

    if(SQRTS== 7.0  && samples[s].Name.find("_7TeV")==string::npos){continue;}
    if(SQRTS== 8.0  && samples[s].Name.find("_8TeV")==string::npos){continue;}
//    if(SQRTS==78.0){if(samples[s].Name.find("_7TeV")==string::npos){continue;}else{samples[s].Name.replace(samples[s].Name.find("_7TeV"),5, ""); } }
    if(SQRTS==78.0){if(samples[s].Name.find("_8TeV")==string::npos){continue;}else{samples[s].Name.replace(samples[s].Name.find("_8TeV"),5, ""); } }

    modelMap[samples[s].ModelName()].push_back(samples[s]);   
    if(modelMap[samples[s].ModelName()].size()==1)modelVector.push_back(samples[s].ModelName());
   }

   //unti we have all the samples at both 7 and 8TeV, add the 7TeV models
//   if(SQRTS== 8.0){
//      for(unsigned int s=0;s<samples.size();s++){
//       if(samples[s].Type!=2)continue;
// 
//        if(modelMap.find(samples[s].ModelName())==modelMap.end()){
//          modelMap[samples[s].ModelName()].push_back(samples[s]);
//          if(modelMap[samples[s].ModelName()].size()==1)modelVector.push_back(samples[s].ModelName());
//        }
//      }      
//   } 

   //based on the modelMap
   DrawRatioBands(TkPattern); 
   DrawRatioBands(MuPattern);
   DrawRatioBands(MOPattern); 
   DrawRatioBands(LQPattern);

   //draw the cross section limit for all model
   DrawModelLimitWithBand(TkPattern);
   DrawModelLimitWithBand(MuPattern);
   DrawModelLimitWithBand(MOPattern);
   DrawModelLimitWithBand(LQPattern);

   //make plots of the observed limit for all signal model (and mass point) and save the result in a latex table
   TCanvas* c1;
   TLegend* LEG;
   double LInt = 0;

   FILE* pFile    = fopen((outpath+string("Analysis_Step6_Result") + ".txt").c_str(),"w");
   FILE* talkFile = fopen((outpath + "TalkPlots" + ".txt").c_str(),"w");

   fprintf(pFile   , "\\documentclass{article}\n");
   fprintf(pFile   , "\\begin{document}\n\n");
   fprintf(talkFile, "\\documentclass{article}\n");
   fprintf(talkFile, "\\usepackage{rotating}\n");
   fprintf(talkFile, "\\begin{document}\n\n");
   fprintf(talkFile, "\\begin{tiny}\n\n");

   fprintf(pFile   , "%% %50s\n", "TkOnly");
   fprintf(pFile   , "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
   fprintf(talkFile, "\\begin{sidewaystable}\n   \\centering\n      \\begin{tabular}{|l|cccccccc|}\n      \\hline\n");
   fprintf(talkFile,"Sample & Mass(GeV) & Pt(GeV) & $I_{as}$ & TOF & Mass Cut (GeV) & N pred & N observed & Eff & Signif \\\\\n");
   fprintf(talkFile, "\\hline\n");
   TGraph** TkGraphs  = new TGraph*[modelVector.size()];
   for(unsigned int k=0; k<modelVector.size(); k++){
      TkGraphs[k] = MakePlot(pFile,talkFile,TkPattern,modelVector[k], 2, modelMap[modelVector[k]], LInt);
   }
   fprintf(pFile   ,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(talkFile,"      \\end{tabular}\n\\end{sidewaystable}\n\n");

   fprintf(pFile   , "%% %50s\n", "TkMuon");
   fprintf(pFile   , "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
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
   fprintf(talkFile,"      \\end{tabular}\n\\end{sidewaystable}\n\n");

   fprintf(pFile   , "%% %50s\n", "MuOnly");
   fprintf(pFile   , "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
   fprintf(talkFile, "\\begin{sidewaystable}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
   fprintf(talkFile,"Sample & Mass(GeV) & Pt(GeV) & $I_{as}$ & $#beta^{-1]$ & Mass Cut (GeV) & N pred & N observed & Eff \\\\\n");
   fprintf(talkFile, "\\hline\n");
   TGraph** MOGraphs = new TGraph*[modelVector.size()];
   for(unsigned int k=0; k<modelVector.size(); k++){
     bool isNeutral = false;if(modelVector[k].find("GluinoN")!=string::npos || modelVector[k].find("StopN")!=string::npos)isNeutral = true;
     if(isNeutral) continue;//skip charged suppressed models                                                                                                                      
     MOGraphs[k] = MakePlot(pFile,talkFile,MOPattern,modelVector[k], 2, modelMap[modelVector[k]], LInt);
   }
   fprintf(pFile   ,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(talkFile,"      \\end{tabular}\n\\end{sidewaystable}\n\n");


   fprintf(pFile   , "%% %50s\n", "multiple charge");
   fprintf(pFile   , "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
   fprintf(talkFile, "\\begin{sidewaystable}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
   fprintf(talkFile,"Sample & Mass(GeV) & Pt(GeV) & $I_{as}$ & $#beta^{-1]$ & Mass Cut (GeV) & N pred & N observed & Eff \\\\\n");
   fprintf(talkFile, "\\hline\n");
   TGraph** HQGraphs = new TGraph*[modelVector.size()];
   for(unsigned int k=0; k<modelVector.size(); k++){
      if(modelVector[k].find("DY")==string::npos)continue;
      bool isFractional = false;if(modelVector[k].find("1o3")!=string::npos || modelVector[k].find("2o3")!=string::npos)isFractional = true;
      if(isFractional) continue;//skip q>=1 charged suppressed models
      HQGraphs[k] = MakePlot(pFile,talkFile,HQPattern,modelVector[k], 2, modelMap[modelVector[k]], LInt);
   }
   fprintf(pFile   ,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(talkFile,"      \\end{tabular}\n\\end{sidewaystable}\n\n");


   fprintf(pFile   , "%% %50s\n", "fractionnally charge");
   fprintf(pFile   , "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
   fprintf(talkFile, "\\begin{sidewaystable}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
   fprintf(talkFile,"Sample & Mass(GeV) & Pt(GeV) & $I_{as}$ & $#beta^{-1]$ & Mass Cut (GeV) & N pred & N observed & Eff \\\\\n");
   fprintf(talkFile, "\\hline\n");
   TGraph** LQGraphs = new TGraph*[modelVector.size()];
   for(unsigned int k=0; k<modelVector.size(); k++){
      bool isFractional = false;if(modelVector[k].find("1o3")!=string::npos || modelVector[k].find("2o3")!=string::npos)isFractional = true;
      if(!isFractional) continue;//skip q>=1 charged suppressed models
      LQGraphs[k] = MakePlot(pFile,talkFile,LQPattern,modelVector[k], 2, modelMap[modelVector[k]], LInt);
   }
   fprintf(pFile   ,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(talkFile,"      \\end{tabular}\n\\end{sidewaystable}\n\n");

   fprintf(pFile   ,"\\end{document}\n\n");
   fprintf(talkFile,"\\end{document}\n\n");


   if(SQRTS==8.0){
      fprintf(pFile,"%%TKONLY\n");
      fprintf(pFile,"Sample & Mass  & Cut   & \\multicolumn{4}{c|}{$\\sqrt{s}=7TeV$} & \\multicolumn{4}{c|}{$\\sqrt{s}=8TeV$} & \\multicolumn{2}{c|}{$\\sqrt{s}=7+8TeV$} \\\\\\hline\n");
      fprintf(pFile,"       & (GeV) & (GeV) & Eff & $\\sigma_{TH}$ & $\\sigma_{obs}$ & $\\sigma_{pred}$ & Eff & $\\sigma_{TH}$ & $\\sigma_{obs}$ & $\\sigma_{pred}$ & $\\mu_{obs}$ & $\\mu_{pred}$ \\\\\\hline\n");
      for(unsigned int k=0; k<modelVector.size(); k++){printSummary(pFile, talkFile, TkPattern , modelVector[k], modelMap[modelVector[k]]); }
      fprintf(pFile,"\\hline\n\n\n");

      fprintf(pFile,"%%TKTOF\n");
      fprintf(pFile,"Sample & Mass  & Cut   & \\multicolumn{4}{c|}{$\\sqrt{s}=7TeV$} & \\multicolumn{4}{c|}{$\\sqrt{s}=8TeV$} & \\multicolumn{2}{c|}{$\\sqrt{s}=7+8TeV$} \\\\\\hline\n");
      fprintf(pFile,"       & (GeV) & (GeV) & Eff & $\\sigma_{TH}$ & $\\sigma_{obs}$ & $\\sigma_{pred}$ & Eff & $\\sigma_{TH}$ & $\\sigma_{obs}$ & $\\sigma_{pred}$ & $\\mu_{obs}$ & $\\mu_{pred}$ \\\\\\hline\n");
      for(unsigned int k=0; k<modelVector.size(); k++){printSummary(pFile, talkFile, MuPattern , modelVector[k], modelMap[modelVector[k]]); }
      fprintf(pFile,"\\hline\n\n\n");

      fprintf(pFile,"%%MUONLY\n");
      fprintf(pFile,"Sample & Mass  & Cut   & \\multicolumn{4}{c|}{$\\sqrt{s}=7TeV$} & \\multicolumn{4}{c|}{$\\sqrt{s}=8TeV$} & \\multicolumn{2}{c|}{$\\sqrt{s}=7+8TeV$} \\\\\\hline\n");
      fprintf(pFile,"       & (GeV) & (GeV) & Eff & $\\sigma_{TH}$ & $\\sigma_{obs}$ & $\\sigma_{pred}$ & Eff & $\\sigma_{TH}$ & $\\sigma_{obs}$ & $\\sigma_{pred}$ & $\\mu_{obs}$ & $\\mu_{pred}$ \\\\\\hline\n");
      for(unsigned int k=0; k<modelVector.size(); k++){printSummary(pFile, talkFile, MOPattern , modelVector[k], modelMap[modelVector[k]]); }
      fprintf(pFile,"\\hline\n\n\n");

      fprintf(pFile,"%%Q>1\n");
      fprintf(pFile,"Sample & Mass  & Cut   & \\multicolumn{4}{c|}{$\\sqrt{s}=7TeV$} & \\multicolumn{4}{c|}{$\\sqrt{s}=8TeV$} & \\multicolumn{2}{c|}{$\\sqrt{s}=7+8TeV$} \\\\\\hline\n");
      fprintf(pFile,"       & (GeV) & (GeV) & Eff & $\\sigma_{TH}$ & $\\sigma_{obs}$ & $\\sigma_{pred}$ & Eff & $\\sigma_{TH}$ & $\\sigma_{obs}$ & $\\sigma_{pred}$ & $\\mu_{obs}$ & $\\mu_{pred}$ \\\\\\hline\n");
      for(unsigned int k=0; k<modelVector.size(); k++){printSummary(pFile, talkFile, HQPattern , modelVector[k], modelMap[modelVector[k]]); }
      fprintf(pFile,"\\hline\n\n\n");

      fprintf(pFile,"%%Q<1\n");
      fprintf(pFile,"Sample & Mass  & Cut   & \\multicolumn{4}{c|}{$\\sqrt{s}=7TeV$} & \\multicolumn{4}{c|}{$\\sqrt{s}=8TeV$} & \\multicolumn{2}{c|}{$\\sqrt{s}=7+8TeV$} \\\\\\hline\n");
      fprintf(pFile,"       & (GeV) & (GeV) & Eff & $\\sigma_{TH}$ & $\\sigma_{obs}$ & $\\sigma_{pred}$ & Eff & $\\sigma_{TH}$ & $\\sigma_{obs}$ & $\\sigma_{pred}$ & $\\mu_{obs}$ & $\\mu_{pred}$ \\\\\\hline\n");
      for(unsigned int k=0; k<modelVector.size(); k++){printSummary(pFile, talkFile, LQPattern , modelVector[k], modelMap[modelVector[k]]); }
      fprintf(pFile,"\\hline\n\n\n");
   }


   //print a table with all uncertainty on signal efficiency

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* TkSystGraphs = new TMultiGraph();

   LEG = new TLegend(0.55,0.75,0.80,0.90);
   LEG->SetFillColor(0);
   LEG->SetFillStyle(0);
   LEG->SetBorderSize(0);

   fprintf(pFile   ,"\n\n %20s \n\n", LegendFromType(TkPattern).c_str());
   fprintf(pFile   ,          "%20s    Eff   --> PScale |  DeDxScale | PUScale | TotalUncertainty     \n","Model");
   fprintf(talkFile, "\\hline\n%20s &  Eff     & PScale &  DeDxScale & PUScale & TotalUncertainty \\\\\n","Model");
   int Graphs=0;

   for(unsigned int k=0; k<modelVector.size(); k++){
     TGraph* Uncertainty = CheckSignalUncertainty(pFile,talkFile,TkPattern, modelVector[k], modelMap[modelVector[k]]);
     if(Uncertainty!=NULL && useSample(0, modelVector[k])) {
       Uncertainty->SetLineColor(Color[Graphs]);  Uncertainty->SetMarkerColor(Color[Graphs]);   Uncertainty->SetMarkerStyle(20); Uncertainty->SetLineWidth(2);
       TkSystGraphs->Add(Uncertainty,"C");
       LEG->AddEntry(Uncertainty,  modelVector[k].c_str() ,"L");
       Graphs++;
     }
   }
   
   if(Graphs>0) {
   TkSystGraphs->Draw("A");
   TkSystGraphs->SetTitle("");
   TkSystGraphs->GetXaxis()->SetTitle("Mass (GeV)");
   TkSystGraphs->GetYaxis()->SetTitle("Relative Uncertainty");
   TkSystGraphs->GetYaxis()->SetTitleOffset(1.70);
   TkSystGraphs->GetYaxis()->SetRangeUser(0., 0.35);
   TkSystGraphs->GetYaxis()->SetNdivisions(520, "X");

   LEG->Draw();
   c1->SetLogy(false);
   c1->SetGridy(false);

   DrawPreliminary(LegendFromType(InputPattern).c_str(), SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,"Results/"+SHAPESTRING+EXCLUSIONDIR+"/", "TkUncertainty");
   delete c1;
   delete TkSystGraphs;
   delete LEG;
   }

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MuSystGraphs = new TMultiGraph();

   LEG = new TLegend(0.55,0.75,0.80,0.90);
   LEG->SetFillColor(0);
   LEG->SetFillStyle(0);
   LEG->SetBorderSize(0);

   fprintf(pFile   ,"\n\n %20s \n\n", LegendFromType(MuPattern).c_str());
   fprintf(pFile,             "%20s   Eff   --> PScale |  DeDxScale | PUScale | TOFScale | TotalUncertainty     \n","Model");
   fprintf(talkFile, "\\hline\n%20s &  Eff    & PScale &  DeDxScale & PUScale & TOFScale & TotalUncertainty \\\\\n","Model");
   Graphs=0;
   for(unsigned int k=0; k<modelVector.size(); k++){
     TGraph* Uncertainty = CheckSignalUncertainty(pFile,talkFile,MuPattern, modelVector[k], modelMap[modelVector[k]]);

     if(Uncertainty!=NULL && useSample(2, modelVector[k])) {
       Uncertainty->SetLineColor(Color[Graphs]);  Uncertainty->SetMarkerColor(Color[Graphs]);   Uncertainty->SetMarkerStyle(20); Uncertainty->SetLineWidth(2);
       MuSystGraphs->Add(Uncertainty,"C");
       LEG->AddEntry(Uncertainty,  modelVector[k].c_str() ,"L");
       Graphs++;
     }
   }

   if(Graphs>0) {
   MuSystGraphs->Draw("A");
   MuSystGraphs->SetTitle("");
   MuSystGraphs->GetXaxis()->SetTitle("Mass (GeV)");
   MuSystGraphs->GetYaxis()->SetTitle("Relative Uncertainty");
   MuSystGraphs->GetYaxis()->SetTitleOffset(1.70);
   MuSystGraphs->GetYaxis()->SetRangeUser(0., 0.35);
   MuSystGraphs->GetYaxis()->SetNdivisions(520, "X");

   LEG->Draw();
   c1->SetLogy(false);
   c1->SetGridy(false);

   DrawPreliminary(LegendFromType(InputPattern).c_str(), SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,"Results/"+SHAPESTRING+EXCLUSIONDIR+"/", "MuUncertainty");
   delete c1;
   delete MuSystGraphs;
   delete LEG;
   }

   fprintf(pFile   ,"\n\n %20s \n\n", LegendFromType(MOPattern).c_str());
   fprintf(pFile,             "%20s   Eff   --> PScale |  DeDxScale | PUScale | TOFScale | TotalUncertainty     \n","Model");
   fprintf(talkFile, "\\hline\n%20s &  Eff    & PScale &  DeDxScale & PUScale & TOFScale & TotalUncertainty \\\\\n","Model");

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MOSystGraphs = new TMultiGraph();

   LEG = new TLegend(0.55,0.75,0.80,0.90);
   LEG->SetFillColor(0);
   LEG->SetFillStyle(0);
   LEG->SetBorderSize(0);

   Graphs=0;
   for(unsigned int k=0; k<modelVector.size(); k++){
     TGraph* Uncertainty = CheckSignalUncertainty(pFile,talkFile,MOPattern, modelVector[k], modelMap[modelVector[k]]);
     if(Uncertainty!=NULL && useSample(3, modelVector[k])) {
       Uncertainty->SetLineColor(Color[Graphs]);  Uncertainty->SetMarkerColor(Color[Graphs]);   Uncertainty->SetMarkerStyle(20); Uncertainty->SetLineWidth(2);
       MOSystGraphs->Add(Uncertainty,"C");
       LEG->AddEntry(Uncertainty,  modelVector[k].c_str() ,"L");
       Graphs++;
     }
   }

   if(Graphs>0) {
   MOSystGraphs->Draw("A");
   MOSystGraphs->SetTitle("");
   MOSystGraphs->GetXaxis()->SetTitle("Mass (GeV)");
   MOSystGraphs->GetYaxis()->SetTitle("Relative Uncertainty");
   MOSystGraphs->GetYaxis()->SetTitleOffset(1.70);
   MOSystGraphs->GetYaxis()->SetRangeUser(0., 0.35);
   MOSystGraphs->GetYaxis()->SetNdivisions(520, "X");

   LEG->Draw();
   c1->SetLogy(false);
   c1->SetGridy(false);

   DrawPreliminary(LegendFromType(InputPattern).c_str(), SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,"Results/"+SHAPESTRING+EXCLUSIONDIR+"/", "MOUncertainty");
   delete c1;
   delete MOSystGraphs;
   delete LEG;
   }

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* LQSystGraphs = new TMultiGraph();

   LEG = new TLegend(0.55,0.75,0.80,0.90);
   LEG->SetFillColor(0);
   LEG->SetFillStyle(0);
   LEG->SetBorderSize(0);

   fprintf(pFile   ,"\n\n %20s \n\n", LegendFromType(LQPattern).c_str());
   fprintf(pFile   ,          "%20s    Eff   --> PScale |  DeDxScale | PUScale | TotalUncertainty     \n","Model");
   fprintf(talkFile, "\\hline\n%20s &  Eff     & PScale &  DeDxScale & PUScale & TotalUncertainty \\\\\n","Model");

   Graphs=0;
   for(unsigned int k=0; k<modelVector.size(); k++){
     TGraph* Uncertainty = CheckSignalUncertainty(pFile,talkFile,LQPattern, modelVector[k], modelMap[modelVector[k]]);
     if(Uncertainty!=NULL && useSample(5, modelVector[k])) {
       Uncertainty->SetLineColor(Color[Graphs]);  Uncertainty->SetMarkerColor(Color[Graphs]);   Uncertainty->SetMarkerStyle(20); Uncertainty->SetLineWidth(2);
       LQSystGraphs->Add(Uncertainty,"C");
       LEG->AddEntry(Uncertainty,  modelVector[k].c_str() ,"L");
       Graphs++;
     }
   }

   if(Graphs>0) {
   LQSystGraphs->Draw("A");
   LQSystGraphs->SetTitle("");
   LQSystGraphs->GetXaxis()->SetTitle("Mass (GeV)");
   LQSystGraphs->GetYaxis()->SetTitle("Relative Uncertainty");
   LQSystGraphs->GetYaxis()->SetTitleOffset(1.70);
   LQSystGraphs->GetYaxis()->SetRangeUser(0., 0.35);
   LQSystGraphs->GetYaxis()->SetNdivisions(520, "X");

   LEG->Draw();
   c1->SetLogy(false);
   c1->SetGridy(false);

   DrawPreliminary(LegendFromType(InputPattern).c_str(), SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c1,"Results/"+SHAPESTRING+EXCLUSIONDIR+"/", "LQUncertainty");
   delete c1;
   delete LQSystGraphs;
   delete LEG;
   }

   //Get Theoretical xsection and error bands
   TGraph** ThXSec    = new TGraph*[modelVector.size()];
   TCutG ** ThXSecErr = new TCutG* [modelVector.size()];
   for(unsigned int k=0; k<modelVector.size(); k++){
     if(modelVector[k].find("Gluino")!=string::npos){
         if(SQRTS==7){
            ThXSec   [k] = new TGraph(sizeof(THXSEC7TeV_Gluino_Mass)/sizeof(double),THXSEC7TeV_Gluino_Mass,THXSEC7TeV_Gluino_Cen);
            ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr",sizeof(THXSEC7TeV_Gluino_Mass)/sizeof(double),THXSEC7TeV_Gluino_Mass,THXSEC7TeV_Gluino_Low,THXSEC7TeV_Gluino_High, PlotMinScale, PlotMaxScale);
         }else if(SQRTS==8){
            ThXSec   [k] = new TGraph(sizeof(THXSEC8TeV_Gluino_Mass)/sizeof(double),THXSEC8TeV_Gluino_Mass,THXSEC8TeV_Gluino_Cen);
            ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr",sizeof(THXSEC8TeV_Gluino_Mass)/sizeof(double),THXSEC8TeV_Gluino_Mass,THXSEC8TeV_Gluino_Low,THXSEC8TeV_Gluino_High, PlotMinScale, PlotMaxScale);
         }
	 else {
	   const int NMass=sizeof(THXSEC8TeV_Gluino_Mass)/sizeof(double);
	   double ones[NMass];
	   for(int i=0; i<NMass; i++) ones[i]=1;
	   ThXSec   [k] = new TGraph(NMass,THXSEC8TeV_Gluino_Mass,ones);
	 }
      }else if(modelVector[k].find("Stop"  )!=string::npos){
         if(SQRTS==7){
            ThXSec   [k] = new TGraph(sizeof(THXSEC7TeV_Stop_Mass)/sizeof(double),THXSEC7TeV_Stop_Mass,THXSEC7TeV_Stop_Cen);
            ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr",sizeof(THXSEC7TeV_Stop_Mass)/sizeof(double),THXSEC7TeV_Stop_Mass,THXSEC7TeV_Stop_Low,THXSEC7TeV_Stop_High, PlotMinScale, PlotMaxScale);
         }else if(SQRTS==8){
            ThXSec   [k] = new TGraph(sizeof(THXSEC8TeV_Stop_Mass)/sizeof(double),THXSEC8TeV_Stop_Mass,THXSEC8TeV_Stop_Cen);
            ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr",sizeof(THXSEC8TeV_Stop_Mass)/sizeof(double),THXSEC8TeV_Stop_Mass,THXSEC8TeV_Stop_Low,THXSEC8TeV_Stop_High, PlotMinScale, PlotMaxScale);
         }
         else {
           const int NMass=sizeof(THXSEC8TeV_Stop_Mass)/sizeof(double);
           double ones[NMass];
           for(int i=0; i<NMass; i++) ones[i]=1;
           ThXSec   [k] = new TGraph(NMass,THXSEC8TeV_Stop_Mass,ones);
         }
      }else if(modelVector[k].find("GMStau"  )!=string::npos){
         if(SQRTS==7){
            ThXSec   [k] = MakePlot(NULL, NULL, TkPattern,modelVector[k], 0, modelMap[modelVector[k]], LInt); 
            ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr", sizeof(THXSEC7TeV_GMStau_Mass)/sizeof(double),THXSEC7TeV_GMStau_Mass,THXSEC7TeV_GMStau_Low,THXSEC7TeV_GMStau_High, PlotMinScale, PlotMaxScale); 
         }else if(SQRTS==8){
            ThXSec   [k] = MakePlot(NULL, NULL, TkPattern,modelVector[k], 0, modelMap[modelVector[k]], LInt);
            ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr", sizeof(THXSEC8TeV_GMStau_Mass)/sizeof(double),THXSEC8TeV_GMStau_Mass,THXSEC8TeV_GMStau_Low,THXSEC8TeV_GMStau_High, PlotMinScale, PlotMaxScale);
         }
         else {
           const int NMass=sizeof(THXSEC8TeV_GMStau_Mass)/sizeof(double);
           double ones[NMass];
           for(int i=0; i<NMass; i++) ones[i]=1;
           ThXSec   [k] = new TGraph(NMass,THXSEC8TeV_GMStau_Mass,ones);
         }
      }else if(modelVector[k].find("PPStau"  )!=string::npos){
         if(SQRTS==7){
            ThXSec   [k] = MakePlot(NULL, NULL, TkPattern,modelVector[k], 0, modelMap[modelVector[k]], LInt);   
            ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr", sizeof(THXSEC7TeV_PPStau_Mass)/sizeof(double),THXSEC7TeV_PPStau_Mass,THXSEC7TeV_PPStau_Low,THXSEC7TeV_PPStau_High, PlotMinScale, PlotMaxScale); 
         }else if(SQRTS==8){
            ThXSec   [k] = MakePlot(NULL, NULL, TkPattern,modelVector[k], 0, modelMap[modelVector[k]], LInt);
            ThXSecErr[k] = GetErrorBand(modelVector[k]+"ThErr", sizeof(THXSEC8TeV_PPStau_Mass)/sizeof(double),THXSEC8TeV_PPStau_Mass,THXSEC8TeV_PPStau_Low,THXSEC8TeV_PPStau_High, PlotMinScale, PlotMaxScale);
         }
         else {
           const int NMass=sizeof(THXSEC8TeV_PPStau_Mass)/sizeof(double);
           double ones[NMass];
           for(int i=0; i<NMass; i++) ones[i]=1;
           ThXSec   [k] = new TGraph(NMass,THXSEC8TeV_PPStau_Mass,ones);
         }
     }else{
         if(modelVector[k].find("o3")!=string::npos){
            ThXSec   [k] = MakePlot(NULL, NULL, LQPattern,modelVector[k], 0, modelMap[modelVector[k]], LInt);
         }else{
            ThXSec   [k] = MakePlot(NULL, NULL, TkPattern,modelVector[k], 0, modelMap[modelVector[k]], LInt);
         }
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
      if(TkGraphs[k]->GetN()==0) continue;
      if(TkGraphs[k]->GetX()[TkGraphs[k]->GetN()-1]<0) continue;
      fprintf(pFile,"%20s --> Excluded mass below %8.3fGeV\n", modelVector[k].c_str(), FindIntersectionBetweenTwoGraphs(TkGraphs[k],  ThXSec[k], TkGraphs[k]->GetX()[0], TkGraphs[k]->GetX()[TkGraphs[k]->GetN()-1], 1, 0.00));
   }

   fprintf(pFile,"-----------------------\n0%% MU+TOF        \n-------------------------\n");
   for(unsigned int k=0; k<modelVector.size(); k++){
      bool isNeutral = false;if(modelVector[k].find("GluinoN")!=string::npos || modelVector[k].find("StopN")!=string::npos)isNeutral = true;
      if(isNeutral) continue;//skip charged suppressed models
      if(MuGraphs[k]->GetN()==0) continue;
      if(MuGraphs[k]->GetX()[MuGraphs[k]->GetN()-1]<0) continue;
      fprintf(pFile,"%20s --> Excluded mass below %8.3fGeV\n", modelVector[k].c_str(), FindIntersectionBetweenTwoGraphs(MuGraphs[k],  ThXSec[k], MuGraphs[k]->GetX()[0], MuGraphs[k]->GetX()[MuGraphs[k]->GetN()-1], 1, 0.00));
   }  

   fprintf(pFile,"-----------------------\n0%% MU+Only        \n-------------------------\n");
   for(unsigned int k=0; k<modelVector.size(); k++){
      bool isNeutral = false;if(modelVector[k].find("GluinoN")!=string::npos || modelVector[k].find("StopN")!=string::npos)isNeutral = true;
      if(isNeutral) continue;//skip charged suppressed models
      if(MOGraphs[k]->GetN()==0) continue;
      if(MOGraphs[k]->GetX()[MOGraphs[k]->GetN()-1]<0) continue;
      fprintf(pFile,"%20s --> Excluded mass below %8.3fGeV\n", modelVector[k].c_str(), FindIntersectionBetweenTwoGraphs(MOGraphs[k],  ThXSec[k], MOGraphs[k]->GetX()[0], MOGraphs[k]->GetX()[MOGraphs[k]->GetN()-1], 1, 0.00));
   }


   fprintf(pFile,"-----------------------\n0%% Q>1            \n-------------------------\n");
   for(unsigned int k=0; k<modelVector.size(); k++){
      if(modelVector[k].find("DY")==string::npos)continue;
      bool isFractional = false;if(modelVector[k].find("1o3")!=string::npos || modelVector[k].find("2o3")!=string::npos)isFractional = true;
      if(isFractional) continue;//skip non fractional charge models
      if(HQGraphs[k]->GetN()==0) continue;
      if(HQGraphs[k]->GetX()[HQGraphs[k]->GetN()-1]<0) continue;
      fprintf(pFile,"%20s --> Excluded mass below %8.3fGeV\n", modelVector[k].c_str(), FindIntersectionBetweenTwoGraphs(HQGraphs[k],  ThXSec[k], HQGraphs[k]->GetX()[0], HQGraphs[k]->GetX()[HQGraphs[k]->GetN()-1], 1, 0.00));
   }


   fprintf(pFile,"-----------------------\n0%% Q<1             \n-------------------------\n");
   for(unsigned int k=0; k<modelVector.size(); k++){
      bool isFractional = false;if(modelVector[k].find("1o3")!=string::npos || modelVector[k].find("2o3")!=string::npos)isFractional = true;
      if(!isFractional) continue;//skip non fractional charge models
      if(LQGraphs[k]->GetN()==0) continue;
      if(LQGraphs[k]->GetX()[LQGraphs[k]->GetN()-1]<0) continue;
      fprintf(pFile,"%20s --> Excluded mass below %8.3fGeV\n", modelVector[k].c_str(), FindIntersectionBetweenTwoGraphs(LQGraphs[k],  ThXSec[k], LQGraphs[k]->GetX()[0], LQGraphs[k]->GetX()[LQGraphs[k]->GetN()-1], 1, 0.00));
   }






   fclose(pFile);


   //Make the final plot with all curves in it
   // I don't like much this part because it is dependent of what is in Analysis_Samples.h in an hardcoded way   
   std::map<string, TGraph*> TkGraphMap;
   std::map<string, TGraph*> MuGraphMap;
   std::map<string, TGraph*> LQGraphMap;
   std::map<string, TGraph*> MOGraphMap;
   std::map<string, TGraph*> ThGraphMap;
   std::map<string, TCutG* > ThErrorMap;
   for(unsigned int k=0; k<modelVector.size(); k++){
      TkGraphMap[modelVector[k]] = TkGraphs [k];
      MuGraphMap[modelVector[k]] = MuGraphs [k];
      LQGraphMap[modelVector[k]] = LQGraphs [k];
      MOGraphMap[modelVector[k]] = MOGraphs [k];
      ThGraphMap[modelVector[k]] = ThXSec   [k];
      ThErrorMap[modelVector[k]] = ThXSecErr[k];
   }

   ThGraphMap["Gluino_f10"   ]->SetLineColor(4);  ThGraphMap["Gluino_f10"   ]->SetMarkerColor(4);   ThGraphMap["Gluino_f10"   ]->SetLineWidth(1);   ThGraphMap["Gluino_f10"   ]->SetLineStyle(1);  ThGraphMap["Gluino_f10"   ]->SetMarkerStyle(1);
   MuGraphMap["Gluino_f10"   ]->SetLineColor(4);  MuGraphMap["Gluino_f10"   ]->SetMarkerColor(4);   MuGraphMap["Gluino_f10"   ]->SetLineWidth(2);   MuGraphMap["Gluino_f10"   ]->SetLineStyle(1);  MuGraphMap["Gluino_f10"   ]->SetMarkerStyle(22);
   MuGraphMap["Gluino_f50"   ]->SetLineColor(4);  MuGraphMap["Gluino_f50"   ]->SetMarkerColor(4);   MuGraphMap["Gluino_f50"   ]->SetLineWidth(2);   MuGraphMap["Gluino_f50"   ]->SetLineStyle(1);  MuGraphMap["Gluino_f50"   ]->SetMarkerStyle(23);
   TkGraphMap["Gluino_f10"   ]->SetLineColor(4);  TkGraphMap["Gluino_f10"   ]->SetMarkerColor(4);   TkGraphMap["Gluino_f10"   ]->SetLineWidth(2);   TkGraphMap["Gluino_f10"   ]->SetLineStyle(1);  TkGraphMap["Gluino_f10"   ]->SetMarkerStyle(22);
   TkGraphMap["Gluino_f50"   ]->SetLineColor(4);  TkGraphMap["Gluino_f50"   ]->SetMarkerColor(4);   TkGraphMap["Gluino_f50"   ]->SetLineWidth(2);   TkGraphMap["Gluino_f50"   ]->SetLineStyle(1);  TkGraphMap["Gluino_f50"   ]->SetMarkerStyle(23);
   TkGraphMap["GluinoN_f10"  ]->SetLineColor(4);  TkGraphMap["GluinoN_f10"  ]->SetMarkerColor(4);   TkGraphMap["GluinoN_f10"  ]->SetLineWidth(2);   TkGraphMap["GluinoN_f10"  ]->SetLineStyle(1);  TkGraphMap["GluinoN_f10"  ]->SetMarkerStyle(26);
   MOGraphMap["Gluino_f10"   ]->SetLineColor(4);  MOGraphMap["Gluino_f10"   ]->SetMarkerColor(4);   MOGraphMap["Gluino_f10"   ]->SetLineWidth(2);   MOGraphMap["Gluino_f10"   ]->SetLineStyle(1);  MOGraphMap["Gluino_f10"   ]->SetMarkerStyle(22);
   MOGraphMap["Gluino_f50"   ]->SetLineColor(4);  MOGraphMap["Gluino_f50"   ]->SetMarkerColor(4);   MOGraphMap["Gluino_f50"   ]->SetLineWidth(2);   MOGraphMap["Gluino_f50"   ]->SetLineStyle(1);  MOGraphMap["Gluino_f50"   ]->SetMarkerStyle(23);
   MOGraphMap["Gluino_f100"  ]->SetLineColor(4);  MOGraphMap["Gluino_f100"  ]->SetMarkerColor(4);   MOGraphMap["Gluino_f100"  ]->SetLineWidth(2);   MOGraphMap["Gluino_f100"  ]->SetLineStyle(1);  MOGraphMap["Gluino_f100"  ]->SetMarkerStyle(26);
   ThGraphMap["Stop"         ]->SetLineColor(2);  ThGraphMap["Stop"         ]->SetMarkerColor(2);   ThGraphMap["Stop"         ]->SetLineWidth(1);   ThGraphMap["Stop"         ]->SetLineStyle(2);  ThGraphMap["Stop"         ]->SetMarkerStyle(1);
   MuGraphMap["Stop"         ]->SetLineColor(2);  MuGraphMap["Stop"         ]->SetMarkerColor(2);   MuGraphMap["Stop"         ]->SetLineWidth(2);   MuGraphMap["Stop"         ]->SetLineStyle(1);  MuGraphMap["Stop"         ]->SetMarkerStyle(21);
   TkGraphMap["Stop"         ]->SetLineColor(2);  TkGraphMap["Stop"         ]->SetMarkerColor(2);   TkGraphMap["Stop"         ]->SetLineWidth(2);   TkGraphMap["Stop"         ]->SetLineStyle(1);  TkGraphMap["Stop"         ]->SetMarkerStyle(21);
   TkGraphMap["StopN"        ]->SetLineColor(2);  TkGraphMap["StopN"        ]->SetMarkerColor(2);   TkGraphMap["StopN"        ]->SetLineWidth(2);   TkGraphMap["StopN"        ]->SetLineStyle(1);  TkGraphMap["StopN"        ]->SetMarkerStyle(25);
   MOGraphMap["Stop"         ]->SetLineColor(2);  MOGraphMap["Stop"         ]->SetMarkerColor(2);   MOGraphMap["Stop"         ]->SetLineWidth(2);   MOGraphMap["Stop"         ]->SetLineStyle(1);  MOGraphMap["Stop"         ]->SetMarkerStyle(21);
   ThGraphMap["GMStau"       ]->SetLineColor(1);  ThGraphMap["GMStau"       ]->SetMarkerColor(1);   ThGraphMap["GMStau"       ]->SetLineWidth(1);   ThGraphMap["GMStau"       ]->SetLineStyle(3);  ThGraphMap["GMStau"       ]->SetMarkerStyle(1);
   ThGraphMap["PPStau"       ]->SetLineColor(6);  ThGraphMap["PPStau"       ]->SetMarkerColor(6);   ThGraphMap["PPStau"       ]->SetLineWidth(1);   ThGraphMap["PPStau"       ]->SetLineStyle(4);  ThGraphMap["PPStau"       ]->SetMarkerStyle(1);
//   ThGraphMap["DCRho08HyperK"]->SetLineColor(4);  ThGraphMap["DCRho08HyperK"]->SetMarkerColor(4);   ThGraphMap["DCRho08HyperK"]->SetLineWidth(1);   ThGraphMap["DCRho08HyperK"]->SetLineStyle(3);  ThGraphMap["DCRho08HyperK"]->SetMarkerStyle(1);
   ThGraphMap["DCRho12HyperK"]->SetLineColor(2);  ThGraphMap["DCRho12HyperK"]->SetMarkerColor(2);   ThGraphMap["DCRho12HyperK"]->SetLineWidth(1);   ThGraphMap["DCRho12HyperK"]->SetLineStyle(2);  ThGraphMap["DCRho12HyperK"]->SetMarkerStyle(1);
   ThGraphMap["DCRho16HyperK"]->SetLineColor(1);  ThGraphMap["DCRho16HyperK"]->SetMarkerColor(1);   ThGraphMap["DCRho16HyperK"]->SetLineWidth(1);   ThGraphMap["DCRho16HyperK"]->SetLineStyle(1);  ThGraphMap["DCRho16HyperK"]->SetMarkerStyle(1);
   MuGraphMap["GMStau"       ]->SetLineColor(1);  MuGraphMap["GMStau"       ]->SetMarkerColor(1);   MuGraphMap["GMStau"       ]->SetLineWidth(2);   MuGraphMap["GMStau"       ]->SetLineStyle(1);  MuGraphMap["GMStau"       ]->SetMarkerStyle(23);
   MuGraphMap["PPStau"       ]->SetLineColor(6);  MuGraphMap["PPStau"       ]->SetMarkerColor(6);   MuGraphMap["PPStau"       ]->SetLineWidth(2);   MuGraphMap["PPStau"       ]->SetLineStyle(1);  MuGraphMap["PPStau"       ]->SetMarkerStyle(23);
//   MuGraphMap["DCRho08HyperK"]->SetLineColor(4);  MuGraphMap["DCRho08HyperK"]->SetMarkerColor(4);   MuGraphMap["DCRho08HyperK"]->SetLineWidth(2);   MuGraphMap["DCRho08HyperK"]->SetLineStyle(1);  MuGraphMap["DCRho08HyperK"]->SetMarkerStyle(22);
   MuGraphMap["DCRho12HyperK"]->SetLineColor(2);  MuGraphMap["DCRho12HyperK"]->SetMarkerColor(2);   MuGraphMap["DCRho12HyperK"]->SetLineWidth(2);   MuGraphMap["DCRho12HyperK"]->SetLineStyle(1);  MuGraphMap["DCRho12HyperK"]->SetMarkerStyle(23);
   MuGraphMap["DCRho16HyperK"]->SetLineColor(1);  MuGraphMap["DCRho16HyperK"]->SetMarkerColor(1);   MuGraphMap["DCRho16HyperK"]->SetLineWidth(2);   MuGraphMap["DCRho16HyperK"]->SetLineStyle(1);  MuGraphMap["DCRho16HyperK"]->SetMarkerStyle(26);
   TkGraphMap["GMStau"       ]->SetLineColor(1);  TkGraphMap["GMStau"       ]->SetMarkerColor(1);   TkGraphMap["GMStau"       ]->SetLineWidth(2);   TkGraphMap["GMStau"       ]->SetLineStyle(1);  TkGraphMap["GMStau"       ]->SetMarkerStyle(20);
   TkGraphMap["PPStau"       ]->SetLineColor(6);  TkGraphMap["PPStau"       ]->SetMarkerColor(6);   TkGraphMap["PPStau"       ]->SetLineWidth(2);   TkGraphMap["PPStau"       ]->SetLineStyle(1);  TkGraphMap["PPStau"       ]->SetMarkerStyle(20);
//   TkGraphMap["DCRho08HyperK"]->SetLineColor(4);  TkGraphMap["DCRho08HyperK"]->SetMarkerColor(4);   TkGraphMap["DCRho08HyperK"]->SetLineWidth(2);   TkGraphMap["DCRho08HyperK"]->SetLineStyle(1);  TkGraphMap["DCRho08HyperK"]->SetMarkerStyle(22);
   TkGraphMap["DCRho12HyperK"]->SetLineColor(2);  TkGraphMap["DCRho12HyperK"]->SetMarkerColor(2);   TkGraphMap["DCRho12HyperK"]->SetLineWidth(2);   TkGraphMap["DCRho12HyperK"]->SetLineStyle(1);  TkGraphMap["DCRho12HyperK"]->SetMarkerStyle(23);
   TkGraphMap["DCRho16HyperK"]->SetLineColor(1);  TkGraphMap["DCRho16HyperK"]->SetMarkerColor(1);   TkGraphMap["DCRho16HyperK"]->SetLineWidth(2);   TkGraphMap["DCRho16HyperK"]->SetLineStyle(1);  TkGraphMap["DCRho16HyperK"]->SetMarkerStyle(26);

   ThGraphMap["DY_Q1o3"      ]->SetLineColor(41); ThGraphMap["DY_Q1o3"      ]->SetMarkerColor(41);  ThGraphMap["DY_Q1o3"      ]->SetLineWidth(1);   ThGraphMap["DY_Q1o3"      ]->SetLineStyle(9);  ThGraphMap["DY_Q1o3"      ]->SetMarkerStyle(1);
   TkGraphMap["DY_Q1o3"      ]->SetLineColor(41); TkGraphMap["DY_Q1o3"      ]->SetMarkerColor(41);  TkGraphMap["DY_Q1o3"      ]->SetLineWidth(2);   TkGraphMap["DY_Q1o3"      ]->SetLineStyle(1);  TkGraphMap["DY_Q1o3"      ]->SetMarkerStyle(33);
   LQGraphMap["DY_Q1o3"      ]->SetLineColor(41); LQGraphMap["DY_Q1o3"      ]->SetMarkerColor(41);  LQGraphMap["DY_Q1o3"      ]->SetLineWidth(2);   LQGraphMap["DY_Q1o3"      ]->SetLineStyle(1);  LQGraphMap["DY_Q1o3"      ]->SetMarkerStyle(33);
   ThGraphMap["DY_Q2o3"      ]->SetLineColor(43); ThGraphMap["DY_Q2o3"      ]->SetMarkerColor(43);  ThGraphMap["DY_Q2o3"      ]->SetLineWidth(1);   ThGraphMap["DY_Q2o3"      ]->SetLineStyle(10); ThGraphMap["DY_Q2o3"      ]->SetMarkerStyle(1);
   TkGraphMap["DY_Q2o3"      ]->SetLineColor(43); TkGraphMap["DY_Q2o3"      ]->SetMarkerColor(43);  TkGraphMap["DY_Q2o3"      ]->SetLineWidth(2);   TkGraphMap["DY_Q2o3"      ]->SetLineStyle(1);  TkGraphMap["DY_Q2o3"      ]->SetMarkerStyle(34);
   LQGraphMap["DY_Q2o3"      ]->SetLineColor(43); LQGraphMap["DY_Q2o3"      ]->SetMarkerColor(43);  LQGraphMap["DY_Q2o3"      ]->SetLineWidth(2);   LQGraphMap["DY_Q2o3"      ]->SetLineStyle(1);  LQGraphMap["DY_Q2o3"      ]->SetMarkerStyle(34);

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGMu = new TMultiGraph();
   if(!Combine) {
   MGMu->Add(ThGraphMap["Gluino_f10" ]      ,"L");
   MGMu->Add(ThGraphMap["Stop"       ]      ,"L");
   MGMu->Add(ThGraphMap["GMStau"     ]      ,"L");
   MGMu->Add(ThGraphMap["PPStau"     ]      ,"L");
   }
   MGMu->Add(MuGraphMap["Gluino_f10" ]      ,"LP");
   MGMu->Add(MuGraphMap["Gluino_f50" ]      ,"LP");
   MGMu->Add(MuGraphMap["Stop"       ]      ,"LP");
   MGMu->Add(MuGraphMap["GMStau"     ]      ,"LP");
   MGMu->Add(MuGraphMap["PPStau"     ]      ,"LP");
   MGMu->Draw("A");

   if(!Combine) {
   ThErrorMap["Gluino_f10"]->Draw("f");
   ThErrorMap["Stop"      ]->Draw("f");
   ThErrorMap["GMStau"    ]->Draw("f");
   ThErrorMap["PPStau"    ]->Draw("f");
   }
   MGMu->Draw("same");
   MGMu->SetTitle("");
   MGMu->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGMu->GetYaxis()->SetTitle(Combine?"#sigma_{obs}/#sigma_{th}":"#sigma (pb)");
   MGMu->GetYaxis()->SetTitleOffset(1.70);
   MGMu->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   MGMu->GetXaxis()->SetRangeUser(50,1550);

   DrawPreliminary("Tracker + TOF", SQRTS, LInt);
   TLegend* LEGMu = !Combine ? new TLegend(0.45,0.58,0.65,0.90) : new TLegend(0.45,0.10,0.65,0.42);
   LEGMu->SetFillColor(0); 
   LEGMu->SetFillStyle(0);
   LEGMu->SetBorderSize(0);
   LEGMu->AddEntry(MuGraphMap["Gluino_f50"] , "gluino; 50% #tilde{g}g"    ,"LP");
   LEGMu->AddEntry(MuGraphMap["Gluino_f10"] , "gluino; 10% #tilde{g}g"    ,"LP");
   LEGMu->AddEntry(MuGraphMap["Stop"      ] , "stop"                      ,"LP");
   LEGMu->AddEntry(MuGraphMap["PPStau"    ] , "Pair Prod. stau"           ,"LP");
   LEGMu->AddEntry(MuGraphMap["GMStau"    ] , "GMSB stau"                 ,"LP");

   TLegend* LEGTh = new TLegend(0.15,0.7,0.48,0.9);
   if(!Combine) {
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
   }
   LEGMu->Draw();
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("MuExclusionLog"));
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGTk = new TMultiGraph();
   if(!Combine) {
   MGTk->Add(ThGraphMap["Gluino_f10" ]     ,"L");
   MGTk->Add(ThGraphMap["Stop"       ]     ,"L");
   MGTk->Add(ThGraphMap["GMStau"     ]     ,"L");
   MGTk->Add(ThGraphMap["PPStau"     ]     ,"L");
   }

   MGTk->Add(TkGraphMap["Gluino_f10" ]     ,"LP");
   MGTk->Add(TkGraphMap["Gluino_f50" ]     ,"LP");
   MGTk->Add(TkGraphMap["GluinoN_f10"]     ,"LP");
   MGTk->Add(TkGraphMap["Stop"       ]     ,"LP");
   MGTk->Add(TkGraphMap["StopN"      ]     ,"LP");
   MGTk->Add(TkGraphMap["GMStau"     ]     ,"LP");
   MGTk->Add(TkGraphMap["PPStau"     ]     ,"LP");
   MGTk->Add(TkGraphMap["DY_Q2o3"    ]     ,"LP");

   MGTk->Draw("A");
   if(!Combine) {
   ThErrorMap["Gluino_f10"]->Draw("f");
   ThErrorMap["Stop"      ]->Draw("f");
   ThErrorMap["GMStau"    ]->Draw("f");
   ThErrorMap["PPStau"    ]->Draw("f");
   }
   MGTk->Draw("same");
   MGTk->SetTitle("");
   MGTk->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGTk->GetYaxis()->SetTitle(Combine?"#sigma_{obs}/#sigma_{th}":"#sigma (pb)");
   MGTk->GetYaxis()->SetTitleOffset(1.70);
   MGTk->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   MGTk->GetXaxis()->SetRangeUser(50,1550);
   
   DrawPreliminary("Tracker - Only", SQRTS, LInt);

   TLegend* LEGTk = !Combine ? new TLegend(0.45,0.58,0.795,0.9) : new TLegend(0.45,0.10,0.795,0.42);
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
   LEGTk->AddEntry(TkGraphMap["DY_Q2o3"    ], "frac. Q=2o3"                       ,"LP");
   if(!Combine) LEGTh->Draw();
   LEGTk->Draw();
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("TkExclusionLog"));
   delete c1;

    c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGDCMu = new TMultiGraph();
   if(!Combine) {
//   MGDCMu->Add(ThGraphMap["DCRho08HyperK"]      ,"L");
   MGDCMu->Add(ThGraphMap["DCRho12HyperK"]      ,"L");
   MGDCMu->Add(ThGraphMap["DCRho16HyperK"]      ,"L");
   }
//   MGDCMu->Add(MuGraphMap["DCRho08HyperK"]      ,"LP");
   MGDCMu->Add(MuGraphMap["DCRho12HyperK"]      ,"LP");
   MGDCMu->Add(MuGraphMap["DCRho16HyperK"]      ,"LP");
   MGDCMu->Draw("A");
   MGDCMu->Draw("same");
   MGDCMu->SetTitle("");
   MGDCMu->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGDCMu->GetYaxis()->SetTitle(Combine?"#sigma_{obs}/#sigma_{th}":"#sigma (pb)");
   MGDCMu->GetYaxis()->SetTitleOffset(1.70);
   MGDCMu->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   MGDCMu->GetXaxis()->SetRangeUser(50,1550);
   DrawPreliminary("Tracker + TOF", SQRTS, LInt);
   
   TLegend* LEGDCMu = new TLegend(0.50,0.65,0.80,0.9);
//   LEGDCMu->SetHeader("Tracker + TOF");
   LEGDCMu->SetFillColor(0); 
   LEGDCMu->SetFillStyle(0);
   LEGDCMu->SetBorderSize(0);
//   LEGDCMu->AddEntry(MuGraphMap["DCRho08HyperK"]   , "Hyper-K, #tilde{#rho} = 0.8 TeV"       ,"LP");
   LEGDCMu->AddEntry(MuGraphMap["DCRho12HyperK"]   , "Hyper-K, #tilde{#rho} = 1.2 TeV"       ,"LP");
   LEGDCMu->AddEntry(MuGraphMap["DCRho16HyperK"]   , "Hyper-K, #tilde{#rho} = 1.6 TeV"       ,"LP");

   TLegend* LEGDCTh = new TLegend(0.15,0.7,0.49,0.9);
   if(!Combine) {
   LEGDCTh->SetHeader("Theoretical Prediction");
   LEGDCTh->SetFillColor(0);
   LEGDCTh->SetFillStyle(0);
   LEGDCTh->SetBorderSize(0);

//   TGraph* DCRho08HyperKThLeg = (TGraph*) ThGraphMap["DCRho08HyperK"]->Clone("DCRho08HyperKThLeg");
//   DCRho08HyperKThLeg->SetFillColor(ThErrorMap["Gluino_f10"]->GetFillColor());
//   LEGDCTh->AddEntry(DCRho08HyperKThLeg   ,"Hyper-K, #tilde{#rho} = 0.8 TeV   (LO)" ,"L");
   TGraph* DCRho12HyperKThLeg = (TGraph*) ThGraphMap["DCRho12HyperK"]->Clone("DCRho12HyperKThLeg");
   DCRho12HyperKThLeg->SetFillColor(ThErrorMap["Gluino_f10"]->GetFillColor());
   LEGDCTh->AddEntry(DCRho12HyperKThLeg   ,"Hyper-K, #tilde{#rho} = 1.2 TeV   (LO)" ,"L");
   TGraph* DCRho16HyperKThLeg = (TGraph*) ThGraphMap["DCRho16HyperK"]->Clone("DCRho16HyperKThLeg");
   DCRho16HyperKThLeg->SetFillColor(ThErrorMap["Gluino_f10"]->GetFillColor());
   LEGDCTh->AddEntry(DCRho16HyperKThLeg   ,"Hyper-K, #tilde{#rho} = 1.6 TeV   (LO)" ,"L");
   LEGDCTh->Draw();
   }

   LEGDCMu->Draw();
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("MuDCExclusionLog"));
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGDCTk = new TMultiGraph();
   if(!Combine) {
//   MGDCTk->Add(ThGraphMap["DCRho08HyperK"]      ,"L");
   MGDCTk->Add(ThGraphMap["DCRho12HyperK"]      ,"L");
   MGDCTk->Add(ThGraphMap["DCRho16HyperK"]      ,"L");
   }
//   MGDCTk->Add(TkGraphMap["DCRho08HyperK"]      ,"LP");
   MGDCTk->Add(TkGraphMap["DCRho12HyperK"]      ,"LP");
   MGDCTk->Add(TkGraphMap["DCRho16HyperK"]      ,"LP");

   MGDCTk->Draw("A");
   MGDCTk->Draw("same");
   MGDCTk->SetTitle("");
   MGDCTk->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGDCTk->GetYaxis()->SetTitle(Combine?"#sigma_{obs}/#sigma_{th}":"#sigma (pb)");
   MGDCTk->GetYaxis()->SetTitleOffset(1.70);
   MGDCTk->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   MGDCTk->GetXaxis()->SetRangeUser(50,1550);
   DrawPreliminary("Tracker - Only", SQRTS, LInt);

   TLegend* LEGDCTk = new TLegend(0.50,0.65,0.80,0.90);
//   LEGDCTk->SetHeader("Tracker - Only");
   LEGDCTk->SetFillColor(0); 
   LEGDCTk->SetFillStyle(0);
   LEGDCTk->SetBorderSize(0);
   LEGDCTk->AddEntry(TkGraphMap["DCRho08HyperK"]   , "Hyper-K, #tilde{#rho} = 0.8 TeV"       ,"LP");
//   LEGDCTk->AddEntry(TkGraphMap["DCRho12HyperK"]   , "Hyper-K, #tilde{#rho} = 1.2 TeV"       ,"LP");
   LEGDCTk->AddEntry(TkGraphMap["DCRho16HyperK"]   , "Hyper-K, #tilde{#rho} = 1.6 TeV"       ,"LP");
   LEGDCTk->Draw();
   LEGDCTh->Draw();

   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("TkDCExclusionLog"));
   delete c1;

   /////////////////////////////// LQ Analysis
   TLegend* LQLEGTh = new TLegend(0.15,0.7,0.48,0.9);
   c1 = new TCanvas("c1", "c1",600,600);
   if(!Combine) {
   LQLEGTh->SetHeader("Theoretical Prediction");
   LQLEGTh->SetFillColor(0);
   LQLEGTh->SetFillStyle(0);
   LQLEGTh->SetBorderSize(0);

   TGraph* DYQ1o3ThLeg = (TGraph*) ThGraphMap["DY_Q1o3"        ]->Clone("DYQ1o3ThLeg");
   DYQ1o3ThLeg->SetFillColor(ThErrorMap["DY_Q1o3"]->GetFillColor());
   LQLEGTh->AddEntry(DYQ1o3ThLeg   ,"Q=1/3   (LO)" ,"LF");
   TGraph* DYQ2o3ThLeg = (TGraph*) ThGraphMap["DY_Q2o3"        ]->Clone("DYQ2o3ThLeg");
   DYQ2o3ThLeg->SetFillColor(ThErrorMap["DY_Q2o3"]->GetFillColor());
   LQLEGTh->AddEntry(DYQ2o3ThLeg   ,"Q=2/3   (LO)" ,"LF");
   LQLEGTh->Draw();
   }

   TMultiGraph* MGLQ = new TMultiGraph();
   if(!Combine) {
   MGLQ->Add(ThGraphMap["DY_Q1o3"    ]     ,"L");
   MGLQ->Add(ThGraphMap["DY_Q2o3"    ]     ,"L");
   }

   MGLQ->Add(LQGraphMap["DY_Q1o3"    ]     ,"LP");
   MGLQ->Add(LQGraphMap["DY_Q2o3"    ]     ,"LP");

   MGLQ->Draw("A");
   if(!Combine) {
   ThErrorMap["DY_Q1o3"   ]->Draw("f");
   ThErrorMap["DY_Q2o3"   ]->Draw("f");
   }
   MGLQ->Draw("same");
   MGLQ->SetTitle("");
   MGLQ->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGLQ->GetYaxis()->SetTitle(Combine?"#sigma_{obs}/#sigma_{th}":"#sigma (pb)");
   MGLQ->GetYaxis()->SetTitleOffset(1.70);
   MGLQ->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   MGLQ->GetXaxis()->SetRangeUser(50,1550);

   DrawPreliminary("frac. charge", SQRTS, LInt);

   TLegend* LEGLQ = !Combine ? new TLegend(0.55,0.80,0.795,0.9) : new TLegend(0.55,0.15,0.795,0.25);
//   LEGLQ->SetHeader("Q<1");
   LEGLQ->SetFillColor(0); 
   LEGLQ->SetFillStyle(0);
   LEGLQ->SetBorderSize(0);
   LEGLQ->AddEntry(TkGraphMap["DY_Q1o3"    ], "Q=1/3"            ,"LP");
   LEGLQ->AddEntry(TkGraphMap["DY_Q2o3"    ], "Q=2/3"            ,"LP");
   if(!Combine) LQLEGTh->Draw();

   LEGLQ->Draw();
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("LQExclusionLog"));
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGMO = new TMultiGraph();
   if(!Combine) {
   MGMO->Add(ThGraphMap["Gluino_f10" ]     ,"L");
   MGMO->Add(ThGraphMap["Stop"       ]     ,"L");
   }

   MGMO->Add(MOGraphMap["Gluino_f10" ]     ,"LP");
   MGMO->Add(MOGraphMap["Gluino_f50" ]     ,"LP");
   MGMO->Add(MOGraphMap["Gluino_f100"]     ,"LP");
   MGMO->Add(MOGraphMap["Stop"       ]     ,"LP");

   MGMO->Draw("A");
   if(!Combine) {
   ThErrorMap["Gluino_f10"]->Draw("f");
   ThErrorMap["Stop"      ]->Draw("f");
   }
   MGMO->Draw("same");
   MGMO->SetTitle("");
   MGMO->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGMO->GetYaxis()->SetTitle(Combine?"#sigma_{obs}/#sigma_{th}":"#sigma (pb)");
   MGMO->GetYaxis()->SetTitleOffset(1.70);
   MGMO->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   MGMO->GetXaxis()->SetRangeUser(50,1550);
   
   DrawPreliminary("Muon - Only", SQRTS, LInt);
   
   TLegend* LEGMO = !Combine ? new TLegend(0.45,0.58,0.795,0.9) : new TLegend(0.45,0.10,0.795,0.42);
   LEGMO->SetFillColor(0); 
   LEGMO->SetFillStyle(0);
   LEGMO->SetBorderSize(0);
   LEGMO->AddEntry(MOGraphMap["Gluino_f100" ], "gluino; 100% #tilde{g}g"            ,"LP");
   LEGMO->AddEntry(MOGraphMap["Gluino_f50" ], "gluino; 50% #tilde{g}g"            ,"LP");
   LEGMO->AddEntry(MOGraphMap["Gluino_f10" ], "gluino; 10% #tilde{g}g"            ,"LP");
   LEGMO->AddEntry(MOGraphMap["Stop"       ], "stop"                              ,"LP");
   if(!Combine) LEGTh->Draw();
   LEGMO->Draw();
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("MOExclusionLog"));
   delete c1;

   return;
}


TGraph* CheckSignalUncertainty(FILE* pFile, FILE* talkFile, string InputPattern, string ModelName, std::vector<stSample>& modelSample){
  int TypeMode = TypeFromPattern(InputPattern);
  string prefix = "BUG";
  switch(TypeMode){
  case 0: prefix   = "Tk"; break;
  case 2: prefix   = "Mu"; break;
  case 3: prefix   = "Mo"; break;
  case 4: prefix   = "HQ"; break;
  case 5: prefix   = "LQ"; break;
  }

   unsigned int N   = 0;

   double* Mass      = new double   [modelSample.size()];
   double* SystP     = new double   [modelSample.size()];
   double* SystI     = new double   [modelSample.size()];
   double* SystPU    = new double   [modelSample.size()];
   double* SystT     = new double   [modelSample.size()];
   double* SystTr    = new double   [modelSample.size()];
   double* SystRe    = new double   [modelSample.size()];
   double* SystTotal = new double   [modelSample.size()];

   for(unsigned int s=0;s<modelSample.size();s++){
      if(modelSample[s].Type!=2)continue;
      bool IsNeutral = (modelSample[s].ModelName().find("N")!=std::string::npos);
      if(TypeMode!=0 && IsNeutral)continue;
      stAllInfo tmp(InputPattern+"/"+SHAPESTRING+EXCLUSIONDIR + "/"+modelSample[s].Name+".txt");
      if(tmp.Eff==0) continue;

      Mass[N]        = tmp.Mass;
      SystP[N]       = (tmp.Eff - tmp.Eff_SYSTP)/tmp.Eff;
      SystI[N]       = (tmp.Eff - tmp.Eff_SYSTI)/tmp.Eff;
      SystPU[N]      = (tmp.Eff - tmp.Eff_SYSTPU)/tmp.Eff;
      SystT[N]       = (tmp.Eff - tmp.Eff_SYSTT)/tmp.Eff;
      SystTr[N]      = 0.05;
      SystRe[N]      = 0.02;

//      double Ptemp=max(SystP[N], 0.0), Itemp=max(SystI[N], 0.0), PUtemp=max(SystPU[N], 0.0), Ttemp=max(SystT[N], 0.0);
      double Ptemp=SystP[N], Itemp=SystI[N], PUtemp=SystPU[N], Ttemp=SystT[N];
      SystTotal[N] = sqrt(Ptemp*Ptemp + Itemp*Itemp + PUtemp*PUtemp + Ttemp*Ttemp + SystTr[N]*SystTr[N] + SystRe[N]*SystRe[N]);

      if(TypeMode==0 || TypeMode==5)fprintf(pFile, "%20s   %7.3f --> %7.3f  |  %7.3f  | %7.3f  | %7.3f\n"        ,modelSample[N].Name.c_str(), tmp.Eff, SystP[N], SystI[N], SystPU[N]           , SystTotal[N]);  
      else          fprintf(pFile, "%20s   %7.3f --> %7.3f  |  %7.3f  | %7.3f  | %7.3f | %7.3f\n",modelSample[N].Name.c_str(), tmp.Eff, SystP[N], SystI[N], SystPU[N], SystT[N], SystTotal[N]);

      //if(TypeMode==0 || TypeMode==5)fprintf(talkFile, "\\hline\n%20s &  %7.1f\\%% & %7.1f\\%%  &  %7.1f\\%%  & %7.1f\\%%  & %7.1f\\%%             \\\\\n",modelSample[N].Name.c_str(), 100.*tmp.Eff, 100.*P, 100.*I, 100.*PU, 100.*sqrt(Ptemp*Ptemp + Itemp*Itemp + PUtemp*PUtemp + Ttemp*Ttemp));	
      //else        fprintf(talkFile, "\\hline\n%20s &  %7.1f\\%% & %7.1f\\%%  &  %7.1f\\%%  & %7.1f\\%%  & %7.1f\\%% & %7.1f\\%% \\\\\n",modelSample[N].Name.c_str(), 100.*tmp.Eff, 100.*P, 100.*I, 100.*PU, 100.*T, 100.*sqrt(Ptemp*Ptemp + Itemp*Itemp + PUtemp*PUtemp + Ttemp*Ttemp));
      N++;
   }

   TGraph* graphSystP = NULL;
   TGraph* graphSystI = NULL;
   TGraph* graphSystPU = NULL;
   TGraph* graphSystT = NULL;
   TGraph* graphSystTr = NULL;
   TGraph* graphSystRe = NULL;
   TGraph* graphSystTotal = NULL;

   if(N>0) {
     TCanvas* c2 = new TCanvas("c2", "c2",600,600);

     graphSystP = new TGraph(N,Mass,SystP);
     graphSystI = new TGraph(N,Mass,SystI);
     graphSystPU = new TGraph(N,Mass,SystPU);
     graphSystT = new TGraph(N,Mass,SystT);
     graphSystTr = new TGraph(N,Mass,SystTr);
     graphSystRe = new TGraph(N,Mass,SystRe);
     graphSystTotal = new TGraph(N,Mass,SystTotal);
     TMultiGraph* SystGraphs = new TMultiGraph();

     graphSystTotal->GetYaxis()->SetTitle("CrossSection ( pb )");
     graphSystTotal->SetLineColor(Color[0]);  graphSystTotal->SetMarkerColor(Color[0]);   graphSystTotal->SetMarkerStyle(20);    graphSystTotal->SetLineWidth(2);
     graphSystP->SetLineColor(Color[1]);      graphSystP->SetMarkerColor(Color[1]);       graphSystP->SetMarkerStyle(Marker[1]); graphSystP->SetLineWidth(2);
     graphSystI->SetLineColor(Color[2]);      graphSystI->SetMarkerColor(Color[2]);       graphSystI->SetMarkerStyle(Marker[2]); graphSystI->SetLineWidth(2);
     graphSystPU->SetLineColor(Color[3]);     graphSystPU->SetMarkerColor(Color[3]);      graphSystPU->SetMarkerStyle(Marker[3]);graphSystPU->SetLineWidth(2);
     graphSystT->SetLineColor(Color[4]);      graphSystT->SetMarkerColor(Color[4]);       graphSystT->SetMarkerStyle(Marker[4]); graphSystT->SetLineWidth(2);
     graphSystTr->SetLineColor(Color[5]);     graphSystTr->SetMarkerColor(Color[5]);      graphSystTr->SetMarkerStyle(Marker[5]);graphSystTr->SetLineWidth(2);
     graphSystRe->SetLineColor(Color[6]);     graphSystRe->SetMarkerColor(Color[6]);      graphSystRe->SetMarkerStyle(Marker[6]);graphSystRe->SetLineWidth(2);
     SystGraphs->Add(graphSystP,"C");

     SystGraphs->Add(graphSystTr,"C");
     SystGraphs->Add(graphSystRe,"C");
     if(TypeMode!=3)SystGraphs->Add(graphSystI,"C");
     SystGraphs->Add(graphSystPU,"C");
     if(TypeMode!=0 && TypeMode!=5)SystGraphs->Add(graphSystT,"C");
     SystGraphs->Add(graphSystTotal,"P");

     SystGraphs->Draw("A");
     SystGraphs->SetTitle("");
     SystGraphs->GetXaxis()->SetTitle("Mass (GeV)");
     SystGraphs->GetYaxis()->SetTitle("Relative Uncertainty");
     SystGraphs->GetYaxis()->SetTitleOffset(1.70);
     SystGraphs->GetYaxis()->SetRangeUser(-0.05, 0.35);
     SystGraphs->GetYaxis()->SetNdivisions(520, "X");

     TLegend* LEG = new TLegend(0.45,0.55,0.80,0.90);
     LEG->SetFillColor(0);
     LEG->SetFillStyle(0);
     LEG->SetBorderSize(0);
     LEG->AddEntry(graphSystTr,  "Trigger" ,"L");
     LEG->AddEntry(graphSystRe,  "Reconstruction" ,"L");
     LEG->AddEntry(graphSystP,  "P" ,"L");
     if(TypeMode!=3)LEG->AddEntry(graphSystI,  "dE/dx" ,"L");
     LEG->AddEntry(graphSystPU,  "Pile Up" ,"L");
     if(TypeMode!=0 && TypeMode!=5)LEG->AddEntry(graphSystT,  "1/#beta" ,"L");
     LEG->AddEntry(graphSystTotal,  "Total" ,"P");
     LEG->Draw();
     c2->SetLogy(false);
     c2->SetGridy(false);

   DrawPreliminary(LegendFromType(InputPattern).c_str(), SQRTS, IntegratedLuminosityFromE(SQRTS));
   SaveCanvas(c2,"Results/"+SHAPESTRING+EXCLUSIONDIR+"/", string(prefix+ ModelName + "Uncertainty"));
   delete c2;
   //delete SystGraphs;
   }

   return graphSystTotal;
}


TGraph* MakePlot(FILE* pFile, FILE* talkFile, string InputPattern, string ModelName, int XSectionType, std::vector<stSample>& modelSamples, double& LInt){
   std::vector<int> signalPoints;
   for(unsigned int i=0;i<modelSamples.size();i++) if(stAllInfo(InputPattern+""+SHAPESTRING+EXCLUSIONDIR+"/" + modelSamples[i].Name +".txt").Mass!=0) {
     signalPoints.push_back(i);
   }
   unsigned int N   = signalPoints.size();

   double* Mass     = new double   [signalPoints.size()];
   double* XSecTh   = new double   [signalPoints.size()];
   double* XSecObs  = new double   [signalPoints.size()];
   double* XSecExp  = new double   [signalPoints.size()];
   stAllInfo* Infos = new stAllInfo[signalPoints.size()];

   bool FileFound=false;

   for(unsigned int i=0;i<signalPoints.size();i++){
     Infos     [i] = stAllInfo(InputPattern+""+SHAPESTRING+EXCLUSIONDIR+"/" + modelSamples[signalPoints[i]].Name +".txt");
     if(Infos[i].Mass!=0) FileFound=true;
     Mass      [i] = Infos[i].Mass;
     XSecTh    [i] = Infos[i].XSec_Th;
     XSecObs   [i] = Infos[i].XSec_Obs;
     XSecExp   [i] = Infos[i].XSec_Exp;
     LInt          = std::max(LInt, Infos[i].LInt);
   }

   if(XSectionType>0 && FileFound){
      //for(unsigned int i=0;i<N;i++)printf("%-18s %5.0f --> Pt>%+6.1f & I>%+5.3f & TOF>%+4.3f & M>%3.0f--> NData=%2.0f  NPred=%6.1E+-%6.1E  NSign=%6.1E (Eff=%3.2f) Local Significance %3.2f\n",ModelName.c_str(),Infos[i].Mass,Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].WP_TOF,Infos[i].MassCut, Infos[i].NData, Infos[i].NPred, Infos[i].NPredErr, Infos[i].NSign, Infos[i].Eff, Infos[i].Significance);

     for(unsigned int i=0;i<signalPoints.size();i++){
       //for(unsigned int i=0;i<N;i++){
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


void printSummary(FILE* pFile, FILE* talkFile, string InputPattern, string ModelName, std::vector<stSample>& modelSamples){
   for(unsigned int i=0;i<modelSamples.size();i++){
      string signal7TeV = modelSamples[i].Name; if(signal7TeV.find("_8TeV")!=string::npos) signal7TeV = signal7TeV.replace(signal7TeV.find("_8TeV"),5, "_7TeV");
      string signal8TeV = modelSamples[i].Name; if(signal8TeV.find("_7TeV")!=string::npos) signal8TeV = signal8TeV.replace(signal8TeV.find("_7TeV"),5, "_8TeV");
      string signal     = signal8TeV;           if(signal    .find("_8TeV")!=string::npos) signal     = signal    .replace(signal    .find("_8TeV"),5, "");
      stAllInfo Infos7(InputPattern+""+SHAPESTRING+"EXCLUSION7TeV"+"/" + signal7TeV +".txt");
      stAllInfo Infos8(InputPattern+""+SHAPESTRING+"EXCLUSION8TeV"+"/" + signal8TeV +".txt");
      stAllInfo InfosC(InputPattern+""+SHAPESTRING+"EXCLUSIONCOMB"+"/" + signal     +".txt");
      if(Infos7.Mass<=0 && Infos8.Mass<=0 && InfosC.Mass<=0)continue;
      if(Infos7.Eff<=0 && Infos8.Eff<=0)continue;
      double Mass = std::max(Infos7.Mass, Infos8.Mass);
      TString ModelNameTS =  ModelName.c_str();  ModelNameTS.ReplaceAll("_"," ");  ModelNameTS.ReplaceAll("8TeV",""); ModelNameTS.ReplaceAll("7TeV","");

      if(ModelNameTS.Contains("Stop")   && ((int)(Mass)/100)%2!=0)continue;
      if(ModelNameTS.Contains("Gluino") && ((int)(Mass)/100)%2!=1)continue;
      if(ModelNameTS.Contains("DY")     && ((int)(Mass)/100)%2!=0)continue;
      if(ModelNameTS.Contains("DC")                              )continue;

      char massCut[255];  if(Infos8.MassCut>0){sprintf(massCut,"$>%.0f$",Infos8.MassCut);}else{sprintf(massCut," - ");}
      char Results7[255]; if(Infos7.Mass>0){sprintf(Results7, "%6.2f & %6.2E & %6.2E & %6.2E", Infos7.Eff, Infos7.XSec_Th,Infos7.XSec_Obs, Infos7.XSec_Exp);}else{sprintf(Results7, "       &          &          &         ");}
      char Results8[255]; if(Infos8.Mass>0){sprintf(Results8, "%6.2f & %6.2E & %6.2E & %6.2E", Infos8.Eff, Infos8.XSec_Th,Infos8.XSec_Obs, Infos8.XSec_Exp);}else{sprintf(Results8, "       &          &          &         ");}
      char ResultsC[255]; if(InfosC.Mass>0){sprintf(ResultsC, "%6.2E & %6.2E", InfosC.XSec_Obs, InfosC.XSec_Exp);}else{sprintf(ResultsC, "         &         ");}

      fprintf(pFile,"%-20s & %4.0f & %-7s & %s & %s & %s\\\\\n", ModelNameTS.Data(), Mass, massCut, Results7, Results8, ResultsC);
   }
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
   int TypeMode = TypeFromPattern(InputPattern);
   string prefix = "BUG";
   switch(TypeMode){
      case 0: prefix   = "Tk"; break;
      case 2: prefix   = "Mu"; break;
      case 3: prefix   = "Mo"; break;
      case 4: prefix   = "HQ"; break;
      case 5: prefix   = "LQ"; break;
   }

   double LInt = 0;
   for(unsigned int k=0; k<modelVector.size(); k++){
      bool isNeutral = false;if(modelVector[k].find("GluinoN")!=string::npos || modelVector[k].find("StopN")!=string::npos)isNeutral = true;
      if(TypeMode!=0 && isNeutral) continue;
      unsigned int N = modelMap[modelVector[k]].size();
      stAllInfo Infos;double Mass[N], XSecTh[N], XSecExp[N],XSecObs[N], XSecExpUp[N],XSecExpDown[N],XSecExp2Up[N],XSecExp2Down[N];
      for(unsigned int i=0;i<N;i++){
         Infos = stAllInfo(InputPattern+""+SHAPESTRING+EXCLUSIONDIR+"/" + modelMap[modelVector[k]][i].Name +".txt");
         Mass        [i]=Infos.Mass;
         XSecTh      [i]=Infos.XSec_Th;
         XSecObs     [i]=Infos.XSec_Obs;
         XSecExp     [i]=Infos.XSec_Exp;
         XSecExpUp   [i]=Infos.XSec_ExpUp;
         XSecExpDown [i]=Infos.XSec_ExpDown;
         XSecExp2Up  [i]=Infos.XSec_Exp2Up;
         XSecExp2Down[i]=Infos.XSec_Exp2Down;
         LInt           =std::max(LInt, Infos.LInt);
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
      
      TLegend* LEG = !Combine ? new TLegend(0.45,0.58,0.65,0.90) : new TLegend(0.45,0.10,0.65,0.42);
      //TLegend* LEG = new TLegend(0.40,0.65,0.8,0.90);
      string headerstr = "95% CL Limits (";
      headerstr += LegendFromType(InputPattern) + string(")");
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

      SaveCanvas(c1,"Results/"+SHAPESTRING+EXCLUSIONDIR+"/", string(prefix+ modelVector[k] + "ExclusionLog"));
      delete c1;
   }
}

// This code make the Expected Limit error band divided by expected limit plot for all signal models
// I don't like much this function... I started to rewrite it, but more work is still needed to improve it.
// I don't think two loops are needed, neither all these arrays...
void DrawRatioBands(string InputPattern)
{
   int TypeMode = TypeFromPattern(InputPattern);
   string prefix = "BUG";
   switch(TypeMode){
      case 0: prefix   = "Tk"; break;
      case 2: prefix   = "Mu"; break;
      case 3: prefix   = "Mo"; break;
      case 4: prefix   = "HQ"; break;
      case 5: prefix   = "LQ"; break;
   }

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
      if(TypeMode!=0 && isNeutral) continue;
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
      if(TypeMode>0 && isNeutral) continue;

      TMultiGraph* MG = new TMultiGraph();
      unsigned int N = modelMap[modelVector[k]].size();
      stAllInfo Infos;double Mass[N], XSecTh[N], XSecExp[N],XSecObs[N], XSecExpUp[N],XSecExpDown[N],XSecExp2Up[N],XSecExp2Down[N];
      for(unsigned int i=0;i<N;i++){
         Infos = stAllInfo(InputPattern+""+SHAPESTRING+EXCLUSIONDIR+"/" + modelMap[modelVector[k]][i].Name +".txt");
         Mass        [i]=Infos.Mass;
         XSecTh      [i]=Infos.XSec_Th;
         XSecObs     [i]=Infos.XSec_Obs     /Infos.XSec_Exp;
         XSecExp     [i]=Infos.XSec_Exp     /Infos.XSec_Exp;
         XSecExpUp   [i]=Infos.XSec_ExpUp   /Infos.XSec_Exp;
         XSecExpDown [i]=Infos.XSec_ExpDown /Infos.XSec_Exp;
         XSecExp2Up  [i]=Infos.XSec_Exp2Up  /Infos.XSec_Exp;
         XSecExp2Down[i]=Infos.XSec_Exp2Down/Infos.XSec_Exp;
         LInt           =std::max(LInt, Infos.LInt);
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
         headerstr = LegendFromType(InputPattern);
         LEG->SetHeader(headerstr.c_str());
         LEG->SetFillStyle(0); 
         LEG->SetBorderSize(0);
         LEG->AddEntry(ExpAErr[0], "Expected #pm 1#sigma","F");
         LEG->SetMargin(0.1);
         LEG->Draw();
      }  

      if(k==1){
         TLegend* LEG;
         LEG = new TLegend(0.13,0.01,0.32,0.99);
	 string headerstr;
	 LEG->SetFillStyle(0);
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
      if(TypeMode==0) {
      if(k!=modelVector.size()-1) pt = new TPaveText(0.45, 0.6, 0.95, 0.87,"LBNDC");
      else pt = new TPaveText(0.45, 0.82, 0.95, 0.935,"LBNDC");
      }
      else {
	if(k!=modelVector.size()-1) pt = new TPaveText(0.55, 0.6, 0.95, 0.87,"LBNDC");
	else pt = new TPaveText(0.55, 0.82, 0.95, 0.935,"LBNDC");
      }

      pt->SetBorderSize(0);
      pt->SetLineWidth(0);
      pt->SetFillStyle(kWhite);
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

   SaveCanvas(c1,"Results/"+SHAPESTRING+EXCLUSIONDIR+"/", string(prefix+"LimitsRatio"));
   delete c1;
}

//will run on all possible selection and try to identify which is the best one for this sample
void Optimize(string InputPattern, string Data, string signal, bool shape, bool cutFromFile){
   printf("Optimize selection for %s in %s\n",signal.c_str(), InputPattern.c_str());fflush(stdout);
  
   //get the typeMode from pattern
   TypeMode = TypeFromPattern(InputPattern); 

   if (TypeMode == 4)    RescaleError = 0.20;

   //Identify the signal sample
   GetSampleDefinition(samples);
   CurrentSampleIndex        = JobIdToIndex(signal,samples); 
   if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return;  } 
  

   if(Data.find("7TeV")!=string::npos){SQRTS=7.0; IntegratedLuminosity = IntegratedLuminosityFromE(SQRTS); }
   if(Data.find("8TeV")!=string::npos){SQRTS=8.0; IntegratedLuminosity = IntegratedLuminosityFromE(SQRTS);  }

 
   //For muon only don't run on neutral samples as near zero efficiency can make jobs take very long time
   if((signal.find("Gluino")!=string::npos || signal.find("Stop")!=string::npos) && signal.find("N")!=string::npos && TypeMode==3) return;

   //Load all input histograms
   TFile*InputFile     = new TFile((InputPattern + "Histos.root").c_str());
   TH1D* HCuts_Pt      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
   TH1D* HCuts_I       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");
   TH1D* HCuts_TOF     = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");
   TH1D* H_Lumi        = (TH1D*)GetObjectFromPath(InputFile, Data+"/IntLumi");
   TH1D* H_A           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_A");
   TH1D* H_B           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_B");
   TH1D* H_C           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_C");
   TH1D* H_D           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_D");
   TH1D* H_E           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_E");
   TH1D* H_F           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_F");
   TH1D* H_G           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_G");
   TH1D* H_H           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_H");
   TH1D* H_P           = (TH1D*)GetObjectFromPath(InputFile, Data+"/H_P");
   TH1D* H_S           = (TH1D*)GetObjectFromPath(InputFile, samples[CurrentSampleIndex].Name + "/TOF");
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



   //If Take the cuts From File --> Load the actual cut index
   int OptimCutIndex = -1;  //int OptimMassWindow;
   if(cutFromFile){
      FILE* pFile = fopen("Analysis_Cuts.txt","r");
      if(!pFile){printf("Can't open %s\n","Analysis_Cuts.txt"); return;}

      while(true){
         char line[4096];  string Name_;  int TypeMode_; double cutPt_; double cutI_; double cutTOF_; int massWindow_;
         if(!fgets(line, 4096, pFile))break;
         char* pch=strtok(line,","); int Arg=0; string tmp;
         while (pch!=NULL){
            switch(Arg){
               case  0: tmp = pch;  Name_     = tmp.substr(tmp.find("\"")+1,tmp.rfind("\"")-tmp.find("\"")-1); break;
               case  1: sscanf(pch, "%d",  &TypeMode_); break;
               case  2: sscanf(pch, "%lf", &cutPt_ ); break;
               case  3: sscanf(pch, "%lf", &cutI_  ); break;
               case  4: sscanf(pch, "%lf", &cutTOF_); break;
               case  5: sscanf(pch, "%d",  &massWindow_);break;
               default:break;
            }
            pch=strtok(NULL,",");Arg++;
         }
         //printf("%s %i %f %f %f %i\n",Name_.c_str(), TypeMode_, cutPt_, cutI_, cutTOF_, massWindow_);
         if(TypeMode_!=TypeMode)continue; //Not reading the cut line for the right TypeMode 

         string signalNameWithoutEnergy = signal;
         char str7TeV[]="_7TeV";
         char str8TeV[]="_8TeV";
         if(signalNameWithoutEnergy.find(str7TeV)!=string::npos)signalNameWithoutEnergy.erase(signalNameWithoutEnergy.find(str7TeV), string(str7TeV).length());
         if(signalNameWithoutEnergy.find(str8TeV)!=string::npos)signalNameWithoutEnergy.erase(signalNameWithoutEnergy.find(str8TeV), string(str8TeV).length()); 

         //printf("%s vs %s\n",Name_.c_str(), signalNameWithoutEnergy.c_str());
         if(Name_!=signalNameWithoutEnergy    )continue; //Not reading the cut line for the right signal sample

         //We are looking at the right sample --> Now loop over all cuts and identify the cut index of the optimal cut
         double MinDistance = 10000;
         for(int CutIndex=0;CutIndex<HCuts_Pt->GetNbinsX();CutIndex++){
            double cutDistance = fabs(cutPt_ - HCuts_Pt ->GetBinContent(CutIndex+1)) + fabs(cutI_ - HCuts_I ->GetBinContent(CutIndex+1)) + fabs(cutTOF_ - HCuts_TOF ->GetBinContent(CutIndex+1));
            if(cutDistance<MinDistance){MinDistance=cutDistance; OptimCutIndex=CutIndex;  }//OptimMassWindow=massWindow_;}
         }
         printf("Closest cut index to the cuts provided: %i\n",OptimCutIndex);
         break; 
      }
      fclose(pFile);
      if(OptimCutIndex<0){printf("DID NOT FIND THE CUT TO USE FOR THIS SAMPLE %s\n",signal.c_str());return;}
   }

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
   string outpath = InputPattern + "/"+SHAPESTRING+EXCLUSIONDIR+"/";
   MakeDirectories(outpath);
   FILE* pFile = fopen((outpath+"/"+signal+".info").c_str(),"w");
   if(!pFile)printf("Can't open file : %s\n",(outpath+"/"+signal+".info").c_str());

   stAllInfo result;
   stAllInfo toReturn;
   //loop on all possible selections and determine which is the best one
   for(int CutIndex=0;CutIndex<MassData->GetNbinsX();CutIndex++){
      //if(CutIndex>25)break; //for debugging purposes

      //if we don't want to optimize but take instead the cuts from a file, we can skip all other cuts
      if(OptimCutIndex>=0 && CutIndex!=OptimCutIndex)continue;

      //make sure the pT cut is high enough to get some statistic for both ABCD and mass shape
      if(HCuts_Pt ->GetBinContent(CutIndex+1) < 50 && TypeMode!=4){printf("Skip cut=%i because of too lose pT cut\n", CutIndex); continue; }

      //make sure we have a reliable prediction of the number of events      
      if(OptimCutIndex<0 && H_E->GetBinContent(CutIndex+1) >0 && (H_A->GetBinContent(CutIndex+1)<25 || H_F->GetBinContent(CutIndex+1)<25 || H_G->GetBinContent(CutIndex+1)<25)){printf("Skip cut=%i because of unreliable ABCD prediction\n", CutIndex); continue;}  //Skip events where Prediction (AFG/EE) is not reliable
      if(OptimCutIndex<0 && H_E->GetBinContent(CutIndex+1)==0 && H_A->GetBinContent(CutIndex+1)>0 && (H_C->GetBinContent(CutIndex+1)<25 || H_B->GetBinContent(CutIndex+1)<25)){printf("Skip cut=%i because of unreliable ABCD prediction\n", CutIndex); continue;}  //Skip events where Prediction (CB/A) is not reliable
      if(OptimCutIndex<0 && H_F->GetBinContent(CutIndex+1)>0 && H_A->GetBinContent(CutIndex+1)==0 && (H_B->GetBinContent(CutIndex+1)<25 || H_H->GetBinContent(CutIndex+1)<25)){printf("Skip cut=%i because of unreliable ABCD prediction\n", CutIndex); continue;}  //Skip events where Prediction (CB/A) is not reliable
      if(OptimCutIndex<0 && H_G->GetBinContent(CutIndex+1)>0 && H_F->GetBinContent(CutIndex+1)==0 && (H_C->GetBinContent(CutIndex+1)<25 || H_H->GetBinContent(CutIndex+1)<25)){printf("Skip cut=%i because of unreliable ABCD prediction\n", CutIndex); continue;}  //Skip events where Prediction (CH/G) is not reliable
   
      //make sure we have a reliable prediction of the shape 
      if(TypeMode<=2){
         double N_P = H_P->GetBinContent(CutIndex+1);       
         if(H_E->GetBinContent(CutIndex+1) >0 && (H_A->GetBinContent(CutIndex+1)<0.25*N_P || H_F->GetBinContent(CutIndex+1)<0.25*N_P || H_G->GetBinContent(CutIndex+1)<0.25*N_P)){printf("Skip cut=%i because of unreliable mass prediction\n", CutIndex); continue;}  //Skip events where Mass Prediction is not reliable
         if(H_E->GetBinContent(CutIndex+1)==0 && (H_C->GetBinContent(CutIndex+1)<0.25*N_P || H_B->GetBinContent(CutIndex+1)<0.25*N_P)){printf("Skip cut=%i because of unreliable mass prediction\n", CutIndex); continue;}  //Skip events where Mass Prediction is not reliable
      }      

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
    //if(TypeMode<=2){if(!runCombine(true, true, false, InputPattern, signal, CutIndex, shape, true, result, MassData, MassPred, MassSign, MassSignP, MassSignI, MassSignM, MassSignT, MassSignPU))continue;
    //}else          {if(!runCombine(true, true, false, InputPattern, signal, CutIndex, shape, true, result, MassData, MassPred, MassSign, MassSignP, MassSignI, MassSignM, MassSignT, MassSignPU))continue;
    //}

      //no need to precompute the reach when not optimizing the cuts
      //if(OptimCutIndex<0){
         //best significance --> is actually best reach
         if(TypeMode<=2){if(!runCombine(true, false, true, InputPattern, signal, CutIndex, shape, true, result, MassData, MassPred, MassSign, MassSignP, MassSignI, MassSignM, MassSignT, MassSignPU)){printf("runCombine did not converge\n"); continue;}
         }else          {if(!runCombine(true, false, true, InputPattern, signal, CutIndex, shape, true, result, H_D, H_P, H_S, MassSignP, MassSignI, MassSignM, MassSignT, MassSignPU)){printf("runCombine did not converge\n"); continue;}
         }
	 //}else{
         //result.XSec_5Sigma=0.0001;//Dummy number --> will be recomputed later on... but it must be >0
	 //}

      //report the result for this point in the log file
      fprintf(pFile  ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.3f TOF>%6.3f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E+-%6.3E SignalEff=%6.3f ExpLimit=%6.3E (%6.3E) Reach=%6.3E",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,result.NData,result.NPred, result.NPredErr,result.Eff,result.XSec_Exp, result.XSec_Obs, result.XSec_5Sigma);fflush(stdout);
      fprintf(stdout ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.3f TOF>%6.3f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E+-%6.3E SignalEff=%6.3f ExpLimit=%6.3E (%6.3E) Reach=%6.3E",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,result.NData,result.NPred, result.NPredErr,result.Eff,result.XSec_Exp, result.XSec_Obs, result.XSec_5Sigma);fflush(stdout);
//      if(result.XSec_Exp<toReturn.XSec_Exp){
      if(OptimCutIndex>=0 || (result.XSec_5Sigma>0 && result.XSec_5Sigma<toReturn.XSec_5Sigma)){
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
   if(TypeMode<=2){runCombine(false, true, true, InputPattern, signal, toReturn.Index, shape, false, toReturn, MassData, MassPred, MassSign, MassSignP, MassSignI, MassSignM, MassSignT, MassSignPU);
   }else          {runCombine(false, true, true, InputPattern, signal, toReturn.Index, shape, false, toReturn, H_D, H_P, H_S, MassSignP, MassSignI, MassSignM, MassSignT, MassSignPU);
   }
  
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
   fprintf(pFile, "rate    %f %f\n",Sign,std::max(1E-4, Pred) );  //if Pred<1E-4 we have troubles when merging datacards
   fprintf(pFile, "-------------------------------\n");
   fprintf(pFile, "%35s    %6s 1.022     1.0  \n","Lumi" , "lnN");
   fprintf(pFile, "%35s    %6s -         %5.3f\n",(ChannelName+"systP").c_str(), "lnN", PredRelErr);
   fprintf(pFile, "%35s    %6s 1.07      -    \n",(ChannelName+"systS").c_str(), "lnN");
   if(Shape){
   fprintf(pFile, "%35s    %6s 1.000     -    \n",(ChannelName+"statS").c_str(), "shapeN2");
   fprintf(pFile, "%35s    %6s -         1    \n",(ChannelName+"statP").c_str(), "shapeN2");
   fprintf(pFile, "%35s    %6s 1.000     -    \n","mom"  , "shapeN2");
   fprintf(pFile, "%35s    %6s 1.000     -    \n","ias"  , "shapeN2");
   fprintf(pFile, "%35s    %6s 1.000     -    \n","ih"   , "shapeN2");
   fprintf(pFile, "%35s    %6s 1.000     -    \n","tof"  , "shapeN2");
   fprintf(pFile, "%35s    %6s 1.000     -    \n","pu"   , "shapeN2");
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

   string outpath = InputPattern + "/"+SHAPESTRING+EXCLUSIONDIR+"/";
   MakeDirectories(outpath);

   //Get Optimal cut from cut&count optimization
   stAllInfo result =  stAllInfo(InputPattern+"/"+EXCLUSIONDIR+"/"+signal+".txt");

   //load all intput histograms
   TFile* InputFile  = new TFile((InputPattern+"/Histos.root").c_str());
   TH1D*  H_Lumi     = (TH1D*)GetObjectFromPath(InputFile, "Data7TeV/IntLumi");
   TH2D*  MassData   = (TH2D*)GetObjectFromPath(InputFile, "Data7TeV/Mass");
   TH2D*  MassPred   = (TH2D*)GetObjectFromPath(InputFile, "Data7TeV/Pred_Mass");
   TH2D*  MassSign   = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass");
   TH2D*  MassSignP  = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass_SystP");
   TH2D*  MassSignI  = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass_SystI");
   TH2D*  MassSignM  = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass_SystM");
   TH2D*  MassSignT  = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass_SystT");
   TH2D*  MassSignPU = (TH2D*)GetObjectFromPath(InputFile, samples[s].Name+"/Mass_SystPU");
   TH1D*  TotalE     = (TH1D*)GetObjectFromPath(InputFile, samples[s].Name+"/TotalE" );
   TH1D*  TotalEPU   = (TH1D*)GetObjectFromPath(InputFile, samples[s].Name+"/TotalEPU" );

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

   bool Shape = true;

   //find range
   if(Shape){
      MinRange = 0;
   }else{
      MinRange = std::max(0.0, result.MassMean-2*result.MassSigma);
      MinRange = MassSign->GetYaxis()->GetBinLowEdge(MassSign->GetYaxis()->FindBin(MinRange)); //Round to a bin value to avoid counting prpoblem due to the binning. 
   }

   //compute shape based limits and save it's output
   runCombine(false, true, true, InputPattern, signal, result.Index, Shape, false, result, MassData, MassPred, MassSign, MassSignP, MassSignI, MassSignM, MassSignT, MassSignPU);

   //all done, save the results to file
   result.Save(InputPattern+"/"+SHAPESTRING+EXCLUSIONDIR+"/"+signal+".txt");
}

//compute the significance using a ProfileLikelihood (assuming datacard is already produced)
double computeSignificance(string datacard, bool expected, string& signal, string massStr, float Strength){
   double toReturn = -1;
   char strengthStr[255]; sprintf(strengthStr,"--expectSignal=%f",Strength);
   string CodeToExecute = "cd /tmp/;";
   if(expected)CodeToExecute += "combine -M ProfileLikelihood -n " + signal + " -m " + massStr + " --significance -t 100 " + strengthStr + " " + datacard + " &> shape_" + signal + ".log;";
   else        CodeToExecute += "combine -M ProfileLikelihood -n " + signal + " -m " + massStr + " --significance                            " + datacard + " &> shape_" + signal + ".log;";
   CodeToExecute += "cd $OLDPWD;";
   system(CodeToExecute.c_str());   

   char line[4096];
   FILE* sFile = fopen((string("/tmp/shape_")+signal + ".log").c_str(), "r");
   if(!sFile)std::cout<<"FILE NOT OPEN:"<< (string("/tmp/shape_")+signal + ".log").c_str() << endl;
   int LineIndex=0; int GarbageI; double GarbageD;     
   while(fgets(line, 4096, sFile)){LineIndex++;       
     if(!expected && LineIndex==3){sscanf(line,"Significance: %lf",&toReturn);     break;}
 //    if( expected && LineIndex==7){sscanf(line,"median expected limit: r < %lf @ 95%%CL (%i toyMC)",&toReturn,&GarbageI); break;}
     if( expected && LineIndex==6){sscanf(line,"mean   expected limit: r < %lf +/- %lf @ 95%%CL (%i toyMC)",&toReturn, &GarbageD, &GarbageI); break;}

     continue;
   }fclose(sFile);
   return toReturn;
}

//run the higgs combine stat tool using predicted mass shape distribution (possibly do shape based analysis and/or cut on mass) OR 1D histogram output from ABCD  (only do cut and count without mass cut)
bool runCombine(bool fastOptimization, bool getXsection, bool getSignificance, string& InputPattern, string& signal, unsigned int CutIndex, bool Shape, bool Temporary, stAllInfo& result, TH1* MassData, TH1* MassPred, TH1* MassSign, TH1* MassSignP, TH1* MassSignI, TH1* MassSignM, TH1* MassSignT, TH1* MassSignPU){
   TH1D *MassDataProj=NULL, *MassPredProj=NULL, *MassSignProj=NULL, *MassSignProjP=NULL, *MassSignProjI=NULL, *MassSignProjM=NULL, *MassSignProjT=NULL, *MassSignProjPU=NULL;
   double NData, NPredErr, NPred, NSign, NSignP, NSignI, NSignM, NSignT, NSignPU;
   double signalsMeanHSCPPerEvent = GetSignalMeanHSCPPerEvent(InputPattern,CutIndex, MinRange, MaxRange);

   //IF 2D histograms --> we get all the information from there (and we can do shape based analysis AND/OR cut on mass)
   if(MassData->InheritsFrom("TH2")){
      //make the projection of all the 2D input histogram to get the shape for this single point
      MassDataProj       = ((TH2D*)MassData  )->ProjectionY("MassDataProj"  ,CutIndex+1,CutIndex+1);
      MassPredProj       = ((TH2D*)MassPred  )->ProjectionY("MassPredProj"  ,CutIndex+1,CutIndex+1);
      MassSignProj       = ((TH2D*)MassSign  )->ProjectionY("MassSignProj"  ,CutIndex+1,CutIndex+1);
      MassSignProjP      = ((TH2D*)MassSignP )->ProjectionY("MassSignProP"  ,CutIndex+1,CutIndex+1);
      MassSignProjI      = ((TH2D*)MassSignI )->ProjectionY("MassSignProI"  ,CutIndex+1,CutIndex+1);
      MassSignProjM      = ((TH2D*)MassSignM )->ProjectionY("MassSignProM"  ,CutIndex+1,CutIndex+1);
      MassSignProjT      = ((TH2D*)MassSignT )->ProjectionY("MassSignProT"  ,CutIndex+1,CutIndex+1);
      MassSignProjPU     = ((TH2D*)MassSignPU)->ProjectionY("MassSignProPU" ,CutIndex+1,CutIndex+1);

      //count events in the allowed range (infinite for shape based and restricted for cut&count)
      NData       = MassDataProj->Integral(MassDataProj->GetXaxis()->FindBin(MinRange), MassDataProj->GetXaxis()->FindBin(MaxRange));
      NPred       = MassPredProj->Integral(MassPredProj->GetXaxis()->FindBin(MinRange), MassPredProj->GetXaxis()->FindBin(MaxRange));
      NPredErr    = pow(NPred*RescaleError,2);
      for(int i=MassPredProj->GetXaxis()->FindBin(MinRange); i<=MassPredProj->GetXaxis()->FindBin(MaxRange) ;i++){NPredErr+=pow(MassPredProj->GetBinError(i),2);}NPredErr=sqrt(NPredErr);
      NSign       = (MassSignProj  ->Integral(MassSignProj  ->GetXaxis()->FindBin(MinRange), MassSignProj  ->GetXaxis()->FindBin(MaxRange))) / signalsMeanHSCPPerEvent;
      NSignP      = (MassSignProjP ->Integral(MassSignProjP ->GetXaxis()->FindBin(MinRange), MassSignProjP ->GetXaxis()->FindBin(MaxRange))) / signalsMeanHSCPPerEvent;
      NSignI      = (MassSignProjI ->Integral(MassSignProjI ->GetXaxis()->FindBin(MinRange), MassSignProjI ->GetXaxis()->FindBin(MaxRange))) / signalsMeanHSCPPerEvent;
      NSignM      = (MassSignProjM ->Integral(MassSignProjM ->GetXaxis()->FindBin(MinRange), MassSignProjM ->GetXaxis()->FindBin(MaxRange))) / signalsMeanHSCPPerEvent;
      NSignT      = (MassSignProjT ->Integral(MassSignProjT ->GetXaxis()->FindBin(MinRange), MassSignProjT ->GetXaxis()->FindBin(MaxRange))) / signalsMeanHSCPPerEvent;
      NSignPU     = (MassSignProjPU->Integral(MassSignProjPU->GetXaxis()->FindBin(MinRange), MassSignProjPU->GetXaxis()->FindBin(MaxRange))) / signalsMeanHSCPPerEvent;

   //IF 1D histograms --> we get all the information from the ABCD method output 
   }else{
      Shape=false; //can not do shape based if we don't get the shapes
      NData       = MassData  ->GetBinContent(CutIndex+1);
      NPredErr    = MassPred  ->GetBinError  (CutIndex+1);
      NPred       = MassPred  ->GetBinContent(CutIndex+1);
      NSign       = MassSign  ->GetBinContent(CutIndex+1) / signalsMeanHSCPPerEvent;

      MassSignProjP      = ((TH2D*)MassSignP )->ProjectionY("MassSignProP"  ,CutIndex+1,CutIndex+1);
      MassSignProjI      = ((TH2D*)MassSignI )->ProjectionY("MassSignProI"  ,CutIndex+1,CutIndex+1);
      MassSignProjM      = ((TH2D*)MassSignM )->ProjectionY("MassSignProM"  ,CutIndex+1,CutIndex+1);
      MassSignProjT      = ((TH2D*)MassSignT )->ProjectionY("MassSignProT"  ,CutIndex+1,CutIndex+1);
      MassSignProjPU     = ((TH2D*)MassSignPU)->ProjectionY("MassSignProPU" ,CutIndex+1,CutIndex+1);

      NSignP      = MassSignProjP ->Integral(0, MassSignProjP ->GetNbinsX()+1) / signalsMeanHSCPPerEvent;
      NSignI      = MassSignProjI ->Integral(0, MassSignProjI ->GetNbinsX()+1) / signalsMeanHSCPPerEvent;
      NSignM      = MassSignProjM ->Integral(0, MassSignProjM ->GetNbinsX()+1) / signalsMeanHSCPPerEvent;
      NSignT      = MassSignProjT ->Integral(0, MassSignProjT ->GetNbinsX()+1) / signalsMeanHSCPPerEvent;
      NSignPU     = MassSignProjPU->Integral(0, MassSignProjPU->GetNbinsX()+1) / signalsMeanHSCPPerEvent;

      NPredErr = sqrt(pow((NPred* RescaleError), 2) + pow(NPredErr,2));      // incorporate background uncertainty
   }

   //skip pathological selection point
   if(isnan((float)NPred))return false;
   if(NPred<=0){return false;} //Is <=0 only when prediction failed or is not meaningful (i.e. WP=(0,0,0) )
   if(!Shape && NPred>1000){return false;}  //When NPred is too big, expected limits just take an infinite time! 

   //compute all efficiencies (not really needed anymore, but it's nice to look at these numbers afterward)
   double Eff         = NSign   / (result.XSec_Th*result.LInt);
   double EffP        = NSignP  / (result.XSec_Th*result.LInt);
   double EffI        = NSignI  / (result.XSec_Th*result.LInt);
   double EffM        = NSignM  / (result.XSec_Th*result.LInt);
   double EffT        = NSignT  / (result.XSec_Th*result.LInt);
   double EffPU       = NSignPU / (result.XSec_Th*result.LInt);
   if(Eff==0)return false;
//   if(Eff<=1E-5)return false; // if Eff<0.001% -> limit will hardly converge and we are probably not interested by this point anyway

   //no way that this point is optimal
   bool pointMayBeOptimal = (fastOptimization && !getXsection && getSignificance && ((NPred-3*NPredErr)<=result.NPred || Eff>=result.Eff));

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
//   NSign/=(result.XSec_Th*1000.0); //normalize xsection to 1fb
   NSign/=(1000.0); //normalize xsection to 1fb

   //for shape based analysis we need to save all histograms into a root file
   char CutIndexStr[255];sprintf(CutIndexStr, "SQRTS%02.0fCut%03.0f",SQRTS, result.Index);
   if(Shape){
      //prepare the histograms and variation
      //scale to 1fb xsection and to observed events instead of observed tracks
      MassSignProj  ->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));
      MassSignProjP ->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));
      MassSignProjI ->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));
      MassSignProjM ->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));
      MassSignProjT ->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));
      MassSignProjPU->Scale(1.0/(result.XSec_Th*signalsMeanHSCPPerEvent*1000));

      //Rebin --> keep CPU time reasonable and error small
      MassDataProj  ->Rebin(2);
      MassPredProj  ->Rebin(2);
      MassSignProj  ->Rebin(2);
      MassSignProjP ->Rebin(2);
      MassSignProjI ->Rebin(2);
      MassSignProjM ->Rebin(2);
      MassSignProjT ->Rebin(2);
      MassSignProjPU->Rebin(2);


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
   if(getSignificance && Temporary){
      if(NPred<0.001) NPred=0.001;
      double SignifValue=0.0; double PrevSignifValue=0; double Strength=0.1*(3*sqrt(NPred)/NSign);  if(result.XSec_5Sigma>0 && result.XSec_5Sigma<1E50)Strength=result.XSec_5Sigma/result.XSec_Th;
//      double SignifValue=0.0;double Strength=0.0005;  if(result.XSec_5Sigma>0 && result.XSec_5Sigma<1E50)Strength=result.XSec_5Sigma/result.XSec_Th;
      double previousXSec_5Sigma=result.XSec_5Sigma; result.XSec_5Sigma = -1;
      //find signal strength needed to get a 5sigma significance
      unsigned int l=0;
      double CountDecrease=0;
      for(l=0;l<10 && pointMayBeOptimal;l++){
         PrevSignifValue = SignifValue;
         SignifValue = computeSignificance(datacardPath, true, signal, massStr, Strength);
         printf("SIGNAL STRENGTH = %E --> SIGNIFICANCE=%E\n",Strength,SignifValue);

         if(SignifValue<=PrevSignifValue || SignifValue<=0){CountDecrease++;}else{CountDecrease=0;}
         if(CountDecrease>=2){result.XSec_5Sigma  = 1E49; break;}

         //we found the signal strength that lead to a significance close enough to the 5sigma to stop the loop 
         //OR we know that this point is not going to be a good one --> can do a coarse approximation since the begining
         if(fabs(SignifValue-5)<0.75 || (fastOptimization && Strength>=previousXSec_5Sigma && SignifValue<5)){
            result.XSec_5Sigma  = Strength * (5/SignifValue) * (result.XSec_Th/1000.0);//xsection in pb
            break;
         }

         //Not yet at the right significance, change the strength to get close
         if(isinf((float)SignifValue)){Strength/=5;                 continue;} //strength is already way too high
         if(SignifValue<=0           ){Strength*=10;                continue;} //strength is already way too low
         if(SignifValue>5            ){Strength*=std::max( 0.1,(4.95/SignifValue)); continue;} //5/significance could be use but converge faster with 4.9
         if(SignifValue<5            ){Strength*=std::min(10.0,(5.05/SignifValue)); continue;} //5/significance could be use, but it converges faster with 5.1
         break;                    
      }
   }

   if(getXsection){
      //prepare and run the script that will run the external "combine" tool from the Higgs group
      //If very low background range too small, set limit at 0.001.  Only affects scanning range not final limit
      if(NPred<0.001) NPred=0.001;
      char rangeStr[255];sprintf(rangeStr," --rMin %f --rMax %f ", 0.0f, 2*(3*sqrt(NPred)/NSign) );
      printf("%f/%f --> %s\n",NSign,NPred,rangeStr);
      string CodeToExecute = "cd /tmp/;";
      CodeToExecute += "combine -M Asymptotic        -n " + signal + " -m " + massStr + rangeStr + " shape_" + signal+".dat &> shape_" + signal + ".log;";   
      CodeToExecute += "cd $OLDPWD;cp /tmp/shape_" + signal + ".* " + InputPattern+"/"+SHAPESTRING+EXCLUSIONDIR+"/." + ";";
      system(CodeToExecute.c_str());

      //if all went well, the combine tool created a new file containing the result of the limit in the form of a TTree
      //we can open this TTree and access the values for the expected limit, uncertainty bands, and observed limits.
      TFile* file = TFile::Open((string("/tmp/")+"higgsCombine"+signal+".Asymptotic.mH"+massStr+".root").c_str());
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
              if(TquantExp==0.025f){ result.XSec_Exp2Down = Tlimit*(result.XSec_Th/1000.0);
        }else if(TquantExp==0.160f){ result.XSec_ExpDown  = Tlimit*(result.XSec_Th/1000.0);
        }else if(TquantExp==0.500f){ result.XSec_Exp      = Tlimit*(result.XSec_Th/1000.0);
        }else if(TquantExp==0.840f){ result.XSec_ExpUp    = Tlimit*(result.XSec_Th/1000.0);
        }else if(TquantExp==0.975f){ result.XSec_Exp2Up   = Tlimit*(result.XSec_Th/1000.0);
        }else if(TquantExp==-1    ){ result.XSec_Obs      = Tlimit*(result.XSec_Th/1000.0); //will be overwritten afterward
        }else{printf("Quantil %f unused by the analysis --> check the code\n", TquantExp);
        }
      }
      file->Close();

      //RUN FULL HYBRID CLS LIMIT (just for observed limit so far, because it is very slow for expected limits --> should be updated --> FIXME)
      CodeToExecute = "cd /tmp/;";
      CodeToExecute += "combine -M HybridNew -n " + signal + " -m " + massStr + rangeStr + " shape_" + signal+".dat &> shape_" + signal + ".log;";
      CodeToExecute += "cd $OLDPWD;cp /tmp/shape_" + signal + ".* " + InputPattern+"/"+SHAPESTRING+EXCLUSIONDIR+"/." + ";";
      system(CodeToExecute.c_str());

      //if all went well, the combine tool created a new file containing the result of the limit in the form of a TTree
      //we can open this TTree and access the values for the expected limit, uncertainty bands, and observed limits.
      file = TFile::Open((string("/tmp/")+"higgsCombine"+signal+".HybridNew.mH"+massStr+".root").c_str());
      if(!file || file->IsZombie())return false;
      tree = (TTree*)file->Get("limit");
      if(!tree)return false;
      tree->GetBranch("mh"              )->SetAddress(&Tmass    );
      tree->GetBranch("limit"           )->SetAddress(&Tlimit   );
      tree->GetBranch("limit"           )->SetAddress(&Tlimit   );
      tree->GetBranch("limitErr"        )->SetAddress(&TlimitErr);
      tree->GetBranch("quantileExpected")->SetAddress(&TquantExp);
      for(int ientry=0;ientry<tree->GetEntriesFast();ientry++){
        tree->GetEntry(ientry);
//              if(TquantExp==0.025f){ result.XSec_Exp2Down = Tlimit*(result.XSec_Th/1000.0);
//        }else if(TquantExp==0.160f){ result.XSec_ExpDown  = Tlimit*(result.XSec_Th/1000.0);
//        }else if(TquantExp==0.500f){ result.XSec_Exp      = Tlimit*(result.XSec_Th/1000.0);
//        }else if(TquantExp==0.840f){ result.XSec_ExpUp    = Tlimit*(result.XSec_Th/1000.0);
//        }else if(TquantExp==0.975f){ result.XSec_Exp2Up   = Tlimit*(result.XSec_Th/1000.0);
//        }else
        if(TquantExp==-1    ){ result.XSec_Obs      = Tlimit*(result.XSec_Th/1000.0);
        }else{printf("Quantil %f unused by the analysis --> check the code\n", TquantExp);
        }
      }
      file->Close();
   }

   if(!Temporary && getSignificance){
       result.Significance = computeSignificance(datacardPath, false, signal, massStr, 1.0);
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

      TH1D* MassPredSignProj = (TH1D*)MassPredProj->Clone("predWithSign");
      MassPredSignProj->Add(MassSignProj);
      MassPredSignProj->SetMarkerStyle(0);
      MassPredSignProj->SetLineColor(kBlue-10);
      MassPredSignProj->SetLineStyle(1);
      MassPredSignProj->SetLineWidth(4);
      MassPredSignProj->SetFillColor(0);
      MassPredSignProj->SetFillStyle(0);
      MassPredSignProj->Draw("same HIST C");

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

      TLegend* leg = new TLegend(0.40,0.75,0.80,0.93);
      leg->SetHeader(NULL);
      leg->SetFillColor(0);
      leg->SetFillStyle(0);
      leg->SetBorderSize(0);
      leg->AddEntry(MassDataProj,"Data", "P");
      leg->AddEntry(MassPredProj,"Prediction", "FP");
      leg->AddEntry(MassSignProj,signal.c_str(), "F");
      leg->AddEntry(MassPredSignProj,(string("Prediction + ")+signal).c_str(), "L");
      leg->Draw();

      (c1->cd(2))->SetLogy(true);
//      (c1->cd(2))->SetGridy(true);

      TH1* MassSignProjRatio  = (TH1D*)MassSignProj ->Clone(signal.c_str() );  MassSignProjRatio ->SetLineColor(1); MassSignProjRatio ->SetMarkerColor(1); MassSignProjRatio ->SetMarkerStyle(0); 
      TH1* MassSignProjPRatio = (TH1D*)MassSignProjP->Clone("mom");            MassSignProjPRatio->SetLineColor(2); MassSignProjPRatio->SetMarkerColor(2); MassSignProjPRatio->SetMarkerStyle(20);
      TH1* MassSignProjIRatio = (TH1D*)MassSignProjI->Clone("Ias");            MassSignProjIRatio->SetLineColor(4); MassSignProjIRatio->SetMarkerColor(4); MassSignProjIRatio->SetMarkerStyle(21);
      TH1* MassSignProjMRatio = (TH1D*)MassSignProjM->Clone("Ih");             MassSignProjMRatio->SetLineColor(3); MassSignProjMRatio->SetMarkerColor(3); MassSignProjMRatio->SetMarkerStyle(22);
      TH1* MassSignProjTRatio = (TH1D*)MassSignProjT->Clone("TOF");            MassSignProjTRatio->SetLineColor(8); MassSignProjTRatio->SetMarkerColor(8); MassSignProjTRatio->SetMarkerStyle(23);
      TH1* MassSignProjLRatio = (TH1D*)MassSignProjPU->Clone("pu");            MassSignProjLRatio->SetLineColor(6); MassSignProjLRatio->SetMarkerColor(6); MassSignProjLRatio->SetMarkerStyle(33);

      //MassSignProjPRatio->Divide(MassSignProjPRatio, MassSignProjRatio,1,1, "B");
      //MassSignProjIRatio->Divide(MassSignProjIRatio, MassSignProjRatio,1,1, "B");
      //MassSignProjMRatio->Divide(MassSignProjMRatio, MassSignProjRatio,1,1, "B");
      //MassSignProjTRatio->Divide(MassSignProjTRatio, MassSignProjRatio,1,1, "B");
      //MassSignProjLRatio->Divide(MassSignProjLRatio, MassSignProjRatio,1,1, "B");
      //MassSignProjRatio ->Divide(MassSignProjRatio , MassSignProjRatio,1,1, "B");

      MassSignProjRatio->SetStats(kFALSE);
      MassSignProjRatio->SetFillColor(0);
      MassSignProjRatio->SetLineWidth(2);
      MassSignProjRatio->GetXaxis()->SetRangeUser(0,1400);
      MassSignProjRatio->GetXaxis()->SetNdivisions(505,"X");
      MassSignProjRatio->GetYaxis()->SetNdivisions(505,"X");
      MassSignProjRatio->SetMaximum(MassSignProjPRatio->GetMaximum()*2.0);
      MassSignProjRatio->SetMinimum(Min);
      //MassSignProjRatio->SetMaximum(2);//Max);
      //MassSignProjRatio->SetMinimum(0);//Min);
      //MassSignProjRatio->Reset(); //use this histogram as a framework only
      MassSignProjRatio->Draw("HIST");
      //TLine l1(0,1,1400,1);l1.SetLineColor(1); l1.SetLineWidth(2); l1.Draw("same");
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

      SaveCanvas(c1, InputPattern+"/"+SHAPESTRING+EXCLUSIONDIR+"/shape", signal, true);
      delete leg2; delete leg; delete c1; delete MassPredSignProj;
      delete MassSignProjRatio; delete MassSignProjPRatio; delete MassSignProjIRatio; delete MassSignProjMRatio; delete MassSignProjTRatio; delete MassSignProjLRatio;
   }

   //all done, clean everything and return true
   delete MassDataProj; delete MassPredProj; delete MassSignProj; delete MassSignProjP; delete MassSignProjI; delete MassSignProjM; delete MassSignProjT; delete MassSignProjPU;
   return true;
}


bool Combine(string InputPattern, string signal7, string signal8){
//   CurrentSampleIndex        = JobIdToIndex(signal, samples); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return false;  }
//   int s = CurrentSampleIndex;

   string outpath = InputPattern + "/"+SHAPESTRING+EXCLUSIONDIR+"/";
   MakeDirectories(outpath);

   //Get Optimal cut from sample11
   stAllInfo result11 =  stAllInfo(InputPattern+"/EXCLUSION7TeV/"+signal7+".txt");
   //Get Optimal cut from sample12
   stAllInfo result12 =  stAllInfo(InputPattern+"/EXCLUSION8TeV/"+signal8+".txt");

   stAllInfo result = result12;
   char massStr[255]; sprintf(massStr,"%.0f",result.Mass);

   string signal = signal7;
   if(signal.find("_7TeV")!=string::npos){signal.replace(signal.find("_7TeV"),5, "");}

   FILE* pFileTmp = NULL;

   bool is7TeVPresent = true;
   pFileTmp = fopen((InputPattern+"/EXCLUSION7TeV/shape_"+signal7+".dat").c_str(), "r");
   if(!pFileTmp){is7TeVPresent=false;}else{fclose(pFileTmp);}
   if(TypeMode==3) is7TeVPresent=false;

   bool is8TeVPresent = true;
   pFileTmp = fopen((InputPattern+"/EXCLUSION8TeV/shape_"+signal8+".dat").c_str(), "r");
   if(!pFileTmp){is8TeVPresent=false;}else{fclose(pFileTmp);}


   string CodeToExecute = "combineCards.py ";
   if(is7TeVPresent)CodeToExecute+="   " + InputPattern+"/EXCLUSION7TeV/shape_"+signal7+".dat ";
   if(is8TeVPresent)CodeToExecute+="   " + InputPattern+"/EXCLUSION8TeV/shape_"+signal8+".dat ";
   CodeToExecute+=" > " + outpath+"shape_"+signal+".dat ";
   system(CodeToExecute.c_str());   
   printf("%s \n",CodeToExecute.c_str());

   result.LInt  = result11.LInt  + result12.LInt ;
   result.NSign = result11.NSign + result12.NSign;
   result.NData = result11.NData + result12.NData;
   result.NPred = result11.NPred + result12.NPred;
   result.NPredErr = sqrt(pow(result11.NPredErr,2) + pow(result12.NPredErr,2));
   result.XSec_Th = 1.0;
   double NPred = result.NPred;
   double NSign = result.NSign / 1000.0;


   //ALL CODE BELOW IS A BIT DIFFERENT THAN THE ONE USED IN runCombined, BECAUSE HERE WE KEEP THE RESULTS ON LIMIT IN TERMS OF SIGNAL STRENGTH (r=SigmaObs/SigmaTH)
   if(true){
      //prepare and run the script that will run the external "combine" tool from the Higgs group
      //If very low background range too small, set limit at 0.001.  Only affects scanning range not final limit
      if(NPred<0.001) NPred=0.001;
      char rangeStr[255];sprintf(rangeStr," --rMin %f --rMax %f ", 0.0f, 2*(3*sqrt(NPred)/NSign) );
      printf("%f/%f --> %s\n",NSign,NPred,rangeStr);
      string CodeToExecute = "cp " + outpath+"shape_"+signal+".dat /tmp/.;";
      CodeToExecute += "cd /tmp/;";
      CodeToExecute += "combine -M Asymptotic        -n " + signal + " -m " + massStr + rangeStr + " shape_" + signal+".dat &> shape_" + signal + ".log;";   
      CodeToExecute += "cd $OLDPWD;cp /tmp/shape_" + signal + ".* " + InputPattern+"/"+SHAPESTRING+EXCLUSIONDIR+"/." + ";";
      system(CodeToExecute.c_str());

      //if all went well, the combine tool created a new file containing the result of the limit in the form of a TTree
      //we can open this TTree and access the values for the expected limit, uncertainty bands, and observed limits.
      TFile* file = TFile::Open((string("/tmp/")+"higgsCombine"+signal+".Asymptotic.mH"+massStr+".root").c_str());
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
/*
      //RUN FULL HYBRID CLS LIMIT (just for observed limit so far, because it is very slow for expected limits --> should be updated --> FIXME)
      CodeToExecute = "cd /tmp/;";
      CodeToExecute += "combine -M HybridNew -n " + signal + " -m " + massStr + rangeStr + " shape_" + signal+".dat &> shape_" + signal + ".log;";
      CodeToExecute += "cd $OLDPWD;cp /tmp/shape_" + signal + ".* " + InputPattern+"/"+SHAPESTRING+EXCLUSIONDIR+"/." + ";";
      system(CodeToExecute.c_str());

      //if all went well, the combine tool created a new file containing the result of the limit in the form of a TTree
      //we can open this TTree and access the values for the expected limit, uncertainty bands, and observed limits.
      file = TFile::Open((string("/tmp/")+"higgsCombine"+signal+".HybridNew.mH"+massStr+".root").c_str());
      if(!file || file->IsZombie())return false;
      tree = (TTree*)file->Get("limit");
      if(!tree)return false;
      tree->GetBranch("mh"              )->SetAddress(&Tmass    );
      tree->GetBranch("limit"           )->SetAddress(&Tlimit   );
      tree->GetBranch("limit"           )->SetAddress(&Tlimit   );
      tree->GetBranch("limitErr"        )->SetAddress(&TlimitErr);
      tree->GetBranch("quantileExpected")->SetAddress(&TquantExp);
      for(int ientry=0;ientry<tree->GetEntriesFast();ientry++){
        tree->GetEntry(ientry);
        if(TquantExp==-1    ){ result.XSec_Obs      = Tlimit/1000.0;
        }else{printf("Quantil %f unused by the analysis --> check the code\n", TquantExp);
        }
      }
      file->Close();*/
   }



   //all done, save the results to file
   result.Save(InputPattern+"/"+SHAPESTRING+EXCLUSIONDIR+"/"+signal+".txt");
   return true;
}


bool useSample(int TypeMode, string sample) {
  if(TypeMode==0 && (sample=="Gluino_f10" || sample=="GluinoN_f10" || sample=="StopN" || sample=="Stop" || sample=="DY_Q2o3")) return true;
  if(TypeMode==2 && (sample=="Gluino_f10" || sample=="Gluino_f50" || sample=="Stop" || sample=="GMStau" || sample=="PPStau" || sample=="DY_Q2o3")) return true;
  if(TypeMode==3 && (sample=="Gluino_f10" || sample=="Gluino_f50" || sample=="Gluino_f100" || sample=="Stop")) return true;
  if(TypeMode==4) return true;
  if(TypeMode==5) return true;
  return false;
}

