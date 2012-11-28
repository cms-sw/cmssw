
#include <exception>
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
#include "TGraphAsymmErrors.h"
#include "TPaveText.h"


#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"

#include "DataFormats/FWLite/interface/LuminosityBlock.h"
#include "DataFormats/FWLite/interface/Run.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"

#include "../../ICHEP_Analysis/Analysis_Global.h"
#include "../../ICHEP_Analysis/Analysis_Samples.h"

using namespace fwlite;

#endif

struct stRun {
   unsigned int runId;
   std::vector<unsigned int> lumiId;   
};

std::map<unsigned int, unsigned int> timeMap;

void GetLumiBlocks_Core(vector<string>& fileNames, std::vector<stRun*>& RunMap);
void DumpJson(const std::vector<stRun*>& RunMap, string FileName);
void DumpTime(const std::map<unsigned int, unsigned int>& TimeMap, string FileName);
void RemoveRunsAfter(unsigned int RunMax, const std::vector<stRun*>& RunMap, std::vector<stRun*>& NewRunMap);

void GetLuminosity()
{
   std::vector<stSample> samples;
   GetSampleDefinition(samples, "../../ICHEP_Analysis/Analysis_Samples.txt");
   keepOnlySamplesOfNameX(samples,"Data8TeV");

   InitBaseDirectory();

   vector<string> inputFiles;
   for(unsigned int s=0;s<samples.size();s++){
      GetInputFiles(samples[s], BaseDirectory, inputFiles);
   }

   std::vector<stRun*> RunMap;
   GetLumiBlocks_Core(inputFiles, RunMap);
   DumpJson(RunMap, "out.json");
   DumpTime(timeMap,"out.time");

   //only used in 2011 data to check luminosity before after RPC trigger change
//   std::vector<stRun*> RunMapBefRPC;
//   RemoveRunsAfter(165970, RunMap, RunMapBefRPC);
//   DumpJson(RunMapBefRPC, "out_beforeRPCChange.json");
}

void GetLumiBlocks_Core(vector<string>& fileNames, std::vector<stRun*>& RunMap)
{
   printf("Running\n");
   for(unsigned int f=0;f<fileNames.size();f++){
     cout << fileNames[f].c_str() << endl;
     //TFile file(fileNames[f].c_str() );
     TFile *file = TFile::Open(fileNames[f].c_str() );
      fwlite::LuminosityBlock ls( file);
      for(ls.toBegin(); !ls.atEnd(); ++ls){
          
        //printf("Run = %i --> Lumi =%lu\n",ls.luminosityBlockAuxiliary().run(), (unsigned long)ls.luminosityBlockAuxiliary().id().value());
        int RunIndex = -1;
        for(unsigned int r=0;r<RunMap.size();r++){
           if(RunMap[r]->runId==ls.luminosityBlockAuxiliary().run()){
              RunIndex = (int)r;
              break;
           }
        }

        if(RunIndex<0){
           stRun* tmp = new stRun();
           tmp->runId=ls.luminosityBlockAuxiliary().run();
           tmp->lumiId.push_back(ls.luminosityBlockAuxiliary().id().value());
           RunMap.push_back(tmp);           
           //std::sort(RunMap.begin(), RunMap.end(), stRunLess);

           timeMap[tmp->runId] = ls.luminosityBlockAuxiliary().beginTime().unixTime()/3600;
        }else{
            stRun* tmp = RunMap[RunIndex];
           int LumiIndex = -1;
           for(unsigned int l=0;l<tmp->lumiId.size();l++){
              //printf("%lu vs %lu\n",tmp->lumiId[l], (unsigned long) ls.luminosityBlockAuxiliary().id().value() );
              if(tmp->lumiId[l]== (unsigned int) ls.luminosityBlockAuxiliary().id().value()){
                 LumiIndex = (int)l;
                 break;
              }
           }
           if(LumiIndex<0){
               tmp->lumiId.push_back((unsigned int) ls.luminosityBlockAuxiliary().id().value());
               std::sort(tmp->lumiId.begin(), tmp->lumiId.end());
            }
        }      
      }printf("\n");
   }
}

void RemoveRunsAfter(unsigned int RunMax, const std::vector<stRun*>& RunMap, std::vector<stRun*>& NewRunMap){
   for(unsigned int r=0;r<RunMap.size();r++){
      if(RunMap[r]->runId<RunMax)NewRunMap.push_back(RunMap[r]);
   }
}


void DumpJson(const std::vector<stRun*>& RunMap, string FileName){
   FILE* json = fopen(FileName.c_str(),"w");
   fprintf(json,"{");
   for(unsigned int r=0;r<RunMap.size();r++){
      stRun* tmp =  RunMap[r];
      fprintf(json,"\"%i\": [",tmp->runId);
      unsigned int l=0;
      while(l<tmp->lumiId.size()){
         unsigned int FirstLumi = tmp->lumiId[l];
         unsigned Size=0; 
         for(unsigned int l2=l;l2<tmp->lumiId.size() && FirstLumi+l2-l==tmp->lumiId[l2]; l2++){Size++;}
         fprintf(json,"[%i, %i]",FirstLumi,FirstLumi+Size-1);
         l+=Size;
         if(l<tmp->lumiId.size()) fprintf(json,",");
      }
      fprintf(json,"] ");
      if(r<RunMap.size()-1)fprintf(json,",");
   }  
   fprintf(json,"}");   
   fclose(json);
}



void DumpTime(const std::map<unsigned int, unsigned int>& TimeMap, string FileName){
   FILE* pFile = fopen(FileName.c_str(),"w");
   for(std::map<unsigned int, unsigned int>::iterator it = timeMap.begin(); it!=timeMap.end(); it++){
      fprintf(pFile, "%i,%i\n", it->first, it->second);
   }
   fclose(pFile);
}





