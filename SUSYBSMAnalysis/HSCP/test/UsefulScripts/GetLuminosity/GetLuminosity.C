
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

using namespace fwlite;

#endif

struct stRun {
   unsigned int runId;
   std::vector<unsigned int> lumiId;   
};
std::vector<stRun*> RunMap;

void GetLumiBlocks_Core(vector<string>& fileNames);
void DumpJson();

void GetLuminosity()
{
  std::string BaseDirectory = "dcap://cmsdca.fnal.gov:24125/pnfs/cms/WAX/11/store/user/farrell3/EDMFiles/";
   vector<string> fileNames;
   fileNames.push_back(BaseDirectory + "Data_RunA_160404_163869.root");
   fileNames.push_back(BaseDirectory + "Data_RunA_165001_166033.root");
   fileNames.push_back(BaseDirectory + "Data_RunA_166034_166500.root");
   fileNames.push_back(BaseDirectory + "Data_RunA_166501_166893.root");
   fileNames.push_back(BaseDirectory + "Data_RunA_166894_167151.root");
   fileNames.push_back(BaseDirectory + "Data_RunA_167153_167913.root");

   GetLumiBlocks_Core(fileNames);
   DumpJson();
}

void GetLumiBlocks_Core(vector<string>& fileNames)
{
   printf("Running\n");
   for(unsigned int f=0;f<fileNames.size();f++){
     cout << fileNames[f].c_str() << endl;
     //TFile file(fileNames[f].c_str() );
     TFile *file = TFile::Open(fileNames[f].c_str() );
      fwlite::LuminosityBlock ls( file);
      for(ls.toBegin(); !ls.atEnd(); ++ls){
  
//        printf("Run = %i --> Lumi =%lu\n",ls.luminosityBlockAuxiliary().run(), (unsigned long)ls.luminosityBlockAuxiliary().id().value());
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



void DumpJson(){
   FILE* json = fopen("out.json","w");
   FILE* json_beforeRPCChange = fopen("out_beforeRPCChange.json","w");

   fprintf(json,"{");
   fprintf(json_beforeRPCChange,"{");
   for(unsigned int r=0;r<RunMap.size();r++){
      stRun* tmp =  RunMap[r];
      fprintf(json,"\"%i\": [",tmp->runId);
      if (tmp->runId<165970) fprintf(json_beforeRPCChange,"\"%i\": [",tmp->runId);

      unsigned int l=0;
      while(l<tmp->lumiId.size()){
         unsigned int FirstLumi = tmp->lumiId[l];
         unsigned Size=0; 
         for(unsigned int l2=l;l2<tmp->lumiId.size() && FirstLumi+l2-l==tmp->lumiId[l2]; l2++){Size++;}
         fprintf(json,"[%i, %i]",FirstLumi,FirstLumi+Size-1);
         if (tmp->runId<165970) fprintf(json_beforeRPCChange,"[%i, %i]",FirstLumi,FirstLumi+Size-1);
         l+=Size;
         if(l<tmp->lumiId.size()) fprintf(json,",");
         if (tmp->runId<165970) if(l<tmp->lumiId.size()) fprintf(json_beforeRPCChange,",");
      }
      fprintf(json,"] ");
      if (tmp->runId<165970) fprintf(json_beforeRPCChange,"] ");
      if(r<RunMap.size()-1)fprintf(json,",");
      if(r<RunMap.size()-1 && tmp->runId<165970)fprintf(json_beforeRPCChange,",");
   }  
   fprintf(json,"}");   
   fclose(json);

   fprintf(json_beforeRPCChange,"}");
   fclose(json_beforeRPCChange);
}
