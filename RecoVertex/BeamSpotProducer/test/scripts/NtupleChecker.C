#include "RecoVertex/BeamSpotProducer/interface/BeamSpotTreeData.h"
#include <TTree.h>
#include <TFile.h>
#include <TString.h>
#include <TList.h>
#include <TSystemFile.h>
#include <TSystemDirectory.h>
#include <iostream>

using namespace std;

void NtupleChecker(){
  TString path = "/uscms_data/d2/uplegger/CMSSW/CMSSW_3_8_0_pre7/src/RecoVertex/BeamSpotProducer/test/scripts/Ntuples/";
  TSystemDirectory sourceDir("fileDir",path); 
  TList* fileList = sourceDir.GetListOfFiles(); 
  TIter next(fileList);
  TSystemFile* fileName;
  int fileNumber = 1;
  int maxFiles = 1000;
  BeamSpotTreeData aData;
  while ((fileName = (TSystemFile*)next()) && fileNumber <= maxFiles){
    if(TString(fileName->GetName()) == "." || TString(fileName->GetName()) == ".."  ){
      continue;
    }
    TTree* aTree = 0;
    TFile file(path+fileName->GetName(),"READ");//STARTUP
    cout << "Opening file: " << path+fileName->GetName() << endl;
    file.cd();
//    aTree = (TTree*)file.Get("PrimaryVertices");
    aTree = (TTree*)file.Get("BeamSpotTree");
    cout << (100*fileNumber)/(fileList->GetSize()-2) << "% of files done." << endl;
    ++fileNumber;
    if(aTree == 0){
      cout << "Can't find the tree" << endl;
      continue;
    }
    aData.setBranchAddress(aTree);
    for(unsigned int entry=0; entry<aTree->GetEntries(); entry++){
      aTree->GetEntry(entry);
      cout << aData.getRun() << endl;
    }
    
  }
}
