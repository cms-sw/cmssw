#include "RecoVertex/BeamSpotProducer/interface/BeamSpotTreeData.h"
#include <TTree.h>


BeamSpotTreeData::BeamSpotTreeData(){}
BeamSpotTreeData::~BeamSpotTreeData(){}


//--------------------------------------------------------------------------------------------------
void BeamSpotTreeData::branch(TTree* tree){
  tree->Branch("run"	      , &run_	       , "run/i");
  tree->Branch("lumi"	      , &lumi_	       , "lumi/i");
  tree->Branch("bunchCrossing", &bunchCrossing_, "bunchCrossing/i");
  tree->Branch("pvData"	      , &pvData_       , "bunchCrossing:position[3]:posError[3]:posCorr[3]/F");
}

//--------------------------------------------------------------------------------------------------
void BeamSpotTreeData::setBranchAddress(TTree* tree){
  tree->SetBranchAddress("run"	        , &run_	         );
  tree->SetBranchAddress("lumi"	        , &lumi_	 );
  tree->SetBranchAddress("bunchCrossing", &bunchCrossing_);
  tree->SetBranchAddress("pvData"	, &pvData_       );
}
