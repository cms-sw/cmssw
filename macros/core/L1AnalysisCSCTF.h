#ifndef __L1Analysis_L1AnalysisCSCTF_H__
#define __L1Analysis_L1AnalysisCSCTF_H__

#include <TTree.h>
#include <vector>
#include <TMatrixD.h>

namespace L1Analysis
{
  class L1AnalysisCSCTF
{

  public : 
  void initTree(TTree * tree);

  public:
  L1AnalysisCSCTF() {}
  void print();
  bool check();    
  
    // ---- L1AnalysisCSCTF information.
   int csctf_trSize;
    std::vector<int> csctf_trEndcap; 
    std::vector<int> csctf_trSector; 
    
    std::vector<int> csctf_trBx; 

    // if set to 1 track has lct from the station 
    std::vector<int> csctf_trME1ID; 
    std::vector<int> csctf_trME2ID; 
    std::vector<int> csctf_trME3ID; 
    std::vector<int> csctf_trME4ID; 
    std::vector<int> csctf_trMB1ID;     

    std::vector<int> csctf_trOutputLink; 
  
    // some input of the PT LUTs 
    std::vector<int> csctf_trCharge; 
    std::vector<int> csctf_trChargeValid; 
    std::vector<int> csctf_trForR; 
    std::vector<int> csctf_trPhi23; 
    std::vector<int> csctf_trPhi12;   
    std::vector<int> csctf_trPhiSign;   

    // in bits... 
    std::vector<int> csctf_trEtaBit;   
    std::vector<int> csctf_trPhiBit;   
    std::vector<int> csctf_trPtBit;   
  
    // ... converted 
    std::vector<float> csctf_trEta;   
    std::vector<float> csctf_trPhi;   
    std::vector<float> csctf_trPhi_02PI; 
    std::vector<float> csctf_trPt;   

    // + useful information
    std::vector<int> csctf_trMode;
    std::vector<int> csctf_trQuality;

    //---------------------------------------------------------------------- 
    // LCT (STUBS FORMING THE TRACK)  
    //---------------------------------------------------------------------- 
    std::vector<int> csctf_trNumLCTs; // it contains the number of LCT forming a track 
       
    TMatrixD csctf_trLctEndcap; 
    TMatrixD csctf_trLctSector; 
    TMatrixD csctf_trLctSubSector; 
    TMatrixD csctf_trLctBx; 
    TMatrixD csctf_trLctBx0; 
       
    TMatrixD csctf_trLctStation; 
    TMatrixD csctf_trLctRing; 
    TMatrixD csctf_trLctChamber; 
    TMatrixD csctf_trLctTriggerCSCID; 
    TMatrixD csctf_trLctFpga;	  

     // note: the SPs return them in bits 
    TMatrixD csctf_trLctlocalPhi; 
    TMatrixD csctf_trLctlocalPhi_bend; 
    TMatrixD csctf_trLctglobalPhi;   
    TMatrixD csctf_trLctglobalEta; 
    TMatrixD csctf_trLctCLCT_pattern; 
    TMatrixD csctf_trLctQuality; 

    TMatrixD csctf_trLctstripNum;   
    TMatrixD csctf_trLctwireGroup;
  
    //---------------------------------------------------------------------- 
    // ALL LCT 
    //---------------------------------------------------------------------- 
    int csctf_lctSize;
    std::vector<int> csctf_lctEndcap; 
    std::vector<int> csctf_lctSector; 
    std::vector<int> csctf_lctSubSector; 
    std::vector<int> csctf_lctBx; 
    std::vector<int> csctf_lctBx0; 
    std::vector<int> csctf_lctStation; 
    std::vector<int> csctf_lctRing; 
    std::vector<int> csctf_lctChamber; 
    std::vector<int> csctf_lctTriggerCSCID; 
    std::vector<int> csctf_lctFpga;     

    // note: the SPs return them in bits 
    std::vector<int> csctf_lctlocalPhi; 
    std::vector<int> csctf_lctlocalPhi_bend; 
    std::vector<int> csctf_lctglobalPhi;   
    std::vector<int> csctf_lctglobalEta; 
    std::vector<int> csctf_lctstripNum;   
    std::vector<int> csctf_lctwireGroup;   

    //--------------------------------------------------------------------------- 
    // BASIC CSCTF information 
    //--------------------------------------------------------------------------- 
    int csctf_nsp; // num of SPs active in the event 
    std::vector<int> csctf_stSPslot; 
    std::vector<int> csctf_stL1A_BXN; 
    std::vector<unsigned long int> csctf_stTrkCounter; 
    std::vector<unsigned long int> csctf_stOrbCounter;   
};
}
#endif
#ifdef l1ntuple_cxx

void L1Analysis::L1AnalysisCSCTF::initTree(TTree * tree)
{
  tree->SetBranchAddress("csctf_trSize",         &csctf_trSize);
  tree->SetBranchAddress("csctf_trEndcap",       &csctf_trEndcap);
  tree->SetBranchAddress("csctf_trSector",       &csctf_trSector);
  tree->SetBranchAddress("csctf_trBx",           &csctf_trBx);
  tree->SetBranchAddress("csctf_trME1ID",        &csctf_trME1ID);
  tree->SetBranchAddress("csctf_trME2ID",        &csctf_trME2ID);
  tree->SetBranchAddress("csctf_trME3ID",        &csctf_trME3ID);
  tree->SetBranchAddress("csctf_trME4ID",        &csctf_trME4ID);
  tree->SetBranchAddress("csctf_trMB1ID",        &csctf_trMB1ID);
  tree->SetBranchAddress("csctf_trOutputLink",   &csctf_trOutputLink);
   tree->SetBranchAddress("csctf_trCharge",      &csctf_trCharge);
   tree->SetBranchAddress("csctf_trChargeValid", &csctf_trChargeValid);
   tree->SetBranchAddress("csctf_trForR",        &csctf_trForR);
   tree->SetBranchAddress("csctf_trPhi23",       &csctf_trPhi23);
   tree->SetBranchAddress("csctf_trPhi12",       &csctf_trPhi12);
   tree->SetBranchAddress("csctf_trPhiSign",     &csctf_trPhiSign);
   tree->SetBranchAddress("csctf_trEtaBit",      &csctf_trEtaBit);
   tree->SetBranchAddress("csctf_trPhiBit",      &csctf_trPhiBit);
   tree->SetBranchAddress("csctf_trPtBit",       &csctf_trPtBit);
   tree->SetBranchAddress("csctf_trEta",         &csctf_trEta);
   tree->SetBranchAddress("csctf_trPhi",         &csctf_trPhi);
   tree->SetBranchAddress("csctf_trPhi_02PI",    &csctf_trPhi_02PI);
   tree->SetBranchAddress("csctf_trPt",          &csctf_trPt);
   tree->SetBranchAddress("csctf_trMode",        &csctf_trMode);
   tree->SetBranchAddress("csctf_trQuality",     &csctf_trQuality);
   tree->SetBranchAddress("csctf_trNumLCTs",     &csctf_trNumLCTs);
   tree->SetBranchAddress("csctf_trLctEndcap",   &csctf_trLctEndcap);
   tree->SetBranchAddress("csctf_trLctSector",   &csctf_trLctSector);
   tree->SetBranchAddress("csctf_trLctSubSector", &csctf_trLctSubSector);
   tree->SetBranchAddress("csctf_trLctBx",        &csctf_trLctBx);
   tree->SetBranchAddress("csctf_trLctBx0",       &csctf_trLctBx0);
   tree->SetBranchAddress("csctf_trLctStation",   &csctf_trLctStation);
   tree->SetBranchAddress("csctf_trLctRing",      &csctf_trLctRing);
   tree->SetBranchAddress("csctf_trLctChamber",   &csctf_trLctChamber);
   tree->SetBranchAddress("csctf_trLctTriggerCSCID", &csctf_trLctTriggerCSCID);
   tree->SetBranchAddress("csctf_trLctFpga",      &csctf_trLctFpga);
   tree->SetBranchAddress("csctf_trLctlocalPhi",  &csctf_trLctlocalPhi);
   //tree->SetBranchAddress("csctf_trLctlocalPhi_bend",  &csctf_trLctlocalPhi_bend);
   tree->SetBranchAddress("csctf_trLctCLCT_pattern",  &csctf_trLctCLCT_pattern);
   tree->SetBranchAddress("csctf_trLctQuality",  &csctf_trLctQuality);
   tree->SetBranchAddress("csctf_trLctglobalPhi", &csctf_trLctglobalPhi);
   tree->SetBranchAddress("csctf_trLctglobalEta", &csctf_trLctglobalEta);
   tree->SetBranchAddress("csctf_trLctstripNum",  &csctf_trLctstripNum);
   tree->SetBranchAddress("csctf_trLctwireGroup", &csctf_trLctwireGroup);
   tree->SetBranchAddress("csctf_lctSize",        &csctf_lctSize);
   tree->SetBranchAddress("csctf_lctEndcap",      &csctf_lctEndcap);
   tree->SetBranchAddress("csctf_lctSector",      &csctf_lctSector);
   tree->SetBranchAddress("csctf_lctSubSector",   &csctf_lctSubSector);
   tree->SetBranchAddress("csctf_lctBx",          &csctf_lctBx);
   tree->SetBranchAddress("csctf_lctBx0",         &csctf_lctBx0);
   tree->SetBranchAddress("csctf_lctStation",     &csctf_lctStation);
   tree->SetBranchAddress("csctf_lctRing",        &csctf_lctRing);
   tree->SetBranchAddress("csctf_lctChamber",     &csctf_lctChamber);
   tree->SetBranchAddress("csctf_lctTriggerCSCID", &csctf_lctTriggerCSCID);
   tree->SetBranchAddress("csctf_lctFpga",        &csctf_lctFpga);
   tree->SetBranchAddress("csctf_lctlocalPhi",    &csctf_lctlocalPhi);
   //tree->SetBranchAddress("csctf_lctlocalPhi_bend",    &csctf_lctlocalPhi_bend);
   tree->SetBranchAddress("csctf_lctCLCT_pattern",    &csctf_lctCLCT_pattern);
   tree->SetBranchAddress("csctf_lctQuality",    &csctf_lctQuality);
   tree->SetBranchAddress("csctf_lctglobalPhi",   &csctf_lctglobalPhi);
   tree->SetBranchAddress("csctf_lctglobalEta",   &csctf_lctglobalEta);
   tree->SetBranchAddress("csctf_lctstripNum",    &csctf_lctstripNum);
   tree->SetBranchAddress("csctf_lctwireGroup",   &csctf_lctwireGroup);
   tree->SetBranchAddress("csctf_nsp",            &csctf_nsp);
   tree->SetBranchAddress("csctf_stSPslot",       &csctf_stSPslot);
   tree->SetBranchAddress("csctf_stL1A_BXN",      &csctf_stL1A_BXN);
   tree->SetBranchAddress("csctf_stTrkCounter",   &csctf_stTrkCounter);
   tree->SetBranchAddress("csctf_stOrbCounter",   &csctf_stOrbCounter);
}


void L1Analysis::L1AnalysisCSCTF::print()
{
}

bool L1Analysis::L1AnalysisCSCTF::check()
{
  bool test=true;
  return test;
}

#endif


