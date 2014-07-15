#ifndef __L1Analysis_L1AnalysisGCTDataFormat_H__
#define __L1Analysis_L1AnalysisGCTDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1TriggerDPG/L1Ntuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisGCTDataFormat
  {
    L1AnalysisGCTDataFormat(){Reset();};
    ~L1AnalysisGCTDataFormat(){};
    
    void Reset() {    
      IsoEmSize = 0;    
      NonIsoEmSize = 0;
      CJetSize = 0;
      FJetSize = 0;
      TJetSize = 0;
      EtMissSize = 0;
      HtMissSize = 0;
      EtHadSize = 0;
      EtTotSize = 0;
      HFRingEtSumSize = 0; 
      HFBitCountsSize = 0;
      
      
      IsoEmEta.clear();
      IsoEmPhi.clear();
      IsoEmRnk.clear();
      IsoEmBx.clear();
      NonIsoEmEta.clear();
      NonIsoEmPhi.clear();
      NonIsoEmRnk.clear();
      NonIsoEmBx.clear();
      CJetEta.clear();
      CJetPhi.clear();
      CJetRnk.clear(); 
      CJetBx.clear();
      FJetEta.clear();
      FJetPhi.clear();
      FJetRnk.clear();
      FJetBx.clear();
      TJetEta.clear();
      TJetPhi.clear();
      TJetRnk.clear(); 
      TJetBx.clear(); 
      EtMiss.clear();
      EtMissPhi.clear();
      EtMissBX.clear();
      HtMiss.clear();
      HtMissPhi.clear();
      HtMissBX.clear();
      EtHad.clear();
      EtHadBX.clear();
      EtTot.clear();
      EtTotBX.clear();
      
      HFRingEtSumEta.clear();
      HFBitCountsEta.clear(); 
    }
    
    void Init() {
      // removed really really stupid stuff from this method - JB, 7 Aug 2012
    }

    // ---- L1AnalysisGCTDataFormat information.
    
    int IsoEmSize;
    std::vector<float> IsoEmEta;
    std::vector<float> IsoEmPhi;
    std::vector<float> IsoEmRnk;
    std::vector<int>   IsoEmBx;
    
    int NonIsoEmSize;
    std::vector<float> NonIsoEmEta;
    std::vector<float> NonIsoEmPhi;
    std::vector<float> NonIsoEmRnk;
    std::vector<int>   NonIsoEmBx;
       
    int CJetSize;    
    std::vector<float> CJetEta;
    std::vector<float> CJetPhi;
    std::vector<float> CJetRnk;
    std::vector<int>   CJetBx;
     
    int FJetSize;    
    std::vector<float> FJetEta;
    std::vector<float> FJetPhi;
    std::vector<float> FJetRnk;
    std::vector<int>   FJetBx;
 
    int TJetSize;
    std::vector<float> TJetEta;
    std::vector<float> TJetPhi;
    std::vector<float> TJetRnk;
    std::vector<int>   TJetBx;
    
    int EtMissSize;
    std::vector<float> EtMiss;
    std::vector<float> EtMissPhi;
    std::vector<float> EtMissBX;

    int HtMissSize;
    std::vector<float> HtMiss;
    std::vector<float> HtMissPhi;
    std::vector<float> HtMissBX;

    int EtHadSize;
    std::vector<float> EtHad;
    std::vector<float> EtHadBX;

    int EtTotSize;
    std::vector<float> EtTot;
    std::vector<float> EtTotBX;
    
    int HFRingEtSumSize;
    std::vector<float> HFRingEtSumEta;

    float HFBitCountsSize;
    std::vector<float> HFBitCountsEta;
    
    
  }; 
} 
#endif


