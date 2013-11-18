#ifndef __L1Analysis_L1AnalysisRecoRpcHit_H__
#define __L1Analysis_L1AnalysisRecoRpcHit_H__

//-------------------------------------------------------------------------------
// Created 21/11/2012 - C. Battilana
// 
//
// Original code : L1TriggerDPG/L1Ntuples/L1RecoMuonProducer - Luigi Guiducci
//-------------------------------------------------------------------------------

#include "FWCore/Framework/interface/Event.h"
#include "L1AnalysisRecoRpcHitDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisRecoRpcHit
  {
  public:
    L1AnalysisRecoRpcHit();
    ~L1AnalysisRecoRpcHit();
    
    void Reset() {recoRpcHit_.Reset();}
    //void Print(std::ostream &os = std::cout) const;
    void Set();
    L1AnalysisRecoRpcHitDataFormat * getData() {return &recoRpcHit_;}

  private :
    L1AnalysisRecoRpcHitDataFormat recoRpcHit_;
  }; 
}
#endif



