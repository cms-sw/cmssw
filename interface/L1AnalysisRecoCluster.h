#ifndef __L1Analysis_L1AnalysisRecoCluster_H__
#define __L1Analysis_L1AnalysisRecoCluster_H__

//-------------------------------------------------------------------------------
// Created 03/03/2010 - A.C. Le Bihan
// 
//
// Addition of reco information
//-------------------------------------------------------------------------------

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "L1AnalysisRecoClusterDataFormat.h"

namespace L1Analysis
{
  struct L1AnalysisRecoCluster
  {
    L1AnalysisRecoCluster(); 
    ~L1AnalysisRecoCluster();
    
    void Set(const reco::CaloClusterCollection &caloClusterCollection, unsigned maxCl);
    void Set(const reco::SuperClusterCollection &superClusterCollection, unsigned maxCl);
    L1AnalysisRecoClusterDataFormat * getData() {return &recoCluster_;}
    void Reset() {recoCluster_.Reset();}

  public :
    L1AnalysisRecoClusterDataFormat recoCluster_;
  }; 
}
#endif


