#ifndef __L1Analysis_L1AnalysisGCT_H__
#define __L1Analysis_L1AnalysisGCT_H__

//-------------------------------------------------------------------------------
// Created 06/01/2010 - A.C. Le Bihan
// 
// 
// Original code : L1TriggerDPG/L1Ntuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/Common/interface/Handle.h"

#include "L1AnalysisGCTDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisGCT
  {
  public:
    L1AnalysisGCT();
    L1AnalysisGCT(bool verbose);
    ~L1AnalysisGCT();
    
    void SetJet(const edm::Handle < L1GctJetCandCollection > l1CenJets,
                const edm::Handle < L1GctJetCandCollection > l1ForJets,
                const edm::Handle < L1GctJetCandCollection > l1TauJets);
		
    void SetES(const edm::Handle < L1GctEtMissCollection > l1EtMiss, const edm::Handle < L1GctHtMissCollection >  l1HtMiss,
               const edm::Handle < L1GctEtHadCollection > l1EtHad, const edm::Handle < L1GctEtTotalCollection > l1EtTotal); 	   
    
    void SetHFminbias(const edm::Handle < L1GctHFRingEtSumsCollection > l1HFSums, 
                      const edm::Handle < L1GctHFBitCountsCollection > l1HFCounts);
		      
    void SetEm(const edm::Handle < L1GctEmCandCollection > l1IsoEm, 
               const edm::Handle < L1GctEmCandCollection > l1NonIsoEm);

    void Reset() {gct_.Reset();}

    L1AnalysisGCTDataFormat * getData() {return &gct_;}
 
  private :
    bool verbose_;
    L1AnalysisGCTDataFormat gct_;
  }; 
} 
#endif


