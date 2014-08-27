#ifndef __L1Analysis_L1AnalysisDTTF_H__
#define __L1Analysis_L1AnalysisDTTF_H__

//-------------------------------------------------------------------------------
// Created 06/01/2010 - A.C. Le Bihan
// 
//  
// Original code : L1TriggerDPG/L1Ntuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"
#include "DataFormats/Common/interface/Handle.h"

#include <vector>
#include "TMatrixD.h"
#include "L1AnalysisDTTFDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisDTTF 
  {
  public:
    L1AnalysisDTTF();
    ~L1AnalysisDTTF();
    
    void SetDTPH(const edm::Handle<L1MuDTChambPhContainer > L1MuDTChambPhContainer, unsigned int maxDTPH);
    void SetDTTH(const edm::Handle<L1MuDTChambThContainer > L1MuDTChambThContainer, unsigned int maxDTTH);
    void SetDTTR(const edm::Handle<L1MuDTTrackContainer >   L1MuDTTrackContainer,   unsigned int maxDTTR);
    void Reset() {dttf_.Reset();}
    L1AnalysisDTTFDataFormat * getData() {return &dttf_;}

  private : 
    L1AnalysisDTTFDataFormat dttf_;
  }; 
} 
#endif


