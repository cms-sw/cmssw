#ifndef __L1Analysis_L1AnalysisRecoTrack_H__
#define __L1Analysis_L1AnalysisRecoTrack_H__

#include <TChain.h>
#include <iostream>
#include <vector>

namespace L1Analysis
{
  class L1AnalysisRecoTrack
  {
    
  public :
    void initTree(TChain * tree, const std::string & className);
  
  public:
    L1AnalysisRecoTrack() {}
    void print();
    
    // ---- General L1AnalysisRecoTrack information.    
    unsigned nTrk;
    unsigned nHighPurity;
    double   fHighPurity;
};
}


#endif

#ifdef l1ntuple_cxx


void L1Analysis::L1AnalysisRecoTrack::initTree(TChain * tree, const std::string & className)
{
  SetBranchAddress(tree,"nTrk",        className, &nTrk );
  SetBranchAddress(tree,"nHighPurity", className, &nHighPurity);
  SetBranchAddress(tree,"fHighPurity", className, &fHighPurity);
}


void L1Analysis::L1AnalysisRecoTrack::print()
{
}

#endif


