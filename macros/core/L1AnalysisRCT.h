#ifndef __L1Analysis_L1AnalysisRCT_H__
#define __L1Analysis_L1AnalysisRCT_H__

#include <TTree.h>
#include <vector>

namespace L1Analysis
{
  class L1AnalysisRCT
{

  public : 
  void initTree(TTree * tree);

  public:
  L1AnalysisRCT() {}
  void print();
  bool check();    
  
    // ---- L1AnalysisRCT information.
    
    int maxRCTREG_;
    
    int rctRegSize;
    std::vector<float> rctRegEta;
    std::vector<float> rctRegPhi;
    std::vector<float> rctRegRnk;
    std::vector<int> rctRegVeto;
    std::vector<int> rctRegBx;
    std::vector<int> rctRegOverFlow;
    std::vector<int> rctRegMip;
    std::vector<int> rctRegFGrain;   
   
    int rctEmSize;   
    std::vector<int> rctIsIsoEm;
    std::vector<float> rctEmEta;
    std::vector<float> rctEmPhi;
    std::vector<float> rctEmRnk;
    std::vector<int> rctEmBx;

};
}


#endif

#ifdef l1ntuple_cxx


void L1Analysis::L1AnalysisRCT::initTree(TTree * tree)
{
   tree->SetBranchAddress("maxRCTREG_",     &maxRCTREG_);
   tree->SetBranchAddress("rctRegSize",     &rctRegSize);
   tree->SetBranchAddress("rctRegEta",      &rctRegEta);
   tree->SetBranchAddress("rctRegPhi",      &rctRegPhi);
   tree->SetBranchAddress("rctRegRnk",      &rctRegRnk);
   tree->SetBranchAddress("rctRegVeto",     &rctRegVeto);
   tree->SetBranchAddress("rctRegBx",       &rctRegBx);
   tree->SetBranchAddress("rctRegOverFlow", &rctRegOverFlow);
   tree->SetBranchAddress("rctRegMip",      &rctRegMip);
   tree->SetBranchAddress("rctRegFGrain",   &rctRegFGrain);
   tree->SetBranchAddress("rctEmSize",      &rctEmSize);
   tree->SetBranchAddress("rctIsIsoEm",     &rctIsIsoEm);
   tree->SetBranchAddress("rctEmEta",       &rctEmEta);
   tree->SetBranchAddress("rctEmPhi",       &rctEmPhi);
   tree->SetBranchAddress("rctEmRnk",       &rctEmRnk);
   tree->SetBranchAddress("rctEmBx",        &rctEmBx);
}


void L1Analysis::L1AnalysisRCT::print()
{
}

bool L1Analysis::L1AnalysisRCT::check()
{
  bool test=true;
  return test;
}

#endif


