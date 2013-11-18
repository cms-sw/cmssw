#ifndef __L1Analysis_L1AnalysisGCT_H__
#define __L1Analysis_L1AnalysisGCT_H__

#include <TTree.h>
#include <vector>

namespace L1Analysis
{
  class L1AnalysisGCT
{

  public : 
  void initTree(TTree * tree);

  public:
  L1AnalysisGCT() {}
  ~L1AnalysisGCT() {}
  void print();
  bool check();    
  
  // ---- L1AnalysisGCT information.
    
  bool verbose_;
    
  int gctIsoEmSize;
  std::vector<float> gctIsoEmEta;
  std::vector<float> gctIsoEmPhi;
  std::vector<float> gctIsoEmRnk;
  std::vector<int>   gctIsoEmBx;
    
  int gctNonIsoEmSize;
  std::vector<float> gctNonIsoEmEta;
  std::vector<float> gctNonIsoEmPhi;
  std::vector<float> gctNonIsoEmRnk;
  std::vector<int>   gctNonIsoEmBx;
 
  int gctCJetSize;    
  std::vector<float> gctCJetEta;
  std::vector<float> gctCJetPhi;
  std::vector<float> gctCJetRnk;
  std::vector<int>   gctCJetBx;
    
  int gctFJetSize;    
  std::vector<float> gctFJetEta;
  std::vector<float> gctFJetPhi;
  std::vector<float> gctFJetRnk;
  std::vector<int>   gctFJetBx;

  int gctTJetSize;
  std::vector<float> gctTJetEta;
  std::vector<float> gctTJetPhi;
  std::vector<float> gctTJetRnk;
  std::vector<int>   gctTJetBx;
    
  float gctEtMiss;
  float gctEtMissPhi;   
  int   gctEtMissBx;
 
  float gctHtMiss;
  float gctHtMissPhi;
  int   gctHtMissBx;
  
  float gctEtHad;
  int   gctEtHadBx;
  
  float gctEtTot; 
  int   gctEtTotBx;

  int                gctHFRingEtSumSize;    
  std::vector<float> gctHFRingEtSumEta; 
  std::vector<int> gctHFRingBx;
  
  float              gctHFBitCountsSize;
  std::vector<float> gctHFBitCountsEta;
  std::vector<int> gctHFBitCountsBx;
};
}

#endif

#ifdef l1ntuple_cxx

void L1Analysis::L1AnalysisGCT::initTree(TTree * tree)
{
   tree->SetBranchAddress("verbose_",           &verbose_);
   tree->SetBranchAddress("gctIsoEmSize",       &gctIsoEmSize      );
   tree->SetBranchAddress("gctIsoEmEta",        &gctIsoEmEta       );
   tree->SetBranchAddress("gctIsoEmPhi",        &gctIsoEmPhi       );
   tree->SetBranchAddress("gctIsoEmRnk",        &gctIsoEmRnk       );
   tree->SetBranchAddress("gctIsoEmBx",         &gctIsoEmBx        );
   
   tree->SetBranchAddress("gctNonIsoEmSize",    &gctNonIsoEmSize   );
   tree->SetBranchAddress("gctNonIsoEmEta",     &gctNonIsoEmEta    );
   tree->SetBranchAddress("gctNonIsoEmPhi",     &gctNonIsoEmPhi    );
   tree->SetBranchAddress("gctNonIsoEmRnk",     &gctNonIsoEmRnk    );
   tree->SetBranchAddress("gctNonIsoEmBx",      &gctNonIsoEmBx     );
   
   tree->SetBranchAddress("gctCJetSize",        &gctCJetSize       );
   tree->SetBranchAddress("gctCJetEta",         &gctCJetEta        );
   tree->SetBranchAddress("gctCJetPhi",         &gctCJetPhi        );
   tree->SetBranchAddress("gctCJetRnk",         &gctCJetRnk        ); 
   tree->SetBranchAddress("gctCJetBx",          &gctCJetBx         );
   
   tree->SetBranchAddress("gctFJetSize",        &gctFJetSize       );
   tree->SetBranchAddress("gctFJetEta",         &gctFJetEta        );
   tree->SetBranchAddress("gctFJetPhi",         &gctFJetPhi        );
   tree->SetBranchAddress("gctFJetRnk",         &gctFJetRnk        );
   tree->SetBranchAddress("gctFJetBx",          &gctFJetBx         );
    
   tree->SetBranchAddress("gctTJetSize",        &gctTJetSize       );
   tree->SetBranchAddress("gctTJetEta",         &gctTJetEta        );
   tree->SetBranchAddress("gctTJetPhi",         &gctTJetPhi        );
   tree->SetBranchAddress("gctTJetRnk",         &gctTJetRnk        ); 
   tree->SetBranchAddress("gctTJetBx",          &gctTJetBx         );
    
   tree->SetBranchAddress("gctEtMiss",          &gctEtMiss         );
   tree->SetBranchAddress("gctEtMissPhi",       &gctEtMissPhi      );
   tree->SetBranchAddress("gctEtMissBx",        &gctEtMissBx       );
   
   tree->SetBranchAddress("gctHtMiss",          &gctHtMiss         );
   tree->SetBranchAddress("gctHtMissPhi",       &gctHtMissPhi      );
   tree->SetBranchAddress("gctHtMissBx",        &gctHtMissBx       );
  
   tree->SetBranchAddress("gctEtHad",           &gctEtHad          );  
   tree->SetBranchAddress("gctEtHadBx",         &gctEtHadBx        );
   tree->SetBranchAddress("gctEtTot",           &gctEtTot          );
   tree->SetBranchAddress("gctEtTotBx",         &gctEtTotBx        );
   
   tree->SetBranchAddress("gctHFRingEtSumSize", &gctHFRingEtSumSize);
   tree->SetBranchAddress("gctHFRingEtSumEta",  &gctHFRingEtSumEta );
   tree->SetBranchAddress("gctHFRingBx",        &gctHFRingBx );
   
   tree->SetBranchAddress("gctHFBitCountsSize", &gctHFBitCountsSize);
   tree->SetBranchAddress("gctHFBitCountsEta",  &gctHFBitCountsEta );
   tree->SetBranchAddress("gctHFBitCountsBx",   &gctHFBitCountsBx );
}




bool L1Analysis::L1AnalysisGCT::check()
{
  bool test=true;
  /* if (gctIsoEmSize!=gctIsoEmEta.size() || gctIsoEmSize!=gctIsoEmPhi.size()  || gctIsoEmSize!=gctIsoEmRnk.size()) test=false;
  if (gctNonIsoEmSize!=gctNonIsoEmEta.size() || gctNonIsoEmSize!=gctNonIsoEmPhi.size() || gctNonIsoEmSize!=gctNonIsoEmRnk.size()) test=false;
  if (gctCJetSize!=gctCJetEta.size() || gctCJetSize!=gctCJetPhi.size() || gctCJetSize!=gctCJetRnk.size()) test=false;
  if (gctFJetSize!=gctFJetEta.size() || gctFJetSize!=gctFJetPhi.size() || gctFJetSize!=gctFJetRnk.size()) test=false;
  if (gctTJetSize!=gctTJetEta.size() || gctTJetSize!=gctTJetPhi.size() || gctTJetSize!=gctTJetRnk.size()) test=false;
  if (gctHFRingEtSumSize!=gctHFRingEtSumEta.size()) test=false;
  if (gctHFBitCountsSize!=gctHFBitCountsEta.size()) test=false;*/
  return test;
}



#endif


