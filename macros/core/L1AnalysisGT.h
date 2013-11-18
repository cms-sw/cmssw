#ifndef __L1Analysis_L1AnalysisGT_H__
#define __L1Analysis_L1AnalysisGT_H__

#include <TTree.h>
#include <vector>

namespace L1Analysis
{
  class L1AnalysisGT
{

  public : 
  void initTree(TTree * tree);

  public:
  L1AnalysisGT() {}
  void print();
  bool check();    
  
  // ---- L1AnalysisGT information.
    std::vector<unsigned long long> gttw1;
    std::vector<unsigned long long> gttw2;
    std::vector<unsigned long long> gttt;

    
    //PSB info
    int            gtNele;
    std::vector<int>    gtBxel;
    std::vector<float>  gtRankel;
    std::vector<float>  gtPhiel;
    std::vector<float>  gtEtael;
    std::vector<bool>   gtIsoel;
    
    int            gtNjet;
    std::vector<int>    gtBxjet;
    std::vector<float>  gtRankjet;
    std::vector<float>  gtPhijet;
    std::vector<float>  gtEtajet;
    std::vector<bool>   gtTaujet;
    std::vector<bool>   gtFwdjet;
};
}


#endif

#ifdef l1ntuple_cxx

void L1Analysis::L1AnalysisGT::initTree(TTree * tree)
{
   tree->SetBranchAddress("gttw1",     &gttw1);
   tree->SetBranchAddress("gttw2",     &gttw2);
   tree->SetBranchAddress("gttt",      &gttt);
   tree->SetBranchAddress("gtNele",    &gtNele);
   tree->SetBranchAddress("gtBxel",    &gtBxel);
   tree->SetBranchAddress("gtRankel",  &gtRankel);
   tree->SetBranchAddress("gtPhiel",   &gtPhiel);
   tree->SetBranchAddress("gtEtael",   &gtEtael);
   tree->SetBranchAddress("gtNjet",    &gtNjet);
   tree->SetBranchAddress("gtBxjet",   &gtBxjet);
   tree->SetBranchAddress("gtRankjet", &gtRankjet);
   tree->SetBranchAddress("gtPhijet",  &gtPhijet);
   tree->SetBranchAddress("gtEtajet",  &gtEtajet);
}


void L1Analysis::L1AnalysisGT::print()
{
}

bool L1Analysis::L1AnalysisGT::check()
{
  bool test=true;
  /*  if (gtNele!=gtBxel.size() || gtNele!=gtRankel.size() ||
      gtNele!=gtPhiel.size() || gtNele!=gtEtael.size() ||
      gtNele!=gtIsoel.size()) test=false;
  if (gtNjet!=gtBxjet.size() || gtNjet!=gtRankjet.size() ||
      gtNjet!=gtPhijet.size() || gtNjet!=gtEtajet.size() ||
      gtNjet!=gtTaujet.size() || gtNjet!=gtFwdjet.size() ) test=false;
  */
  return test;
}




#endif


