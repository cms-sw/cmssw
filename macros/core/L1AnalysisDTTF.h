#ifndef __L1Analysis_L1AnalysisDTTF_H__
#define __L1Analysis_L1AnalysisDTTF_H__

#include <TTree.h>
#include <TMatrixD.h>

namespace L1Analysis
{
  class L1AnalysisDTTF
{

  public : 
  void initTree(TTree * tree);

  public:
  L1AnalysisDTTF() {}
  void print(); 
  bool check();   
  
    // ---- L1AnalysisDTTF information.
    int dttf_phSize;    
    std::vector<int>   dttf_phBx; 
    std::vector<int>   dttf_phWh; 
    std::vector<int>   dttf_phSe; 
    std::vector<int>   dttf_phSt; 
    std::vector<float> dttf_phAng;
    std::vector<float> dttf_phBandAng;
    std::vector<int>   dttf_phCode; 
    std::vector<float> dttf_phX;
    std::vector<float> dttf_phY;
    
    int dttf_thSize;
    std::vector<int>   dttf_thBx;
    std::vector<int>   dttf_thWh;
    std::vector<int>   dttf_thSe;
    std::vector<int>   dttf_thSt;
    std::vector<float> dttf_thX; 
    std::vector<float> dttf_thY; 
    
    TMatrixD dttf_thTheta;
    TMatrixD dttf_thCode; 
    
    int dttf_trSize;
    std::vector<int>   dttf_trBx; 
    std::vector<int>   dttf_trTag;
    std::vector<int>   dttf_trQual; 
    std::vector<int>   dttf_trPtPck;
    std::vector<float> dttf_trPtVal;
    std::vector<int>   dttf_trPhiPck; 
    std::vector<float> dttf_trPhiVal; 
    std::vector<int>   dttf_trPhiGlob; 
    std::vector<int>   dttf_trChPck;
    std::vector<int>   dttf_trWh; 
    std::vector<int>   dttf_trSc; 
};
}

#endif

#ifdef l1ntuple_cxx

void L1Analysis::L1AnalysisDTTF::initTree(TTree * tree)
{
  tree->SetBranchAddress("dttf_phSize",    &dttf_phSize);
  tree->SetBranchAddress("dttf_phBx",      &dttf_phBx);
  tree->SetBranchAddress("dttf_phWh",      &dttf_phWh);
  tree->SetBranchAddress("dttf_phSe",      &dttf_phSe);
  tree->SetBranchAddress("dttf_phSt",      &dttf_phSt);
  tree->SetBranchAddress("dttf_phAng",     &dttf_phAng);
  tree->SetBranchAddress("dttf_phBandAng", &dttf_phBandAng);
  tree->SetBranchAddress("dttf_phCode",    &dttf_phCode);
  tree->SetBranchAddress("dttf_phX",       &dttf_phX);
  tree->SetBranchAddress("dttf_phY",       &dttf_phY);
  tree->SetBranchAddress("dttf_thSize",    &dttf_thSize);
  tree->SetBranchAddress("dttf_thBx",      &dttf_thBx);
  tree->SetBranchAddress("dttf_thWh",      &dttf_thWh);
  tree->SetBranchAddress("dttf_thSe",      &dttf_thSe);
  tree->SetBranchAddress("dttf_thSt",      &dttf_thSt);
  tree->SetBranchAddress("dttf_thX",       &dttf_thX);
  tree->SetBranchAddress("dttf_thY",       &dttf_thY);
  tree->SetBranchAddress("dttf_thTheta",   &dttf_thTheta);
  tree->SetBranchAddress("dttf_thCode",    &dttf_thCode);
  tree->SetBranchAddress("dttf_trSize",    &dttf_trSize);
  tree->SetBranchAddress("dttf_trBx",      &dttf_trBx);
  tree->SetBranchAddress("dttf_trTag",     &dttf_trTag);
  tree->SetBranchAddress("dttf_trQual",    &dttf_trQual);
  tree->SetBranchAddress("dttf_trPtPck",   &dttf_trPtPck);
  tree->SetBranchAddress("dttf_trPtVal",   &dttf_trPtVal);
  tree->SetBranchAddress("dttf_trPhiPck",  &dttf_trPhiPck);
  tree->SetBranchAddress("dttf_trPhiVal",  &dttf_trPhiVal);
  tree->SetBranchAddress("dttf_trPhiGlob", &dttf_trPhiGlob);
  tree->SetBranchAddress("dttf_trChPck",   &dttf_trChPck);
  tree->SetBranchAddress("dttf_trWh",      &dttf_trWh);
  tree->SetBranchAddress("dttf_trSc",      &dttf_trSc);
}


void L1Analysis::L1AnalysisDTTF::print()
{
}

bool L1Analysis::L1AnalysisDTTF::check()
{
  bool test=true;
  return test;
}
#endif


