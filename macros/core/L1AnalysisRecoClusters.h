#ifndef __L1Analysis_L1AnalysisRecoClusters_H__
#define __L1Analysis_L1AnalysisRecoClusters_H__

#include <TTree.h>
#include <iostream>
#include <vector>

namespace L1Analysis
{
  class L1AnalysisRecoClusters
  {
    
  public :
    void initTree(TTree * tree, const std::string & className);
  
  public:
    L1AnalysisRecoClusters() {}
    void print();
    
    // ---- General L1AnalysisRecoClusters information.    
    unsigned maxCl_;
    unsigned nClusters;
    std::vector<double> clusterEta;
    std::vector<double> clusterPhi;
    std::vector<double> clusterEt;
    std::vector<double> clusterE;
};
}


#endif

#ifdef l1ntuple_cxx


void L1Analysis::L1AnalysisRecoClusters::initTree(TTree * tree, const std::string & className)
{
  SetBranchAddress(tree,"nClusters",  className, &nClusters );
  SetBranchAddress(tree,"clusterEta", className, &clusterEta);
  SetBranchAddress(tree,"clusterPhi", className, &clusterPhi);
  SetBranchAddress(tree,"clusterEt",  className, &clusterEt );
  SetBranchAddress(tree,"clusterE",   className, &clusterE  );
}


void L1Analysis::L1AnalysisRecoClusters::print()
{
}

#endif


