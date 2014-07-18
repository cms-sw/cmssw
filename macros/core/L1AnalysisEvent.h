#ifndef __L1Analysis_L1AnalysisEvent_H__
#define __L1Analysis_L1AnalysisEvent_H__

#include <TTree.h>
#include <iostream>

namespace L1Analysis
{
  class L1AnalysisEvent
  {
    
  public :
    void initTree(TTree * tree, std::string className);
  
  public:
    L1AnalysisEvent() {reset();}
    ~L1AnalysisEvent() {}
    void print();
    void reset();
    
    // ---- General L1AnalysisEvent information.    
    int     run;
    int     event;
    int     lumi;
    int     bx;
    ULong64_t orbit;
    ULong64_t time;
};
}

//#endif

//#ifdef l1ntuple_cxx

void L1Analysis::L1AnalysisEvent::reset()
{
  run=-999; event=-999; lumi=-999; bx=-999; orbit=999; time=+999;
}

void L1Analysis::L1AnalysisEvent::initTree(TTree * tree, std::string className)
    {
      SetBranchAddress(tree, "run",  className, &run);
      SetBranchAddress(tree, "event",className, &event);
      SetBranchAddress(tree, "lumi", className, &lumi);
      SetBranchAddress(tree, "bx",   className, &bx);
      SetBranchAddress(tree, "orbit",className, &orbit);
      SetBranchAddress(tree, "time", className, &time);
    }

void L1Analysis::L1AnalysisEvent::print()
{
  std::cout << "run="<<run<<" "
            << "event="<<event<<" "
            << "lumi="<<lumi<<" "
            << "bx="<<bx<<" "
            << "orbit="<<orbit<<" "
            << "time="<<time<<std::endl;
}

#endif


