#ifndef __L1Analysis_L1AnalysisGeneratorDataFormat_H__
#define __L1Analysis_L1AnalysisGeneratorDataFormat_H__

//-------------------------------------------------------------------------------
// Created 15/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1TriggerDPG/L1Ntuples/L1NtupleProducer
//-------------------------------------------------------------------------------
#include <TROOT.h>
#include <vector>
//#include <TString.h>


namespace L1Analysis
{
  struct L1AnalysisGeneratorDataFormat
  {
  
    L1AnalysisGeneratorDataFormat(){Reset();};
    ~L1AnalysisGeneratorDataFormat(){};
    
    void Reset()
    {
     weight = -999.;
     pthat  = -999.;
                id.resize(0);
		status.resize(0);
		parent_id.resize(0);
		px.resize(0);
		py.resize(0);
		pz.resize(0);
		e.resize(0);

    }

                   
    // ---- L1AnalysisGeneratorDataFormat information.
    
    float weight;
    float pthat;
    std::vector<int> id;
    std::vector<int> status;
    std::vector<int> parent_id;
    std::vector<float> px;
    std::vector<float> py;
    std::vector<float> pz;
    std::vector<float> e;
            
  }; 
} 
#endif


