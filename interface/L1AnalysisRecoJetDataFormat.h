#ifndef __L1Analysis_L1AnalysisRecoJetDataFormat_H__
#define __L1Analysis_L1AnalysisRecoJetDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
//
//
// Original code : L1TriggerDPG/L1Ntuples/L1RecoJetNtupleProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisRecoJetDataFormat
  {
    L1AnalysisRecoJetDataFormat(){Reset();};
    ~L1AnalysisRecoJetDataFormat(){Reset();};

    void Reset()
    {
    nJets=0;

    e.clear();
    et.clear();
    etCorr.clear();
    corrFactor.clear();
    eta.clear();
    phi.clear();
    eEMF.clear();
    eEmEB.clear();
    eEmEE.clear();
    eEmHF.clear();
    eHadHB.clear();
    eHadHE.clear();
    eHadHO.clear();
    eHadHF.clear();
    eMaxEcalTow.clear();
    eMaxHcalTow.clear();
    towerArea.clear();
    towerSize.clear();
    n60.clear();
    n90.clear();

    n90hits.clear();
    fHPD.clear();
    fRBX.clear();
    }

    unsigned nJets;
    std::vector<double> e;
    std::vector<double> et;
    std::vector<double> etCorr;
    std::vector<double> corrFactor;
    std::vector<double> eta;
    std::vector<double> phi;
    std::vector<double> eEMF;
    std::vector<double> eHadHB;
    std::vector<double> eHadHE;
    std::vector<double> eHadHO;
    std::vector<double> eHadHF;
    std::vector<double> eEmEB;
    std::vector<double> eEmEE;
    std::vector<double> eEmHF;
    std::vector<double> eMaxEcalTow;
    std::vector<double> eMaxHcalTow;
    std::vector<double> towerArea;
    std::vector<int> towerSize;
    std::vector<int> n60;
    std::vector<int> n90;

    std::vector<int> n90hits;
    std::vector<double> fHPD;
    std::vector<double> fRBX;

  };
}
#endif


