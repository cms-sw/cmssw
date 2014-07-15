#ifndef __L1Analysis_L1AnalysisL1ExtraDataFormat_H__
#define __L1Analysis_L1AnalysisL1ExtraDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1TriggerDPG/L1Ntuples/L1ExtraTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------


#include <vector>

namespace L1Analysis
{
  struct L1AnalysisL1ExtraDataFormat
  {
    L1AnalysisL1ExtraDataFormat(){Reset();};
    ~L1AnalysisL1ExtraDataFormat(){};
    
    void Reset()
    {
      nIsoEm = 0;
      isoEmEt.clear();
      isoEmEta.clear();
      isoEmPhi.clear();
      isoEmBx.clear();

      nNonIsoEm = 0;
      nonIsoEmEt.clear();
      nonIsoEmEta.clear();
      nonIsoEmPhi.clear();
      nonIsoEmBx.clear();
      
      nCenJets = 0;
      cenJetEt.clear();
      cenJetEta.clear();
      cenJetPhi.clear();
      cenJetBx.clear();

      nFwdJets = 0;
      fwdJetEt.clear();
      fwdJetEta.clear();
      fwdJetPhi.clear();
      fwdJetBx.clear();

      nTauJets = 0;
      tauJetEt.clear();
      tauJetEta.clear();
      tauJetPhi.clear(); 
      tauJetBx.clear();

      nMuons = 0;
      muonEt.clear();
      muonEta.clear();
      muonPhi.clear();
      muonChg.clear();
      muonIso.clear();
      muonFwd.clear();
      muonMip.clear();
      muonRPC.clear();
      muonBx.clear();
      muonQuality.clear();

      nMet = 0;
      et.clear();
      met.clear();
      metPhi.clear();
      metBx.clear();

      nMht = 0;
      ht.clear();
      mht.clear();
      mhtPhi.clear();
      mhtBx.clear();

      hfEtSum.clear();
      hfBitCnt.clear();
      hfBx.clear();

    }
   
    unsigned int nIsoEm;
    std::vector<double> isoEmEt;
    std::vector<double> isoEmEta;
    std::vector<double> isoEmPhi;
    std::vector<int>    isoEmBx;
 
    unsigned int nNonIsoEm;
    std::vector<double> nonIsoEmEt;
    std::vector<double> nonIsoEmEta;
    std::vector<double> nonIsoEmPhi;
    std::vector<int>    nonIsoEmBx;
 
    unsigned int nCenJets;
    std::vector<double> cenJetEt;
    std::vector<double> cenJetEta;
    std::vector<double> cenJetPhi;
    std::vector<int>    cenJetBx;
 
    unsigned int nFwdJets;
    std::vector<double> fwdJetEt;
    std::vector<double> fwdJetEta;
    std::vector<double> fwdJetPhi;
    std::vector<int>    fwdJetBx;

    unsigned int nTauJets;
    std::vector<double> tauJetEt;
    std::vector<double> tauJetEta;
    std::vector<double> tauJetPhi;
    std::vector<int>    tauJetBx;

    unsigned int nMuons;
    std::vector<double>   muonEt;
    std::vector<double>   muonEta;
    std::vector<double>   muonPhi;
    std::vector<int>      muonChg;
    std::vector<unsigned int> muonIso;
    std::vector<unsigned int> muonFwd;
    std::vector<unsigned int> muonMip;
    std::vector<unsigned int> muonRPC;
    std::vector<int>      muonBx;
    std::vector<int>      muonQuality;
 
    std::vector<double> hfEtSum;
    std::vector<unsigned int> hfBitCnt;
    std::vector<int> hfBx;
    
    unsigned int nMet;
    std::vector<double> et;
    std::vector<double> met;
    std::vector<double> metPhi;
    std::vector<double> metBx;

    unsigned int nMht;
    std::vector<double> ht;
    std::vector<double> mht;
    std::vector<double> mhtPhi;
    std::vector<double> mhtBx;

  }; 
}
#endif


