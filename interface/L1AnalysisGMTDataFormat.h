#ifndef __L1Analysis_L1AnalysisGMTDataFormat_H__
#define __L1Analysis_L1AnalysisGMTDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
//
// Original code : L1TriggerDPG/L1Ntuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include <vector>


namespace L1Analysis
{
  struct L1AnalysisGMTDataFormat
  {
    L1AnalysisGMTDataFormat(){Reset();};
    ~L1AnalysisGMTDataFormat(){};
    
    void Reset()
    {
     Ndt = 0; Ncsc = 0; Nrpcb = 0;
     Nrpcf = 0; N = 0;

  Bxdt.clear();
  Ptdt.clear();
  Chadt.clear();
  Etadt.clear();
  FineEtadt.clear();
  Phidt.clear();
  Qualdt.clear();
  Dwdt.clear();
  Chdt.clear();

 //CSC Trigger block

  Bxcsc.clear();
  Ptcsc.clear();
  Chacsc.clear();
  Etacsc.clear();
  Phicsc.clear();
  Qualcsc.clear();
  Dwcsc.clear();

 //RPCb Trigger

  Bxrpcb.clear();
  Ptrpcb.clear();
  Charpcb.clear();
  Etarpcb.clear();
  Phirpcb.clear();
  Qualrpcb.clear();
  Dwrpcb.clear();

 //RPCf Trigger

  Bxrpcf.clear();
  Ptrpcf.clear();
  Charpcf.clear();
  Etarpcf.clear();
  Phirpcf.clear();
  Qualrpcf.clear();
  Dwrpcf.clear();

 //Global Muon Trigger

  CandBx.clear();
  Pt.clear();
  Cha.clear();
  Eta.clear();
  Phi.clear();

 //RPCb Trigger

  Bxrpcb.clear();
  Ptrpcb.clear();
  Charpcb.clear();
  Etarpcb.clear();
  Phirpcb.clear();
  Qualrpcb.clear();
  Dwrpcb.clear();

 //RPCf Trigger

  Bxrpcf.clear();
  Ptrpcf.clear();
  Charpcf.clear();
  Etarpcf.clear();
  Phirpcf.clear();
  Qualrpcf.clear();
  Dwrpcf.clear();

 //Global Muon Trigger

  CandBx.clear();
  Pt.clear();
  Cha.clear();
  Eta.clear();
  Phi.clear();
  Qual.clear();
  Det.clear();
  Rank.clear();
  Isol.clear();
  Mip.clear();
  Dw.clear();
  IdxRPCb.clear();
  IdxRPCf.clear();
  IdxDTBX.clear();
  IdxCSC.clear();
  }
      
    // ---- General L1AnalysisGMTDataFormat information.
    
    int EvBx;
    
    //DTBX Trigger block
    int Ndt;
    std::vector<int>   Bxdt;
    std::vector<float> Ptdt;
    std::vector<int>   Chadt;
    std::vector<float> Etadt;
    std::vector<int>   FineEtadt;
    std::vector<float> Phidt;
    std::vector<int>   Qualdt;
    std::vector<int>   Dwdt;
    std::vector<int>   Chdt;

    //CSC Trigger block
    int	Ncsc;
    std::vector<int>   Bxcsc;
    std::vector<float> Ptcsc;
    std::vector<int>   Chacsc;
    std::vector<float> Etacsc;
    std::vector<float> Phicsc;
    std::vector<int>   Qualcsc;
    std::vector<int>   Dwcsc;
   
    //RPCb Trigger
    int	Nrpcb ;
    std::vector<int>   Bxrpcb;
    std::vector<float> Ptrpcb;
    std::vector<int>   Charpcb;
    std::vector<float> Etarpcb;
    std::vector<float> Phirpcb;
    std::vector<int>   Qualrpcb;
    std::vector<int>   Dwrpcb;
    
    //RPCf Trigger
    int	Nrpcf ;
    std::vector<int>   Bxrpcf;
    std::vector<float> Ptrpcf;
    std::vector<int>   Charpcf;
    std::vector<float> Etarpcf;
    std::vector<float> Phirpcf;
    std::vector<int>   Qualrpcf;
    std::vector<int>   Dwrpcf;
    		  
    //Global Muon Trigger
    int N;
    std::vector<int>	  CandBx;
    std::vector<float>    Pt;
    std::vector<int>	  Cha;
    std::vector<float>    Eta;
    std::vector<float>    Phi;
    std::vector<int>	  Qual;
    std::vector<int>	  Det;
    std::vector<int>	  Rank;
    std::vector<int>	  Isol;
    std::vector<int>	  Mip;
    std::vector<int>	  Dw;
    std::vector<int>	  IdxRPCb;
    std::vector<int>	  IdxRPCf;
    std::vector<int>	  IdxDTBX;
    std::vector<int>	  IdxCSC;
    
  }; 
} 
#endif


