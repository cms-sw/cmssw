#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisGMT.h"


L1Analysis::L1AnalysisGMT::L1AnalysisGMT()
{
}


L1Analysis::L1AnalysisGMT::~L1AnalysisGMT()
{
}

void L1Analysis::L1AnalysisGMT::Set(const L1MuGMTReadoutCollection* gmtrc, unsigned maxDTBX, unsigned maxCSC, unsigned maxRPC, unsigned maxGMT, bool physVal)
{
   
  std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  std::vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;
  for(igmtrr=gmt_records.begin(); igmtrr!=gmt_records.end(); igmtrr++) {
 
   std::vector<L1MuRegionalCand>::const_iterator iter1;
   std::vector<L1MuRegionalCand> rmc;
   
   if(igmtrr->getBxInEvent()==0) {
     gmt_.EvBx = igmtrr->getBxNr();
   }
   
   
  //
  // DTBX Trigger
  //    
  
  int iidt = 0;
  rmc = igmtrr->getDTBXCands();
  for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
    if ( (unsigned) gmt_.Ndt < maxDTBX && !(*iter1).empty() ) {
      gmt_.Bxdt.push_back((*iter1).bx());
      if(physVal) {
	gmt_.Etadt.push_back(float((*iter1).etaValue()));
	gmt_.Phidt.push_back(float((*iter1).phiValue()));
	gmt_.Ptdt.push_back(float((*iter1).ptValue()));
      } else {
	gmt_.Etadt.push_back(float((*iter1).eta_packed()));
	gmt_.Phidt.push_back(float((*iter1).phi_packed()));
	gmt_.Ptdt.push_back(float((*iter1).pt_packed()));
      }
      gmt_.Chadt.push_back((*iter1).chargeValue()); if(!(*iter1).chargeValid()) gmt_.Chadt.push_back(0);
      gmt_.FineEtadt.push_back((*iter1).isFineHalo());
      gmt_.Qualdt.push_back((*iter1).quality());
      gmt_.Dwdt.push_back((*iter1).getDataWord());
      gmt_.Chdt.push_back(iidt);
	
      gmt_.Ndt++;
    }
    iidt++;
  }

   //
   // CSC Trigger
   //  
  
   rmc = igmtrr->getCSCCands();
   for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
     if ( (unsigned) gmt_.Ncsc < maxCSC && !(*iter1).empty() ) {
       gmt_.Bxcsc.push_back((*iter1).bx());
       if(physVal) {  	  
  	 gmt_.Etacsc.push_back((*iter1).etaValue());
  	 gmt_.Phicsc.push_back((*iter1).phiValue());
  	 gmt_.Ptcsc.push_back((*iter1).ptValue());
       } else {
  	 gmt_.Etacsc.push_back((*iter1).eta_packed());
  	 gmt_.Phicsc.push_back((*iter1).phi_packed());
  	 gmt_.Ptcsc.push_back((*iter1).pt_packed());
       }
       gmt_.Chacsc.push_back((*iter1).chargeValue()); if(!(*iter1).chargeValid()) gmt_.Chacsc.push_back(0);
       gmt_.Qualcsc.push_back((*iter1).quality());
       gmt_.Dwcsc.push_back((*iter1).getDataWord());
 
       gmt_.Ncsc++;
     }
   }
    
 
 
    //
    // RPCb Trigger
    //
    
    rmc = igmtrr->getBrlRPCCands();
    for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
      if ( (unsigned) gmt_.Nrpcb < maxRPC && !(*iter1).empty() ) {
        gmt_.Bxrpcb.push_back((*iter1).bx());
        if(physVal) {	    
          gmt_.Etarpcb.push_back((*iter1).etaValue());
          gmt_.Phirpcb.push_back((*iter1).phiValue());
          gmt_.Ptrpcb.push_back((*iter1).ptValue());
        } else {
          gmt_.Etarpcb.push_back((*iter1).eta_packed());
          gmt_.Phirpcb.push_back((*iter1).phi_packed());
          gmt_.Ptrpcb.push_back((*iter1).pt_packed());
        }
        gmt_.Charpcb.push_back((*iter1).chargeValue()); if(!(*iter1).chargeValid()) gmt_.Charpcb.push_back(0);
        gmt_.Qualrpcb.push_back((*iter1).quality());
        gmt_.Dwrpcb.push_back((*iter1).getDataWord());

        gmt_.Nrpcb++;
      }
    }

    
    //
    // RPCf Trigger
    // 
    
    rmc = igmtrr->getFwdRPCCands();
    for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
      if ( (unsigned) gmt_.Nrpcf < maxRPC && !(*iter1).empty() ) {
        gmt_.Bxrpcf.push_back((*iter1).bx());
        if(physVal) {
          gmt_.Etarpcf.push_back((*iter1).etaValue());
          gmt_.Phirpcf.push_back((*iter1).phiValue());
          gmt_.Ptrpcf.push_back((*iter1).ptValue());
        } else {
          gmt_.Etarpcf.push_back((*iter1).eta_packed());
          gmt_.Phirpcf.push_back((*iter1).phi_packed());
          gmt_.Ptrpcf.push_back((*iter1).pt_packed());
        }
        gmt_.Charpcf.push_back((*iter1).chargeValue()); if(!(*iter1).chargeValid()) gmt_.Charpcf.push_back(0);
        gmt_.Qualrpcf.push_back((*iter1).quality());
        gmt_.Dwrpcf.push_back((*iter1).getDataWord());

        gmt_.Nrpcf++;
      }
    }
  
  //
  // GMT_. Trigger
  //  
  
  std::vector<L1MuGMTExtendedCand>::const_iterator gmt_iter;
  std::vector<L1MuGMTExtendedCand> exc = igmtrr->getGMTCands();
  for(gmt_iter=exc.begin(); gmt_iter!=exc.end(); gmt_iter++) {
    if ( (unsigned) gmt_.N < maxGMT && !(*gmt_iter).empty() ) {
      gmt_.CandBx.push_back((*gmt_iter).bx());
      if(physVal) {
        gmt_.Eta.push_back((*gmt_iter).etaValue());
        gmt_.Phi.push_back((*gmt_iter).phiValue()); 
        gmt_.Pt.push_back((*gmt_iter).ptValue());
      } else {
        gmt_.Eta.push_back((*gmt_iter).etaIndex());
        gmt_.Phi.push_back((*gmt_iter).phiIndex()); 
        gmt_.Pt.push_back((*gmt_iter).ptIndex());
      }
      gmt_.Cha.push_back((*gmt_iter).charge()); if(!(*gmt_iter).charge_valid()) gmt_.Cha.push_back(0);
      gmt_.Qual.push_back((*gmt_iter).quality());
      gmt_.Det.push_back((*gmt_iter).detector());
      gmt_.Rank.push_back((*gmt_iter).rank());
      gmt_.Isol.push_back((*gmt_iter).isol());
      gmt_.Mip.push_back((*gmt_iter).mip());
      gmt_.Dw.push_back((*gmt_iter).getDataWord());
            
      gmt_.IdxRPCb.push_back(-1);
      gmt_.IdxRPCf.push_back(-1);
      gmt_.IdxDTBX.push_back(-1);
      gmt_.IdxCSC.push_back(-1);

      if ( (*gmt_iter).isMatchedCand() || (*gmt_iter).isRPC() ) {
        if((*gmt_iter).isFwd()) {
          gmt_.IdxRPCf.back() = (*gmt_iter).getRPCIndex();
        } else {
          gmt_.IdxRPCb.back() = (*gmt_iter).getRPCIndex();
        }
      }

      if ( (*gmt_iter).isMatchedCand() || ( !(*gmt_iter).isRPC() ) ) {
        if ( (*gmt_iter).isFwd() )  {
          gmt_.IdxCSC.back() = (*gmt_iter).getDTCSCIndex();
        } else {
          gmt_.IdxDTBX.back() = (*gmt_iter).getDTCSCIndex();
        }
      }
      gmt_.N++;
    }
  }
    
  }
   
  
}

