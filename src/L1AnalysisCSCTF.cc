#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisCSCTF.h"



L1Analysis::L1AnalysisCSCTF::L1AnalysisCSCTF()
{
}

L1Analysis::L1AnalysisCSCTF::~L1AnalysisCSCTF()
{
}



void L1Analysis::L1AnalysisCSCTF::SetTracks(const edm::Handle<L1CSCTrackCollection> csctfTrks,
                                            const L1MuTriggerScales *ts, const L1MuTriggerPtScale *tpts, 
                                            CSCSectorReceiverLUT* srLUTs_[5][2],
                                            CSCTFPtLUT* ptLUTs_)
{
   
      //for (int i =0; i<MAXCSCTFTR; i++) csctf_.trSector[i]=0;

      csctf_.trSize = csctfTrks->size();
      //cout << " csctf_.trSize: " << csctf_.trSize << endl;

      int nTrk=0;
      for(L1CSCTrackCollection::const_iterator trk=csctfTrks->begin(); 
	  trk<csctfTrks->end(); trk++){

	nTrk++;

	// trk->first.endcap() = 2 for - endcap
	//                     = 1 for + endcap
	csctf_.trEndcap.push_back(trk->first.endcap()==2 ? trk->first.endcap()-3 : trk->first.endcap());
	//sectors: 1->6 (plus endcap), 7->12 (minus endcap)
	csctf_.trSector.push_back(6* (trk->first.endcap()-1) + trk->first.sector());
	csctf_.trBx.push_back(trk->first.BX());	 

	csctf_.trME1ID.push_back(trk->first.me1ID());
	csctf_.trME2ID.push_back(trk->first.me2ID());
	csctf_.trME3ID.push_back(trk->first.me3ID());
	csctf_.trME4ID.push_back(trk->first.me4ID());
	csctf_.trMB1ID.push_back(trk->first.mb1ID());

 	csctf_.trME1TBin.push_back(trk->first.me1Tbin());
 	csctf_.trME2TBin.push_back(trk->first.me2Tbin());
 	csctf_.trME3TBin.push_back(trk->first.me3Tbin());
 	csctf_.trME4TBin.push_back(trk->first.me4Tbin());
 	csctf_.trMB1TBin.push_back(trk->first.mb1Tbin());

	// output link of the track
	csctf_.trOutputLink.push_back(trk->first.outputLink());
    
	// PtAddress gives an handle on other parameters
  	ptadd thePtAddress(trk->first.ptLUTAddress());
	
	csctf_.trPhiSign.push_back(thePtAddress.delta_phi_sign);
	csctf_.trPhi12.push_back(thePtAddress.delta_phi_12);
	csctf_.trPhi23.push_back(thePtAddress.delta_phi_23);
	csctf_.trMode.push_back(thePtAddress.track_mode);
	csctf_.trForR.push_back(thePtAddress.track_fr);
	csctf_.trCharge.push_back(trk->first.chargeValue());


	//Pt needs some more workaround since it is not in the unpacked data
	////ptdat thePtData  = ptLUT->Pt(thePtAddress);
        ptdat thePtData  = ptLUTs_->Pt(thePtAddress);
	
 	// front or rear bit? 
 	if (thePtAddress.track_fr) {
 	  csctf_.trPtBit.push_back(thePtData.front_rank&0x1f);
 	  csctf_.trQuality.push_back((thePtData.front_rank>>5)&0x3);
 	  csctf_.trChargeValid.push_back(thePtData.charge_valid_front);
	  } else {
 	  csctf_.trPtBit.push_back(thePtData.rear_rank&0x1f);
 	  csctf_.trQuality.push_back((thePtData.rear_rank>>5)&0x3);
 	  csctf_.trChargeValid.push_back(thePtData.charge_valid_rear);
	  }
	  

	// convert the Pt in human readable values (GeV/c)
	csctf_.trPt.push_back(tpts->getPtScale()->getLowEdge(csctf_.trPtBit.back())); 

 	// track's Eta and Phi in bit... 
 	csctf_.trEtaBit.push_back(trk->first.eta_packed());
 	csctf_.trPhiBit.push_back(trk->first.localPhi());

 	//... in radians
 	// Type 2 is CSC
 	//csctf_.trEta.push_back(theTriggerScales->getRegionalEtaScale(2)->getCenter(trk->first.eta_packed()));
        //csctf_.trPhi.push_back(theTriggerScales->getPhiScale()->getLowEdge(trk->first.localPhi()));
        csctf_.trEta.push_back(ts->getRegionalEtaScale(2)->getCenter(trk->first.eta_packed()));
 	csctf_.trPhi.push_back(ts->getPhiScale()->getLowEdge(trk->first.localPhi()));
 	//Phi in one sector varies from [0,62] degrees -> Rescale manually to global coords.
 	csctf_.trPhi_02PI.push_back(fmod(csctf_.trPhi[nTrk-1]  + 
 					((trk->first.sector()-1)*TMath::Pi()/3) + //sector 1 starts at 15 degrees 
 					(TMath::Pi()/12) , 2*TMath::Pi()));


 //	//csctf lcts of tracks
  //	if( csctfLCTSource_.label() != "none" ){

  	  // For each trk, get the list of its LCTs
  	  CSCCorrelatedLCTDigiCollection lctsOfTracks = trk -> second;
  
  	  int LctTrkId_ = 0;

  	  for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator lctOfTrks = lctsOfTracks.begin(); 
  	      lctOfTrks  != lctsOfTracks.end()  ; lctOfTrks++){
	  
  	    int lctTrkId = 0;	
			
  	    CSCCorrelatedLCTDigiCollection::Range lctRange = 
  	      lctsOfTracks.get((*lctOfTrks).first);
	  
  	    for(CSCCorrelatedLCTDigiCollection::const_iterator 
  		  lctTrk = lctRange.first ; 
  		lctTrk  != lctRange.second; lctTrk++, lctTrkId++){


  	      csctf_.trLctEndcap[nTrk-1][LctTrkId_] = (*lctOfTrks).first.zendcap();
 	      if ((*lctOfTrks).first.zendcap() > 0)
 		csctf_.trLctSector[nTrk-1][LctTrkId_] = (*lctOfTrks).first.triggerSector();
 	      else
 		csctf_.trLctSector[nTrk-1][LctTrkId_] = 6+(*lctOfTrks).first.triggerSector();
   	      csctf_.trLctSubSector[nTrk-1][LctTrkId_] = CSCTriggerNumbering::triggerSubSectorFromLabels((*lctOfTrks).first);;
   	      csctf_.trLctBx[nTrk-1][LctTrkId_] = lctTrk -> getBX();
   	      csctf_.trLctBx0[nTrk-1][LctTrkId_] = lctTrk -> getBX0();
	      
  	      csctf_.trLctStation[nTrk-1][LctTrkId_] = (*lctOfTrks).first.station();
  	      csctf_.trLctRing[nTrk-1][LctTrkId_] = (*lctOfTrks).first.ring();
  	      csctf_.trLctChamber[nTrk-1][LctTrkId_] = (*lctOfTrks).first.chamber();
  	      csctf_.trLctTriggerCSCID[nTrk-1][LctTrkId_] = (*lctOfTrks).first.triggerCscId();
  	      csctf_.trLctFpga[nTrk-1][LctTrkId_] = 
 		( csctf_.trLctSubSector[nTrk-1][LctTrkId_] ? csctf_.trLctSubSector[nTrk-1][LctTrkId_] : (*lctOfTrks).first.station()+1);


  	      // Check if DetId is within range
  	      if( csctf_.trLctSector[nTrk-1][LctTrkId_] < 1 || csctf_.trLctSector[nTrk-1][LctTrkId_] > 12 || 
  		  csctf_.trLctStation[nTrk-1][LctTrkId_] < 1 || csctf_.trLctStation[nTrk-1][LctTrkId_] >  4 || 
  		  csctf_.trLctTriggerCSCID[nTrk-1][LctTrkId_] < 1 || csctf_.trLctTriggerCSCID[nTrk-1][LctTrkId_] >  9 || 
  		  lctTrkId < 0 || lctTrkId >  1 ){
		
  		edm::LogInfo("L1NtupleProducer")<<"  TRACK ERROR: CSC digi are out of range: ";
		
  		continue;
  	      }

  	      // handles not to overload the method: mostly for readability	      
 	      int endcap = (*lctOfTrks).first.zendcap();
 	      if (endcap < 0) endcap = 0; 

 	      int StationLctTrk  = (*lctOfTrks).first.station();
 	      int CscIdLctTrk    = (*lctOfTrks).first.triggerCscId();
 	      int SubSectorLctTrk = 
 		CSCTriggerNumbering::triggerSubSectorFromLabels((*lctOfTrks).first);
	      
 	      int FPGALctTrk    = 
 		( SubSectorLctTrk ? SubSectorLctTrk-1 : StationLctTrk );

	      
 	      // local Phi
  	      lclphidat lclPhi;
		
   	      try {
		
 		csctf_.trLctstripNum[nTrk-1][LctTrkId_] = lctTrk->getStrip();
 		lclPhi = srLUTs_[FPGALctTrk][endcap] -> localPhi(lctTrk->getStrip(), 
 								 lctTrk->getPattern(), 
 								 lctTrk->getQuality(), 
 								 lctTrk->getBend() );
		
 		csctf_.trLctlocalPhi[nTrk-1][LctTrkId_] = lclPhi.phi_local;
 		//csctf_.trLctlocalPhi_bend[nTrk-1][LctTrkId_] = lclPhi.phi_bend_local;
 		//csctf_.trLctCLCT_pattern[nTrk-1][LctTrkId_] = lctTrk->getPattern();
 		csctf_.trLctQuality[nTrk-1][LctTrkId_] = lctTrk->getQuality();

                //std::cout <<"lctTrk->getPattern() =  " << lctTrk->getPattern() << std::endl;
   	      } 
   	      catch(...) { 
   		bzero(&lclPhi,sizeof(lclPhi)); 
   		csctf_.trLctlocalPhi[nTrk-1][LctTrkId_] = -999;
   		//csctf_.trLctlocalPhi_bend[nTrk-1][LctTrkId_] = -999;
   		//csctf_.trLctCLCT_pattern[nTrk-1][LctTrkId_] = -999;
   		csctf_.trLctQuality[nTrk-1][LctTrkId_] = -999;
   	      }
              // clct pattern
              lclphiadd lclPattern;
              try{
                //std::cout <<"lclPattern.clct_pattern = " << lclPattern.clct_pattern << std::endl;
                //std::cout <<"lclPattern.pattern_type = " << lclPattern.pattern_type << std::endl;
                 
              }
              catch(...){
              }
		
		
   	      // Global Phi
   	      gblphidat gblPhi;
	      		
   	      try {
	      
 		csctf_.trLctwireGroup[nTrk-1][LctTrkId_] = lctTrk->getKeyWG();
 		gblPhi = srLUTs_[FPGALctTrk][endcap] -> globalPhiME(lclPhi.phi_local  , 
 								    lctTrk->getKeyWG(), 
 								    CscIdLctTrk);
		
 		csctf_.trLctglobalPhi[nTrk-1][LctTrkId_] = gblPhi.global_phi;
		
 	      } catch(...) { 
   		bzero(&gblPhi,sizeof(gblPhi)); 
   		csctf_.trLctglobalPhi[nTrk-1][LctTrkId_] = -999;
   	      }
		
    	      // Global Eta
    	      gbletadat gblEta;
		
   	      try {
    		gblEta = srLUTs_[FPGALctTrk][endcap] -> globalEtaME(lclPhi.phi_bend_local, 
 								    lclPhi.phi_local     , 
 								    lctTrk->getKeyWG()   , 
 								    CscIdLctTrk);
    		csctf_.trLctglobalEta[nTrk-1][LctTrkId_] = gblEta.global_eta;
 		csctf_.trLctCLCT_pattern[nTrk-1][LctTrkId_] = gblEta.global_bend;
 	      } 	  
    	      catch(...) { 
    		bzero(&gblEta,sizeof(gblEta)); 
    		csctf_.trLctglobalEta[nTrk-1][LctTrkId_] = -999;
   		csctf_.trLctCLCT_pattern[nTrk-1][LctTrkId_] = -999;
    	      } 
	      
  	      ++LctTrkId_;
	      
  	    } // for(CSCCorrelatedLCTDigiCollection::const_iterator lctTrk 
  	  } // for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator lctOfTrks
	  
  	  csctf_.trNumLCTs.push_back(LctTrkId_);
  	//}
  	//else 
  	 // edm::LogInfo("L1NtupleProducer")<<"  No valid CSCCorrelatedLCTDigiCollection products found";
      
          ////delete ptLUT;
     
      } //for(L1CSCTrackCollection::const_iterator trk=csctfTrks->begin(); trk<csctfTrks->end(); trk++,nTrk++){
}



//ALL csctf lcts
void L1Analysis::L1AnalysisCSCTF::SetLCTs(const edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts, CSCSectorReceiverLUT* srLUTs_[5][2])
{
  
    int nLCT=0;
    for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator 
	  corrLct = corrlcts.product()->begin(); 
	corrLct != corrlcts.product()->end()  ; corrLct++){
      
      nLCT++;
 
      int lctId = 0;	
      
      CSCCorrelatedLCTDigiCollection::Range lctRange = 
	corrlcts.product()->get((*corrLct).first);
			
      for(CSCCorrelatedLCTDigiCollection::const_iterator 
	   lct = lctRange.first ; 
	  lct != lctRange.second; lct++, lctId++){
		
	csctf_.lctEndcap.push_back((*corrLct).first.zendcap());
	if ((*corrLct).first.zendcap() > 0)
	  csctf_.lctSector.push_back((*corrLct).first.triggerSector());
	else
	  csctf_.lctSector.push_back(6+(*corrLct).first.triggerSector());
	
        csctf_.lctSubSector.push_back(CSCTriggerNumbering::triggerSubSectorFromLabels((*corrLct).first));
	csctf_.lctBx.push_back(lct->getBX());
	csctf_.lctBx0.push_back(lct->getBX0());
	
	csctf_.lctStation.push_back((*corrLct).first.station());
	csctf_.lctRing.push_back((*corrLct).first.ring());
	csctf_.lctChamber.push_back((*corrLct).first.chamber());
	csctf_.lctTriggerCSCID.push_back((*corrLct).first.triggerCscId());
	csctf_.lctFpga.push_back((csctf_.lctSubSector.back() ? csctf_.lctSubSector.back() : (*corrLct).first.station()+1));
	

	// Check if DetId is within range
	if( csctf_.lctSector.back() < 1 || csctf_.lctSector.back() > 12 || 
	    csctf_.lctStation.back() < 1 || csctf_.lctStation.back() >  4 || 
	    csctf_.lctTriggerCSCID.back() < 1 || csctf_.lctTriggerCSCID.back() >  9 || 
	    lctId < 0 || lctId >  1 ){
	  
	  edm::LogInfo("L1NtupleProducer")<<"  LCT ERROR: CSC digi are out of range: ";
	  
	  continue;
	}

	// handles not to overload the method: mostly for readability	      
	int endcap = (*corrLct).first.zendcap();
	if (endcap < 0) endcap = 0; 
	
	int StationLct  = (*corrLct).first.station();
	int CscIdLct    = (*corrLct).first.triggerCscId();
	int SubSectorLct = 
	  CSCTriggerNumbering::triggerSubSectorFromLabels((*corrLct).first);
	      
	int FPGALct    = ( SubSectorLct ? SubSectorLct-1 : StationLct );
	
	
	// local Phi
	lclphidat lclPhi;
	
/*
	try {
	  
	  csctf_.lctstripNum.push_back(lct->getStrip());

	  
	  csctf_.lctlocalPhi.push_back(lclPhi.phi_local);
	} 
	catch(...) { 
	  bzero(&lclPhi,sizeof(lclPhi)); 
	  csctf_.lctlocalPhi.push_back(-999);
	}
		
*/
        try {

	  csctf_.lctstripNum.push_back(lct->getStrip());
          lclPhi = srLUTs_[FPGALct][endcap] -> localPhi(lct->getStrip(),
                                                        lct->getPattern(),
                                                        lct->getQuality(),
                                                        lct->getBend() );

          csctf_.lctlocalPhi.push_back(lclPhi.phi_local);
          //csctf_.lctlocalPhi_bend.push_back(lclPhi.phi_bend_local);
          //csctf_.lctCLCT_pattern.push_back(lct->getPattern());
          csctf_.lctQuality.push_back(lct->getPattern());
          //std::cout <<"localPhi: lclPhi.phi_bend_local = " << lclPhi.phi_bend_local << std::endl;
          //std::cout <<"localPhi: lct->getBend() = " << lct->getBend() << std::endl;
          
        }
        catch(...) {
	  bzero(&lclPhi,sizeof(lclPhi)); 
	  csctf_.lctlocalPhi.push_back(-999);
	  //csctf_.lctlocalPhi_bend.push_back(-999);
	  //csctf_.lctCLCT_pattern.push_back(-999);
	  csctf_.lctQuality.push_back(-999);
        }
	
	// Global Phi
	gblphidat gblPhi;
	
	try {
	  
	  csctf_.lctwireGroup.push_back(lct->getKeyWG());

	  //std::cout << "lclPhi.phi_local: " << lclPhi.phi_local << std::endl;
	  //std::cout << "lct->getKeyWG(): " << lct->getKeyWG() << std::endl;
	  //std::cout << "CscIdLct: " << CscIdLct << std::endl;
	  
          gblPhi = srLUTs_[FPGALct][endcap] -> globalPhiME(lclPhi.phi_local  ,
                                                           lct->getKeyWG(),
                                                           CscIdLct);
	  csctf_.lctglobalPhi.push_back(gblPhi.global_phi);


	} catch(...) { 
	  bzero(&gblPhi,sizeof(gblPhi)); 
	  csctf_.lctglobalPhi.push_back(-999);
	}
	
	// Global Eta
	gbletadat gblEta;
	
	try {
	  gblEta = srLUTs_[FPGALct][endcap] -> globalEtaME(lclPhi.phi_bend_local, 
							   lclPhi.phi_local     , 
							   lct->getKeyWG()   , 
							   CscIdLct);
          //std::cout <<"gblEta: lclPhi.phi_bend_local = " << lclPhi.phi_bend_local << std::endl;
	  csctf_.lctglobalEta.push_back(gblEta.global_eta);
          csctf_.lctCLCT_pattern.push_back(gblEta.global_bend);
	} 	  
	catch(...) { 
	  bzero(&gblEta,sizeof(gblEta)); 
	  csctf_.lctglobalEta.push_back(-999);
	  csctf_.lctCLCT_pattern.push_back(-999);
	} 
	
      } // for(CSCCorrelatedLCTDigiCollection::const_iterator lct 
    } // for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator lct

    csctf_.lctSize = nLCT;
    
}


void L1Analysis::L1AnalysisCSCTF::SetStatus(const edm::Handle<L1CSCStatusDigiCollection> status)
{
   int nStat=0;
   for(std::vector<L1CSCSPStatusDigi>::const_iterator stat=status->second.begin();
      stat!=status->second.end(); stat++){

    // fill the Ntuple
    if (stat->VPs() != 0) {
      
      csctf_.stSPslot.push_back(stat->slot());   
      csctf_.stL1A_BXN.push_back(stat->BXN());
      csctf_.stTrkCounter.push_back((const_cast<L1CSCSPStatusDigi*>(&(*stat)))->track_counter());
      csctf_.stOrbCounter.push_back((const_cast<L1CSCSPStatusDigi*>(&(*stat)))->orbit_counter());
      
      nStat++;
    }
   }
  
   csctf_.nsp = nStat;
}

//DT Stubs added by Alex Ji
//Code modeled from DQM/L1TMonitor/src/L1TCSCTF.cc, v1.36
void L1Analysis::L1AnalysisCSCTF::SetDTStubs(const edm::Handle<CSCTriggerContainer<csctf::TrackStub> > dtStubs) {
  //get the vector of DT Stubs
  std::vector<csctf::TrackStub> vstubs = dtStubs->get();
  //iterate through DT Stubs
  for (std::vector<csctf::TrackStub>::const_iterator stub = vstubs.begin();
       stub != vstubs.end(); stub++) {
    csctf_.dtBXN.push_back(stub->BX());
    csctf_.dtFLAG.push_back(stub->getStrip()); //getStrip() is actually the "FLAG" bit
    csctf_.dtCAL.push_back(stub->getKeyWG());  //getKeyWG() is actually the "CAL" bit

    csctf_.dtSector.push_back( 6*(stub->endcap()-1) + stub->sector() );
    csctf_.dtSubSector.push_back(stub->subsector());

    csctf_.dtBX0.push_back(stub->getBX0());    //it is unclear what this variable is...
    csctf_.dtPhiBend.push_back(stub->getBend());
    csctf_.dtPhiPacked.push_back(stub->phiPacked());
    csctf_.dtQuality.push_back(stub->getQuality());
  }

  csctf_.dtSize = vstubs.size();
}
