#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisGCT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

L1Analysis::L1AnalysisGCT::L1AnalysisGCT():verbose_(false)
{
}

L1Analysis::L1AnalysisGCT::L1AnalysisGCT(bool verbose)
{
  verbose_ = verbose;
}

L1Analysis::L1AnalysisGCT::~L1AnalysisGCT()
{

}
void L1Analysis::L1AnalysisGCT::SetJet(const edm::Handle < L1GctJetCandCollection > l1CenJets,
                    		       const edm::Handle < L1GctJetCandCollection > l1ForJets,
                  		       const edm::Handle < L1GctJetCandCollection > l1TauJets)
{   
   
    // Central jets
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1NtupleProducer: number of central jets = " 
		<< l1CenJets->size() << std::endl;
    }
    gct_.CJetSize= l1CenJets->size();//1
    for (L1GctJetCandCollection::const_iterator cj = l1CenJets->begin();
	 cj != l1CenJets->end(); cj++) {
      gct_.CJetEta.push_back(cj->regionId().ieta());//2
      gct_.CJetPhi.push_back(cj->regionId().iphi());//3
      gct_.CJetRnk.push_back(cj->rank());//4
      gct_.CJetBx .push_back(cj->bx());//4
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1NtupleProducer: Central jet " 
		  << cj->regionId().iphi() << ", " << cj->regionId().ieta()
		  << ", " << cj->rank() << std::endl;
      }
    }

    // Forward jets
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1NtupleProducer: number of forward jets = " 
		<< l1ForJets->size() << std::endl;
    }
    gct_.FJetSize= l1ForJets->size();//5
    for (L1GctJetCandCollection::const_iterator fj = l1ForJets->begin();
	 fj != l1ForJets->end(); fj++) {
      gct_.FJetEta.push_back(fj->regionId().ieta());//6
      gct_.FJetPhi.push_back(fj->regionId().iphi());//7
      gct_.FJetRnk.push_back(fj->rank());//8
      gct_.FJetBx .push_back(fj->bx());//8
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1NtupleProducer: Forward jet " 
		  << fj->regionId().iphi() << ", " << fj->regionId().ieta()
		  << ", " << fj->rank() << std::endl;
      }
    }

    // Tau jets
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1NtupleProducer: number of tau jets = " 
		<< l1TauJets->size() << std::endl;
    }
    gct_.FJetSize= l1TauJets->size();//9
     for (L1GctJetCandCollection::const_iterator tj = l1TauJets->begin();
	 tj != l1TauJets->end(); tj++) {
      //if ( tj->rank() == 0 ) continue;
      gct_.TJetEta.push_back(tj->regionId().ieta());//10
      gct_.TJetPhi.push_back(tj->regionId().iphi());//11
      gct_.TJetRnk.push_back(tj->rank());//12
      gct_.TJetBx .push_back(tj->bx());//
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1NtupleProducer: Tau jet " 
			       << tj->regionId().iphi() << ", " << tj->regionId().ieta()
			       << ", " << tj->rank() << std::endl;
      }
    }
  
}
 
void L1Analysis::L1AnalysisGCT::SetES(const edm::Handle < L1GctEtMissCollection > l1EtMiss, 
                                      const edm::Handle < L1GctHtMissCollection >  l1HtMiss,
                                      const edm::Handle < L1GctEtHadCollection > l1EtHad, 
				      const edm::Handle < L1GctEtTotalCollection > l1EtTotal)
{ 
  
  // Energy sums
  for (L1GctEtMissCollection::const_iterator etm = l1EtMiss->begin();
       etm != l1EtMiss->end();
       ++etm) {

    gct_.EtMiss.push_back( etm->et() );
    gct_.EtMissPhi.push_back( etm->phi() );
    gct_.EtMissBX.push_back( etm->bx() );
    gct_.EtMissSize++;
    
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1NtupleProducer: Et Miss " 
			       << etm->et() << ", " <<  etm->phi()
			       << ", " <<  etm->bx() << std::endl;
    }
  }

   for (L1GctHtMissCollection::const_iterator htm = l1HtMiss->begin();
       htm != l1HtMiss->end();
       ++htm) {

    gct_.HtMiss.push_back( htm->et() );
    gct_.HtMissPhi.push_back( htm->phi() );
    gct_.HtMissBX.push_back( htm->bx() );
    gct_.HtMissSize++;
    
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1NtupleProducer: Ht Miss " 
			       << htm->et() << ", " <<  htm->phi()
			       << ", " <<  htm->bx() << std::endl;
    }
  }

   for (L1GctEtHadCollection::const_iterator ht = l1EtHad->begin();
       ht != l1EtHad->end();
       ++ht) {

    gct_.EtHad.push_back( ht->et() );
    gct_.EtHadBX.push_back( ht->bx() );
    gct_.EtHadSize++;
    
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1NtupleProducer: Ht Total " 
			       << ht->et()
			       << ", " <<  ht->bx() << std::endl;
    }
  }

  for (L1GctEtTotalCollection::const_iterator ett = l1EtTotal->begin();
       ett != l1EtTotal->end();
       ++ett) {

    gct_.EtTot.push_back( ett->et() );
    gct_.EtTotBX.push_back( ett->bx() );
    gct_.EtTotSize++;
    
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1NtupleProducer: Et Total " 
			       << ett->et()
			       << ", " <<  ett->bx() << std::endl;
    }
  }

}

void L1Analysis::L1AnalysisGCT::SetHFminbias(const edm::Handle < L1GctHFRingEtSumsCollection > l1HFSums, 
                                             const edm::Handle < L1GctHFBitCountsCollection > l1HFCounts)
{   
    
   //Fill HF Ring Histograms
    gct_.HFRingEtSumSize=l1HFSums->size();
    int ies=0;
    for (L1GctHFRingEtSumsCollection::const_iterator hfs=l1HFSums->begin(); hfs!=l1HFSums->end(); hfs++){ 
       gct_.HFRingEtSumEta.push_back(hfs->etSum(ies));
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1NtupleProducer: HF Sums " 
			         << l1HFSums->size() << ", " << hfs->etSum(ies) << std::endl;
      }
      ies++;
    }
    
    int ibc=0;
    gct_.HFBitCountsSize=l1HFCounts->size();
    for (L1GctHFBitCountsCollection::const_iterator hfc=l1HFCounts->begin(); hfc!=l1HFCounts->end(); hfc++){ 
      gct_.HFBitCountsEta.push_back(hfc->bitCount(ibc));
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1NtupleProducer: HF Counts " 
			         << l1HFCounts->size() << ", " << hfc->bitCount(ibc) << std::endl;
      }
      ibc++;
    }
   
}
  
void L1Analysis::L1AnalysisGCT::SetEm(const edm::Handle < L1GctEmCandCollection > l1IsoEm, 
                                      const edm::Handle < L1GctEmCandCollection > l1NonIsoEm)
{   
        
      // Isolated EM
      if ( verbose_ ) {
    	edm::LogInfo("L1Prompt") << "L1TGCT: number of iso em cands: " 
    		  << l1IsoEm->size() << std::endl;
      }
      
      gct_.IsoEmSize = l1IsoEm->size();
      for (L1GctEmCandCollection::const_iterator ie=l1IsoEm->begin(); ie!=l1IsoEm->end(); ie++) {
    	
    	gct_.IsoEmEta.push_back(ie->regionId().ieta());
    	gct_.IsoEmPhi.push_back(ie->regionId().iphi());
    	gct_.IsoEmRnk.push_back(ie->rank());
        gct_.IsoEmBx.push_back(ie->bx());
      } 

      // Non-isolated EM
      if ( verbose_ ) {
    	edm::LogInfo("L1Prompt") << "L1TGCT: number of non-iso em cands: " 
    		  << l1NonIsoEm->size() << std::endl;
      }
      gct_.NonIsoEmSize = l1NonIsoEm->size();
      
      for (L1GctEmCandCollection::const_iterator ne=l1NonIsoEm->begin(); ne!=l1NonIsoEm->end(); ne++) {
    	gct_.NonIsoEmEta.push_back(ne->regionId().ieta());
    	gct_.NonIsoEmPhi.push_back(ne->regionId().iphi());
    	gct_.NonIsoEmRnk.push_back(ne->rank());
	gct_.NonIsoEmBx.push_back(ne->bx());
    	  
      } 
   
}


