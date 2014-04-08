//-------------------------------------------------
//
//   Class: L1NtupleProducer
//
//
//   \class L1NtupleProducer
/**
 *   Description:  This code is designed for l1 prompt analysis
//                 starting point is a GMTTreeMaker By Ivan Mikulec now
//                 extended from Lorenzo Agostino. 
*/
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerDPG/L1Ntuples/interface/L1NtupleProducer.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"


//----------------
// Constructor  --
//----------------
L1NtupleProducer::L1NtupleProducer(const edm::ParameterSet& ps) : csctfPtLUTs_(NULL), tree_(0) {

  hltSource_           = ps.getParameter<edm::InputTag>("hltSource");

//gt, gmt  
  gmtSource_           = ps.getParameter<edm::InputTag>("gmtSource");
  gtEvmSource_         = ps.getParameter<edm::InputTag>("gtEvmSource");
  gtSource_	       = ps.getParameter<edm::InputTag>("gtSource");
  generatorSource_     = ps.getParameter<edm::InputTag>("generatorSource");
  simulationSource_    = ps.getParameter<edm::InputTag>("simulationSource");
  physVal_             = ps.getParameter< bool >("physVal");
  
//gct
  gctCenJetsSource_    = ps.getParameter<edm::InputTag>("gctCentralJetsSource");
  gctForJetsSource_    = ps.getParameter<edm::InputTag>("gctForwardJetsSource");
  gctTauJetsSource_    = ps.getParameter<edm::InputTag>("gctTauJetsSource");
  gctEnergySumsSource_ = ps.getParameter<edm::InputTag>("gctEnergySumsSource");
  gctIsoEmSource_      = ps.getParameter<edm::InputTag>("gctIsoEmSource");
  gctNonIsoEmSource_   = ps.getParameter<edm::InputTag>("gctNonIsoEmSource");
  verbose_             = ps.getUntrackedParameter< bool >("verbose", false);

//rct
  rctSource_           = ps.getParameter<edm::InputTag>("rctSource");

//dt  
  dttfSource_          = ps.getParameter<edm::InputTag>("dttfSource");
  
//ecal/hcal
  ecalSource_          = ps.getParameter<edm::InputTag>("ecalSource");
  hcalSource_          = ps.getParameter<edm::InputTag>("hcalSource");

//csctf
  csctfTrkSource_      = ps.getParameter<edm::InputTag>("csctfTrkSource");	
  csctfLCTSource_      = ps.getParameter<edm::InputTag>("csctfLCTSource"); 
  csctfStatusSource_   = ps.getParameter<edm::InputTag>("csctfStatusSource"); 
  csctfDTStubsSource_  = ps.getParameter<edm::InputTag>("csctfDTStubsSource");

  initCSCTFPtLutsPSet  = ps.getParameter< bool >("initCSCTFPtLutsPSet");
  csctfPtLutsPSet      = ps.getParameter<edm::ParameterSet>("csctfPtLutsPSet");
   
//maximum allowed size of tree vectors
  maxRPC_              = ps.getParameter<unsigned int>("maxRPC");
  maxDTBX_             = ps.getParameter<unsigned int>("maxDTBX");
  maxCSC_              = ps.getParameter<unsigned int>("maxCSC");
  maxGMT_              = ps.getParameter<unsigned int>("maxGMT");
  maxGT_               = ps.getParameter<unsigned int>("maxGT");
  maxRCTREG_           = ps.getParameter<unsigned int>("maxRCTREG");
  maxDTPH_             = ps.getParameter<unsigned int>("maxDTPH");
  maxDTTH_             = ps.getParameter<unsigned int>("maxDTTH");
  maxDTTR_             = ps.getParameter<unsigned int>("maxDTTR");
  maxGEN_              = ps.getParameter<unsigned int>("maxGEN");
  maxCSCTFTR_          = ps.getParameter<unsigned int>("maxCSCTFTR");
  maxCSCTFLCTSTR_      = ps.getParameter<unsigned int>("maxCSCTFLCTSTR");
  maxCSCTFLCTS_        = ps.getParameter<unsigned int>("maxCSCTFLCTS");
  maxCSCTFSPS_         = ps.getParameter<unsigned int>("maxCSCTFSPS");

  hcalScaleCacheID_    = 0;
  ecalScaleCacheID_    = 0;

  std::string puMCFile   = ps.getUntrackedParameter<std::string>("puMCFile", "");
  std::string puMCHist   = ps.getUntrackedParameter<std::string>("puMCHist", "pileup");
  std::string puDataFile = ps.getUntrackedParameter<std::string>("puDataFile", "");
  std::string puDataHist = ps.getUntrackedParameter<std::string>("puDataHist", "pileup");

  bool useAvgVtx          = ps.getUntrackedParameter<bool>("useAvgVtx", true);
  double maxAllowedWeight = ps.getUntrackedParameter<double>("maxAllowedWeight", -1);

  pL1evt                = new L1Analysis::L1AnalysisEvent(puMCFile, puMCHist, 
							  puDataFile, puDataHist,
							  useAvgVtx, maxAllowedWeight);
  pL1gmt                = new L1Analysis::L1AnalysisGMT();
  pL1rct                = new L1Analysis::L1AnalysisRCT(maxRCTREG_);
  pL1gt                 = new L1Analysis::L1AnalysisGT();
  pL1gct                = new L1Analysis::L1AnalysisGCT(verbose_);
  pL1dttf               = new L1Analysis::L1AnalysisDTTF();
  pL1csctf              = new L1Analysis::L1AnalysisCSCTF();
  pL1calotp             = new L1Analysis::L1AnalysisCaloTP();
  pL1generator          = new L1Analysis::L1AnalysisGenerator(); 
  pL1simulation         = new L1Analysis::L1AnalysisSimulation();


  pL1evt_data           = pL1evt->getData();
  pL1gmt_data           = pL1gmt->getData();
  pL1rct_data           = pL1rct->getData();
  pL1gt_data            = pL1gt->getData();
  pL1gct_data           = pL1gct->getData();
  pL1dttf_data          = pL1dttf->getData();
  pL1csctf_data         = pL1csctf->getData();
  pL1calotp_data         = pL1calotp->getData();
  pL1generator_data     = pL1generator->getData();
  pL1simulation_data    = pL1simulation->getData();
  
  initCSCTF(); 
  
  tree_ = tfs_->make<TTree>("L1Tree", "L1Tree");

  book();
}

//--------------
// Destructor --
//--------------
L1NtupleProducer::~L1NtupleProducer() { 
  
  //free the CSCTF array of pointers
  for(unsigned int j=0; j<2; j++) 
    for(unsigned int i=0; i<5; i++) 
     delete srLUTs_[i][j]; 
  delete csctfPtLUTs_; 
  
  delete pL1evt;
  delete pL1gmt;
  delete pL1rct;
  delete pL1gct;
  delete pL1gt;
  delete pL1dttf; 
  delete pL1csctf;
  delete pL1calotp;
  delete pL1generator;
  delete pL1simulation;
}

void L1NtupleProducer::beginJob(void) {
}

void L1NtupleProducer::endJob() {

}

//--------------
// Operations --
//--------------

void L1NtupleProducer::analyze(const edm::Event& e, const edm::EventSetup& es) {
  
   //add if "none" ..
  analyzeEvent(e);
  analyzeGenerator(e);
  analyzeSimulation(e);
  analyzeGMT(e);
  analyzeGT(e);
  analyzeGCT(e);
  analyzeRCT(e);
  analyzeDTTF(e);
  analyzeCSCTF(e,es); 
  pL1calotp->Reset();
  analyzeECAL(e, es);
  analyzeHCAL(e, es);                           

  tree_->Fill();

}


void L1NtupleProducer::analyzeEvent(const edm::Event& e) { 
  

    pL1evt->Reset();
    pL1evt->Set(e,hltSource_); 

}

void L1NtupleProducer::analyzeGenerator(const edm::Event& e) {

  if(generatorSource_.label() != "none") {
    pL1generator->Reset();
    pL1generator->Set(e);
  }  

}

void L1NtupleProducer::analyzeSimulation(const edm::Event& e) {

  if(simulationSource_.label() != "none") {
    pL1simulation->Reset();
    pL1simulation->Set(e);
  } 

}


void L1NtupleProducer::analyzeGMT(const edm::Event& e) {
  
  pL1gmt->Reset();
  
  if (gmtSource_.label() != "none") {
    edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle; 
    e.getByLabel(gmtSource_,gmtrc_handle);
    L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();
    
    pL1gmt->Set(gmtrc, maxDTBX_, maxCSC_, maxRPC_, maxGMT_, physVal_);
  }
}


void L1NtupleProducer::analyzeGT(const edm::Event& e) {
  
  pL1gt->Reset();
  
  if (gtEvmSource_.label() != "none") {
    edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtevmrr_handle;
    e.getByLabel(gtEvmSource_.label(), gtevmrr_handle);
    L1GlobalTriggerEvmReadoutRecord const* gtevmrr = gtevmrr_handle.product();

    L1TcsWord tcsw = gtevmrr->tcsWord();

    pL1evt_data->bx = tcsw.bxNr();
    pL1evt_data->orbit = tcsw.orbitNr();
    //pL1evt->lumi = tcsw.luminositySegmentNr();
    //pL1evt->run = tcsw.partRunNr();
    //pL1evt->event = tcsw.partTrigNr();

    pL1gt->SetEvm(gtevmrr);
  }

   
  if (gtSource_.label() != "none") {
    edm::Handle<L1GlobalTriggerReadoutRecord> gtrr_handle;
    e.getByLabel(gtSource_.label(), gtrr_handle);
    L1GlobalTriggerReadoutRecord const* gtrr = gtrr_handle.product();

    pL1gt->Set(gtrr);
  }
}


  
void L1NtupleProducer::analyzeRCT(const edm::Event& e) {
  
  pL1rct->Reset();
  
  edm::Handle < L1CaloEmCollection > em;
  e.getByLabel(rctSource_,em);
  if (!em.isValid()) {
    pL1rct_data->InitEmRCT();
    edm::LogInfo("L1Prompt") << "can't find L1CaloEmCollection with label "
			     << rctSource_.label() ;}			     
  else pL1rct->SetEmRCT(em);
  
  edm::Handle < L1CaloRegionCollection > rgn;
  e.getByLabel(rctSource_,rgn);
  if (!rgn.isValid()) { 
    pL1rct_data->InitHdRCT();
    edm::LogInfo("L1Prompt") << "can't find L1CaloRegionCollection with label "
			     << rctSource_.label() ;}
  else pL1rct->SetHdRCT(rgn);

}
  
  
void L1NtupleProducer::analyzeDTTF(const edm::Event& e)
{
  
  pL1dttf->Reset();

  edm::Handle<L1MuDTChambPhContainer > myL1MuDTChambPhContainer;  
  e.getByLabel(dttfSource_,myL1MuDTChambPhContainer);
  if (!myL1MuDTChambPhContainer.isValid()) {
    edm::LogInfo("L1Prompt") << "can't find L1MuDTChambPhContainer with label " << dttfSource_.label() ;   
    }
  else pL1dttf->SetDTPH(myL1MuDTChambPhContainer, maxDTPH_);

  edm::Handle<L1MuDTChambThContainer > myL1MuDTChambThContainer;  
  e.getByLabel(dttfSource_,myL1MuDTChambThContainer);
  if (!myL1MuDTChambThContainer.isValid()) {
    edm::LogInfo("L1Prompt") << "can't find L1MuDTChambThContainer with label " << dttfSource_.label() ;
    edm::LogInfo("L1Prompt") << "if this fails try to add DATA to the process name." ;   
    }
  else pL1dttf->SetDTTH(myL1MuDTChambThContainer, maxDTTH_);


  edm::Handle<L1MuDTTrackContainer > myL1MuDTTrackContainer;
  std::string trstring;
  trstring = dttfSource_.label()+":"+"DATA"+":"+dttfSource_.process();
  edm::InputTag trInputTag(trstring);
  e.getByLabel(trInputTag,myL1MuDTTrackContainer);
  if (myL1MuDTTrackContainer.isValid()) {
    pL1dttf->SetDTTR(myL1MuDTTrackContainer, maxDTTR_);
  }
  else {
    trstring = dttfSource_.label()+":"+"DTTF"+":"+dttfSource_.process();
    edm::InputTag trInputTag(trstring);
    e.getByLabel(trInputTag,myL1MuDTTrackContainer);
    if (myL1MuDTTrackContainer.isValid()) {
      pL1dttf->SetDTTR(myL1MuDTTrackContainer, maxDTTR_);
    }
    else {
      edm::LogInfo("L1Prompt") << "can't find L1MuDTTrackContainer " << dttfSource_.label();
    }
  }

}


  
void L1NtupleProducer::analyzeGCT(const edm::Event& e)
{
 
  pL1gct->Reset();
  
  ///
    
  edm::Handle < L1GctEmCandCollection > l1IsoEm;
  e.getByLabel(gctIsoEmSource_, l1IsoEm);
  if (!l1IsoEm.isValid() && gctIsoEmSource_.label() != "none") 
    edm::LogWarning("DataNotFound") << " Could not find l1IsoEm "" elements, label was " << gctIsoEmSource_ ;
  
  edm::Handle < L1GctEmCandCollection > l1NonIsoEm;
  e.getByLabel(gctNonIsoEmSource_, l1NonIsoEm);
  if (!l1NonIsoEm.isValid() && gctNonIsoEmSource_.label() != "none") 
    edm::LogWarning("DataNotFound") << " Could not find l1NonIsoEm "" elements, label was " << gctNonIsoEmSource_ ;
    
  if (l1IsoEm.isValid() && l1NonIsoEm.isValid())
    pL1gct->SetEm(l1IsoEm, l1NonIsoEm);
  else 
    pL1gct_data->Init();
  
  ///
  
  edm::Handle < L1GctJetCandCollection > l1CenJets;
  e.getByLabel(gctCenJetsSource_, l1CenJets);
  if (!l1CenJets.isValid() && gctCenJetsSource_.label() != "none")  
    edm::LogWarning("DataNotFound") << " Could not find l1CenJets"", label was " << gctCenJetsSource_ ;
    
  edm::Handle < L1GctJetCandCollection > l1ForJets;
  e.getByLabel(gctForJetsSource_, l1ForJets);
  if (!l1ForJets.isValid() && gctForJetsSource_.label() != "none")  
    edm::LogWarning("DataNotFound") << " Could not find l1ForJets"", label was " << gctForJetsSource_ ;  
  
  edm::Handle < L1GctJetCandCollection > l1TauJets;
  e.getByLabel(gctTauJetsSource_, l1TauJets);
  if (!l1TauJets.isValid() && gctTauJetsSource_.label() != "none")  
    edm::LogWarning("DataNotFound") << " Could not find l1TauJets"", label was " << gctTauJetsSource_;
  
  if (l1CenJets.isValid() && l1ForJets.isValid() && l1TauJets.isValid()) 
    pL1gct->SetJet(l1CenJets, l1ForJets, l1TauJets);
  
  ///
  
  edm::Handle < L1GctHFRingEtSumsCollection > l1HFSums; 
  e.getByLabel(gctEnergySumsSource_, l1HFSums); 
  if (!l1HFSums.isValid() && gctEnergySumsSource_.label() != "none")  
    edm::LogWarning("DataNotFound") << " Could not find l1HFSums"", label was " << gctEnergySumsSource_ ;
    
  edm::Handle < L1GctHFBitCountsCollection > l1HFCounts;
  e.getByLabel(gctEnergySumsSource_, l1HFCounts);
  if (!l1HFCounts.isValid() && gctEnergySumsSource_.label() != "none")  
    edm::LogWarning("DataNotFound") << " Could not find l1HFCounts"", label was " << gctEnergySumsSource_ ;
 
  if (l1HFSums.isValid() && l1HFCounts.isValid()) 
  pL1gct->SetHFminbias(l1HFSums, l1HFCounts);    
  
  ///
  
  edm::Handle < L1GctEtMissCollection >  l1EtMiss;
  e.getByLabel(gctEnergySumsSource_, l1EtMiss);
  if (!l1EtMiss.isValid() && gctEnergySumsSource_.label() != "none")  
    edm::LogWarning("DataNotFound") << " Could not find l1EtMiss"", label was " << gctEnergySumsSource_ ;

  edm::Handle < L1GctHtMissCollection >  l1HtMiss;
  e.getByLabel(gctEnergySumsSource_, l1HtMiss);
  if (!l1HtMiss.isValid() && gctEnergySumsSource_.label() != "none")  
    edm::LogWarning("DataNotFound") << " Could not find l1HtMiss"", label was " << gctEnergySumsSource_ ;
  
  edm::Handle < L1GctEtHadCollection >   l1EtHad;
  e.getByLabel(gctEnergySumsSource_, l1EtHad);
  if (!l1EtHad.isValid() && gctEnergySumsSource_.label() != "none")  
    edm::LogWarning("DataNotFound") << " Could not find l1EtHad"", label was " << gctEnergySumsSource_ ;
  
  edm::Handle < L1GctEtTotalCollection > l1EtTotal;
  e.getByLabel(gctEnergySumsSource_, l1EtTotal);
  if (!l1EtTotal.isValid() && gctEnergySumsSource_.label() != "none")  
    edm::LogWarning("DataNotFound") << " Could not find l1EtTotal"", label was " << gctEnergySumsSource_ ;
 
  if (l1EtMiss.isValid() && l1HtMiss.isValid() && l1EtHad.isValid() && l1EtTotal.isValid())
    pL1gct->SetES(l1EtMiss, l1HtMiss, l1EtHad, l1EtTotal);

}

void L1NtupleProducer::analyzeCSCTF(const edm::Event& e, const edm::EventSetup& es)
{
 
  pL1csctf->Reset();

  //csctf (tracks)
  if( csctfTrkSource_.label() != "none" )
  {   
     if( es.get< L1MuTriggerScalesRcd > ().cacheIdentifier() != m_scalesCacheID ||
         es.get< L1MuTriggerPtScaleRcd >().cacheIdentifier() != m_ptScaleCacheID ){
      
      edm::ESHandle< L1MuTriggerScales > scales;
      es.get< L1MuTriggerScalesRcd >().get(scales);
      ts = scales.product();
      edm::ESHandle< L1MuTriggerPtScale > ptscales;
      es.get< L1MuTriggerPtScaleRcd >().get(ptscales);
      tpts = ptscales.product();
      m_scalesCacheID  = es.get< L1MuTriggerScalesRcd  >().cacheIdentifier();
      m_ptScaleCacheID = es.get< L1MuTriggerPtScaleRcd >().cacheIdentifier();
      
      edm::LogInfo("L1NtupleProducer") << "Changing triggerscales and triggerptscales for CSCTF...\n";
     }    

     // initialize ptLUTs from a PSet if the flag is True
     if( initCSCTFPtLutsPSet ) 
       {
         // set it only if the pointer is empty
         if (csctfPtLUTs_ == NULL) {
           edm::LogInfo("L1NtupleProducer") << "Initializing csctf pt LUTs from PSet...";
           csctfPtLUTs_ = new CSCTFPtLUT(csctfPtLutsPSet, ts, tpts);
         }
       } 

     // otherwise use the O2O mechanism
     // else if (es.get< L1MuCSCPtLutRcd > ().cacheIdentifier() != m_csctfptlutCacheID )
     //   {
     //     edm::LogInfo("L1NtupleProducer") << "  Initializing the CSCTF ptLUTs via O2O mechanism...";
     //     // initializing the ptLUT from O2O
     //     csctfPtLUTs_ = new CSCTFPtLUT(es);
         
     //     m_csctfptlutCacheID = es.get< L1MuCSCPtLutRcd > ().cacheIdentifier();
     //     edm::LogInfo("L1NtupleProducer") << "  Changed the cache ID for CSCTF ptLUTs...";
     //   }
     
     if (csctfPtLUTs_ == NULL)
       edm::LogWarning("L1NtupleProducer")<<"  No valid CSCTFPtLUT initialized!";

    
     edm::Handle<L1CSCTrackCollection> csctfTrks;
     e.getByLabel(csctfTrkSource_.label(),csctfTrkSource_.instance(),csctfTrks);
     
     if( csctfTrks.isValid() && csctfPtLUTs_) 
       pL1csctf->SetTracks(csctfTrks, ts, tpts, srLUTs_, csctfPtLUTs_);
     else 
       edm::LogInfo("L1NtupleProducer")<<" No valid L1CSCTrackCollection products found"
                                       <<" or ptLUT pointer(" << csctfPtLUTs_ <<") null";
       
          
   } else 
    edm::LogInfo("L1NtupleProducer")<<" No valid L1CSCTrackCollection products found";
     
     
      
  //ALL csctf lcts
  if( csctfLCTSource_.label() != "none" )
  {
     edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts;
     e.getByLabel(csctfLCTSource_.label(),csctfLCTSource_.instance(),corrlcts);
     if( corrlcts.isValid() ) pL1csctf->SetLCTs(corrlcts, srLUTs_);
   } else 
    edm::LogInfo("L1NtupleProducer")<<"  No valid CSCCorrelatedLCTDigiCollection products found";

           
  //csctf status
  if( csctfStatusSource_.label() != "none" )
  {
     edm::Handle<L1CSCStatusDigiCollection> status;
     e.getByLabel(csctfStatusSource_.label(),csctfStatusSource_.instance(),status);  
     if( status.isValid() ) pL1csctf->SetStatus(status);
   } else 
     edm::LogInfo("L1NtupleProducer")<<"  No valid L1CSCTrackCollection products found";  

  //dt stubs
  if( csctfDTStubsSource_.label() != "none" )    {
    edm::Handle<CSCTriggerContainer<csctf::TrackStub> > dtStubs;
    e.getByLabel(csctfDTStubsSource_.label(),csctfDTStubsSource_.instance(),dtStubs);  
    if( dtStubs.isValid() ) {
      pL1csctf->SetDTStubs(dtStubs);
    } else {
      edm::LogInfo("L1NtupleProducer")<<"  No valid DT TrackStub products found";    
    }
  } else
    edm::LogInfo("L1NtupleProducer")<<"  Label for DTStubsSource is set to none";
}  

void L1NtupleProducer::analyzeECAL(const edm::Event& e, const edm::EventSetup& es) {

  if( es.get< L1CaloEcalScaleRcd > ().cacheIdentifier() != ecalScaleCacheID_ ) {
    edm::ESHandle<L1CaloEcalScale> ecalScale;
    es.get<L1CaloEcalScaleRcd>().get(ecalScale);
    pL1calotp->setEcalScale( ecalScale.product() );
    ecalScaleCacheID_ = es.get< L1CaloEcalScaleRcd > ().cacheIdentifier();
  }

  if( ecalSource_.label() != "none" ) {

    edm::Handle< EcalTrigPrimDigiCollection > ecalTPs;
    e.getByLabel(ecalSource_,ecalTPs);

    if( ecalTPs.isValid() ) {
      pL1calotp->SetECAL( *ecalTPs.product());
    } else {
      edm::LogInfo("L1NtupleProducer")<<"  No valid ECAL trigger primitives found";
    }
  } else
    edm::LogInfo("L1NtupleProducer")<<"  Label for ECAL trig prims is set to none";

}

void L1NtupleProducer::analyzeHCAL(const edm::Event& e, const edm::EventSetup& es) {

  //  if( es.get< L1CaloHcalScaleRcd > ().cacheIdentifier() != hcalScaleCacheID_ ) {
    edm::ESHandle<L1CaloHcalScale> hcalScale;
    es.get<L1CaloHcalScaleRcd>().get(hcalScale);
    pL1calotp->setHcalScale( hcalScale.product() );
    hcalScaleCacheID_ = es.get< L1CaloHcalScaleRcd > ().cacheIdentifier();
    //  }  

  if( hcalSource_.label() != "none" ) {

    edm::Handle< HcalTrigPrimDigiCollection > hcalTPs;
    e.getByLabel(hcalSource_,hcalTPs);

    if( hcalTPs.isValid() ) {
      pL1calotp->SetHCAL(*hcalTPs.product());
    } else {
      edm::LogInfo("L1NtupleProducer")<<"  No valid HCAL trigger primitives found";
    }
  } else
    edm::LogInfo("L1NtupleProducer")<<"  Label for HCAL trig prims is set to none";

}


//--------------
// Operations --
//--------------
void L1NtupleProducer::book() {

  tree_->Branch("Event", "L1Analysis::L1AnalysisEventDataFormat", &pL1evt_data, 32000, 3);
   
  if (gctCenJetsSource_.label() != "none" || 
      gctForJetsSource_.label() != "none" || 
      gctTauJetsSource_.label() != "none" || 
      gctEnergySumsSource_.label() != "none" || 
      gctIsoEmSource_.label() != "none" || 
      gctNonIsoEmSource_.label() != "none")
  tree_->Branch("GCT", "L1Analysis::L1AnalysisGCTDataFormat", &pL1gct_data, 32000, 3);

  if (generatorSource_.label() != "none") 
    tree_->Branch("Generator", "L1Analysis::L1AnalysisGeneratorDataFormat", &pL1generator_data, 32000, 3);
  
  if (simulationSource_.label() != "none") 
    tree_->Branch("Simulation", "L1Analysis::L1AnalysisSimulationDataFormat", &pL1simulation_data, 32000, 3);
  
  if (gmtSource_.label() != "none")   
    tree_->Branch("GMT", "L1Analysis::L1AnalysisGMTDataFormat", &pL1gmt_data, 32000, 3);
    
  if (gtSource_.label() != "none") 
    tree_->Branch("GT", "L1Analysis::L1AnalysisGTDataFormat", &pL1gt_data, 32000, 3);
     
  if (rctSource_.label() != "none") 
    tree_->Branch("RCT", "L1Analysis::L1AnalysisRCTDataFormat", &pL1rct_data, 32000, 3);

  if (dttfSource_.label() != "none") 
  tree_->Branch("DTTF", "L1Analysis::L1AnalysisDTTFDataFormat", &pL1dttf_data, 32000, 3);
  if (csctfTrkSource_.label() != "none" ||
      csctfLCTSource_.label() != "none" ||
      csctfStatusSource_.label() != "none" ||
      csctfDTStubsSource_.label() != "none" ) 
  tree_->Branch("CSCTF", "L1Analysis::L1AnalysisCSCTFDataFormat", &pL1csctf_data, 32000, 3);

  if (!(ecalSource_.label()=="none" && hcalSource_.label()=="none")) {
    tree_->Branch("CALO", "L1Analysis::L1AnalysisCaloTPDataFormat", &pL1calotp_data, 32000, 3);
  }

}

void L1NtupleProducer::initCSCTF() {
  
  // csctf does not preserve information about the LCT (stubs) which forms
  // the track so we need to retrieve this information. In order to do so
  // we need to initialize the Sector Receiver LUTs in the software
  
  bzero(srLUTs_ , sizeof(srLUTs_));
  int sector=1;    // assume SR LUTs are all same for every sector
  bool TMB07=true; // specific TMB firmware
  // Create a pset for SR/PT LUTs: if you do not change the value in the 
  // configuration file, it will load the default minitLUTs
  edm::ParameterSet srLUTset;
  srLUTset.addUntrackedParameter<bool>("ReadLUTs", false);
  srLUTset.addUntrackedParameter<bool>("Binary",   false);
  srLUTset.addUntrackedParameter<std::string>("LUTPath", "./");
 
   // positive endcap
  int endcap = 1; 
  for(int station=1,fpga=0; station<=4 && fpga<5; station++)
    {
      if(station==1)
	for(int subSector=0; subSector<2 && fpga<5; subSector++)
	  srLUTs_[fpga++][1] = new CSCSectorReceiverLUT(endcap,sector,subSector+1,
							station, srLUTset, TMB07);
      else
	srLUTs_[fpga++][1]   = new CSCSectorReceiverLUT(endcap,  sector,   0, 
							station, srLUTset, TMB07);
    }

  // negative endcap
  endcap = 2; 
  for(int station=1,fpga=0; station<=4 && fpga<5; station++)
    {
      if(station==1)
	for(int subSector=0; subSector<2 && fpga<5; subSector++)
	  srLUTs_[fpga++][0] = new CSCSectorReceiverLUT(endcap,sector,subSector+1,
							station, srLUTset, TMB07);
      else
	srLUTs_[fpga++][0]   = new CSCSectorReceiverLUT(endcap,  sector,   0, 
							station, srLUTset, TMB07);
    }
 
}

DEFINE_FWK_MODULE( L1NtupleProducer );
