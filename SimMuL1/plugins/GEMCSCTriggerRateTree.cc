#include "GEMCode/SimMuL1/plugins/GEMCSCTriggerRateTree.h"

const int GEMCSCTriggerRateTree::pbend[CSCConstants::NUM_CLCT_PATTERNS]= 
   { -999,  -5,  4, -4,  3, -3,  2, -2,  1, -1,  0}; // "signed" pattern (== phiBend)
const double GEMCSCTriggerRateTree::PT_THRESHOLDS[N_PT_THRESHOLDS] = {0,10,20,30,40,50};
const double GEMCSCTriggerRateTree::PT_THRESHOLDS_FOR_ETA[N_PT_THRESHOLDS] = {10,15,30,40,55,70};

// ================================================================================================
GEMCSCTriggerRateTree::GEMCSCTriggerRateTree(const edm::ParameterSet& iConfig):
  CSCTFSPset(iConfig.getParameter<edm::ParameterSet>("SectorProcessor")),
  ptLUTset(CSCTFSPset.getParameter<edm::ParameterSet>("PTLUT")),
  ptLUT(0),
  matchAllTrigPrimitivesInChamber_(iConfig.getUntrackedParameter<bool>("matchAllTrigPrimitivesInChamber", false)),
  debugRATE(iConfig.getUntrackedParameter<int>("debugRATE", 0)),
  minBX_(iConfig.getUntrackedParameter<int>("minBX",-6)),
  maxBX_(iConfig.getUntrackedParameter<int>("maxBX",6)),
  minTMBBX_(iConfig.getUntrackedParameter<int>("minTMBBX",-6)),
  maxTMBBX_(iConfig.getUntrackedParameter<int>("maxTMBBX",6)),
  minRateBX_(iConfig.getUntrackedParameter<int>("minRateBX",-1)),
  maxRateBX_(iConfig.getUntrackedParameter<int>("maxRateBX",1)),
  minBxALCT_(iConfig.getUntrackedParameter<int>("minBxALCT",5)),
  maxBxALCT_(iConfig.getUntrackedParameter<int>("maxBxALCT",7)),
  minBxCLCT_(iConfig.getUntrackedParameter<int>("minBxCLCT",5)),
  maxBxCLCT_(iConfig.getUntrackedParameter<int>("maxBxCLCT",7)),
  minBxLCT_(iConfig.getUntrackedParameter<int>("minBxLCT",5)),
  maxBxLCT_(iConfig.getUntrackedParameter<int>("maxBxLCT",7)),
  minBxMPLCT_(iConfig.getUntrackedParameter<int>("minBxMPLCT",5)),
  maxBxMPLCT_(iConfig.getUntrackedParameter<int>("maxBxMPLCT",7)),
  minBxGMT_(iConfig.getUntrackedParameter<int>("minBxGMT",-1)),
  maxBxGMT_(iConfig.getUntrackedParameter<int>("maxBxGMT",1)),
  centralBxOnlyGMT_(iConfig.getUntrackedParameter< bool >("centralBxOnlyGMT",false)),
  doSelectEtaForGMTRates_(iConfig.getUntrackedParameter< bool >("doSelectEtaForGMTRates",false)),
  doME1a_(iConfig.getUntrackedParameter< bool >("doME1a",false)),
  // special treatment of matching in ME1a for the case of the default emulator
  defaultME1a(iConfig.getUntrackedParameter<bool>("defaultME1a", false))
{
  edm::ParameterSet srLUTset = CSCTFSPset.getParameter<edm::ParameterSet>("SRLUT");

  for(int e=0; e<2; e++) 
    for (int s=0; s<6; s++) 
      my_SPs[e][s] = nullptr;
  
  bool TMB07 = true;
  for(int endcap = 1; endcap<=2; endcap++)
  {
    for(int sector=1; sector<=6; sector++)
    {
      for(int station=1,fpga=0; station<=4 && fpga<5; station++)
      {
	if(station==1) for(int subSector=0; subSector<2; subSector++)
	  srLUTs_[fpga++][sector-1][endcap-1] = new CSCSectorReceiverLUT(endcap, sector, subSector+1, station, srLUTset, TMB07);
	else
	  srLUTs_[fpga++][sector-1][endcap-1] = new CSCSectorReceiverLUT(endcap, sector, 0, station, srLUTset, TMB07);
      }
    }
  }

  my_dtrc = new CSCTFDTReceiver();

  // cache flags for event setup records
  muScalesCacheID_ = 0ULL ;
  muPtScaleCacheID_ = 0ULL ;

  bookALCTTree();
  bookCLCTTree();
  bookLCTTree();
  bookMPCLCTTree();
  bookTFTrackTree();
  bookTFCandTree();
  bookGMTRegionalTree();
  bookGMTCandTree();
}

// ================================================================================================
GEMCSCTriggerRateTree::~GEMCSCTriggerRateTree()
{
  if(ptLUT) delete ptLUT;
  ptLUT = nullptr;

  for(int e=0; e<2; e++) for (int s=0; s<6; s++){
      if  (my_SPs[e][s]) delete my_SPs[e][s];
      my_SPs[e][s] = nullptr;

      for(int fpga=0; fpga<5; fpga++)
	{
	  if (srLUTs_[fpga][s][e]) delete srLUTs_[fpga][s][e];
	  srLUTs_[fpga][s][e] = nullptr;
	}
    }
  
  if(my_dtrc) delete my_dtrc;
  my_dtrc = nullptr;
}

// ================================================================================================
void
GEMCSCTriggerRateTree::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup)
{
  edm::ESHandle< CSCGeometry > cscGeom;
  iSetup.get<MuonGeometryRecord>().get(cscGeom);
  cscGeometry = &*cscGeom;
  CSCTriggerGeometry::setGeometry(cscGeometry);
}

// ================================================================================================
void 
GEMCSCTriggerRateTree::beginJob()
{
}


// ================================================================================================
void 
GEMCSCTriggerRateTree::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // need to reset here

  analyzeALCTRate(iEvent);
  analyzeCLCTRate(iEvent);
  analyzeLCTRate(iEvent);
  analyzeMPCLCTRate(iEvent);
  analyzeTFTrackRate(iEvent);
  analyzeTFCandRate(iEvent);
  analyzeGMTRegCandRate(iEvent);
  analyzeGMTCandRate(iEvent);


//   // DT primitives for input to TF
//   edm::Handle<L1MuDTChambPhContainer> dttrig;
//   iEvent.getByLabel("simDtTriggerPrimitiveDigis", dttrig);
//   const L1MuDTChambPhContainer* dttrigs = dttrig.product();

//   // L1 muon candidates after CSC sorter
//   edm::Handle< std::vector< L1MuRegionalCand > > hl1TfCands;
//   iEvent.getByLabel("simCsctfDigis", "CSC", hl1TfCands);
//   const std::vector< L1MuRegionalCand > *l1TfCands = hl1TfCands.product();

//   // GMT readout collection
//   edm::Handle< L1MuGMTReadoutCollection > hl1GmtCands;
//   iEvent.getByLabel("simGmtDigis", hl1GmtCands ) ;// InputTag("simCsctfDigis","CSC")

//   //const L1MuGMTReadoutCollection* l1GmtCands = hl1GmtCands.product();
//   std::vector<L1MuGMTExtendedCand> l1GmtCands;
//   std::vector<L1MuGMTExtendedCand> l1GmtfCands;
//   std::vector<L1MuRegionalCand>    l1GmtCSCCands;
//   std::vector<L1MuRegionalCand>    l1GmtRPCfCands;
//   std::vector<L1MuRegionalCand>    l1GmtRPCbCands;
//   std::vector<L1MuRegionalCand>    l1GmtDTCands;

//   // key = BX
//   std::map<int, std::vector<L1MuRegionalCand> >  l1GmtCSCCandsInBXs;

//   // TOCHECK
//   if ( centralBxOnlyGMT_ )
//   {
//     // Get GMT candidates from central bunch crossing only
//     l1GmtCands = hl1GmtCands->getRecord().getGMTCands() ;
//     l1GmtfCands = hl1GmtCands->getRecord().getGMTFwdCands() ;
//     l1GmtCSCCands = hl1GmtCands->getRecord().getCSCCands() ;
//     l1GmtRPCfCands = hl1GmtCands->getRecord().getFwdRPCCands() ;
//     l1GmtRPCbCands = hl1GmtCands->getRecord().getBrlRPCCands() ;
//     l1GmtDTCands = hl1GmtCands->getRecord().getDTBXCands() ;
//     l1GmtCSCCandsInBXs[hl1GmtCands->getRecord().getBxInEvent()] = l1GmtCSCCands;
//   }
//   else
//   {
//     // Get GMT candidates from all bunch crossings
//     std::vector<L1MuGMTReadoutRecord> gmt_records = hl1GmtCands->getRecords();
//     for ( std::vector< L1MuGMTReadoutRecord >::const_iterator rItr=gmt_records.begin(); rItr!=gmt_records.end() ; ++rItr )
//       {
// 	if (rItr->getBxInEvent() < minBxGMT_ || rItr->getBxInEvent() > maxBxGMT_) continue;
	
// 	std::vector<L1MuGMTExtendedCand> GMTCands = rItr->getGMTCands();
// 	for ( std::vector<L1MuGMTExtendedCand>::const_iterator  cItr = GMTCands.begin() ; cItr != GMTCands.end() ; ++cItr )
// 	  if (!cItr->empty()) l1GmtCands.push_back(*cItr);
	
// 	std::vector<L1MuGMTExtendedCand> GMTfCands = rItr->getGMTFwdCands();
// 	for ( std::vector<L1MuGMTExtendedCand>::const_iterator  cItr = GMTfCands.begin() ; cItr != GMTfCands.end() ; ++cItr )
// 	  if (!cItr->empty()) l1GmtfCands.push_back(*cItr);
	
// 	//std::cout<<" ggg: "<<GMTCands.size()<<" "<<GMTfCands.size()<<std::endl;
	
// 	std::vector<L1MuRegionalCand> CSCCands = rItr->getCSCCands();
// 	l1GmtCSCCandsInBXs[rItr->getBxInEvent()] = CSCCands;
// 	for ( std::vector<L1MuRegionalCand>::const_iterator  cItr = CSCCands.begin() ; cItr != CSCCands.end() ; ++cItr )
// 	  if (!cItr->empty()) l1GmtCSCCands.push_back(*cItr);
	
// 	std::vector<L1MuRegionalCand> RPCfCands = rItr->getFwdRPCCands();
// 	for ( std::vector<L1MuRegionalCand>::const_iterator  cItr = RPCfCands.begin() ; cItr != RPCfCands.end() ; ++cItr )
// 	  if (!cItr->empty()) l1GmtRPCfCands.push_back(*cItr);
	
// 	std::vector<L1MuRegionalCand> RPCbCands = rItr->getBrlRPCCands();
// 	for ( std::vector<L1MuRegionalCand>::const_iterator  cItr = RPCbCands.begin() ; cItr != RPCbCands.end() ; ++cItr )
// 	  if (!cItr->empty()) l1GmtRPCbCands.push_back(*cItr);
	
// 	std::vector<L1MuRegionalCand> DTCands = rItr->getDTBXCands();
// 	for ( std::vector<L1MuRegionalCand>::const_iterator  cItr = DTCands.begin() ; cItr != DTCands.end() ; ++cItr )
// 	  if (!cItr->empty()) l1GmtDTCands.push_back(*cItr);
//       }
//     //std::cout<<" sizes: "<<l1GmtCands.size()<<" "<<l1GmtfCands.size()<<" "<<l1GmtCSCCands.size()<<" "<<l1GmtRPCfCands.size()<<std::endl;
//   }
  
//   // does the trigger sccale need to be defined in the beginrun or analyze method?
//   if (iSetup.get< L1MuTriggerScalesRcd >().cacheIdentifier() != muScalesCacheID_ ||
//       iSetup.get< L1MuTriggerPtScaleRcd >().cacheIdentifier() != muPtScaleCacheID_ )
//     {
//       iSetup.get< L1MuTriggerScalesRcd >().get( muScales );

//       iSetup.get< L1MuTriggerPtScaleRcd >().get( muPtScale );

//       if (ptLUT) delete ptLUT;  
//       ptLUT = new CSCTFPtLUT(ptLUTset, muScales.product(), muPtScale.product());
  
//       for(int e=0; e<2; e++) for (int s=0; s<6; s++){
//   	  if  (my_SPs[e][s]) delete my_SPs[e][s];
//   	  my_SPs[e][s] = new CSCTFSectorProcessor(e+1, s+1, CSCTFSPset, true, muScales.product(), muPtScale.product());
//   	  my_SPs[e][s]->initialize(iSetup);
//   	}
//       muScalesCacheID_  = iSetup.get< L1MuTriggerScalesRcd >().cacheIdentifier();
//       muPtScaleCacheID_ = iSetup.get< L1MuTriggerPtScaleRcd >().cacheIdentifier();
//     }

/*
  //============ RATE GMT REGIONAL ==================

  int ngmtcsc=0, ngmtcscpt10=0;
  if (debugRATE) std::cout<< "----- statring ngmt csc"<<std::endl;
  std::vector<MatchCSCMuL1::GMTREGCAND> rtGMTREGCands;
  float max_pt_2s = -1, max_pt_3s = -1, max_pt_2q = -1, max_pt_3q = -1;
  float max_pt_2s_eta = -111, max_pt_3s_eta = -111, max_pt_2q_eta = -111, max_pt_3q_eta = -111;
  float max_pt_me42_2s = -1, max_pt_me42_3s = -1, max_pt_me42_2q = -1, max_pt_me42_3q = -1;
  float max_pt_me42r_2s = -1, max_pt_me42r_3s = -1, max_pt_me42r_2q = -1, max_pt_me42r_3q = -1;

  float max_pt_2s_2s1b = -1, max_pt_2s_2s1b_eta = -111; 
  float max_pt_2s_no1a = -1;//, max_pt_2s_eta_no1a = -111;
  float max_pt_2s_1b = -1;//,   max_pt_2s_eta_1b = -111;
  float max_pt_3s_no1a = -1, max_pt_3s_eta_no1a = -111;
  float max_pt_3s_1b = -1,   max_pt_3s_eta_1b = -111;
  float max_pt_3s_1ab = -1,   max_pt_3s_eta_1ab = -111;

  float max_pt_3s_2s1b = -1,      max_pt_3s_2s1b_eta = -111;
  float max_pt_3s_2s1b_no1a = -1, max_pt_3s_2s1b_eta_no1a = -111;
  float max_pt_3s_2s123_no1a = -1, max_pt_3s_2s123_eta_no1a = -111;
  float max_pt_3s_2s13_no1a = -1, max_pt_3s_2s13_eta_no1a = -111;
  float max_pt_3s_2s1b_1b = -1,   max_pt_3s_2s1b_eta_1b = -111;
  float max_pt_3s_2s123_1b = -1, max_pt_3s_2s123_eta_1b = -111;
  float max_pt_3s_2s13_1b = -1, max_pt_3s_2s13_eta_1b = -111;

  float max_pt_3s_3s1b = -1,      max_pt_3s_3s1b_eta = -111;
  float max_pt_3s_3s1b_no1a = -1, max_pt_3s_3s1b_eta_no1a = -111;
  float max_pt_3s_3s1b_1b = -1,   max_pt_3s_3s1b_eta_1b = -111;


  float max_pt_3s_3s1ab = -1,      max_pt_3s_3s1ab_eta = -111;
  float max_pt_3s_3s1ab_no1a = -1;//, max_pt_3s_3s1ab_eta_no1a = -111;
  float max_pt_3s_3s1ab_1b = -1;//,   max_pt_3s_3s1ab_eta_1b = -111;

  MatchCSCMuL1::TFTRACK *trk__max_pt_3s_3s1b_eta = nullptr;
  //  MatchCSCMuL1::TFTRACK *trk__max_pt_3s_3s1ab_eta = nullptr;
  MatchCSCMuL1::TFTRACK *trk__max_pt_2s1b_1b = nullptr;
  const CSCCorrelatedLCTDigi * the_me1_stub = nullptr;
  CSCDetId the_me1_id;
  std::map<int,int> bx2n;
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
  for ( std::vector<L1MuRegionalCand>::const_iterator trk = l1GmtCSCCands.begin(); trk != l1GmtCSCCands.end(); trk++)
    {
      if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
  	{
  	  if (debugRATE) std::cout<<"discarding BX = "<< trk->bx() <<std::endl;
  	  continue;
  	}
      double sign_eta = ( (trk->eta_packed() & 0x20) == 0) ? 1.:-1;
      if (doSelectEtaForGMTRates_ && sign_eta<0) continue;

      bx2n[trk->bx()] += 1;

      MatchCSCMuL1::GMTREGCAND myGMTREGCand;
      myGMTREGCand.init( &*trk , muScales, muPtScale);
      myGMTREGCand.dr = 999.;

      myGMTREGCand.tfcand = nullptr;
      for (unsigned i=0; i< rtTFCands.size(); i++)
  	{
  	  if ( trk->bx()          != rtTFCands[i].l1cand->bx()         ||
  	       trk->phi_packed()  != rtTFCands[i].l1cand->phi_packed() ||
  	       trk->eta_packed()  != rtTFCands[i].l1cand->eta_packed()   ) continue;
  	  myGMTREGCand.tfcand = &(rtTFCands[i]);
  	  myGMTREGCand.ids = rtTFCands[i].ids;
  	  myGMTREGCand.nTFStubs = rtTFCands[i].nTFStubs;
  	  break;
  	}
      rtGMTREGCands.push_back(myGMTREGCand);

      float geta = fabs(myGMTREGCand.eta);
      float gpt = myGMTREGCand.pt;

      bool eta_me42 = mugeo::isME42EtaRegion(myGMTREGCand.eta);
      bool eta_me42r = mugeo::isME42RPCEtaRegion(myGMTREGCand.eta);
      //if (geta>=1.2 && geta<=1.8) eta_me42 = 1;
      bool eta_q = (geta > 1.2);

      bool has_me1_stub = false;
      size_t n_stubs = 0;

      if (myGMTREGCand.tfcand != nullptr)
  	{
  	  //rtGMTREGCands.push_back(myGMTREGCand);

  	  if (myGMTREGCand.tfcand->tftrack != nullptr)
  	    {
  	      has_me1_stub = myGMTREGCand.tfcand->tftrack->hasStub(1);
  	    }

  	  bool has_1b_stub = false;
  	  for (auto& id: myGMTREGCand.ids) if (id.iChamberType() == 2) {
  	      has_1b_stub = true;
  	      continue;
  	    }

  	  bool has_1a_stub = false;
  	  for (auto& id: myGMTREGCand.ids) if (id.iChamberType() == 1) {
  	      has_1a_stub = true;
  	      continue;
  	    }

  	  bool eta_me1b = mugeo::isME1bEtaRegion(myGMTREGCand.eta);
  	  bool eta_me1ab = mugeo::isME1abEtaRegion(myGMTREGCand.eta);
  	  bool eta_me1a = mugeo::isME1aEtaRegion(myGMTREGCand.eta);
  	  bool eta_me1b_whole = mugeo::isME1bEtaRegion(myGMTREGCand.eta, 1.6, 2.14);
  	  bool eta_no1a = (geta >= 1.2 && geta < 2.14);
	  
  	  n_stubs = myGMTREGCand.nTFStubs;
  	  size_t n_stubs_id = myGMTREGCand.ids.size();
  	  //if (n_stubs == n_stubs_id) std::cout<<"n_stubs good"<<std::endl;
  	  if (n_stubs != n_stubs_id) std::cout<<"n_stubs bad: "<<eta_q<<" "<<n_stubs<<" != "<<n_stubs_id<<" "<< geta  <<std::endl;
	  
  	  auto stub_ids = myGMTREGCand.tfcand->tftrack->trgids;
  	  for (size_t i=0; i<stub_ids.size(); ++i)
  	    {
  	      // pick up the ME11 stub of this track
  	      if ( !(stub_ids[i].station() == 1 && (stub_ids[i].ring() == 1 || stub_ids[i].ring() == 4) ) ) continue;
  	      the_me1_stub = (myGMTREGCand.tfcand->tftrack->trgdigis)[i];
  	      the_me1_id = stub_ids[i];
  	    }

  	  int tf_mode = myGMTREGCand.tfcand->tftrack->mode();
  	  bool ok_2s123 = (tf_mode != 0xd); // excludes ME1-ME4 stub tf tracks
  	  bool ok_2s13 = (ok_2s123 && (tf_mode != 0x6)); // excludes ME1-ME2 and ME1-ME4 stub tf tracks

  	  if (n_stubs >= 2)
  	    {
  	      h_rt_gmt_csc_pt_2st->Fill(gpt);
  	      if (eta_me42) h_rt_gmt_csc_pt_2s42->Fill(gpt);
  	      if (eta_me42r) h_rt_gmt_csc_pt_2s42r->Fill(gpt);
  	      if (            gpt > max_pt_2s     ) { max_pt_2s = gpt; max_pt_2s_eta = geta; }
  	      if (eta_me1b && gpt > max_pt_2s_1b  ) { max_pt_2s_1b = gpt; max_pt_2s_eta_1b = geta; }
  	      if (eta_no1a && gpt > max_pt_2s_no1a) { max_pt_2s_no1a = gpt; max_pt_2s_eta_no1a = geta; }
  	      if (eta_me42 && gpt > max_pt_me42_2s) max_pt_me42_2s = gpt;
  	      if (eta_me42r && gpt>max_pt_me42r_2s) max_pt_me42r_2s = gpt;
  	    }
  	  if ( (has_1b_stub && n_stubs >=2) || ( !has_1b_stub && !eta_me1b_whole && n_stubs >=2 ) )
  	    {
  	      if (            gpt > max_pt_2s_2s1b) { max_pt_2s_2s1b = gpt; max_pt_2s_2s1b_eta = geta; }
  	    }

  	  if (n_stubs >= 3)
  	    {
  	      h_rt_gmt_csc_pt_3st->Fill(gpt);
  	      if (eta_me42) h_rt_gmt_csc_pt_3s42->Fill(gpt);
  	      if (eta_me42r) h_rt_gmt_csc_pt_3s42r->Fill(gpt);
  	      if (            gpt > max_pt_3s     ) { max_pt_3s = gpt; max_pt_3s_eta = geta; }


  	      if (eta_me1b && gpt > max_pt_3s_1b  ) { max_pt_3s_1b = gpt; max_pt_3s_eta_1b = geta; }
  	      if (eta_me1ab && gpt > max_pt_3s_1ab  ) { max_pt_3s_1ab = gpt; max_pt_3s_eta_1ab = geta; }

  	      if (eta_no1a && gpt > max_pt_3s_no1a) { max_pt_3s_no1a = gpt; max_pt_3s_eta_no1a = geta; }
  	      if (eta_me42 && gpt > max_pt_me42_3s) max_pt_me42_3s = gpt;
  	      if (eta_me42r && gpt>max_pt_me42r_3s) max_pt_me42r_3s = gpt;
  	    }

  	  if ( (has_1b_stub && n_stubs >=2) || ( !has_1b_stub && !eta_me1b_whole && n_stubs >=3 ) )
  	    {
  	      if (            gpt > max_pt_3s_2s1b     ) { max_pt_3s_2s1b = gpt; max_pt_3s_2s1b_eta = geta; }

  	      if (eta_me1b && gpt > max_pt_3s_2s1b_1b  ) { max_pt_3s_2s1b_1b = gpt; max_pt_3s_2s1b_eta_1b = geta; 
  		trk__max_pt_2s1b_1b = myGMTREGCand.tfcand->tftrack; }
  	      if (eta_me1b && gpt > max_pt_3s_2s123_1b && ok_2s123 ) 
  		{ max_pt_3s_2s123_1b = gpt; max_pt_3s_2s123_eta_1b = geta; }
  	      if (eta_me1b && gpt > max_pt_3s_2s13_1b && ok_2s13 ) 
  		{ max_pt_3s_2s13_1b = gpt; max_pt_3s_2s13_eta_1b = geta; }

  	      if (eta_no1a && gpt > max_pt_3s_2s1b_no1a) { max_pt_3s_2s1b_no1a = gpt; max_pt_3s_2s1b_eta_no1a = geta; }
  	      if (eta_no1a && gpt > max_pt_3s_2s123_no1a && (!eta_me1b || (eta_me1b && ok_2s123) ) )
  		{ max_pt_3s_2s123_no1a = gpt; max_pt_3s_2s123_eta_no1a = geta; }
  	      if (eta_no1a && gpt > max_pt_3s_2s13_no1a && (!eta_me1b || (eta_me1b && ok_2s13) ) )
  		{ max_pt_3s_2s13_no1a = gpt; max_pt_3s_2s13_eta_no1a = geta; }
  	    }

  	  if ( (has_1b_stub && n_stubs >=3) || ( !has_1b_stub && !eta_me1b_whole && n_stubs >=3 ) )
  	    {
  	      if (            gpt > max_pt_3s_3s1b      ) { max_pt_3s_3s1b = gpt; max_pt_3s_3s1b_eta = geta;
  		trk__max_pt_3s_3s1b_eta = myGMTREGCand.tfcand->tftrack; }
  	      if (eta_me1b && gpt > max_pt_3s_3s1b_1b   ) { max_pt_3s_3s1b_1b = gpt; max_pt_3s_3s1b_eta_1b = geta; }
  	      if (eta_no1a && gpt > max_pt_3s_3s1b_no1a ) { max_pt_3s_3s1b_no1a = gpt; max_pt_3s_3s1b_eta_no1a = geta; }
  	    }

  	  if (n_stubs >=3 && ( (eta_me1a && has_1a_stub) || (eta_me1b && has_1b_stub) || (!has_1a_stub && !has_1b_stub && !eta_me1ab) ) )
  	    {
  	      if (            gpt > max_pt_3s_3s1ab      ) { max_pt_3s_3s1ab = gpt; max_pt_3s_3s1ab_eta = geta;
  		//trk__max_pt_3s_3s1ab_eta = myGMTREGCand.tfcand->tftrack; 
	      }
  	      if (eta_me1b && gpt > max_pt_3s_3s1ab_1b   ) { max_pt_3s_3s1ab_1b = gpt; 
		//max_pt_3s_3s1ab_eta_1b = geta; 
	      }
  	      if (eta_no1a && gpt > max_pt_3s_3s1ab_no1a ) { max_pt_3s_3s1ab_no1a = gpt; 
		//max_pt_3s_3s1ab_eta_no1a = geta; 
	      }
  	    }


  	} else { 
  	std::cout<<"GMTCSC match not found pt="<<gpt<<" eta="<<myGMTREGCand.eta<<"  packed: "<<trk->phi_packed()<<" "<<trk->eta_packed()<<std::endl;
  	for (unsigned i=0; i< rtTFCands.size(); i++) std::cout<<"    "<<rtTFCands[i].l1cand->phi_packed()<<" "<<rtTFCands[i].l1cand->eta_packed();
  	std::cout<<std::endl;
  	std::cout<<"  all tfcands:";
  	for ( std::vector< L1MuRegionalCand >::const_iterator ctrk = l1TfCands->begin(); ctrk != l1TfCands->end(); ctrk++)
  	  if (!( ctrk->bx() < minRateBX_ || ctrk->bx() > maxRateBX_ )) std::cout<<"    "<<ctrk->phi_packed()<<" "<<ctrk->eta_packed();
  	std::cout<<std::endl;
      }
    
      if (trk->quality()>=2) {
  	h_rt_gmt_csc_pt_2q->Fill(gpt);
  	if (eta_me42) h_rt_gmt_csc_pt_2q42->Fill(gpt);
  	if (eta_me42r) h_rt_gmt_csc_pt_2q42r->Fill(gpt);
  	if (gpt > max_pt_2q) {max_pt_2q = gpt; max_pt_2q_eta = geta;}
  	if (eta_me42 && gpt > max_pt_me42_2q) max_pt_me42_2q = gpt;
  	if (eta_me42r && gpt > max_pt_me42r_2q) max_pt_me42r_2q = gpt;
      }
      if ((!eta_q && trk->quality()>=2) || ( eta_q && trk->quality()>=3) ) {
  	h_rt_gmt_csc_pt_3q->Fill(gpt);
  	if (eta_me42) h_rt_gmt_csc_pt_3q42->Fill(gpt);
  	if (eta_me42r) h_rt_gmt_csc_pt_3q42r->Fill(gpt);
  	if (gpt > max_pt_3q) {max_pt_3q = gpt; max_pt_3q_eta = geta;}
  	if (eta_me42 && gpt > max_pt_me42_3q) max_pt_me42_3q = gpt;
  	if (eta_me42r && gpt > max_pt_me42r_3q) max_pt_me42r_3q = gpt;
      }
    
      //if (trk->quality()>=3 && !(myGMTREGCand.ids.size()>=3) ) {
      //  std::cout<<"weird stubs number "<<myGMTREGCand.ids.size()<<" for q="<<trk->quality()<<std::endl;
      //  if (myGMTREGCand.tfcand->tftrack != nullptr) myGMTREGCand.tfcand->tftrack->print("");
      //  else std::cout<<"null tftrack!"<<std::endl;
      //}

      //    if (trk->quality()>=3 && gpt >=40. && mugeo::isME1bEtaRegion(myGMTREGCand.eta) ) {
      //      std::cout<<"highpt csctf in ME1b "<<std::endl;
      //      myGMTREGCand.tfcand->tftrack->print("");
      //    }
      if (has_me1_stub && n_stubs > 2 && gpt >= 30. && geta> 1.6 && geta < 2.15 ) {
  	std::cout<<"highpt csctf in ME1b "<<std::endl;
  	myGMTREGCand.tfcand->tftrack->print("");
      }


      ngmtcsc++;
      if (gpt>=10.) ngmtcscpt10++;
      h_rt_gmt_csc_pt->Fill(gpt);
      h_rt_gmt_csc_eta->Fill(geta);
      h_rt_gmt_csc_bx->Fill(trk->bx());
  
      h_rt_gmt_csc_q->Fill(trk->quality());
      if (eta_me42) h_rt_gmt_csc_q_42->Fill(trk->quality());
      if (eta_me42r) h_rt_gmt_csc_q_42r->Fill(trk->quality());
    }

  h_rt_ngmt_csc->Fill(ngmtcsc);
  h_rt_ngmt_csc_pt10->Fill(ngmtcscpt10);
  if (max_pt_2s>0) h_rt_gmt_csc_ptmax_2s->Fill(max_pt_2s);
  if (max_pt_3s>0) h_rt_gmt_csc_ptmax_3s->Fill(max_pt_3s);

  if (max_pt_2s_1b>0) h_rt_gmt_csc_ptmax_2s_1b->Fill(max_pt_2s_1b);
  if (max_pt_2s_no1a>0) h_rt_gmt_csc_ptmax_2s_no1a->Fill(max_pt_2s_no1a);
  if (max_pt_3s_1b>0) h_rt_gmt_csc_ptmax_3s_1b->Fill(max_pt_3s_1b);
  if (max_pt_3s_no1a>0) h_rt_gmt_csc_ptmax_3s_no1a->Fill(max_pt_3s_no1a);
  if (max_pt_3s_2s1b>0) h_rt_gmt_csc_ptmax_3s_2s1b->Fill(max_pt_3s_2s1b);
  if (max_pt_3s_2s1b_1b>0) h_rt_gmt_csc_ptmax_3s_2s1b_1b->Fill(max_pt_3s_2s1b_1b);
  if (max_pt_3s_2s123_1b>0) h_rt_gmt_csc_ptmax_3s_2s123_1b->Fill(max_pt_3s_2s123_1b);
  if (max_pt_3s_2s13_1b>0) h_rt_gmt_csc_ptmax_3s_2s13_1b->Fill(max_pt_3s_2s13_1b);
  if (max_pt_3s_2s1b_no1a>0) h_rt_gmt_csc_ptmax_3s_2s1b_no1a->Fill(max_pt_3s_2s1b_no1a);
  if (max_pt_3s_2s123_no1a>0) h_rt_gmt_csc_ptmax_3s_2s123_no1a->Fill(max_pt_3s_2s123_no1a);
  if (max_pt_3s_2s13_no1a>0) h_rt_gmt_csc_ptmax_3s_2s13_no1a->Fill(max_pt_3s_2s13_no1a);
  if (max_pt_3s_3s1b>0) h_rt_gmt_csc_ptmax_3s_3s1b->Fill(max_pt_3s_3s1b);
  if (max_pt_3s_3s1b_1b>0) h_rt_gmt_csc_ptmax_3s_3s1b_1b->Fill(max_pt_3s_3s1b_1b);
  if (max_pt_3s_3s1b_no1a>0) h_rt_gmt_csc_ptmax_3s_3s1b_no1a->Fill(max_pt_3s_3s1b_no1a);

  if (max_pt_2q>0) h_rt_gmt_csc_ptmax_2q->Fill(max_pt_2q);
  if (max_pt_3q>0) h_rt_gmt_csc_ptmax_3q->Fill(max_pt_3q);

  if (max_pt_2s>=10.) h_rt_gmt_csc_ptmax10_eta_2s->Fill(max_pt_2s_eta);
  if (max_pt_2s_2s1b>=10.) h_rt_gmt_csc_ptmax10_eta_2s_2s1b->Fill(max_pt_2s_2s1b_eta);
  if (max_pt_3s>=10.) h_rt_gmt_csc_ptmax10_eta_3s->Fill(max_pt_3s_eta);
  if (max_pt_3s_1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_1b->Fill(max_pt_3s_eta_1b);
  if (max_pt_3s_no1a>=10.) h_rt_gmt_csc_ptmax10_eta_3s_no1a->Fill(max_pt_3s_eta_no1a);
  if (max_pt_3s_2s1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s1b->Fill(max_pt_3s_2s1b_eta);
  if (max_pt_3s_2s1b_1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s1b_1b->Fill(max_pt_3s_2s1b_eta_1b);
  if (max_pt_3s_2s123_1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s123_1b->Fill(max_pt_3s_2s123_eta_1b);
  if (max_pt_3s_2s13_1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s13_1b->Fill(max_pt_3s_2s13_eta_1b);
  if (max_pt_3s_2s1b_no1a>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s1b_no1a->Fill(max_pt_3s_2s1b_eta_no1a);
  if (max_pt_3s_2s123_no1a>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s123_no1a->Fill(max_pt_3s_2s123_eta_no1a);
  if (max_pt_3s_2s13_no1a>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s13_no1a->Fill(max_pt_3s_2s13_eta_no1a);
  if (max_pt_3s_3s1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_3s1b->Fill(max_pt_3s_3s1b_eta);
  if (max_pt_3s_3s1b_1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_3s1b_1b->Fill(max_pt_3s_3s1b_eta_1b);
  if (max_pt_3s_3s1b_no1a>=10.) h_rt_gmt_csc_ptmax10_eta_3s_3s1b_no1a->Fill(max_pt_3s_3s1b_eta_no1a);
  if (max_pt_2q>=10.) h_rt_gmt_csc_ptmax10_eta_2q->Fill(max_pt_2q_eta);
  if (max_pt_3q>=10.) h_rt_gmt_csc_ptmax10_eta_3q->Fill(max_pt_3q_eta);

  if (max_pt_2s>=20.) h_rt_gmt_csc_ptmax20_eta_2s->Fill(max_pt_2s_eta);
  if (max_pt_2s_2s1b>=20.) h_rt_gmt_csc_ptmax20_eta_2s_2s1b->Fill(max_pt_2s_2s1b_eta);
  if (max_pt_3s>=20.) h_rt_gmt_csc_ptmax20_eta_3s->Fill(max_pt_3s_eta);

  if (max_pt_3s_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_1b->Fill(max_pt_3s_eta_1b);
  if (max_pt_3s_1ab>=20.) h_rt_gmt_csc_ptmax20_eta_3s_1ab->Fill(max_pt_3s_eta_1ab);

  if (max_pt_3s_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_no1a->Fill(max_pt_3s_eta_no1a);
  if (max_pt_3s_2s1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s1b->Fill(max_pt_3s_2s1b_eta);
  if (max_pt_3s_2s1b_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s1b_1b->Fill(max_pt_3s_2s1b_eta_1b);
  if (max_pt_3s_2s123_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s123_1b->Fill(max_pt_3s_2s123_eta_1b);
  if (max_pt_3s_2s13_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s13_1b->Fill(max_pt_3s_2s13_eta_1b);
  if (max_pt_3s_2s1b_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s1b_no1a->Fill(max_pt_3s_2s1b_eta_no1a);
  if (max_pt_3s_2s123_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s123_no1a->Fill(max_pt_3s_2s123_eta_no1a);
  if (max_pt_3s_2s13_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s13_no1a->Fill(max_pt_3s_2s13_eta_no1a);

  if (max_pt_3s_3s1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_3s1b->Fill(max_pt_3s_3s1b_eta);
  if (max_pt_3s_3s1ab>=20.) h_rt_gmt_csc_ptmax20_eta_3s_3s1ab->Fill(max_pt_3s_3s1b_eta);

  if (max_pt_3s_3s1b_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_3s1b_1b->Fill(max_pt_3s_3s1b_eta_1b);
  if (max_pt_3s_3s1b_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_3s1b_no1a->Fill(max_pt_3s_3s1b_eta_no1a);
  if (max_pt_2q>=20.) h_rt_gmt_csc_ptmax20_eta_2q->Fill(max_pt_2q_eta);
  if (max_pt_3q>=20.) h_rt_gmt_csc_ptmax20_eta_3q->Fill(max_pt_3q_eta);

  if (max_pt_2s>=30.) h_rt_gmt_csc_ptmax30_eta_2s->Fill(max_pt_2s_eta);
  if (max_pt_2s_2s1b>=30.) h_rt_gmt_csc_ptmax30_eta_2s_2s1b->Fill(max_pt_2s_2s1b_eta);
  if (max_pt_3s>=30.) h_rt_gmt_csc_ptmax30_eta_3s->Fill(max_pt_3s_eta);

  if (max_pt_3s_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_1b->Fill(max_pt_3s_eta_1b);
  if (max_pt_3s_1ab>=30.) h_rt_gmt_csc_ptmax30_eta_3s_1ab->Fill(max_pt_3s_eta_1ab);


  if (max_pt_3s_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_no1a->Fill(max_pt_3s_eta_no1a);
  if (max_pt_3s_2s1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s1b->Fill(max_pt_3s_2s1b_eta);
  if (max_pt_3s_2s1b_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s1b_1b->Fill(max_pt_3s_2s1b_eta_1b);
  if (max_pt_3s_2s123_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s123_1b->Fill(max_pt_3s_2s123_eta_1b);
  if (max_pt_3s_2s13_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s13_1b->Fill(max_pt_3s_2s13_eta_1b);
  if (max_pt_3s_2s1b_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s1b_no1a->Fill(max_pt_3s_2s1b_eta_no1a);
  if (max_pt_3s_2s123_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s123_no1a->Fill(max_pt_3s_2s123_eta_no1a);
  if (max_pt_3s_2s13_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s13_no1a->Fill(max_pt_3s_2s13_eta_no1a);


  if (max_pt_3s_3s1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_3s1b->Fill(max_pt_3s_3s1b_eta);
  if (max_pt_3s_3s1ab>=30.) h_rt_gmt_csc_ptmax30_eta_3s_3s1ab->Fill(max_pt_3s_3s1ab_eta);



  if (max_pt_3s_3s1b_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_3s1b_1b->Fill(max_pt_3s_3s1b_eta_1b);
  if (max_pt_3s_3s1b_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_3s1b_no1a->Fill(max_pt_3s_3s1b_eta_no1a);

  if (max_pt_2q>=30.) h_rt_gmt_csc_ptmax30_eta_2q->Fill(max_pt_2q_eta);
  if (max_pt_3q>=30.) h_rt_gmt_csc_ptmax30_eta_3q->Fill(max_pt_3q_eta);

  if (max_pt_me42_2s>0) h_rt_gmt_csc_ptmax_2s42->Fill(max_pt_me42_2s);
  if (max_pt_me42_3s>0) h_rt_gmt_csc_ptmax_3s42->Fill(max_pt_me42_3s);
  if (max_pt_me42_2q>0) h_rt_gmt_csc_ptmax_2q42->Fill(max_pt_me42_2q);
  if (max_pt_me42_3q>0) h_rt_gmt_csc_ptmax_3q42->Fill(max_pt_me42_3q);
  if (max_pt_me42r_2s>0) h_rt_gmt_csc_ptmax_2s42r->Fill(max_pt_me42r_2s);
  if (max_pt_me42r_3s>0) h_rt_gmt_csc_ptmax_3s42r->Fill(max_pt_me42r_3s);
  if (max_pt_me42r_2q>0) h_rt_gmt_csc_ptmax_2q42r->Fill(max_pt_me42r_2q);
  if (max_pt_me42r_3q>0) h_rt_gmt_csc_ptmax_3q42r->Fill(max_pt_me42r_3q);
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) h_rt_ngmt_csc_per_bx->Fill(bx2n[bx]);
  if (debugRATE) std::cout<< "----- end ngmt csc/ngmtpt10="<<ngmtcsc<<"/"<<ngmtcscpt10<<std::endl;

  if (max_pt_3s_3s1b>=30.) 
    {
      std::cout<<"filled h_rt_gmt_csc_ptmax30_eta_3s_3s1b eta "<<max_pt_3s_3s1b_eta<<std::endl;
      if (trk__max_pt_3s_3s1b_eta) trk__max_pt_3s_3s1b_eta->print("");
    }

  if (max_pt_3s_2s1b_1b >= 10. && trk__max_pt_2s1b_1b)
    {
      const int Nthr = 6;
      float tfc_pt_thr[Nthr] = {10., 15., 20., 25., 30., 40.};
      for (int i=0; i<Nthr; ++i) if (max_pt_3s_2s1b_1b >= tfc_pt_thr[i])
  				   {
  				     h_rt_gmt_csc_mode_2s1b_1b[i]->Fill(trk__max_pt_2s1b_1b->mode());
  				   }
      if (the_me1_stub) std::cout<<"DBGMODE "<<the_me1_id.endcap()<<" "<<the_me1_id.chamber()<<" "<<trk__max_pt_2s1b_1b->pt<<" "<<trk__max_pt_2s1b_1b->mode()<<" "<<pbend[the_me1_stub->getPattern()] <<" "<<the_me1_stub->getGEMDPhi()<<std::endl;
    }

  int ngmtrpcf=0, ngmtrpcfpt10=0;
  if (debugRATE) std::cout<< "----- statring ngmt rpcf"<<std::endl;
  std::vector<MatchCSCMuL1::GMTREGCAND> rtGMTRPCfCands;
  float max_pt_me42 = -1, max_pt = -1, max_pt_eta = -111;
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
  for ( std::vector<L1MuRegionalCand>::const_iterator trk = l1GmtRPCfCands.begin(); trk != l1GmtRPCfCands.end(); trk++)
    {
      if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
  	{
  	  if (debugRATE) std::cout<<"discarding BX = "<< trk->bx() <<std::endl;
  	  continue;
  	}
      double sign_eta = ( (trk->eta_packed() & 0x20) == 0) ? 1.:-1;
      if (doSelectEtaForGMTRates_ && sign_eta<0) continue;

      bx2n[trk->bx()] += 1;
      MatchCSCMuL1::GMTREGCAND myGMTREGCand;

      myGMTREGCand.init( &*trk , muScales, muPtScale);
      myGMTREGCand.dr = 999.;

      myGMTREGCand.tfcand = nullptr;
      rtGMTRPCfCands.push_back(myGMTREGCand);

      ngmtrpcf++;
      if (myGMTREGCand.pt>=10.) ngmtrpcfpt10++;
      h_rt_gmt_rpcf_pt->Fill(myGMTREGCand.pt);
      h_rt_gmt_rpcf_eta->Fill(fabs(myGMTREGCand.eta));
      h_rt_gmt_rpcf_bx->Fill(trk->bx());

      bool eta_me42 = mugeo::isME42RPCEtaRegion(myGMTREGCand.eta);
      //if (fabs(myGMTREGCand.eta)>=1.2 && fabs(myGMTREGCand.eta)<=1.8) eta_me42 = 1;

      if(eta_me42) h_rt_gmt_rpcf_pt_42->Fill(myGMTREGCand.pt);
      if(eta_me42 && myGMTREGCand.pt > max_pt_me42) max_pt_me42 = myGMTREGCand.pt;
      if(myGMTREGCand.pt > max_pt) { max_pt = myGMTREGCand.pt;  max_pt_eta = fabs(myGMTREGCand.eta);}
    
      h_rt_gmt_rpcf_q->Fill(trk->quality());
      if (eta_me42) h_rt_gmt_rpcf_q_42->Fill(trk->quality());
    }
  h_rt_ngmt_rpcf->Fill(ngmtrpcf);
  h_rt_ngmt_rpcf_pt10->Fill(ngmtrpcfpt10);
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) h_rt_ngmt_rpcf_per_bx->Fill(bx2n[bx]);
  if (max_pt>0) h_rt_gmt_rpcf_ptmax->Fill(max_pt);
  if (max_pt>=10.) h_rt_gmt_rpcf_ptmax10_eta->Fill(max_pt_eta);
  if (max_pt>=20.) h_rt_gmt_rpcf_ptmax20_eta->Fill(max_pt_eta);
  if (max_pt_me42>0) h_rt_gmt_rpcf_ptmax_42->Fill(max_pt_me42);
  if (debugRATE) std::cout<< "----- end ngmt rpcf/ngmtpt10="<<ngmtrpcf<<"/"<<ngmtrpcfpt10<<std::endl;


  int ngmtrpcb=0, ngmtrpcbpt10=0;
  if (debugRATE) std::cout<< "----- statring ngmt rpcb"<<std::endl;
  std::vector<MatchCSCMuL1::GMTREGCAND> rtGMTRPCbCands;
  max_pt = -1, max_pt_eta = -111;
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
  for ( std::vector<L1MuRegionalCand>::const_iterator trk = l1GmtRPCbCands.begin(); trk != l1GmtRPCbCands.end(); trk++)
    {
      if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
  	{
  	  if (debugRATE) std::cout<<"discarding BX = "<< trk->bx() <<std::endl;
  	  continue;
  	}
      double sign_eta = ( (trk->eta_packed() & 0x20) == 0) ? 1.:-1;
      if (doSelectEtaForGMTRates_ && sign_eta<0) continue;

      bx2n[trk->bx()] += 1;
      MatchCSCMuL1::GMTREGCAND myGMTREGCand;

      myGMTREGCand.init( &*trk , muScales, muPtScale);
      myGMTREGCand.dr = 999.;

      myGMTREGCand.tfcand = nullptr;
      rtGMTRPCbCands.push_back(myGMTREGCand);

      ngmtrpcb++;
      if (myGMTREGCand.pt>=10.) ngmtrpcbpt10++;
      h_rt_gmt_rpcb_pt->Fill(myGMTREGCand.pt);
      h_rt_gmt_rpcb_eta->Fill(fabs(myGMTREGCand.eta));
      h_rt_gmt_rpcb_bx->Fill(trk->bx());

      if(myGMTREGCand.pt > max_pt) { max_pt = myGMTREGCand.pt;  max_pt_eta = fabs(myGMTREGCand.eta);}

      h_rt_gmt_rpcb_q->Fill(trk->quality());
    }
  h_rt_ngmt_rpcb->Fill(ngmtrpcb);
  h_rt_ngmt_rpcb_pt10->Fill(ngmtrpcbpt10);
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) h_rt_ngmt_rpcb_per_bx->Fill(bx2n[bx]);
  if (max_pt>0) h_rt_gmt_rpcb_ptmax->Fill(max_pt);
  if (max_pt>=10.) h_rt_gmt_rpcb_ptmax10_eta->Fill(max_pt_eta);
  if (max_pt>=20.) h_rt_gmt_rpcb_ptmax20_eta->Fill(max_pt_eta);
  if (debugRATE) std::cout<< "----- end ngmt rpcb/ngmtpt10="<<ngmtrpcb<<"/"<<ngmtrpcbpt10<<std::endl;


  int ngmtdt=0, ngmtdtpt10=0;
  if (debugRATE) std::cout<< "----- statring ngmt dt"<<std::endl;
  std::vector<MatchCSCMuL1::GMTREGCAND> rtGMTDTCands;
  max_pt = -1, max_pt_eta = -111;
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
  for ( std::vector<L1MuRegionalCand>::const_iterator trk = l1GmtDTCands.begin(); trk != l1GmtDTCands.end(); trk++)
    {
      if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
  	{
  	  if (debugRATE) std::cout<<"discarding BX = "<< trk->bx() <<std::endl;
  	  continue;
  	}
      double sign_eta = ( (trk->eta_packed() & 0x20) == 0) ? 1.:-1;
      if (doSelectEtaForGMTRates_ && sign_eta<0) continue;

      bx2n[trk->bx()] += 1;
      MatchCSCMuL1::GMTREGCAND myGMTREGCand;

      myGMTREGCand.init( &*trk , muScales, muPtScale);
      myGMTREGCand.dr = 999.;

      myGMTREGCand.tfcand = nullptr;
      rtGMTDTCands.push_back(myGMTREGCand);

      ngmtdt++;
      if (myGMTREGCand.pt>=10.) ngmtdtpt10++;
      h_rt_gmt_dt_pt->Fill(myGMTREGCand.pt);
      h_rt_gmt_dt_eta->Fill(fabs(myGMTREGCand.eta));
      h_rt_gmt_dt_bx->Fill(trk->bx());

      if(myGMTREGCand.pt > max_pt) { max_pt = myGMTREGCand.pt;  max_pt_eta = fabs(myGMTREGCand.eta);}

      h_rt_gmt_dt_q->Fill(trk->quality());
    }
  h_rt_ngmt_dt->Fill(ngmtdt);
  h_rt_ngmt_dt_pt10->Fill(ngmtdtpt10);
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) h_rt_ngmt_dt_per_bx->Fill(bx2n[bx]);
  if (max_pt>0) h_rt_gmt_dt_ptmax->Fill(max_pt);
  if (max_pt>=10.) h_rt_gmt_dt_ptmax10_eta->Fill(max_pt_eta);
  if (max_pt>=20.) h_rt_gmt_dt_ptmax20_eta->Fill(max_pt_eta);
  if (debugRATE) std::cout<< "----- end ngmt dt/ngmtpt10="<<ngmtdt<<"/"<<ngmtdtpt10<<std::endl;


  //============ RATE GMT ==================

  int ngmt=0;
  if (debugRATE) std::cout<< "----- statring ngmt"<<std::endl;
  std::vector<MatchCSCMuL1::GMTCAND> rtGMTCands;
  max_pt_me42_2s = -1; max_pt_me42_3s = -1;  max_pt_me42_2q = -1; max_pt_me42_3q = -1;
  max_pt_me42r_2s = -1; max_pt_me42r_3s = -1;  max_pt_me42r_2q = -1; max_pt_me42r_3q = -1;
  float max_pt_me42_2s_sing = -1, max_pt_me42_3s_sing = -1, max_pt_me42_2q_sing = -1, max_pt_me42_3q_sing = -1;
  float max_pt_me42r_2s_sing = -1, max_pt_me42r_3s_sing = -1, max_pt_me42r_2q_sing = -1, max_pt_me42r_3q_sing = -1;
  max_pt = -1, max_pt_eta = -999;

  float max_pt_sing = -1, max_pt_eta_sing = -999, max_pt_sing_3s = -1, max_pt_eta_sing_3s = -999;
  float max_pt_sing_csc = -1., max_pt_eta_sing_csc = -999.;
  float max_pt_sing_dtcsc = -1., max_pt_eta_sing_dtcsc = -999.;
  float max_pt_sing_1b = -1.;//, max_pt_eta_sing_1b = -999;
  float max_pt_sing_no1a = -1.;//, max_pt_eta_sing_no1a = -999.;

  float max_pt_sing6 = -1, max_pt_eta_sing6 = -999, max_pt_sing6_3s = -1, max_pt_eta_sing6_3s = -999;
  float max_pt_sing6_csc = -1., max_pt_eta_sing6_csc = -999.;
  float max_pt_sing6_1b = -1.;//, max_pt_eta_sing6_1b = -999;
  float max_pt_sing6_no1a = -1.;//, max_pt_eta_sing6_no1a = -999.;
  float max_pt_sing6_3s1b_no1a = -1.;//, max_pt_eta_sing6_3s1b_no1a = -999.;

  float max_pt_dbl = -1, max_pt_eta_dbl = -999;

  std::vector<L1MuGMTReadoutRecord> gmt_records = hl1GmtCands->getRecords();
  for ( std::vector< L1MuGMTReadoutRecord >::const_iterator rItr=gmt_records.begin(); rItr!=gmt_records.end() ; ++rItr )
  {
    if (rItr->getBxInEvent() < minBxGMT_ || rItr->getBxInEvent() > maxBxGMT_) continue;
    
    std::vector<L1MuRegionalCand> CSCCands = rItr->getCSCCands();
    std::vector<L1MuRegionalCand> DTCands  = rItr->getDTBXCands();
    std::vector<L1MuRegionalCand> RPCfCands = rItr->getFwdRPCCands();
    std::vector<L1MuRegionalCand> RPCbCands = rItr->getBrlRPCCands();
    std::vector<L1MuGMTExtendedCand> GMTCands = rItr->getGMTCands();
    for ( std::vector<L1MuGMTExtendedCand>::const_iterator  muItr = GMTCands.begin() ; muItr != GMTCands.end() ; ++muItr )
    {
      if( muItr->empty() ) continue;
      
      if ( muItr->bx() < minRateBX_ || muItr->bx() > maxRateBX_ )
      {
	if (debugRATE) std::cout<<"discarding BX = "<< muItr->bx() <<std::endl;
	continue;
      }
      
      MatchCSCMuL1::GMTCAND myGMTCand;
      myGMTCand.init( &*muItr , muScales, muPtScale);
      myGMTCand.dr = 999.;
      if (doSelectEtaForGMTRates_ && myGMTCand.eta<0) continue;
      
      myGMTCand.regcand = nullptr;
      myGMTCand.regcand_rpc = nullptr;
      
      float gpt = myGMTCand.pt;
      float geta = fabs(myGMTCand.eta);
      
      MatchCSCMuL1::GMTREGCAND * gmt_csc = nullptr;
      if (muItr->isFwd() && ( muItr->isMatchedCand() || !muItr->isRPC())) 
      {
	L1MuRegionalCand rcsc = CSCCands[muItr->getDTCSCIndex()];
	unsigned my_i = 999;
	for (unsigned i=0; i< rtGMTREGCands.size(); i++)
	{
	  if (rcsc.getDataWord()!=rtGMTREGCands[i].l1reg->getDataWord()) continue;
	  my_i = i;
	  break;
	}
	if (my_i<99) gmt_csc = &rtGMTREGCands[my_i];
	else std::cout<<"DOES NOT EXIST IN rtGMTREGCands! Should not happen!"<<std::endl;
	myGMTCand.regcand = gmt_csc;
	myGMTCand.ids = gmt_csc->ids;
      }
      
      MatchCSCMuL1::GMTREGCAND * gmt_rpcf = nullptr;
      if (muItr->isFwd() && (muItr->isMatchedCand() || muItr->isRPC())) 
      {
	L1MuRegionalCand rrpcf = RPCfCands[muItr->getRPCIndex()];
	unsigned my_i = 999;
	for (unsigned i=0; i< rtGMTRPCfCands.size(); i++)
	{
	  if (rrpcf.getDataWord()!=rtGMTRPCfCands[i].l1reg->getDataWord()) continue;
	  my_i = i;
	  break;
	}
	if (my_i<99) gmt_rpcf = &rtGMTRPCfCands[my_i];
	else std::cout<<"DOES NOT EXIST IN rtGMTRPCfCands! Should not happen!"<<std::endl;
	myGMTCand.regcand_rpc = gmt_rpcf;
      }
      
      MatchCSCMuL1::GMTREGCAND * gmt_rpcb = nullptr;
      if (!(muItr->isFwd()) && (muItr->isMatchedCand() || muItr->isRPC()))
      {
	L1MuRegionalCand rrpcb = RPCbCands[muItr->getRPCIndex()];
	unsigned my_i = 999;
	for (unsigned i=0; i< rtGMTRPCbCands.size(); i++)
	{
	  if (rrpcb.getDataWord()!=rtGMTRPCbCands[i].l1reg->getDataWord()) continue;
	  my_i = i;
	  break;
	}
	if (my_i<99) gmt_rpcb = &rtGMTRPCbCands[my_i];
	else std::cout<<"DOES NOT EXIST IN rtGMTRPCbCands! Should not happen!"<<std::endl;
	myGMTCand.regcand_rpc = gmt_rpcb;
      }
      
      MatchCSCMuL1::GMTREGCAND * gmt_dt = nullptr;
      if (!(muItr->isFwd()) && (muItr->isMatchedCand() || !(muItr->isRPC())))
      {
	L1MuRegionalCand rdt = DTCands[muItr->getDTCSCIndex()];
	unsigned my_i = 999;
	for (unsigned i=0; i< rtGMTDTCands.size(); i++)
	  {
	    if (rdt.getDataWord()!=rtGMTDTCands[i].l1reg->getDataWord()) continue;
	    my_i = i;
	    break;
	  }
	if (my_i<99) gmt_dt = &rtGMTDTCands[my_i];
	else std::cout<<"DOES NOT EXIST IN rtGMTDTCands! Should not happen!"<<std::endl;
	myGMTCand.regcand = gmt_dt;
      }
      
      if ( (gmt_csc != nullptr && gmt_rpcf != nullptr) && !muItr->isMatchedCand() ) std::cout<<"csc&rpcf but not matched!"<<std::endl;
      
      bool eta_me42 = mugeo::isME42EtaRegion(myGMTCand.eta);
      bool eta_me42r = mugeo::isME42RPCEtaRegion(myGMTCand.eta);
      //if (geta>=1.2 && geta<=1.8) eta_me42 = 1;
      bool eta_q = (geta > 1.2);
      
      bool eta_me1b = mugeo::isME1bEtaRegion(myGMTCand.eta);
      //bool eta_me1b_whole = mugeo::isME1bEtaRegion(myGMTCand.eta, 1.6, 2.14);
      bool eta_no1a = (geta >= 1.2 && geta < 2.14);
      //bool eta_csc = (geta > 0.9);
      //
      
      size_t n_stubs = 0;
      if (gmt_csc) n_stubs = gmt_csc->nTFStubs;
      
      bool has_me1_stub = false;
      if (gmt_csc && gmt_csc->tfcand && gmt_csc->tfcand->tftrack)
      {
	has_me1_stub = gmt_csc->tfcand->tftrack->hasStub(1);
      }
      
      
      if (eta_me42) h_rt_gmt_gq_42->Fill(muItr->quality());
      if (eta_me42r) {
	int gtype = 0;
	if (muItr->isMatchedCand()) gtype = 6;
	else if (gmt_csc!=0) gtype = gmt_csc->l1reg->quality()+2;
	else if (gmt_rpcf!=0) gtype = gmt_rpcf->l1reg->quality()+1;
	if (gtype==0) std::cout<<"weird: gtype=0 That shouldn't happen!";
	h_rt_gmt_gq_vs_type_42r->Fill(muItr->quality(), gtype);
	h_rt_gmt_gq_vs_pt_42r->Fill(muItr->quality(), gpt);
	h_rt_gmt_gq_42r->Fill(muItr->quality());
      }
      h_rt_gmt_gq->Fill(muItr->quality());
      
      h_rt_gmt_bx->Fill(muItr->bx());
      
      //if (muItr->quality()<4) continue; // not good for single muon trigger!
      
      bool isSingleTrigOk = muItr->useInSingleMuonTrigger(); // good for single trigger
      bool isDoubleTrigOk = muItr->useInDiMuonTrigger(); // good for single trigger
      
      bool isSingle6TrigOk = (muItr->quality() >= 6); // unmatched or matched CSC or DT
      
      if (muItr->quality()<3) continue; // not good for neither single nor dimuon triggers
      
      bool isCSC = (gmt_csc != nullptr);
      bool isDT  = (gmt_dt  != nullptr);
      bool isRPCf = (gmt_rpcf != nullptr);
      bool isRPCb = (gmt_rpcb != nullptr);
      
      if (isCSC && gmt_csc->tfcand != nullptr && gmt_csc->tfcand->tftrack == nullptr) std::cout<<"warning: gmt_csc->tfcand->tftrack == nullptr"<<std::endl;
      if (isCSC && gmt_csc->tfcand != nullptr && gmt_csc->tfcand->tftrack != nullptr && gmt_csc->tfcand->tftrack->l1trk == nullptr)
	std::cout<<"warning: gmt_csc->tfcand->tftrack->l1trk == nullptr"<<std::endl;
      //bool isCSC2s = (isCSC && gmt_csc->tfcand != nullptr && myGMTCand.ids.size()>=2);
      //bool isCSC3s = (isCSC && gmt_csc->tfcand != nullptr && myGMTCand.ids.size()>=3);
      bool isCSC2s = (isCSC && gmt_csc->tfcand != nullptr && gmt_csc->tfcand->tftrack != nullptr && gmt_csc->tfcand->tftrack->nStubs()>=2);
      bool isCSC3s = (isCSC && gmt_csc->tfcand != nullptr && gmt_csc->tfcand->tftrack != nullptr
		      && ( (!eta_q && isCSC2s) || (eta_q && gmt_csc->tfcand->tftrack->nStubs()>=3) ) );
      bool isCSC2q = (isCSC && gmt_csc->l1reg != nullptr && gmt_csc->l1reg->quality()>=2);
      bool isCSC3q = (isCSC && gmt_csc->l1reg != nullptr 
		      && ( (!eta_q && isCSC2q) || (eta_q && gmt_csc->l1reg->quality()>=3) ) );
      
      myGMTCand.isCSC = isCSC;
      myGMTCand.isDT = isDT;
      myGMTCand.isRPCf = isRPCf;
      myGMTCand.isRPCb = isRPCb;
      myGMTCand.isCSC2s = isCSC2s;
      myGMTCand.isCSC3s = isCSC3s;
      myGMTCand.isCSC2q = isCSC2q;
      myGMTCand.isCSC3q = isCSC3q;
      
      rtGMTCands.push_back(myGMTCand);
      
      
      if (isCSC2q || isRPCf) {
	h_rt_gmt_pt_2q->Fill(gpt);
	if (eta_me42) {
	  h_rt_gmt_pt_2q42->Fill(gpt);
	  if (gpt > max_pt_me42_2q) max_pt_me42_2q = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42_2q_sing) max_pt_me42_2q_sing = gpt;
	}
	if (eta_me42r) {
	  h_rt_gmt_pt_2q42r->Fill(gpt);
	  if (gpt > max_pt_me42r_2q) max_pt_me42r_2q = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42r_2q_sing) max_pt_me42r_2q_sing = gpt;
	}
      }
      if (isCSC3q || isRPCf) {
	h_rt_gmt_pt_3q->Fill(gpt);
	if (eta_me42) {
	  h_rt_gmt_pt_3q42->Fill(gpt);
	  if (gpt > max_pt_me42_3q) max_pt_me42_3q = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42_3q_sing) max_pt_me42_3q_sing = gpt;
	}
	if (eta_me42r) {
	  h_rt_gmt_pt_3q42r->Fill(gpt);
	  if (gpt > max_pt_me42r_3q) max_pt_me42r_3q = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42r_3q_sing) max_pt_me42r_3q_sing = gpt;
	}
      }

      if (isCSC2s || isRPCf) {
	h_rt_gmt_pt_2st->Fill(gpt);
	if (eta_me42) {
	  h_rt_gmt_pt_2s42->Fill(gpt);
	  if (gpt > max_pt_me42_2s) max_pt_me42_2s = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42_2s_sing) max_pt_me42_2s_sing = gpt;
	}
	if (eta_me42r) {
	  h_rt_gmt_pt_2s42r->Fill(gpt);
	  if (gpt > max_pt_me42r_2s) max_pt_me42r_2s = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42r_2s_sing) max_pt_me42r_2s_sing = gpt;
	}
      }
      if (isCSC3s || isRPCf) {
	h_rt_gmt_pt_3st->Fill(gpt);
	if (eta_me42) {
	  h_rt_gmt_pt_3s42->Fill(gpt);
	  if (gpt > max_pt_me42_3s) max_pt_me42_3s = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42_3s_sing) max_pt_me42_3s_sing = gpt;
	}
	if (eta_me42r) {
	  h_rt_gmt_pt_3s42r->Fill(gpt);
	  if (gpt > max_pt_me42r_3s) max_pt_me42r_3s = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42r_3s_sing) max_pt_me42r_3s_sing = gpt;
	}
      }

      ngmt++;
      h_rt_gmt_pt->Fill(gpt);
      h_rt_gmt_eta->Fill(geta);
      if (gpt > max_pt) {max_pt = gpt; max_pt_eta = geta;}
      if (isDoubleTrigOk && gpt > max_pt_dbl) {max_pt_dbl = gpt; max_pt_eta_dbl = geta;}
      if (isSingleTrigOk)
	{
	  if (            gpt > max_pt_sing     ) { max_pt_sing = gpt;     max_pt_eta_sing = geta;}
	  if (isCSC    && gpt > max_pt_sing_csc ) { max_pt_sing_csc = gpt; max_pt_eta_sing_csc = geta; }
	  if ((isCSC||isDT) && gpt > max_pt_sing_dtcsc ) { max_pt_sing_dtcsc = gpt; max_pt_eta_sing_dtcsc = geta; }
	  if (gpt > max_pt_sing_3s && ( !isCSC || isCSC3s ) ) {max_pt_sing_3s = gpt; max_pt_eta_sing_3s = geta;}
	  if (eta_me1b && gpt > max_pt_sing_1b  ) { max_pt_sing_1b = gpt; max_pt_eta_sing_1b = geta; }
	  if (eta_no1a && gpt > max_pt_sing_no1a) { max_pt_sing_no1a = gpt; max_pt_eta_sing_no1a = geta; }
	}
      if (isSingle6TrigOk)
	{
	  if (            gpt > max_pt_sing6     ) { max_pt_sing6 = gpt;     max_pt_eta_sing6 = geta;}
	  if (isCSC    && gpt > max_pt_sing6_csc ) { max_pt_sing6_csc = gpt; max_pt_eta_sing6_csc = geta; }
	  if (gpt > max_pt_sing6_3s && ( !isCSC || isCSC3s ) ) {max_pt_sing6_3s = gpt; max_pt_eta_sing6_3s = geta;}
	  if (eta_me1b && gpt > max_pt_sing6_1b  ) { max_pt_sing6_1b = gpt; max_pt_eta_sing6_1b = geta; }
	  if (eta_no1a && gpt > max_pt_sing6_no1a) { max_pt_sing6_no1a = gpt; max_pt_eta_sing6_no1a = geta; }
	  if (eta_no1a && gpt > max_pt_sing6_3s1b_no1a && 
	      (!eta_me1b  || (eta_me1b && has_me1_stub && n_stubs >=3) ) ) { max_pt_sing6_3s1b_no1a = gpt; max_pt_eta_sing6_no1a = geta; }
	}
    }
  }
  h_rt_ngmt->Fill(ngmt);
  if (max_pt_me42_2s>0) h_rt_gmt_ptmax_2s42->Fill(max_pt_me42_2s);
  if (max_pt_me42_3s>0) h_rt_gmt_ptmax_3s42->Fill(max_pt_me42_3s);
  if (max_pt_me42_2q>0) h_rt_gmt_ptmax_2q42->Fill(max_pt_me42_2q);
  if (max_pt_me42_3q>0) h_rt_gmt_ptmax_3q42->Fill(max_pt_me42_3q);
  if (max_pt_me42_2s_sing>0) h_rt_gmt_ptmax_2s42_sing->Fill(max_pt_me42_2s_sing);
  if (max_pt_me42_3s_sing>0) h_rt_gmt_ptmax_3s42_sing->Fill(max_pt_me42_3s_sing);
  if (max_pt_me42_2q_sing>0) h_rt_gmt_ptmax_2q42_sing->Fill(max_pt_me42_2q_sing);
  if (max_pt_me42_3q_sing>0) h_rt_gmt_ptmax_3q42_sing->Fill(max_pt_me42_3q_sing);
  if (max_pt_me42r_2s>0) h_rt_gmt_ptmax_2s42r->Fill(max_pt_me42r_2s);
  if (max_pt_me42r_3s>0) h_rt_gmt_ptmax_3s42r->Fill(max_pt_me42r_3s);
  if (max_pt_me42r_2q>0) h_rt_gmt_ptmax_2q42r->Fill(max_pt_me42r_2q);
  if (max_pt_me42r_3q>0) h_rt_gmt_ptmax_3q42r->Fill(max_pt_me42r_3q);
  if (max_pt_me42r_2s_sing>0) h_rt_gmt_ptmax_2s42r_sing->Fill(max_pt_me42r_2s_sing);
  if (max_pt_me42r_3s_sing>0) h_rt_gmt_ptmax_3s42r_sing->Fill(max_pt_me42r_3s_sing);
  if (max_pt_me42r_2q_sing>0) h_rt_gmt_ptmax_2q42r_sing->Fill(max_pt_me42r_2q_sing);
  if (max_pt_me42r_3q_sing>0) h_rt_gmt_ptmax_3q42r_sing->Fill(max_pt_me42r_3q_sing);
  if (max_pt>0) h_rt_gmt_ptmax->Fill(max_pt);
  if (max_pt>=10.) h_rt_gmt_ptmax10_eta->Fill(max_pt_eta);
  if (max_pt>=20.) h_rt_gmt_ptmax20_eta->Fill(max_pt_eta);

  if (max_pt_sing>0) h_rt_gmt_ptmax_sing->Fill(max_pt_sing);
  if (max_pt_sing_3s>0) h_rt_gmt_ptmax_sing_3s->Fill(max_pt_sing_3s);
  if (max_pt_sing>=10.) h_rt_gmt_ptmax10_eta_sing->Fill(max_pt_eta_sing);
  if (max_pt_sing_3s>=10.) h_rt_gmt_ptmax10_eta_sing_3s->Fill(max_pt_eta_sing_3s);
  if (max_pt_sing>=20.) h_rt_gmt_ptmax20_eta_sing->Fill(max_pt_eta_sing);
  if (max_pt_sing_csc>=20.) h_rt_gmt_ptmax20_eta_sing_csc->Fill(max_pt_eta_sing_csc);
  if (max_pt_sing_dtcsc>=20.) h_rt_gmt_ptmax20_eta_sing_dtcsc->Fill(max_pt_eta_sing_dtcsc);
  if (max_pt_sing_3s>=20.) h_rt_gmt_ptmax20_eta_sing_3s->Fill(max_pt_eta_sing_3s);
  if (max_pt_sing>=30.) h_rt_gmt_ptmax30_eta_sing->Fill(max_pt_eta_sing);
  if (max_pt_sing_csc>=30.) h_rt_gmt_ptmax30_eta_sing_csc->Fill(max_pt_eta_sing_csc);
  if (max_pt_sing_dtcsc>=30.) h_rt_gmt_ptmax30_eta_sing_dtcsc->Fill(max_pt_eta_sing_dtcsc);
  if (max_pt_sing_3s>=30.) h_rt_gmt_ptmax30_eta_sing_3s->Fill(max_pt_eta_sing_3s);
  if (max_pt_sing_csc > 0.) h_rt_gmt_ptmax_sing_csc->Fill(max_pt_sing_csc);
  if (max_pt_sing_1b > 0. ) h_rt_gmt_ptmax_sing_1b->Fill(max_pt_sing_1b);
  if (max_pt_sing_no1a > 0.) h_rt_gmt_ptmax_sing_no1a->Fill(max_pt_sing_no1a);

  if (max_pt_sing6>0) h_rt_gmt_ptmax_sing6->Fill(max_pt_sing6);
  if (max_pt_sing6_3s>0) h_rt_gmt_ptmax_sing6_3s->Fill(max_pt_sing6_3s);
  if (max_pt_sing6>=10.) h_rt_gmt_ptmax10_eta_sing6->Fill(max_pt_eta_sing6);
  if (max_pt_sing6_3s>=10.) h_rt_gmt_ptmax10_eta_sing6_3s->Fill(max_pt_eta_sing6_3s);
  if (max_pt_sing6>=20.) h_rt_gmt_ptmax20_eta_sing6->Fill(max_pt_eta_sing6);
  if (max_pt_sing6_csc>=20.) h_rt_gmt_ptmax20_eta_sing6_csc->Fill(max_pt_eta_sing6_csc);
  if (max_pt_sing6_3s>=20.) h_rt_gmt_ptmax20_eta_sing6_3s->Fill(max_pt_eta_sing6_3s);
  if (max_pt_sing6>=30.) h_rt_gmt_ptmax30_eta_sing6->Fill(max_pt_eta_sing6);
  if (max_pt_sing6_csc>=30.) h_rt_gmt_ptmax30_eta_sing6_csc->Fill(max_pt_eta_sing6_csc);
  if (max_pt_sing6_3s>=30.) h_rt_gmt_ptmax30_eta_sing6_3s->Fill(max_pt_eta_sing6_3s);
  if (max_pt_sing6_csc > 0.) h_rt_gmt_ptmax_sing6_csc->Fill(max_pt_sing6_csc);
  if (max_pt_sing6_1b > 0. ) h_rt_gmt_ptmax_sing6_1b->Fill(max_pt_sing6_1b);
  if (max_pt_sing6_no1a > 0.) h_rt_gmt_ptmax_sing6_no1a->Fill(max_pt_sing6_no1a);
  if (max_pt_sing6_3s1b_no1a > 0.) h_rt_gmt_ptmax_sing6_3s1b_no1a->Fill(max_pt_sing6_3s1b_no1a);

  if (max_pt_dbl>0) h_rt_gmt_ptmax_dbl->Fill(max_pt_dbl);
  if (max_pt_dbl>=10.) h_rt_gmt_ptmax10_eta_dbl->Fill(max_pt_eta_dbl);
  if (max_pt_dbl>=20.) h_rt_gmt_ptmax20_eta_dbl->Fill(max_pt_eta_dbl);
  if (debugRATE) std::cout<< "----- end ngmt="<<ngmt<<std::endl;
  */
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::bookALCTTree()
{
  edm::Service< TFileService > fs;
  alct_tree_ = fs->make<TTree>("ALCTs", "ALCTs");
  alct_tree_->Branch("event",&alct_.event);
  alct_tree_->Branch("bx",&alct_.bx);
  alct_tree_->Branch("endcap",&alct_.endcap);
  alct_tree_->Branch("station",&alct_.station);
  alct_tree_->Branch("ring",&alct_.ring);
  alct_tree_->Branch("chamber",&alct_.chamber);
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::bookCLCTTree()
{
  edm::Service< TFileService > fs;
  clct_tree_ = fs->make<TTree>("CLCTs", "CLCTs");
  clct_tree_->Branch("event",&clct_.event);
  clct_tree_->Branch("bx",&clct_.bx);
  clct_tree_->Branch("endcap",&clct_.endcap);
  clct_tree_->Branch("station",&clct_.station);
  clct_tree_->Branch("ring",&clct_.ring);
  clct_tree_->Branch("chamber",&clct_.chamber);
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::bookLCTTree()
{
  edm::Service< TFileService > fs;
  lct_tree_ = fs->make<TTree>("LCTs", "LCTs");
  lct_tree_->Branch("event",&lct_.event);
  lct_tree_->Branch("bx",&lct_.bx);
  lct_tree_->Branch("endcap",&lct_.endcap);
  lct_tree_->Branch("station",&lct_.station);
  lct_tree_->Branch("ring",&lct_.ring);
  lct_tree_->Branch("chamber",&lct_.chamber);
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::bookMPCLCTTree()
{
  edm::Service< TFileService > fs;
  mplct_tree_ = fs->make<TTree>("MPLCTs", "MPLCTs");
  mplct_tree_->Branch("event",&mplct_.event);
  mplct_tree_->Branch("bx",&mplct_.bx);
  mplct_tree_->Branch("endcap",&mplct_.endcap);
  mplct_tree_->Branch("station",&mplct_.station);
  mplct_tree_->Branch("ring",&mplct_.ring);
  mplct_tree_->Branch("chamber",&mplct_.chamber);
  mplct_tree_->Branch("etalut",&mplct_.etalut);
  mplct_tree_->Branch("philut",&mplct_.philut);
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::bookTFTrackTree()
{
  edm::Service< TFileService > fs;
  tftrack_tree_ = fs->make<TTree>("TFTracks", "TFTracks");
  tftrack_tree_->Branch("event",&tftrack_.event);
  tftrack_tree_->Branch("bx",&tftrack_.bx);
  tftrack_tree_->Branch("pt",&tftrack_.pt);
  tftrack_tree_->Branch("eta",&tftrack_.eta);
  tftrack_tree_->Branch("phi",&tftrack_.phi);
  tftrack_tree_->Branch("hasME1a",&tftrack_.hasME1a);
  tftrack_tree_->Branch("hasME1b",&tftrack_.hasME1b);
  tftrack_tree_->Branch("hasME12",&tftrack_.hasME12);
  tftrack_tree_->Branch("hasME13",&tftrack_.hasME13);
  tftrack_tree_->Branch("hasME21",&tftrack_.hasME21);
  tftrack_tree_->Branch("hasME22",&tftrack_.hasME22);
  tftrack_tree_->Branch("hasME31",&tftrack_.hasME31);
  tftrack_tree_->Branch("hasME32",&tftrack_.hasME32);
  tftrack_tree_->Branch("hasME41",&tftrack_.hasME41);
  tftrack_tree_->Branch("hasME42",&tftrack_.hasME42);
  tftrack_tree_->Branch("hasRE12",&tftrack_.hasRE12);
  tftrack_tree_->Branch("hasRE13",&tftrack_.hasRE13);
  tftrack_tree_->Branch("hasRE22",&tftrack_.hasRE22);
  tftrack_tree_->Branch("hasRE23",&tftrack_.hasRE23);
  tftrack_tree_->Branch("hasRE31",&tftrack_.hasRE31);
  tftrack_tree_->Branch("hasRE32",&tftrack_.hasRE32);
  tftrack_tree_->Branch("hasRE33",&tftrack_.hasRE33);
  tftrack_tree_->Branch("hasRE41",&tftrack_.hasRE41);
  tftrack_tree_->Branch("hasRE42",&tftrack_.hasRE42);
  tftrack_tree_->Branch("hasRE43",&tftrack_.hasRE43);
  tftrack_tree_->Branch("hasGE11",&tftrack_.hasGE11);
  tftrack_tree_->Branch("hasGE21",&tftrack_.hasGE21);
  tftrack_tree_->Branch("hasME0",&tftrack_.hasME0);
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::bookTFCandTree()
{
  edm::Service< TFileService > fs;
  tfcand_tree_ = fs->make<TTree>("TFCands", "TFCands");
  tfcand_tree_->Branch("event",&tfcand_.event);
  tfcand_tree_->Branch("bx",&tfcand_.bx);
  tfcand_tree_->Branch("pt",&tfcand_.pt);
  tfcand_tree_->Branch("eta",&tfcand_.eta);
  tfcand_tree_->Branch("phi",&tfcand_.phi);
  tfcand_tree_->Branch("hasME1a",&tfcand_.hasME1a);
  tfcand_tree_->Branch("hasME1b",&tfcand_.hasME1b);
  tfcand_tree_->Branch("hasME12",&tfcand_.hasME12);
  tfcand_tree_->Branch("hasME13",&tfcand_.hasME13);
  tfcand_tree_->Branch("hasME21",&tfcand_.hasME21);
  tfcand_tree_->Branch("hasME22",&tfcand_.hasME22);
  tfcand_tree_->Branch("hasME31",&tfcand_.hasME31);
  tfcand_tree_->Branch("hasME32",&tfcand_.hasME32);
  tfcand_tree_->Branch("hasME41",&tfcand_.hasME41);
  tfcand_tree_->Branch("hasME42",&tfcand_.hasME42);
  tfcand_tree_->Branch("hasRE12",&tfcand_.hasRE12);
  tfcand_tree_->Branch("hasRE13",&tfcand_.hasRE13);
  tfcand_tree_->Branch("hasRE22",&tfcand_.hasRE22);
  tfcand_tree_->Branch("hasRE23",&tfcand_.hasRE23);
  tfcand_tree_->Branch("hasRE31",&tfcand_.hasRE31);
  tfcand_tree_->Branch("hasRE32",&tfcand_.hasRE32);
  tfcand_tree_->Branch("hasRE33",&tfcand_.hasRE33);
  tfcand_tree_->Branch("hasRE41",&tfcand_.hasRE41);
  tfcand_tree_->Branch("hasRE42",&tfcand_.hasRE42);
  tfcand_tree_->Branch("hasRE43",&tfcand_.hasRE43);
  tfcand_tree_->Branch("hasGE11",&tfcand_.hasGE11);
  tfcand_tree_->Branch("hasGE21",&tfcand_.hasGE21);
  tfcand_tree_->Branch("hasME0",&tfcand_.hasME0);
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::bookGMTRegionalTree()
{
  edm::Service< TFileService > fs;
  gmtreg_tree_ = fs->make<TTree>("GMTRegs", "GMTRegs");
  gmtreg_tree_->Branch("event",&gmtreg_.event);
  gmtreg_tree_->Branch("bx",&gmtreg_.bx);
  gmtreg_tree_->Branch("pt",&gmtreg_.pt);
  gmtreg_tree_->Branch("eta",&gmtreg_.eta);
  gmtreg_tree_->Branch("phi",&gmtreg_.phi);
  gmtreg_tree_->Branch("hasME1a",&gmtreg_.hasME1a);
  gmtreg_tree_->Branch("hasME1b",&gmtreg_.hasME1b);
  gmtreg_tree_->Branch("hasME12",&gmtreg_.hasME12);
  gmtreg_tree_->Branch("hasME13",&gmtreg_.hasME13);
  gmtreg_tree_->Branch("hasME21",&gmtreg_.hasME21);
  gmtreg_tree_->Branch("hasME22",&gmtreg_.hasME22);
  gmtreg_tree_->Branch("hasME31",&gmtreg_.hasME31);
  gmtreg_tree_->Branch("hasME32",&gmtreg_.hasME32);
  gmtreg_tree_->Branch("hasME41",&gmtreg_.hasME41);
  gmtreg_tree_->Branch("hasME42",&gmtreg_.hasME42);
  gmtreg_tree_->Branch("hasRE12",&gmtreg_.hasRE12);
  gmtreg_tree_->Branch("hasRE13",&gmtreg_.hasRE13);
  gmtreg_tree_->Branch("hasRE22",&gmtreg_.hasRE22);
  gmtreg_tree_->Branch("hasRE23",&gmtreg_.hasRE23);
  gmtreg_tree_->Branch("hasRE31",&gmtreg_.hasRE31);
  gmtreg_tree_->Branch("hasRE32",&gmtreg_.hasRE32);
  gmtreg_tree_->Branch("hasRE33",&gmtreg_.hasRE33);
  gmtreg_tree_->Branch("hasRE41",&gmtreg_.hasRE41);
  gmtreg_tree_->Branch("hasRE42",&gmtreg_.hasRE42);
  gmtreg_tree_->Branch("hasRE43",&gmtreg_.hasRE43);
  gmtreg_tree_->Branch("hasGE11",&gmtreg_.hasGE11);
  gmtreg_tree_->Branch("hasGE21",&gmtreg_.hasGE21);
  gmtreg_tree_->Branch("hasME0",&gmtreg_.hasME0);
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::bookGMTCandTree()
{
  edm::Service< TFileService > fs;
  gmt_tree_ = fs->make<TTree>("GMTs", "GMTs");
  gmt_tree_->Branch("event",&gmt_.event);
  gmt_tree_->Branch("bx",&gmt_.bx);
  gmt_tree_->Branch("pt",&gmt_.pt);
  gmt_tree_->Branch("eta",&gmt_.eta);
  gmt_tree_->Branch("phi",&gmt_.phi);
  gmt_tree_->Branch("hasME1a",&gmt_.hasME1a);
  gmt_tree_->Branch("hasME1b",&gmt_.hasME1b);
  gmt_tree_->Branch("hasME12",&gmt_.hasME12);
  gmt_tree_->Branch("hasME13",&gmt_.hasME13);
  gmt_tree_->Branch("hasME21",&gmt_.hasME21);
  gmt_tree_->Branch("hasME22",&gmt_.hasME22);
  gmt_tree_->Branch("hasME31",&gmt_.hasME31);
  gmt_tree_->Branch("hasME32",&gmt_.hasME32);
  gmt_tree_->Branch("hasME41",&gmt_.hasME41);
  gmt_tree_->Branch("hasME42",&gmt_.hasME42);
  gmt_tree_->Branch("hasRE12",&gmt_.hasRE12);
  gmt_tree_->Branch("hasRE13",&gmt_.hasRE13);
  gmt_tree_->Branch("hasRE22",&gmt_.hasRE22);
  gmt_tree_->Branch("hasRE23",&gmt_.hasRE23);
  gmt_tree_->Branch("hasRE31",&gmt_.hasRE31);
  gmt_tree_->Branch("hasRE32",&gmt_.hasRE32);
  gmt_tree_->Branch("hasRE33",&gmt_.hasRE33);
  gmt_tree_->Branch("hasRE41",&gmt_.hasRE41);
  gmt_tree_->Branch("hasRE42",&gmt_.hasRE42);
  gmt_tree_->Branch("hasRE43",&gmt_.hasRE43);
  gmt_tree_->Branch("hasGE11",&gmt_.hasGE11);
  gmt_tree_->Branch("hasGE21",&gmt_.hasGE21);
  gmt_tree_->Branch("hasME0",&gmt_.hasME0);
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::analyzeALCTRate(const edm::Event& iEvent)
{
  edm::Handle< CSCALCTDigiCollection > halcts;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  halcts);
  const CSCALCTDigiCollection* alcts = halcts.product();
  
  for (CSCALCTDigiCollection::DigiRangeIterator  adetUnitIt = alcts->begin(); adetUnitIt != alcts->end(); ++adetUnitIt)
  {
    CSCDetId detId((*adetUnitIt).first);
    auto range = (*adetUnitIt).second;
    for (CSCALCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt)
    {
      if (!(*digiIt).isValid()) continue;
      const int bx((*digiIt).getBX());
      if (bx < minBxALCT_ or bx > maxBxALCT_) continue;
      alct_.event = iEvent.id().event();
      alct_.endcap = detId.zendcap();
      alct_.station = detId.station();
      alct_.ring = detId.ring();
      alct_.chamber = detId.chamber();
      alct_.bx = bx - 6;
      alct_tree_->Fill();
    }
  }
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::analyzeCLCTRate(const edm::Event& iEvent)
{
  edm::Handle< CSCCLCTDigiCollection > hclcts;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  hclcts);
  const CSCCLCTDigiCollection* clcts = hclcts.product();

  for (CSCCLCTDigiCollection::DigiRangeIterator  adetUnitIt = clcts->begin(); adetUnitIt != clcts->end(); ++adetUnitIt)
  {
    CSCDetId detId((*adetUnitIt).first);
    auto range = (*adetUnitIt).second;
    for (CSCCLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt)
    {
      if (!(*digiIt).isValid()) continue;
      const int bx((*digiIt).getBX());
      if (bx < minBxCLCT_ or bx > maxBxCLCT_) continue;
      clct_.event = iEvent.id().event();
      clct_.endcap = detId.zendcap();
      clct_.station = detId.station();
      clct_.ring = detId.ring();
      clct_.chamber = detId.chamber();
      clct_.bx = bx - 6;
      clct_tree_->Fill();
    }
  }
}


// ================================================================================================
void  
GEMCSCTriggerRateTree::analyzeLCTRate(const edm::Event& iEvent)
{
  edm::Handle< CSCCorrelatedLCTDigiCollection > lcts_tmb;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  lcts_tmb);
  const CSCCorrelatedLCTDigiCollection* lcts = lcts_tmb.product();

  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt = lcts->begin(); detUnitIt != lcts->end(); detUnitIt++) 
  {
    const CSCDetId& detId = (*detUnitIt).first;
    auto range = (*detUnitIt).second;
    for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
    {
      if (!(*digiIt).isValid()) continue;
      const int bx((*digiIt).getBX());
      if (bx < minBxLCT_ or bx > maxBxLCT_) continue;
      lct_.event = iEvent.id().event();
      lct_.endcap = detId.zendcap();
      lct_.station = detId.station();
      lct_.ring = detId.ring();
      lct_.chamber = detId.chamber();
      lct_.bx = bx - 6;
      lct_tree_->Fill();
    }
  }
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::analyzeMPCLCTRate(const edm::Event& iEvent)
{
  edm::Handle< CSCCorrelatedLCTDigiCollection > lcts_mpc;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis", "MPCSORTED", lcts_mpc);
  const CSCCorrelatedLCTDigiCollection* mplcts = lcts_mpc.product();

  for (auto detUnitIt = mplcts->begin(); detUnitIt != mplcts->end(); detUnitIt++) 
  {
    const CSCDetId& detId = (*detUnitIt).first;
    auto range = (*detUnitIt).second;
    for (auto digiIt = range.first; digiIt != range.second; digiIt++) 
    {
      if (!(*digiIt).isValid()) continue;
      const int bx((*digiIt).getBX());
      if (bx < minBxMPLCT_ or bx > maxBxMPLCT_) continue;
      mplct_.event = iEvent.id().event();
      mplct_.endcap = detId.zendcap();
      mplct_.station = detId.station();
      mplct_.ring = detId.ring();
      mplct_.chamber = detId.chamber();
      mplct_.bx = bx - 6;
      const csctf::TrackStub stub(buildTrackStub((*digiIt), detId));
      mplct_.etalut = stub.etaValue();
      mplct_.philut = stub.phiValue();
      mplct_tree_->Fill();
    }
  }
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::analyzeTFTrackRate(const edm::Event& iEvent)
{
  edm::Handle< L1CSCTrackCollection > hl1Tracks;
  iEvent.getByLabel("simCsctfTrackDigis",hl1Tracks);
  const L1CSCTrackCollection* l1Tracks = hl1Tracks.product();

  for (auto  trk = l1Tracks->begin(); trk != l1Tracks->end(); trk++) {
    if (trk->first.bx() < minRateBX_ or trk->first.bx() > maxRateBX_) continue;
    const bool endcapOnly(true);
    if (endcapOnly and trk->first.endcap()!=1) continue;
    
    MatchCSCMuL1::TFTRACK myTFTrk;
    myTFTrk.init( &(trk->first) , ptLUT, muScales, muPtScale);
    myTFTrk.dr = 999.;
    // add the TFTrack to the list
    rtTFTracks_.push_back(myTFTrk);

    tftrack_.event = iEvent.id().event();
    tftrack_.bx = trk->first.bx();
    tftrack_.pt = myTFTrk.pt;
    tftrack_.eta = myTFTrk.pt;
    tftrack_.phi = myTFTrk.eta;
    
    for (auto detUnitIt = trk->second.begin(); detUnitIt != trk->second.end(); detUnitIt++) {
      const CSCDetId& id = (*detUnitIt).first;
      auto range = (*detUnitIt).second;
      for (auto  digiIt = range.first; digiIt != range.second; digiIt++) {
	if (!((*digiIt).isValid())) continue;
	myTFTrk.trgdigis.push_back(&*digiIt);
	myTFTrk.trgids.push_back(id);
	myTFTrk.trgetaphis.push_back(intersectionEtaPhi(id,(*digiIt).getKeyWG(),(*digiIt).getStrip()));
	myTFTrk.trgstubs.push_back( buildTrackStub((*digiIt),id));

	// stub analysis
	if (id.station()==1 and id.ring()==4) tftrack_.hasME1a |= 1;
	if (id.station()==1 and id.ring()==1) tftrack_.hasME1b |= 1;
	if (id.station()==1 and id.ring()==2) tftrack_.hasME12 |= 1;
	if (id.station()==1 and id.ring()==3) tftrack_.hasME13 |= 1; 
	if (id.station()==2 and id.ring()==1) tftrack_.hasME21 |= 1;
	if (id.station()==2 and id.ring()==2) tftrack_.hasME22 |= 1;
	if (id.station()==3 and id.ring()==1) tftrack_.hasME31 |= 1;
	if (id.station()==3 and id.ring()==2) tftrack_.hasME32 |= 1;
	if (id.station()==4 and id.ring()==1) tftrack_.hasME41 |= 1;
	if (id.station()==4 and id.ring()==2) tftrack_.hasME42 |= 1;
      }
    }
  }
}

// ================================================================================================
void  
GEMCSCTriggerRateTree::analyzeTFCandRate(const edm::Event& iEvent)
{
  edm::Handle< std::vector< L1MuRegionalCand > > hl1TfCands;
  iEvent.getByLabel("simCsctfDigis", "CSC", hl1TfCands);
  const std::vector< L1MuRegionalCand > *l1TfCands = hl1TfCands.product();

  for (auto trk = l1TfCands->begin(); trk != l1TfCands->end(); trk++){
    if ( trk->bx() < minRateBX_ or trk->bx() > maxRateBX_ ) continue;
    //    const int sign_eta(((trk->eta_packed() & 0x20) == 0) ? 1.:-1);
    MatchCSCMuL1::TFCAND myTFCand;
    myTFCand.init( &*trk , ptLUT, muScales, muPtScale);
    myTFCand.dr = 999.;
    rtTFCands_.push_back(myTFCand);
    // associate the TFTracks to this TFCand
    for (size_t tt = 0; tt<rtTFTracks_.size(); tt++){
      if (trk->bx()         != rtTFTracks_[tt].l1trk->bx() or
	  trk->phi_packed() != rtTFTracks_[tt].phi_packed or
	  trk->pt_packed()  != rtTFTracks_[tt].pt_packed or
	  trk->eta_packed() != rtTFTracks_[tt].eta_packed) continue;
      myTFCand.tftrack = &(rtTFTracks_[tt]);
      // ids now hold *trigger segments IDs*
      myTFCand.ids = rtTFTracks_[tt].trgids;
      myTFCand.nTFStubs = rtTFTracks_[tt].nStubs(1,1,1,1,1);
    }
    
    // analysis
    if (myTFCand.tftrack != nullptr) continue;
    tfcand_.event = iEvent.id().event();
    tfcand_.bx = trk->bx();
    tfcand_.pt = myTFCand.tftrack->pt;
    tfcand_.eta = myTFCand.tftrack->eta;
    tfcand_.phi = myTFCand.tftrack->phi;
    auto trgids(myTFCand.tftrack->trgids);
 
	// stub analysis
    for (auto id : trgids){
      if (id.station()==1 and id.ring()==4) tftrack_.hasME1a |= 1;
      if (id.station()==1 and id.ring()==1) tftrack_.hasME1b |= 1;
      if (id.station()==1 and id.ring()==2) tftrack_.hasME12 |= 1;
      if (id.station()==1 and id.ring()==3) tftrack_.hasME13 |= 1; 
      if (id.station()==2 and id.ring()==1) tftrack_.hasME21 |= 1;
      if (id.station()==2 and id.ring()==2) tftrack_.hasME22 |= 1;
      if (id.station()==3 and id.ring()==1) tftrack_.hasME31 |= 1;
      if (id.station()==3 and id.ring()==2) tftrack_.hasME32 |= 1;
      if (id.station()==4 and id.ring()==1) tftrack_.hasME41 |= 1;
      if (id.station()==4 and id.ring()==2) tftrack_.hasME42 |= 1;
    }
  }
}


// ================================================================================================
void  
GEMCSCTriggerRateTree::analyzeGMTRegCandRate(const edm::Event& iEvent)
{
}


// ================================================================================================
void  
GEMCSCTriggerRateTree::analyzeGMTCandRate(const edm::Event& iEvent)
{
}




// ================================================================================================
void 
GEMCSCTriggerRateTree::runCSCTFSP(const CSCCorrelatedLCTDigiCollection* mplcts, const L1MuDTChambPhContainer* dttrig)
   //, L1CSCTrackCollection*, CSCTriggerContainer<csctf::TrackStub>*)
{
// Just run it for the sake of its debug printout, do not return any results

  // Create csctf::TrackStubs collection from MPC LCTs
  CSCTriggerContainer<csctf::TrackStub> stub_list;
  CSCCorrelatedLCTDigiCollection::DigiRangeIterator Citer;
  for(Citer = mplcts->begin(); Citer != mplcts->end(); Citer++)
  {
    CSCCorrelatedLCTDigiCollection::const_iterator Diter = (*Citer).second.first;
    CSCCorrelatedLCTDigiCollection::const_iterator Dend = (*Citer).second.second;
    for(; Diter != Dend; Diter++)
    {
      csctf::TrackStub theStub((*Diter),(*Citer).first);
      stub_list.push_back(theStub);
    }
  }
  
  // Now we append the track stubs the the DT Sector Collector
  // after processing from the DT Receiver.
  CSCTriggerContainer<csctf::TrackStub> dtstubs = my_dtrc->process(dttrig);
  stub_list.push_many(dtstubs);
  
  //for(int e=0; e<2; e++) for (int s=0; s<6; s++) {
  int e=0;
  for (int s=0; s<6; s++) 
  {
    CSCTriggerContainer<csctf::TrackStub> current_e_s = stub_list.get(e+1, s+1);
    if (current_e_s.get().size()>0) 
    {
      std::cout<<"sector "<<s+1<<":"<<std::endl<<std::endl;
      my_SPs[e][s]->run(current_e_s);
    }
  }
}

// ================================================================================================
// Returns chamber type (0-9) according to the station and ring number
int 
GEMCSCTriggerRateTree::getCSCType(CSCDetId &id) 
{
  int type = -999;

  if (id.station() == 1) {
    type = (id.triggerCscId()-1)/3;
    if (id.ring() == 4) {
      type = 3;
    }
  }
  else { // stations 2-4
    type = 3 + id.ring() + 2*(id.station()-2);
  }
  assert(type >= 0 && type < CSC_TYPES); // include ME4/2
  return type;
}

// ================================================================================================
int
GEMCSCTriggerRateTree::isME11(int t)
{
  if (t==0 || t==3) return CSC_TYPES;
  return 0;
}

// ================================================================================================
// Returns chamber type (0-9) according to CSCChamberSpecs type
// 1..10 -> 1/a, 1/b, 1/2, 1/3, 2/1...
int
GEMCSCTriggerRateTree::getCSCSpecsType(CSCDetId &id)
{
  return cscGeometry->chamber(id)->specs()->chamberType();
}

// ================================================================================================
int
GEMCSCTriggerRateTree::cscTriggerSubsector(CSCDetId &id)
{
  if(id.station() != 1) return 0; // only station one has subsectors
  int chamber = id.chamber();
  switch(chamber) // first make things easier to deal with
  {
    case 1:
      chamber = 36;
      break;
    case 2:
      chamber = 35;
      break;
    default:
      chamber -= 2;
  }
  chamber = ((chamber-1)%6) + 1; // renumber all chambers to 1-6
  return ((chamber-1) / 3) + 1; // [1,3] -> 1 , [4,6]->2
}


// ================================================================================================
// void 
// GEMCSCTriggerRateTree::setupTFModeHisto(TH1D* h)
// {
//   if (h==0) return;
//   if (h->GetXaxis()->GetNbins()<16) {
//     std::cout<<"TF mode histogram should have 16 bins, nbins="<<h->GetXaxis()->GetNbins()<<std::endl;
//     return;
//   }
//   h->GetXaxis()->SetTitle("Track Type");
//   h->GetXaxis()->SetTitleOffset(1.2);
//   h->GetXaxis()->SetBinLabel(1,"No Track");
//   h->GetXaxis()->SetBinLabel(2,"Bad Phi Road");
//   h->GetXaxis()->SetBinLabel(3,"ME1-2-3(-4)");
//   h->GetXaxis()->SetBinLabel(4,"ME1-2-4");
//   h->GetXaxis()->SetBinLabel(5,"ME1-3-4");
//   h->GetXaxis()->SetBinLabel(6,"ME2-3-4");
//   h->GetXaxis()->SetBinLabel(7,"ME1-2");
//   h->GetXaxis()->SetBinLabel(8,"ME1-3");
//   h->GetXaxis()->SetBinLabel(9,"ME2-3");
//   h->GetXaxis()->SetBinLabel(10,"ME2-4");
//   h->GetXaxis()->SetBinLabel(11,"ME3-4");
//   h->GetXaxis()->SetBinLabel(12,"B1-ME3,B1-ME1-");
//   h->GetXaxis()->SetBinLabel(13,"B1-ME2(-3)");
//   h->GetXaxis()->SetBinLabel(14,"ME1-4");
//   h->GetXaxis()->SetBinLabel(15,"B1-ME1(-2)(-3)");
//   h->GetXaxis()->SetBinLabel(16,"Halo Trigger");
// }

// ================================================================================================
std::pair<float, float> 
GEMCSCTriggerRateTree::intersectionEtaPhi(CSCDetId id, int wg, int hs)
{
  const CSCDetId layerId(id.endcap(), id.station(), id.ring(), id.chamber(), CSCConstants::KEY_CLCT_LAYER);
  const CSCLayer* csclayer(cscGeometry->layer(layerId));
  const CSCLayerGeometry* layer_geo(csclayer->geometry());
    
  // LCT::getKeyWG() starts from 0
  const float wire(layer_geo->middleWireOfGroup(wg + 1));

  // half-strip to strip
  // note that LCT's HS starts from 0, but in geometry strips start from 1
  const float fractional_strip(0.5 * (hs + 1) - 0.25);
  const LocalPoint csc_intersect(layer_geo->intersectionOfStripAndWire(fractional_strip, wire));
  const GlobalPoint csc_gp(cscGeometry->idToDet(layerId)->surface().toGlobal(csc_intersect));
  return std::make_pair(csc_gp.eta(), csc_gp.phi());
}

// ================================================================================================
csctf::TrackStub 
GEMCSCTriggerRateTree::buildTrackStub(const CSCCorrelatedLCTDigi &d, CSCDetId id)
{
  const unsigned fpga((id.station() == 1) ? CSCTriggerNumbering::triggerSubSectorFromLabels(id) - 1 : id.station());
  const CSCSectorReceiverLUT* srLUT(srLUTs_[fpga][id.triggerSector()-1][id.endcap()-1]);
  const unsigned cscid(CSCTriggerNumbering::triggerCscIdFromLabels(id));
  const unsigned cscid_special(id.station()==1 and id.ring()==4 ? cscid + 9 : cscid);
  const lclphidat lclPhi(srLUT->localPhi(d.getStrip(), d.getPattern(), d.getQuality(), d.getBend()));
  const gblphidat gblPhi(srLUT->globalPhiME(lclPhi.phi_local, d.getKeyWG(), cscid_special));
  const gbletadat gblEta(srLUT->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, d.getKeyWG(), cscid));
  return csctf::TrackStub(d, id, gblPhi.global_phi, gblEta.global_eta);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMCSCTriggerRateTree);
