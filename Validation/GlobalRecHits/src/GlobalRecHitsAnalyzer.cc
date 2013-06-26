/** \file GlobalRecHitsAnalyzer.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2013/05/28 18:03:30 $
 *  $Revision: 1.21 $
 *  \author M. Strang SUNY-Buffalo
 *  Testing by Ken Smith
 */
using namespace std;
#include "Validation/GlobalRecHits/interface/GlobalRecHitsAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

GlobalRecHitsAnalyzer::GlobalRecHitsAnalyzer(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), frequency(0), label(""), getAllProvenances(false),
  printProvenanceInfo(false), count(0)
{
  std::string MsgLoggerCat = "GlobalRecHitsAnalyzer_GlobalRecHitsAnalyzer";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");
  hitsProducer = iPSet.getParameter<std::string>("hitsProducer");

  //get Labels to use to extract information
  ECalEBSrc_ = iPSet.getParameter<edm::InputTag>("ECalEBSrc");
  ECalUncalEBSrc_ = iPSet.getParameter<edm::InputTag>("ECalUncalEBSrc");
  ECalEESrc_ = iPSet.getParameter<edm::InputTag>("ECalEESrc");
  ECalUncalEESrc_ = iPSet.getParameter<edm::InputTag>("ECalUncalEESrc");
  ECalESSrc_ = iPSet.getParameter<edm::InputTag>("ECalESSrc");
  HCalSrc_ = iPSet.getParameter<edm::InputTag>("HCalSrc");
  SiStripSrc_ = iPSet.getParameter<edm::InputTag>("SiStripSrc"); 
  SiPxlSrc_ = iPSet.getParameter<edm::InputTag>("SiPxlSrc");
  MuDTSrc_ = iPSet.getParameter<edm::InputTag>("MuDTSrc");
  MuDTSimSrc_ = iPSet.getParameter<edm::InputTag>("MuDTSimSrc");
  MuCSCSrc_ = iPSet.getParameter<edm::InputTag>("MuCSCSrc");
  MuRPCSrc_ = iPSet.getParameter<edm::InputTag>("MuRPCSrc");
  MuRPCSimSrc_ = iPSet.getParameter<edm::InputTag>("MuRPCSimSrc");

  conf_ = iPSet;

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // create persistent object
  // produces<PGlobalRecHit>(label);

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDProducer with parameter values:\n"
      << "    Name           = " << fName << "\n"
      << "    Verbosity      = " << verbosity << "\n"
      << "    Frequency      = " << frequency << "\n"
      << "    GetProv        = " << getAllProvenances << "\n"
      << "    PrintProv      = " << printProvenanceInfo << "\n"
      << "    ECalEBSrc      = " << ECalEBSrc_.label() 
      << ":" << ECalEBSrc_.instance() << "\n"
      << "    ECalUncalEBSrc = " << ECalUncalEBSrc_.label() 
      << ":" << ECalUncalEBSrc_.instance() << "\n"
      << "    ECalEESrc      = " << ECalEESrc_.label() 
      << ":" << ECalUncalEESrc_.instance() << "\n"
      << "    ECalUncalEESrc = " << ECalUncalEESrc_.label() 
      << ":" << ECalEESrc_.instance() << "\n"
      << "    ECalESSrc      = " << ECalESSrc_.label() 
      << ":" << ECalESSrc_.instance() << "\n"
      << "    HCalSrc        = " << HCalSrc_.label() 
      << ":" << HCalSrc_.instance() << "\n"
      << "    SiStripSrc     = " << SiStripSrc_.label() 
      << ":" << SiStripSrc_.instance() << "\n" 
      << "    SiPixelSrc     = " << SiPxlSrc_.label()
      << ":" << SiPxlSrc_.instance() << "\n"
      << "    MuDTSrc        = " << MuDTSrc_.label()
      << ":" << MuDTSrc_.instance() << "\n"
      << "    MuDTSimSrc     = " << MuDTSimSrc_.label()
      << ":" << MuDTSimSrc_.instance() << "\n"
      << "    MuCSCSrc       = " << MuCSCSrc_.label()
      << ":" << MuCSCSrc_.instance() << "\n"
      << "    MuRPCSrc       = " << MuRPCSrc_.label()
      << ":" << MuRPCSrc_.instance() << "\n"
      << "    MuRPCSimSrc    = " << MuRPCSimSrc_.label()
      << ":" << MuRPCSimSrc_.instance() << "\n"
      << "===============================\n";
  }
  //Put in analyzer stuff here....

  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
  if (dbe) {
    if (verbosity > 0 ) {
      dbe->setVerbose(1);
    } else {
      dbe->setVerbose(0);
    }
  }
  if (dbe) {
    if (verbosity > 0 ) dbe->showDirStructure();
  }

  //monitor elements 
  
  //Si Strip
  if(dbe) {
    string SiStripString[19] = {"TECW1", "TECW2", "TECW3", "TECW4", "TECW5", 
				"TECW6", "TECW7", "TECW8", "TIBL1", "TIBL2", 
				"TIBL3", "TIBL4", "TIDW1", "TIDW2", "TIDW3", 
				"TOBL1", "TOBL2", "TOBL3", "TOBL4"};
    for(int i = 0; i<19; ++i) {
      mehSiStripn[i]=0;
      mehSiStripResX[i]=0;
      mehSiStripResY[i]=0;
    }
    string hcharname, hchartitle;
    dbe->setCurrentFolder("GlobalRecHitsV/SiStrips");
    for(int amend = 0; amend < 19; ++amend) { 
      hcharname = "hSiStripn_"+SiStripString[amend];
      hchartitle= SiStripString[amend]+"  rechits";
      mehSiStripn[amend] = dbe->book1D(hcharname,hchartitle,200,0.,200.);
      mehSiStripn[amend]->setAxisTitle("Number of hits in "+
				       SiStripString[amend],1);
      mehSiStripn[amend]->setAxisTitle("Count",2);
      hcharname = "hSiStripResX_"+SiStripString[amend];
      hchartitle= SiStripString[amend]+" rechit x resolution";
      mehSiStripResX[amend] = dbe->book1D(hcharname,hchartitle,200,-0.02,.02);
      mehSiStripResX[amend]->setAxisTitle("X-resolution in "
					  +SiStripString[amend],1);
      mehSiStripResX[amend]->setAxisTitle("Count",2);
      hcharname = "hSiStripResY_"+SiStripString[amend];
      hchartitle= SiStripString[amend]+" rechit y resolution";
      mehSiStripResY[amend] = dbe->book1D(hcharname,hchartitle,200,-0.02,.02);
      mehSiStripResY[amend]->setAxisTitle("Y-resolution in "+
					  SiStripString[amend],1);
      mehSiStripResY[amend]->setAxisTitle("Count",2);
    }
    
    
    //HCal
    //string hcharname, hchartitle;
    string HCalString[4]={"HB", "HE", "HF", "HO"};
    float HCalnUpper[4]={3000.,3000.,3000.,3000.}; 
    float HCalnLower[4]={0.,0.,0.,0.};
    for(int j =0; j <4; ++j) {
      mehHcaln[j]=0;
      mehHcalRes[j]=0;
    }
    
    dbe->setCurrentFolder("GlobalRecHitsV/HCals");
    for(int amend = 0; amend < 4; ++amend) {
      hcharname = "hHcaln_"+HCalString[amend];
      hchartitle= HCalString[amend]+"  rechits";
      mehHcaln[amend] = dbe->book1D(hcharname,hchartitle, 1000, HCalnLower[amend], 
				    HCalnUpper[amend]);
      mehHcaln[amend]->setAxisTitle("Number of RecHits",1);
      mehHcaln[amend]->setAxisTitle("Count",2);
      hcharname = "hHcalRes_"+HCalString[amend];
      hchartitle= HCalString[amend]+"  rechit resolution";
      mehHcalRes[amend] = dbe->book1D(hcharname,hchartitle, 25, -2., 2.);
      mehHcalRes[amend]->setAxisTitle("RecHit E - SimHit E",1);
      mehHcalRes[amend]->setAxisTitle("Count",2);
    }
    
    
    //Ecal
    string ECalString[3] = {"EB","EE", "ES"}; 
    int ECalnBins[3] = {1000,3000,150};
    float ECalnUpper[3] = {20000., 62000., 3000.};
    float ECalnLower[3] = {0., 0., 0.};
    int ECalResBins[3] = {200,200,200};
    float ECalResUpper[3] = {1., 0.3, .0002};
    float ECalResLower[3] = {-1., -0.3, -.0002};
    for(int i =0; i<3; ++i) {
      mehEcaln[i]=0;
      mehEcalRes[i]=0;
    }
    dbe->setCurrentFolder("GlobalRecHitsV/ECals");
    
    for(int amend = 0; amend < 3; ++amend) {
      hcharname = "hEcaln_"+ECalString[amend];
      hchartitle= ECalString[amend]+"  rechits";
      mehEcaln[amend] = dbe->book1D(hcharname,hchartitle, ECalnBins[amend], 
				    ECalnLower[amend], ECalnUpper[amend]);
      mehEcaln[amend]->setAxisTitle("Number of RecHits",1);
      mehEcaln[amend]->setAxisTitle("Count",2);
      hcharname = "hEcalRes_"+ECalString[amend];
      hchartitle= ECalString[amend]+"  rechit resolution";
      mehEcalRes[amend] = dbe->book1D(hcharname,hchartitle,ECalResBins[amend], 
				      ECalResLower[amend], 
				      ECalResUpper[amend]);
      mehEcalRes[amend]->setAxisTitle("RecHit E - SimHit E",1);
      mehEcalRes[amend]->setAxisTitle("Count",2);
    }
    
    //Si Pixels
    string SiPixelString[7] = {"BRL1", "BRL2", "BRL3", "FWD1n", "FWD1p", 
			       "FWD2n", "FWD2p"};
    for(int j =0; j<7; ++j) {
      mehSiPixeln[j]=0;
      mehSiPixelResX[j]=0;
      mehSiPixelResY[j]=0;
    }
    
    dbe->setCurrentFolder("GlobalRecHitsV/SiPixels");
    for(int amend = 0; amend < 7; ++amend) {
      hcharname = "hSiPixeln_"+SiPixelString[amend];
      hchartitle= SiPixelString[amend]+" rechits";
      mehSiPixeln[amend] = dbe->book1D(hcharname,hchartitle,200,0.,200.);
      mehSiPixeln[amend]->setAxisTitle("Number of hits in "+
				       SiPixelString[amend],1);
      mehSiPixeln[amend]->setAxisTitle("Count",2);
      hcharname = "hSiPixelResX_"+SiPixelString[amend];
      hchartitle= SiPixelString[amend]+" rechit x resolution";
      mehSiPixelResX[amend] = dbe->book1D(hcharname,hchartitle,200,-0.02,.02);
      mehSiPixelResX[amend]->setAxisTitle("X-resolution in "+
					  SiPixelString[amend],1);
      mehSiPixelResX[amend]->setAxisTitle("Count",2);
      hcharname = "hSiPixelResY_"+SiPixelString[amend];
      hchartitle= SiPixelString[amend]+" rechit y resolution";
      
      mehSiPixelResY[amend] = dbe->book1D(hcharname,hchartitle,200,-0.02,.02);
      mehSiPixelResY[amend]->setAxisTitle("Y-resolution in "+
					  SiPixelString[amend],1);
      mehSiPixelResY[amend]->setAxisTitle("Count",2);
    }
 
    //Muons 
    dbe->setCurrentFolder("GlobalRecHitsV/Muons");
    
    mehDtMuonn = 0;
    mehCSCn = 0;
    mehRPCn = 0;
    
    string n_List[3] = {"hDtMuonn", "hCSCn", "hRPCn"};
    string hist_string[3] = {"Dt", "CSC", "RPC"};
    
    for(int amend=0; amend<3; ++amend) {
      hchartitle = hist_string[amend]+" rechits";
      if(amend==0) {
	mehDtMuonn=dbe->book1D(n_List[amend],hchartitle,50, 0., 500.);
	mehDtMuonn->setAxisTitle("Number of Rechits",1);
	mehDtMuonn->setAxisTitle("Count",2);
      }
      if(amend==1) {
	mehCSCn=dbe->book1D(n_List[amend],hchartitle,50, 0., 500.);
	mehCSCn->setAxisTitle("Number of Rechits",1);
	mehCSCn->setAxisTitle("Count",2);
      }
      if(amend==2){
	mehRPCn=dbe->book1D(n_List[amend],hchartitle,50, 0., 500.);
	mehRPCn->setAxisTitle("Number of Rechits",1);
	mehRPCn->setAxisTitle("Count",2);
      }
    }
    
    mehDtMuonRes=0;
    mehCSCResRDPhi=0;
    mehRPCResX=0;
    
    hcharname = "hDtMuonRes";
    hchartitle = "DT wire distance resolution";
    mehDtMuonRes = dbe->book1D(hcharname, hchartitle, 200, -0.2, 0.2);
    hcharname = "CSCResRDPhi";
    hchartitle = "CSC perp*dphi resolution";
    mehCSCResRDPhi = dbe->book1D(hcharname, hchartitle, 200, -0.2, 0.2);
    hcharname = "hRPCResX";
    hchartitle = "RPC rechits x resolution";
    mehRPCResX = dbe->book1D(hcharname, hchartitle, 50, -5., 5.);
  } 
}

GlobalRecHitsAnalyzer::~GlobalRecHitsAnalyzer() {}

void GlobalRecHitsAnalyzer::beginJob()
{
  return;
}

void GlobalRecHitsAnalyzer::endJob()
{
  std::string MsgLoggerCat = "GlobalRecHitsAnalyzer_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " events.";
  return;
}

void GlobalRecHitsAnalyzer::analyze(const edm::Event& iEvent, 
				    const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsAnalyzer_analyze";
  
  // keep track of number of events processed
  ++count;
  
  // get event id information
  int nrun = iEvent.id().run();
  int nevt = iEvent.id().event();
  
  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << ", event " << nevt
      << " (" << count << " events total)";
  } else if (verbosity == 0) {
    if (nevt%frequency == 0 || nevt == 1) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << ", event " << nevt
	<< " (" << count << " events total)";
    }
  }
  
  // look at information available in the event
  if (getAllProvenances) {
    
    std::vector<const edm::Provenance*> AllProv;
    iEvent.getAllProvenance(AllProv);
    
    if (verbosity >= 0)
      edm::LogInfo(MsgLoggerCat)
	<< "Number of Provenances = " << AllProv.size();
    
    if (printProvenanceInfo && (verbosity >= 0)) {
      TString eventout("\nProvenance info:\n");      
      
      for (unsigned int i = 0; i < AllProv.size(); ++i) {
	eventout += "\n       ******************************";
	eventout += "\n       Module       : ";
	eventout += AllProv[i]->moduleLabel();
	eventout += "\n       ProductID    : ";
	eventout += AllProv[i]->productID().id();
	eventout += "\n       ClassName    : ";
	eventout += AllProv[i]->className();
	eventout += "\n       InstanceName : ";
	eventout += AllProv[i]->productInstanceName();
	eventout += "\n       BranchName   : ";
	eventout += AllProv[i]->branchName();
      }
      eventout += "\n       ******************************\n";
      edm::LogInfo(MsgLoggerCat) << eventout << "\n";
      printProvenanceInfo = false;
    }
    getAllProvenances = false;
  }
  
  // call fill functions
  // gather Ecal information from event
  fillECal(iEvent, iSetup);
  // gather Hcal information from event
  fillHCal(iEvent, iSetup);
  // gather Track information from event
  fillTrk(iEvent, iSetup);
  // gather Muon information from event
  fillMuon(iEvent, iSetup);
  
  if (verbosity > 0)
    edm::LogInfo (MsgLoggerCat)
      << "Done gathering data from event.";
  
  return;
}

void GlobalRecHitsAnalyzer::fillECal(const edm::Event& iEvent, 
				     const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsAnalyzer_fillECal";
  
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  
  
  // extract crossing frame from event
  edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;
  
  ////////////////////////
  //extract EB information
  ////////////////////////
  edm::Handle<EBUncalibratedRecHitCollection> EcalUncalibRecHitEB;
  iEvent.getByLabel(ECalUncalEBSrc_, EcalUncalibRecHitEB);
  bool validUncalibRecHitEB = true;
  if (!EcalUncalibRecHitEB.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find EcalUncalRecHitEB in event!";
    validUncalibRecHitEB = false;
  }  
  
  edm::Handle<EBRecHitCollection> EcalRecHitEB;
  iEvent.getByLabel(ECalEBSrc_, EcalRecHitEB);
  bool validRecHitEB = true;
  if (!EcalRecHitEB.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find EcalRecHitEB in event!";
    validRecHitEB = false;
  }  

  // loop over simhits
  const std::string barrelHitsName(hitsProducer+"EcalHitsEB");
  iEvent.getByLabel("mix",barrelHitsName,crossingFrame);
  bool validXFrame = true;
  if (!crossingFrame.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find cal barrel crossingFrame in event!";
    validXFrame = false;
  }

  MapType ebSimMap;
  if (validXFrame) {
    std::auto_ptr<MixCollection<PCaloHit> >
      barrelHits(new MixCollection<PCaloHit>(crossingFrame.product()));  
    
    // keep track of sum of simhit energy in each crystal
    for (MixCollection<PCaloHit>::MixItr hitItr 
	   = barrelHits->begin();
	 hitItr != barrelHits->end();
	 ++hitItr) {
      
      EBDetId ebid = EBDetId(hitItr->id());
      
      uint32_t crystid = ebid.rawId();
      ebSimMap[crystid] += hitItr->energy();
    }
  }  

  int nEBRecHits = 0;
  // loop over RecHits
  if (validUncalibRecHitEB && validRecHitEB) {
    const EBUncalibratedRecHitCollection *EBUncalibRecHit = 
      EcalUncalibRecHitEB.product();
    const EBRecHitCollection *EBRecHit = EcalRecHitEB.product();
  
    for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit =
	   EBUncalibRecHit->begin();
	 uncalibRecHit != EBUncalibRecHit->end();
	 ++uncalibRecHit) {
      
      EBDetId EBid = EBDetId(uncalibRecHit->id());
      
      EcalRecHitCollection::const_iterator myRecHit = EBRecHit->find(EBid);
      
      if (myRecHit != EBRecHit->end()) {
	++nEBRecHits;
	mehEcalRes[1]->Fill(myRecHit->energy()-ebSimMap[EBid.rawId()]);
      }
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of EBRecHits collected:............ ";
      eventout += nEBRecHits;
    }
    mehEcaln[1]->Fill((float)nEBRecHits); 
  }

  ////////////////////////
  //extract EE information
  ////////////////////////
  edm::Handle<EEUncalibratedRecHitCollection> EcalUncalibRecHitEE;
  iEvent.getByLabel(ECalUncalEESrc_, EcalUncalibRecHitEE);
  bool validuncalibRecHitEE = true;
  if (!EcalUncalibRecHitEE.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find EcalUncalRecHitEE in event!";
    validuncalibRecHitEE = false;
  }  
  
  edm::Handle<EERecHitCollection> EcalRecHitEE;
  iEvent.getByLabel(ECalEESrc_, EcalRecHitEE);
  bool validRecHitEE = true;
  if (!EcalRecHitEE.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find EcalRecHitEE in event!";
    validRecHitEE = false;
  }  
  
  // loop over simhits
  const std::string endcapHitsName(hitsProducer+"EcalHitsEE");
  iEvent.getByLabel("mix",endcapHitsName,crossingFrame);
  validXFrame = true;
  if (!crossingFrame.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find cal endcap crossingFrame in event!";
    validXFrame = false;
  }

  MapType eeSimMap;
  if (validXFrame) {
    std::auto_ptr<MixCollection<PCaloHit> >
      endcapHits(new MixCollection<PCaloHit>(crossingFrame.product()));  
    
    // keep track of sum of simhit energy in each crystal
    for (MixCollection<PCaloHit>::MixItr hitItr 
	   = endcapHits->begin();
	 hitItr != endcapHits->end();
	 ++hitItr) {
      
      EEDetId eeid = EEDetId(hitItr->id());
      
      uint32_t crystid = eeid.rawId();
      eeSimMap[crystid] += hitItr->energy();
    }
  }    

  int nEERecHits = 0;
  if (validuncalibRecHitEE && validRecHitEE) {
    // loop over RecHits
    const EEUncalibratedRecHitCollection *EEUncalibRecHit = 
      EcalUncalibRecHitEE.product();
    const EERecHitCollection *EERecHit = EcalRecHitEE.product();
    
    for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit =
	   EEUncalibRecHit->begin();
	 uncalibRecHit != EEUncalibRecHit->end();
	 ++uncalibRecHit) {
      
      EEDetId EEid = EEDetId(uncalibRecHit->id());
      
      EcalRecHitCollection::const_iterator myRecHit = EERecHit->find(EEid);
      
      if (myRecHit != EERecHit->end()) {
	++nEERecHits;
	mehEcalRes[0]->Fill(myRecHit->energy()-eeSimMap[EEid.rawId()]);
      }
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of EERecHits collected:............ ";
      eventout += nEERecHits;
    }
    mehEcaln[0]->Fill((float)nEERecHits);
  }    

  ////////////////////////
  //extract ES information
  ////////////////////////
  edm::Handle<ESRecHitCollection> EcalRecHitES;
  iEvent.getByLabel(ECalESSrc_, EcalRecHitES);
  bool validRecHitES = true;
  if (!EcalRecHitES.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find EcalRecHitES in event!";
    validRecHitES = false;
  }  

  // loop over simhits
  const std::string preshowerHitsName(hitsProducer+"EcalHitsES");
  iEvent.getByLabel("mix",preshowerHitsName,crossingFrame);
  validXFrame = true;
  if (!crossingFrame.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find cal preshower crossingFrame in event!";
    validXFrame = false;
  }

  MapType esSimMap;
  if (validXFrame) {
    std::auto_ptr<MixCollection<PCaloHit> >
      preshowerHits(new MixCollection<PCaloHit>(crossingFrame.product()));  
    
    // keep track of sum of simhit energy in each crystal
    for (MixCollection<PCaloHit>::MixItr hitItr 
	   = preshowerHits->begin();
	 hitItr != preshowerHits->end();
	 ++hitItr) {
      
      ESDetId esid = ESDetId(hitItr->id());
      
      uint32_t crystid = esid.rawId();
      esSimMap[crystid] += hitItr->energy();
    }
  }

  int nESRecHits = 0;
  if (validRecHitES) {
    // loop over RecHits
    const ESRecHitCollection *ESRecHit = EcalRecHitES.product();
    for (EcalRecHitCollection::const_iterator recHit =
	   ESRecHit->begin();
	 recHit != ESRecHit->end();
	 ++recHit) {
      
      ESDetId ESid = ESDetId(recHit->id());
      
      ++nESRecHits;
      mehEcalRes[2]->Fill(recHit->energy()-esSimMap[ESid.rawId()]);
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of ESRecHits collected:............ ";
      eventout += nESRecHits;
    }
    mehEcaln[2]->Fill(float(nESRecHits));
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  
  return;
}

void GlobalRecHitsAnalyzer::fillHCal(const edm::Event& iEvent, 
				     const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsAnalyzer_fillHCal";
  
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  
  
  // get geometry
  edm::ESHandle<CaloGeometry> geometry;
  iSetup.get<CaloGeometryRecord>().get(geometry);
  if (!geometry.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find CaloGeometry in event!";
    return;
  }
  
  // iterator to access containers
  edm::PCaloHitContainer::const_iterator itHit;
  
  ///////////////////////
  // extract simhit info
  //////////////////////
  edm::Handle<edm::PCaloHitContainer> hcalHits;
  iEvent.getByLabel(HCalSrc_,hcalHits);
  bool validhcalHits = true;
  if (!hcalHits.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find hcalHits in event!";
    validhcalHits = false;
  }  

  MapType fHBEnergySimHits;
  MapType fHEEnergySimHits;
  MapType fHOEnergySimHits;
  MapType fHFEnergySimHits;
  if (validhcalHits) {
    const edm::PCaloHitContainer *simhitResult = hcalHits.product();
  
    for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin();
	 simhits != simhitResult->end();
	 ++simhits) {
      
      HcalDetId detId(simhits->id());
      uint32_t cellid = detId.rawId();
      
      if (detId.subdet() == sdHcalBrl){  
	fHBEnergySimHits[cellid] += simhits->energy(); 
      }
      if (detId.subdet() == sdHcalEC){  
	fHEEnergySimHits[cellid] += simhits->energy(); 
      }    
      if (detId.subdet() == sdHcalOut){  
	fHOEnergySimHits[cellid] += simhits->energy(); 
      }    
      if (detId.subdet() == sdHcalFwd){  
	fHFEnergySimHits[cellid] += simhits->energy(); 
      }    
    }
  }

  // max values to be used (HO is found in HB)
  Double_t maxHBEnergy = 0.;
  Double_t maxHEEnergy = 0.;
  Double_t maxHFEnergy = 0.;
  
  Double_t maxHBPhi = -1000.;
  Double_t maxHEPhi = -1000.;
  Double_t maxHOPhi = -1000.;
  Double_t maxHFPhi = -1000.;
  
  
  Double_t PI = 3.141592653589;
  
  ////////////////////////
  // get HBHE information
  ///////////////////////
  std::vector<edm::Handle<HBHERecHitCollection> > hbhe;
  iEvent.getManyByType(hbhe);
  bool validHBHE = true;
  if (!hbhe[0].isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find any HBHERecHitCollections in event!";
    validHBHE = false;
  } 

  if (validHBHE) {
    std::vector<edm::Handle<HBHERecHitCollection> >::iterator ihbhe;
    
    int iHB = 0;
    int iHE = 0; 
    for (ihbhe = hbhe.begin(); ihbhe != hbhe.end(); ++ihbhe) {
      
      // find max values
      for (HBHERecHitCollection::const_iterator jhbhe = (*ihbhe)->begin();
	   jhbhe != (*ihbhe)->end(); ++jhbhe) {
	
	HcalDetId cell(jhbhe->id());
	
	if (cell.subdet() == sdHcalBrl) {
	  
	  const CaloCellGeometry* cellGeometry =
	    geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	  double fPhi = cellGeometry->getPosition().phi () ;
	  if ( (jhbhe->energy()) > maxHBEnergy ) {
	    maxHBEnergy = jhbhe->energy();
	    maxHBPhi = fPhi;
	    maxHOPhi = maxHBPhi;
	  }	  
	}
      
	if (cell.subdet() == sdHcalEC) {
	  
	  const CaloCellGeometry* cellGeometry =
	    geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	  double fPhi = cellGeometry->getPosition().phi () ;
	  if ( (jhbhe->energy()) > maxHEEnergy ) {
	    maxHEEnergy = jhbhe->energy();
	    maxHEPhi = fPhi;
	  }	  
	}
      } // end find max values
      
      for (HBHERecHitCollection::const_iterator jhbhe = (*ihbhe)->begin();
	   jhbhe != (*ihbhe)->end(); ++jhbhe) {
	
	HcalDetId cell(jhbhe->id());
	
	if (cell.subdet() == sdHcalBrl) {
	  
	  ++iHB;
	  
	  const CaloCellGeometry* cellGeometry =
	    geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	  double fPhi = cellGeometry->getPosition().phi () ;
	  
	  float deltaphi = maxHBPhi - fPhi;
	  if (fPhi > maxHBPhi) { deltaphi = fPhi - maxHBPhi;}
	  if (deltaphi > PI) { deltaphi = 2.0 * PI - deltaphi;}
	  
	  mehHcalRes[0]->Fill(jhbhe->energy() - 
			      fHBEnergySimHits[cell.rawId()]);
	}
	
	if (cell.subdet() == sdHcalEC) {
	  
	  ++iHE;
	  
	  const CaloCellGeometry* cellGeometry =
	    geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	  double fPhi = cellGeometry->getPosition().phi () ;
	  
	  float deltaphi = maxHEPhi - fPhi;
	  if (fPhi > maxHEPhi) { deltaphi = fPhi - maxHEPhi;}
	  if (deltaphi > PI) { deltaphi = 2.0 * PI - deltaphi;}
	  mehHcalRes[1]->Fill(jhbhe->energy() - 
			      fHEEnergySimHits[cell.rawId()]);
	}
      }
    } // end loop through collection
  
    
    if (verbosity > 1) {
      eventout += "\n          Number of HBRecHits collected:............ ";
      eventout += iHB;
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of HERecHits collected:............ ";
      eventout += iHE;
    }
    mehHcaln[0]->Fill((float)iHB);
    mehHcaln[1]->Fill((float)iHE);
  }

  ////////////////////////
  // get HF information
  ///////////////////////
  std::vector<edm::Handle<HFRecHitCollection> > hf;
  iEvent.getManyByType(hf);
  bool validHF = true;
  if (!hf[0].isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find any HFRecHitCollections in event!";
    validHF = false;
  } 
  if (validHF) {
    std::vector<edm::Handle<HFRecHitCollection> >::iterator ihf;
    
    int iHF = 0; 
    for (ihf = hf.begin(); ihf != hf.end(); ++ihf) {
      
      // find max values
      for (HFRecHitCollection::const_iterator jhf = (*ihf)->begin();
	   jhf != (*ihf)->end(); ++jhf) {
	
	HcalDetId cell(jhf->id());
	
	if (cell.subdet() == sdHcalFwd) {
	  
	  const CaloCellGeometry* cellGeometry =
	    geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	  double fPhi = cellGeometry->getPosition().phi () ;
	  if ( (jhf->energy()) > maxHFEnergy ) {
	    maxHFEnergy = jhf->energy();
	    maxHFPhi = fPhi;
	  }	  
	}
      } // end find max values
      
      for (HFRecHitCollection::const_iterator jhf = (*ihf)->begin();
	   jhf != (*ihf)->end(); ++jhf) {
	
	HcalDetId cell(jhf->id());
	
	if (cell.subdet() == sdHcalFwd) {
	  
	  ++iHF;
	  
	  const CaloCellGeometry* cellGeometry =
	    geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	  double fPhi = cellGeometry->getPosition().phi () ;
	  
	  float deltaphi = maxHBPhi - fPhi;
	  if (fPhi > maxHFPhi) { deltaphi = fPhi - maxHFPhi;}
	  if (deltaphi > PI) { deltaphi = 2.0 * PI - deltaphi;}
	  
	  mehHcalRes[2]->Fill(jhf->energy()-fHFEnergySimHits[cell.rawId()]);
	}
      }
    } // end loop through collection
    
    if (verbosity > 1) {
      eventout += "\n          Number of HFDigis collected:.............. ";
      eventout += iHF;
    }
    mehHcaln[2]->Fill((float)iHF);
  }    

  ////////////////////////
  // get HO information
  ///////////////////////
  std::vector<edm::Handle<HORecHitCollection> > ho;
  iEvent.getManyByType(ho);
  bool validHO = true;
  if (!ho[0].isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find any HORecHitCollections in event!";
    validHO = false;
  } 

  if (validHO) {
    std::vector<edm::Handle<HORecHitCollection> >::iterator iho;
    
    int iHO = 0; 
    for (iho = ho.begin(); iho != ho.end(); ++iho) {
      
      for (HORecHitCollection::const_iterator jho = (*iho)->begin();
	   jho != (*iho)->end(); ++jho) {
	
	HcalDetId cell(jho->id());
	
	if (cell.subdet() == sdHcalOut) {
	  
	  ++iHO;
	  
	  const CaloCellGeometry* cellGeometry =
	    geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	  double fPhi = cellGeometry->getPosition().phi () ;
	  
	  float deltaphi = maxHOPhi - fPhi;
	  if (fPhi > maxHOPhi) { deltaphi = fPhi - maxHOPhi;}
	  if (deltaphi > PI) { deltaphi = 2.0 * PI - deltaphi;}
	  mehHcalRes[3]->Fill(jho->energy()-fHOEnergySimHits[cell.rawId()]);
	}
      }
    } // end loop through collection
    
    if (verbosity > 1) {
      eventout += "\n          Number of HODigis collected:.............. ";
      eventout += iHO;
    }
    mehHcaln[3]->Fill((float)iHO);
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  
  return;
}

void GlobalRecHitsAnalyzer::fillTrk(const edm::Event& iEvent, 
				    const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();


  std::string MsgLoggerCat = "GlobalRecHitsAnalyzer_fillTrk";
  
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  
  
  // get strip information
  edm::Handle<SiStripMatchedRecHit2DCollection> rechitsmatched;
  iEvent.getByLabel(SiStripSrc_, rechitsmatched);
  bool validstrip = true;
  if (!rechitsmatched.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find stripmatchedrechits in event!";
    validstrip = false;
  }  
  
  TrackerHitAssociator associate(iEvent,conf_);
  
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  if (!pDD.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerDigiGeometry in event!";
    return;
  }
  const TrackerGeometry &tracker(*pDD);
  
  if (validstrip) {
    int nStripBrl = 0, nStripFwd = 0;
    
    // loop over det units
    for (TrackerGeometry::DetContainer::const_iterator it = 
	   pDD->dets().begin();
	 it != pDD->dets().end(); ++it) {
      
      uint32_t myid = ((*it)->geographicalId()).rawId();
      DetId detid = ((*it)->geographicalId());
      
      //loop over rechits-matched in the same subdetector
      SiStripMatchedRecHit2DCollection::const_iterator rechitmatchedMatch = rechitsmatched->find(detid);
      
      if (rechitmatchedMatch != rechitsmatched->end()) {
        SiStripMatchedRecHit2DCollection::DetSet rechitmatchedRange = *rechitmatchedMatch;
        SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitmatchedRangeIteratorBegin = rechitmatchedRange.begin();
        SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitmatchedRangeIteratorEnd   = rechitmatchedRange.end();
        SiStripMatchedRecHit2DCollection::DetSet::const_iterator itermatched = rechitmatchedRangeIteratorBegin;
	
	for ( itermatched = rechitmatchedRangeIteratorBegin; 
	      itermatched != rechitmatchedRangeIteratorEnd;
	      ++itermatched) {
	  
	  SiStripMatchedRecHit2D const rechit = *itermatched;
	  LocalPoint position = rechit.localPosition();
	  
	  float mindist = 999999.;
	  float distx = 999999.;
	  float disty = 999999.;
	  float dist = 999999.;
	  std::pair<LocalPoint,LocalVector> closestPair;
	  matched.clear();
	  
	  float rechitmatchedx = position.x();
	  float rechitmatchedy = position.y();
	  
	  matched = associate.associateHit(rechit);
	  
	  if (!matched.empty()) {
	    //project simhit;
	    const GluedGeomDet* gluedDet = 
	      (const GluedGeomDet*)tracker.idToDet(rechit.geographicalId());
	    const StripGeomDetUnit* partnerstripdet =
	      (StripGeomDetUnit*) gluedDet->stereoDet();
	    std::pair<LocalPoint,LocalVector> hitPair;
	    
	    for(std::vector<PSimHit>::const_iterator m = matched.begin(); 
		m != matched.end(); m++){
	      //project simhit;
	      hitPair = projectHit((*m),partnerstripdet,gluedDet->surface());
	      distx = fabs(rechitmatchedx - hitPair.first.x());
	      disty = fabs(rechitmatchedy - hitPair.first.y());
	      dist = sqrt(distx*distx+disty*disty);
	      
	      if(dist < mindist){
		mindist = dist;
		closestPair = hitPair;
	      }
	    }
	    
	    // get TIB
	    if (detid.subdetId() == sdSiTIB) {
	      
	      
	      ++nStripBrl;
	      
	      if (tTopo->tibLayer(myid) == 1) {
		mehSiStripResX[8]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[8]->Fill(rechitmatchedy-closestPair.first.y());
	      }
	      if (tTopo->tibLayer(myid) == 2) {
		mehSiStripResX[9]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[9]->Fill(rechitmatchedy-closestPair.first.y());
	      }	
	      if (tTopo->tibLayer(myid) == 3) {
		mehSiStripResX[10]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[10]->Fill(rechitmatchedy-closestPair.first.y());
		
	      }
	      if (tTopo->tibLayer(myid) == 4) {
		mehSiStripResX[11]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[11]->Fill(rechitmatchedy-closestPair.first.y());
	      }
	    }
	    
	    // get TOB
	    if (detid.subdetId() == sdSiTOB) {
	      
	      
	      ++nStripBrl;
	      
	      if (tTopo->tobLayer(myid) == 1) {
		mehSiStripResX[15]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[15]->Fill(rechitmatchedy-closestPair.first.y());
	      }
	      if (tTopo->tobLayer(myid) == 2) {
		mehSiStripResX[16]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[16]->Fill(rechitmatchedy-closestPair.first.y());
	      }	
	      if (tTopo->tobLayer(myid) == 3) {
		mehSiStripResX[17]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[17]->Fill(rechitmatchedy-closestPair.first.y());
	      }
	      if (tTopo->tobLayer(myid) == 4) {
		mehSiStripResX[18]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[18]->Fill(rechitmatchedy-closestPair.first.y());
	      }
	    }
	    
	    // get TID
	    if (detid.subdetId() == sdSiTID) {
	      
	      
	      ++nStripFwd;
	      
	      if (tTopo->tidWheel(myid) == 1) {
		mehSiStripResX[12]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[12]->Fill(rechitmatchedy-closestPair.first.y());
	      }
	      if (tTopo->tidWheel(myid) == 2) {
		mehSiStripResX[13]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[13]->Fill(rechitmatchedy-closestPair.first.y());
	      }	
	      if (tTopo->tidWheel(myid) == 3) {
		mehSiStripResX[14]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[14]->Fill(rechitmatchedy-closestPair.first.y());
	      }
	    }
	    
	    // get TEC
	    if (detid.subdetId() == sdSiTEC) {
	      
	      
	      ++nStripFwd;
	      
	      if (tTopo->tecWheel(myid) == 1) {
		mehSiStripResX[0]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[0]->Fill(rechitmatchedy-closestPair.first.y());
	      }
	      if (tTopo->tecWheel(myid) == 2) {
		mehSiStripResX[1]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[1]->Fill(rechitmatchedy-closestPair.first.y());
	      }	
	      if (tTopo->tecWheel(myid) == 3) {
		mehSiStripResX[2]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[2]->Fill(rechitmatchedy-closestPair.first.y());
	      }
	      if (tTopo->tecWheel(myid) == 4) {
		mehSiStripResX[3]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[3]->Fill(rechitmatchedy-closestPair.first.y());
		
	      }
	      if (tTopo->tecWheel(myid) == 5) {
		mehSiStripResX[4]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[4]->Fill(rechitmatchedy-closestPair.first.y());
	      }	
	      if (tTopo->tecWheel(myid) == 6) {
		mehSiStripResX[5]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[5]->Fill(rechitmatchedy-closestPair.first.y());
	      }
	      if (tTopo->tecWheel(myid) == 7) {
		mehSiStripResX[6]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[6]->Fill(rechitmatchedy-closestPair.first.y());
	      }	
	      if (tTopo->tecWheel(myid) == 8) {
		mehSiStripResX[7]->Fill(rechitmatchedx-closestPair.first.x());
		mehSiStripResY[7]->Fill(rechitmatchedy-closestPair.first.y()); 
	      }
	    }
	    
	  } // end if matched empty
	}
      }
    } // end loop over det units
                                                                      
    if (verbosity > 1) {
      eventout += "\n          Number of BrlStripRecHits collected:...... ";
      eventout += nStripBrl;
    }
    
    for(int i =8; i<12; ++i)
      {mehSiStripn[i]->Fill((float)nStripBrl);}
    for(int i =16; i<19; ++i)
      {mehSiStripn[i]->Fill((float)nStripBrl);}
    
    if (verbosity > 1) {
      eventout += "\n          Number of FrwdStripRecHits collected:..... ";
      eventout += nStripFwd;
    }
    for(int i =0; i<8; ++i)
      {mehSiStripn[i]->Fill((float)nStripFwd);}
    for(int i =12; i<16; ++i)
      {mehSiStripn[i]->Fill((float)nStripFwd);}
  }

  // get pixel information
  //Get RecHits
  edm::Handle<SiPixelRecHitCollection> recHitColl;
  iEvent.getByLabel(SiPxlSrc_, recHitColl);
  bool validpixel = true;
  if (!recHitColl.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find SiPixelRecHitCollection in event!";
    validpixel = false;
  }  
  
  //Get event setup
  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get(geom); 
  if (!geom.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerDigiGeometry in event!";
    return;
  }

  if (validpixel) {
    int nPxlBrl = 0, nPxlFwd = 0;    
    //iterate over detunits
    for (TrackerGeometry::DetContainer::const_iterator it = 
	   geom->dets().begin();
	 it != geom->dets().end(); ++it) {
      
      uint32_t myid = ((*it)->geographicalId()).rawId();
      DetId detId = ((*it)->geographicalId());
      int subid = detId.subdetId();
      
      if (! ((subid == sdPxlBrl) || (subid == sdPxlFwd))) continue;
      
      SiPixelRecHitCollection::const_iterator pixeldet = recHitColl->find(detId);
      if (pixeldet == recHitColl->end()) continue;
      SiPixelRecHitCollection::DetSet pixelrechitRange = *pixeldet;
      SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.begin();
      SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorEnd   = pixelrechitRange.end();
      SiPixelRecHitCollection::DetSet::const_iterator pixeliter = pixelrechitRangeIteratorBegin;

      
      std::vector<PSimHit> matched;
      
      //----Loop over rechits for this detId
      for ( ; pixeliter != pixelrechitRangeIteratorEnd; ++pixeliter) {
	
	matched.clear();
	matched = associate.associateHit(*pixeliter);
	
	if ( !matched.empty() ) {
	  
	  float closest = 9999.9;
	  LocalPoint lp = pixeliter->localPosition();
	  float rechit_x = lp.x();
	  float rechit_y = lp.y();
	  
	  float sim_x = 0.;
	  float sim_y = 0.;
	  
	  //loop over sim hits and fill closet
	  for (std::vector<PSimHit>::const_iterator m = matched.begin(); 
	       m != matched.end(); ++m) {
	    
	    float sim_x1 = (*m).entryPoint().x();
	    float sim_x2 = (*m).exitPoint().x();
	    float sim_xpos = 0.5*(sim_x1+sim_x2);
	    
	    float sim_y1 = (*m).entryPoint().y();
	    float sim_y2 = (*m).exitPoint().y();
	    float sim_ypos = 0.5*(sim_y1+sim_y2);
	    
	    float x_res = fabs(sim_xpos - rechit_x);
	    float y_res = fabs(sim_ypos - rechit_y);
	    
	    float dist = sqrt(x_res*x_res + y_res*y_res);
	    
	    if ( dist < closest ) {
	      closest = dist;
	      sim_x = sim_xpos;
	      sim_y = sim_ypos;
	    }
	  } // end sim hit loop
	  
	  // get Barrel pixels ***************Pixel STuff******************
	  if (subid == sdPxlBrl) {
	    
	    ++nPxlBrl;
	    
	    if (tTopo->pxbLayer(myid) == 1) {
	      mehSiPixelResX[0]->Fill(rechit_x-sim_x);
	      mehSiPixelResY[0]->Fill(rechit_y-sim_y); 
	      
	    }
	    if (tTopo->pxbLayer(myid) == 2) {
	      mehSiPixelResX[1]->Fill(rechit_x-sim_x);
	      mehSiPixelResY[1]->Fill(rechit_y-sim_y); 
	    }
	    if (tTopo->pxbLayer(myid) == 3) {
	      mehSiPixelResX[2]->Fill(rechit_x-sim_x);
	      mehSiPixelResY[2]->Fill(rechit_y-sim_y); 
	    }
	  }
	  
	  // get Forward pixels
	  if (subid == sdPxlFwd) {
	    
	    ++nPxlFwd;
	    
	    if (tTopo->pxfDisk(myid) == 1) {
	      if (tTopo->pxfSide(myid) == 1) {
		mehSiPixelResX[3]->Fill(rechit_x-sim_x);
		mehSiPixelResY[3]->Fill(rechit_y-sim_y);
	      }
	      if (tTopo->pxfSide(myid) == 2) {
		mehSiPixelResX[4]->Fill(rechit_x-sim_x);
		mehSiPixelResY[4]->Fill(rechit_y-sim_y); 
	      }
	    }
	    if (tTopo->pxfDisk(myid) == 2) {
	      if (tTopo->pxfSide(myid) == 1) {
		mehSiPixelResX[5]->Fill(rechit_x-sim_x);
		mehSiPixelResY[5]->Fill(rechit_y-sim_y);
	      }
	      if (tTopo->pxfSide(myid) == 2) {
		mehSiPixelResX[6]->Fill(rechit_x-sim_x);
		mehSiPixelResY[6]->Fill(rechit_y-sim_y); 
	      }
	    }
	  }      
	} // end matched emtpy
      } // <-----end rechit loop 
    } // <------ end detunit loop  
    
    
    if (verbosity > 1) {
      eventout += "\n          Number of BrlPixelRecHits collected:...... ";
      eventout += nPxlBrl;
    }
    for(int i=0; i<3; ++i) {
      mehSiPixeln[i]->Fill((float)nPxlBrl);
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of FrwdPixelRecHits collected:..... ";
      eventout += nPxlFwd;
    }
    
    for(int i=3; i<7; ++i) {
      mehSiPixeln[i]->Fill((float)nPxlFwd);
    }
  }
   
  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  
  return;
}

void GlobalRecHitsAnalyzer::fillMuon(const edm::Event& iEvent, 
				     const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsAnalyzer_fillMuon";
  
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  

  // get DT information
  edm::ESHandle<DTGeometry> dtGeom;
  iSetup.get<MuonGeometryRecord>().get(dtGeom);
  if (!dtGeom.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find DTMuonGeometryRecord in event!";
    return;
  }  

  edm::Handle<edm::PSimHitContainer> dtsimHits;
  iEvent.getByLabel(MuDTSimSrc_, dtsimHits);
  bool validdtsim = true;
  if (!dtsimHits.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find dtsimHits in event!";
    validdtsim = false;
  } 

  edm::Handle<DTRecHitCollection> dtRecHits;
  iEvent.getByLabel(MuDTSrc_, dtRecHits);
  bool validdtrec = true;
  if (!dtRecHits.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find dtRecHits in event!";
    validdtrec = false;
  }   
    
  if (validdtsim && validdtrec) {

    std::map<DTWireId, edm::PSimHitContainer> simHitsPerWire =
      DTHitQualityUtils::mapSimHitsPerWire(*(dtsimHits.product()));

    std::map<DTWireId, std::vector<DTRecHit1DPair> > recHitsPerWire =
      map1DRecHitsPerWire(dtRecHits.product());

    int nDt = compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, 1);
    
    if (verbosity > 1) {
      eventout += "\n          Number of DtMuonRecHits collected:........ ";
      eventout += nDt;
    }
    mehDtMuonn->Fill(float(nDt));
  }

  // get CSC Strip information
  // get map of sim hits
  theMap.clear();
  edm::Handle<CrossingFrame<PSimHit> > cf;

  iEvent.getByLabel("mix",hitsProducer+"MuonCSCHits",cf);
  bool validXFrame = true;
  if (!cf.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find muo CSC crossingFrame in event!";
    validXFrame = false;
  }
  if (validXFrame) {
    MixCollection<PSimHit> simHits(cf.product());
    
    // arrange the hits by detUnit
    for(MixCollection<PSimHit>::MixItr hitItr = simHits.begin();
	hitItr != simHits.end(); ++hitItr) {
      theMap[hitItr->detUnitId()].push_back(*hitItr);
    }  
  }

  // get geometry
  edm::ESHandle<CSCGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  if (!hGeom.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find CSCMuonGeometryRecord in event!";
    return;
  }    
  const CSCGeometry *theCSCGeometry = &*hGeom;

  // get rechits
  edm::Handle<CSCRecHit2DCollection> hRecHits;
  iEvent.getByLabel(MuCSCSrc_, hRecHits);
  bool validCSC = true;
  if (!hRecHits.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find CSC RecHits in event!";
    validCSC = false;
  }    

  if (validCSC) {
    const CSCRecHit2DCollection *cscRecHits = hRecHits.product();
    
    int nCSC = 0;
    for (CSCRecHit2DCollection::const_iterator recHitItr = cscRecHits->begin();
	 recHitItr != cscRecHits->end(); ++recHitItr) {
      
      int detId = (*recHitItr).cscDetId().rawId();
      
      edm::PSimHitContainer simHits;   
      std::map<int, edm::PSimHitContainer>::const_iterator mapItr = 
	theMap.find(detId);
      if (mapItr != theMap.end()) {
	simHits = mapItr->second;
      }
      
      if (simHits.size() == 1) {
	++nCSC;
	
	const GeomDetUnit* detUnit = 
	  theCSCGeometry->idToDetUnit(CSCDetId(detId));
	const CSCLayer *layer = dynamic_cast<const CSCLayer *>(detUnit); 
	
	int chamberType = layer->chamber()->specs()->chamberType();
	plotResolution(simHits[0], *recHitItr, layer, chamberType);
      }
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of CSCRecHits collected:........... ";
      eventout += nCSC;
    }
    mehCSCn->Fill((float)nCSC);
  }

  // get RPC information
  std::map<double, int> mapsim, maprec;
  std::map<int, double> nmapsim, nmaprec;

  edm::ESHandle<RPCGeometry> rpcGeom;
  iSetup.get<MuonGeometryRecord>().get(rpcGeom);
  if (!rpcGeom.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find RPCMuonGeometryRecord in event!";
    return;
  }  

  edm::Handle<edm::PSimHitContainer> simHit;
  iEvent.getByLabel(MuRPCSimSrc_, simHit);
  bool validrpcsim = true;
  if (!simHit.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find RPCSimHit in event!";
    validrpcsim = false;
  }    

  edm::Handle<RPCRecHitCollection> recHit;
  iEvent.getByLabel(MuRPCSrc_, recHit);
  bool validrpc = true;
  if (!simHit.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find RPCRecHit in event!";
    validrpc = false;
  } 

  if (validrpc) {
    int nRPC = 0;
    RPCRecHitCollection::const_iterator recIt;
    int nrec = 0;
    for (recIt = recHit->begin(); recIt != recHit->end(); ++recIt) {
      RPCDetId Rid = (RPCDetId)(*recIt).rpcId();
      const RPCRoll *roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(Rid));
      if (roll->isForward()) {
	
	if (verbosity > 1) {
	  eventout += 
	    "\n          Number of RPCRecHits collected:........... ";
	  eventout += nRPC;
	}
	
	if (verbosity > 0)
	  edm::LogInfo(MsgLoggerCat) << eventout << "\n";
	return;
      }
      nrec = nrec + 1;
      LocalPoint rhitlocal = (*recIt).localPosition();
      double rhitlocalx = rhitlocal.x();
      maprec[rhitlocalx] = nrec; 
    }
    
    int i = 0;
    for (std::map<double,int>::iterator iter = maprec.begin();
	 iter != maprec.end(); ++iter) {
      i = i + 1;
      nmaprec[i] = (*iter).first;
    }
    
    int nsim = 0;
    if (validrpcsim) {
      edm::PSimHitContainer::const_iterator simIt;
      for (simIt = simHit->begin(); simIt != simHit->end(); simIt++) {
	int ptype = (*simIt).particleType();
	if (ptype == 13 || ptype == -13) {
	  nsim = nsim + 1;
	  LocalPoint shitlocal = (*simIt).localPosition();
	  double shitlocalx = shitlocal.x();
	  mapsim[shitlocalx] = nsim;
	}
      }
      
      i = 0;
      for (std::map<double,int>::iterator iter = mapsim.begin();
	   iter != mapsim.end(); ++iter) {
	i = i + 1;
	nmapsim[i] = (*iter).first;
      }
    }

    if (nsim == nrec) {
      for (int r = 0; r < nsim; r++) {
	++nRPC;
	mehRPCResX->Fill(nmaprec[r+1]-nmapsim[r+1]);
      }
    }
                                                                  
    if (verbosity > 1) {
      eventout += "\n          Number of RPCRecHits collected:........... ";
      eventout += nRPC;
    }
    mehRPCn->Fill((float)nRPC);
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  
  return;
}

//needed by to do the residual for matched hits in SiStrip
std::pair<LocalPoint,LocalVector> 
GlobalRecHitsAnalyzer::projectHit(const PSimHit& hit, 
				  const StripGeomDetUnit* stripDet,
				  const BoundPlane& plane) 
{
  
  const StripTopology& topol = stripDet->specificTopology();
  GlobalPoint globalpos= stripDet->surface().toGlobal(hit.localPosition());
  LocalPoint localHit = plane.toLocal(globalpos);
  //track direction
  LocalVector locdir=hit.localDirection();
  //rotate track in new frame
  
  GlobalVector globaldir= stripDet->surface().toGlobal(locdir);
  LocalVector dir=plane.toLocal(globaldir);
  float scale = -localHit.z() / dir.z();
  
  LocalPoint projectedPos = localHit + scale*dir;
    
  float selfAngle = topol.stripAngle( topol.strip( hit.localPosition()));

  // vector along strip in hit frame 
  LocalVector stripDir( sin(selfAngle), cos(selfAngle), 0); 
  
  LocalVector 
    localStripDir(plane.toLocal(stripDet->surface().toGlobal(stripDir)));
  
  return std::pair<LocalPoint,LocalVector>( projectedPos, localStripDir);
}

// Return a map between DTRecHit1DPair and wireId
std::map<DTWireId, std::vector<DTRecHit1DPair> >
GlobalRecHitsAnalyzer::map1DRecHitsPerWire(const DTRecHitCollection* 
					   dt1DRecHitPairs) {
  std::map<DTWireId, std::vector<DTRecHit1DPair> > ret;
  
  for(DTRecHitCollection::const_iterator rechit = dt1DRecHitPairs->begin();
      rechit != dt1DRecHitPairs->end(); rechit++) {
    ret[(*rechit).wireId()].push_back(*rechit);
  }
  
  return ret;
}

// Compute SimHit distance from wire (cm)
float GlobalRecHitsAnalyzer::simHitDistFromWire(const DTLayer* layer,
						DTWireId wireId,
						const PSimHit& hit) {
  float xwire = layer->specificTopology().wirePosition(wireId.wire());
  LocalPoint entryP = hit.entryPoint();
  LocalPoint exitP = hit.exitPoint();
  float xEntry = entryP.x()-xwire;
  float xExit  = exitP.x()-xwire;

  //FIXME: check...  
  return fabs(xEntry - (entryP.z()*(xExit-xEntry))/(exitP.z()-entryP.z()));
}

// Find the RecHit closest to the muon SimHit
template  <typename type>
const type* 
GlobalRecHitsAnalyzer::findBestRecHit(const DTLayer* layer,
				      DTWireId wireId,
				      const std::vector<type>& recHits,
				      const float simHitDist) {
  float res = 99999;
  const type* theBestRecHit = 0;
  // Loop over RecHits within the cell
  for(typename std::vector<type>::const_iterator recHit = recHits.begin();
      recHit != recHits.end();
      recHit++) {
    float distTmp = recHitDistFromWire(*recHit, layer);
    if(fabs(distTmp-simHitDist) < res) {
      res = fabs(distTmp-simHitDist);
      theBestRecHit = &(*recHit);
    }
  } // End of loop over RecHits within the cell
  
  return theBestRecHit;
}

// Compute the distance from wire (cm) of a hits in a DTRecHit1DPair
float 
GlobalRecHitsAnalyzer::recHitDistFromWire(const DTRecHit1DPair& hitPair, 
					  const DTLayer* layer) {
  // Compute the rechit distance from wire
  return fabs(hitPair.localPosition(DTEnums::Left).x() -
	      hitPair.localPosition(DTEnums::Right).x())/2.;
}

// Compute the distance from wire (cm) of a hits in a DTRecHit1D
float 
GlobalRecHitsAnalyzer::recHitDistFromWire(const DTRecHit1D& recHit, 
					  const DTLayer* layer) {
  return fabs(recHit.localPosition().x() - 
	      layer->specificTopology().wirePosition(recHit.wireId().wire()));
}

template  <typename type>
int GlobalRecHitsAnalyzer::compute(const DTGeometry *dtGeom,
				   const std::map<DTWireId, std::vector<PSimHit> >& 
				   _simHitsPerWire,
				   const std::map<DTWireId, std::vector<type> >& 
				   _recHitsPerWire,
				   int step) {
  
std::map<DTWireId, std::vector<PSimHit> >  simHitsPerWire = _simHitsPerWire;
std::map<DTWireId, std::vector<type> > recHitsPerWire = _recHitsPerWire;
  int nDt = 0;
  // Loop over cells with a muon SimHit
  for(std::map<DTWireId, std::vector<PSimHit> >::const_iterator wireAndSHits = 
	simHitsPerWire.begin();
      wireAndSHits != simHitsPerWire.end();
      wireAndSHits++) {
    DTWireId wireId = (*wireAndSHits).first;
    std::vector<PSimHit> simHitsInCell = (*wireAndSHits).second;
    
    // Get the layer
    const DTLayer* layer = dtGeom->layer(wireId);
    
    // Look for a mu hit in the cell
    const PSimHit* muSimHit = DTHitQualityUtils::findMuSimHit(simHitsInCell);
    if (muSimHit==0) {
      continue; // Skip this cell
    }

    // Find the distance of the simhit from the wire
    float simHitWireDist = simHitDistFromWire(layer, wireId, *muSimHit);
    // Skip simhits out of the cell
    if(simHitWireDist>2.1) {
      continue; // Skip this cell
    }

    // Look for RecHits in the same cell
    if(recHitsPerWire.find(wireId) == recHitsPerWire.end()) {
      continue; // No RecHit found in this cell
    } else {

      std::vector<type> recHits = recHitsPerWire[wireId];
	 
      // Find the best RecHit
      const type* theBestRecHit = 
	findBestRecHit(layer, wireId, recHits, simHitWireDist);
 
      float recHitWireDist =  recHitDistFromWire(*theBestRecHit, layer);
      
      ++nDt;

      mehDtMuonRes->Fill(recHitWireDist-simHitWireDist);
      
    } // find rechits
  } // loop over simhits

  return nDt;
}

void 
GlobalRecHitsAnalyzer::plotResolution(const PSimHit & simHit, 
				      const CSCRecHit2D & recHit,
				      const CSCLayer * layer, 
				      int chamberType) {
  GlobalPoint simHitPos = layer->toGlobal(simHit.localPosition());
  GlobalPoint recHitPos = layer->toGlobal(recHit.localPosition());
  
  mehCSCResRDPhi->Fill(recHitPos.phi()-simHitPos.phi());
}

