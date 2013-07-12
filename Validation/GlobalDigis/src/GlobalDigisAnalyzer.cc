/** \file GlobalDigisAnalyzer.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2012/12/26 22:47:50 $
 *  $Revision: 1.21 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "Validation/GlobalDigis/interface/GlobalDigisAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

GlobalDigisAnalyzer::GlobalDigisAnalyzer(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), frequency(0), label(""), getAllProvenances(false),
  printProvenanceInfo(false), hitsProducer(""), theCSCStripPedestalSum(0),
  theCSCStripPedestalCount(0), count(0)
{
  std::string MsgLoggerCat = "GlobalDigisAnalyzer_GlobalDigisAnalyzer";

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
  ECalEESrc_ = iPSet.getParameter<edm::InputTag>("ECalEESrc");
  ECalESSrc_ = iPSet.getParameter<edm::InputTag>("ECalESSrc");
  HCalSrc_ = iPSet.getParameter<edm::InputTag>("HCalSrc");
  HCalDigi_ = iPSet.getParameter<edm::InputTag>("HCalDigi");
  SiStripSrc_ = iPSet.getParameter<edm::InputTag>("SiStripSrc"); 
  SiPxlSrc_ = iPSet.getParameter<edm::InputTag>("SiPxlSrc");
  MuDTSrc_ = iPSet.getParameter<edm::InputTag>("MuDTSrc");
  MuCSCStripSrc_ = iPSet.getParameter<edm::InputTag>("MuCSCStripSrc");
  MuCSCWireSrc_ = iPSet.getParameter<edm::InputTag>("MuCSCWireSrc");
  MuRPCSrc_ = iPSet.getParameter<edm::InputTag>("MuRPCSrc");
  
  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;
  
  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "    GetProv       = " << getAllProvenances << "\n"
      << "    PrintProv     = " << printProvenanceInfo << "\n"
      << "    ECalEBSrc     = " << ECalEBSrc_.label() 
      << ":" << ECalEBSrc_.instance() << "\n"
      << "    ECalEESrc     = " << ECalEESrc_.label() 
      << ":" << ECalEESrc_.instance() << "\n"
      << "    ECalESSrc     = " << ECalESSrc_.label() 
      << ":" << ECalESSrc_.instance() << "\n"
      << "    HCalSrc       = " << HCalSrc_.label() 
      << ":" << HCalSrc_.instance() << "\n"
      << "    HCalDigi       = " << HCalDigi_.label() 
      << ":" << HCalDigi_.instance() << "\n"
      << "    SiStripSrc    = " << SiStripSrc_.label() 
      << ":" << SiStripSrc_.instance() << "\n" 
      << "    SiPixelSrc    = " << SiPxlSrc_.label()
      << ":" << SiPxlSrc_.instance() << "\n"
      << "    MuDTSrc       = " << MuDTSrc_.label()
      << ":" << MuDTSrc_.instance() << "\n"
      << "    MuCSCStripSrc = " << MuCSCStripSrc_.label()
      << ":" << MuCSCStripSrc_.instance() << "\n"
      << "    MuCSCWireSrc  = " << MuCSCWireSrc_.label()
      << ":" << MuCSCWireSrc_.instance() << "\n"
      << "    MuRPCSrc      = " << MuRPCSrc_.label()
      << ":" << MuRPCSrc_.instance() << "\n"
      << "===============================\n";
  }
  
  //Put in analyzer stuff here.... Pasted from Rec Hits... 
  
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
  if (dbe) {
    std::string SiStripString[19] = {"TECW1", "TECW2", "TECW3", "TECW4", 
				     "TECW5", "TECW6", "TECW7", "TECW8", 
				     "TIBL1", "TIBL2", "TIBL3", "TIBL4", 
				     "TIDW1", "TIDW2", "TIDW3", "TOBL1", 
				     "TOBL2", "TOBL3", "TOBL4"};
    for(int i = 0; i<19; ++i) {
      mehSiStripn[i]=0;
      mehSiStripADC[i]=0;
      mehSiStripStrip[i]=0;
    }
    std::string hcharname, hchartitle;
    dbe->setCurrentFolder("GlobalDigisV/SiStrips");
    for(int amend = 0; amend < 19; ++amend) { 
      hcharname = "hSiStripn_"+SiStripString[amend];
      hchartitle= SiStripString[amend]+"  Digis";
      mehSiStripn[amend] = dbe->book1D(hcharname,hchartitle,5000,0.,10000.);
      mehSiStripn[amend]->setAxisTitle("Number of Digis",1);
      mehSiStripn[amend]->setAxisTitle("Count",2);
      
      hcharname = "hSiStripADC_"+SiStripString[amend];
      hchartitle= SiStripString[amend]+" ADC";
      mehSiStripADC[amend] = dbe->book1D(hcharname,hchartitle,150,0.0,300.);
      mehSiStripADC[amend]->setAxisTitle("ADC",1);
      mehSiStripADC[amend]->setAxisTitle("Count",2);
      
      hcharname = "hSiStripStripADC_"+SiStripString[amend];
      hchartitle= SiStripString[amend]+" Strip";
      mehSiStripStrip[amend] = dbe->book1D(hcharname,hchartitle,200,0.0,800.);
      mehSiStripStrip[amend]->setAxisTitle("Strip Number",1);
      mehSiStripStrip[amend]->setAxisTitle("Count",2);
    }
    
    //HCal
    std::string HCalString[4] = {"HB", "HE", "HO","HF"}; 
    float calnUpper[4] = {30000.,30000.,30000.,20000.}; 
    float calnLower[4]={0.,0.,0.,0.}; 
    float SHEUpper[4]={1.,1.,1.,1.};
    float SHEvAEEUpper[4] = {5000, 5000, 5000, 5000}; 
    float SHEvAEELower[4] = {-5000, -5000, -5000, -5000}; 
    int SHEvAEEnBins[4] = {200,200,200,200};
    double ProfileUpper[4] = {1.,1.,1.,1.};  
    
    for(int i =0; i<4; ++i) {
      mehHcaln[i]=0;
      mehHcalAEE[i]=0;
      mehHcalSHE[i]=0;
      mehHcalAEESHE[i]=0;
      mehHcalSHEvAEE[i]=0;
    }
    dbe->setCurrentFolder("GlobalDigisV/HCals");
    
    for(int amend = 0; amend < 4; ++amend) {
      hcharname = "hHcaln_"+HCalString[amend];
      hchartitle= HCalString[amend]+"  digis";
      mehHcaln[amend] = dbe->book1D(hcharname,hchartitle, 10000, calnLower[amend], 
				    calnUpper[amend]);
      mehHcaln[amend]->setAxisTitle("Number of Digis",1);
      mehHcaln[amend]->setAxisTitle("Count",2);

      hcharname = "hHcalAEE_"+HCalString[amend];
      hchartitle= HCalString[amend]+"Cal AEE";
      mehHcalAEE[amend] = dbe->book1D(hcharname,hchartitle, 60, -10., 50.);
      mehHcalAEE[amend]->setAxisTitle("Analog Equivalent Energy",1);
      mehHcalAEE[amend]->setAxisTitle("Count",2);

      hcharname = "hHcalSHE_"+HCalString[amend];
      hchartitle= HCalString[amend]+"Cal SHE";
      mehHcalSHE[amend] = dbe->book1D(hcharname,hchartitle, 1000, 0.0, 
				      SHEUpper[amend]);
      mehHcalSHE[amend]->setAxisTitle("Simulated Hit Energy",1);
      mehHcalSHE[amend]->setAxisTitle("Count",2);

      hcharname = "hHcalAEESHE_"+HCalString[amend];
      hchartitle= HCalString[amend]+"Cal AEE/SHE";
      mehHcalAEESHE[amend] = dbe->book1D(hcharname, hchartitle, SHEvAEEnBins[amend], 
					 SHEvAEELower[amend], 
					 SHEvAEEUpper[amend]);
      mehHcalAEESHE[amend]->setAxisTitle("ADC / SHE",1);
      mehHcalAEESHE[amend]->setAxisTitle("Count",2);
      
      hcharname = "hHcalSHEvAEE_"+HCalString[amend];
      hchartitle= HCalString[amend]+"Cal SHE vs. AEE";
      mehHcalSHEvAEE[amend] = dbe->bookProfile(hcharname,hchartitle, 60, -10., 
					       50., 100, 0., 
					       (float)ProfileUpper[amend],"");
      mehHcalSHEvAEE[amend]->setAxisTitle("AEE / SHE",1);
      mehHcalSHEvAEE[amend]->setAxisTitle("SHE",2);

    }
    
    //Ecal
    std::string ECalString[2] = {"EB","EE"}; 
    
    for(int i =0; i<2; ++i) {
      mehEcaln[i]=0;
      mehEcalAEE[i]=0;
      mehEcalSHE[i]=0;
      mehEcalMaxPos[i]=0;
      mehEcalMultvAEE[i]=0;
      mehEcalSHEvAEESHE[i]=0;
    }
    dbe->setCurrentFolder("GlobalDigisV/ECals");
    
    for(int amend = 0; amend < 2; ++amend) {
      hcharname = "hEcaln_"+ECalString[amend];
      hchartitle= ECalString[amend]+"  digis";
      mehEcaln[amend] = dbe->book1D(hcharname,hchartitle, 3000, 0., 40000.);
      mehEcaln[amend]->setAxisTitle("Number of Digis",1);
      mehEcaln[amend]->setAxisTitle("Count",2);

      hcharname = "hEcalAEE_"+ECalString[amend];
      hchartitle= ECalString[amend]+"Cal AEE";
      mehEcalAEE[amend] = dbe->book1D(hcharname,hchartitle, 1000, 0., 100.);
      mehEcalAEE[amend]->setAxisTitle("Analog Equivalent Energy",1);
      mehEcalAEE[amend]->setAxisTitle("Count",2);

      hcharname = "hEcalSHE_"+ECalString[amend];
      hchartitle= ECalString[amend]+"Cal SHE";
      mehEcalSHE[amend] = dbe->book1D(hcharname,hchartitle, 500, 0., 50.);
      mehEcalSHE[amend]->setAxisTitle("Simulated Hit Energy",1);
      mehEcalSHE[amend]->setAxisTitle("Count",2);

      hcharname = "hEcalMaxPos_"+ECalString[amend];
      hchartitle= ECalString[amend]+"Cal MaxPos";
      mehEcalMaxPos[amend] = dbe->book1D(hcharname,hchartitle,10, 0., 10.);
      mehEcalMaxPos[amend]->setAxisTitle("Maximum Position",1);
      mehEcalMaxPos[amend]->setAxisTitle("Count",2);
      
      hcharname = "hEcalSHEvAEESHE_"+ECalString[amend];
      hchartitle= ECalString[amend]+"Cal SHE vs. AEE/SHE";
      mehEcalSHEvAEESHE[amend] = dbe->bookProfile(hcharname,hchartitle,1000, 0., 100., 
						  500, 0., 50.,"");
      mehEcalSHEvAEESHE[amend]->setAxisTitle("AEE / SHE",1);
      mehEcalSHEvAEESHE[amend]->setAxisTitle("SHE",2);

      hcharname = "hEcalMultvAEE_"+ECalString[amend];
      hchartitle= ECalString[amend]+"Cal Multi vs. AEE";
      mehEcalMultvAEE[amend] = dbe->bookProfile(hcharname,hchartitle, 1000, 0., 100., 
						4000, 0., 40000.,"");
      mehEcalMultvAEE[amend]->setAxisTitle("Analog Equivalent Energy",1);
      mehEcalMultvAEE[amend]->setAxisTitle("Number of Digis",2);      
    }
    mehEScaln = 0;

    hcharname = "hEcaln_ES";
    hchartitle= "ESCAL  digis";
    mehEScaln = dbe->book1D(hcharname,hchartitle, 1000, 0., 5000.);
    mehEScaln->setAxisTitle("Number of Digis",1);
    mehEScaln->setAxisTitle("Count",2);

    std::string ADCNumber[3] = {"0", "1", "2"};
    for(int i =0; i<3; ++i) {
      mehEScalADC[i] = 0;
      hcharname = "hEcalADC"+ADCNumber[i]+"_ES";
      hchartitle= "ESCAL  ADC"+ADCNumber[i];
      mehEScalADC[i] = dbe->book1D(hcharname,hchartitle, 1500, 0., 1500.);
      mehEScalADC[i]->setAxisTitle("ADC"+ADCNumber[i],1);
      mehEScalADC[i]->setAxisTitle("Count",2);
    }
    
    //Si Pixels ***DONE***  
    std::string SiPixelString[7] = {"BRL1", "BRL2", "BRL3", "FWD1n", "FWD1p", 
				    "FWD2n", "FWD2p"};
    for(int j =0; j<7; ++j) {
      mehSiPixeln[j]=0;
      mehSiPixelADC[j]=0;
      mehSiPixelRow[j]=0;
      mehSiPixelCol[j]=0;
    }
    
    dbe->setCurrentFolder("GlobalDigisV/SiPixels");
    for(int amend = 0; amend < 7; ++amend) {
      hcharname = "hSiPixeln_"+SiPixelString[amend];
      hchartitle= SiPixelString[amend]+" Digis";
      if(amend<3) mehSiPixeln[amend] = dbe->book1D(hcharname,hchartitle,500,0.,1000.);
      else mehSiPixeln[amend] = dbe->book1D(hcharname,hchartitle,500,0.,1000.);
      mehSiPixeln[amend]->setAxisTitle("Number of Digis",1);
      mehSiPixeln[amend]->setAxisTitle("Count",2);
      
      hcharname = "hSiPixelADC_"+SiPixelString[amend];
      hchartitle= SiPixelString[amend]+" ADC";
      mehSiPixelADC[amend] = dbe->book1D(hcharname,hchartitle,150,0.0,300.);
      mehSiPixelADC[amend]->setAxisTitle("ADC",1);
      mehSiPixelADC[amend]->setAxisTitle("Count",2);

      hcharname = "hSiPixelRow_"+SiPixelString[amend];
      hchartitle= SiPixelString[amend]+" Row";
      mehSiPixelRow[amend] = dbe->book1D(hcharname,hchartitle,100,0.0,100.);
      mehSiPixelRow[amend]->setAxisTitle("Row Number",1);
      mehSiPixelRow[amend]->setAxisTitle("Count",2);

      hcharname = "hSiPixelColumn_"+SiPixelString[amend];
      hchartitle= SiPixelString[amend]+" Column";
      mehSiPixelCol[amend] = dbe->book1D(hcharname,hchartitle,200,0.0,500.);
      mehSiPixelCol[amend]->setAxisTitle("Column Number",1);
      mehSiPixelCol[amend]->setAxisTitle("Count",2);
    }

    //Muons
    dbe->setCurrentFolder("GlobalDigisV/Muons");

    //DT
    std::string MuonString[4] = {"MB1", "MB2", "MB3", "MB4"};
    
    for(int i =0; i < 4; ++i) {
      mehDtMuonn[i] = 0;
      mehDtMuonLayer[i] = 0;
      mehDtMuonTime[i] = 0;
      mehDtMuonTimevLayer[i] = 0;
    }
    
    for(int j = 0; j < 4; ++j) {
      hcharname = "hDtMuonn_"+MuonString[j];
      hchartitle= MuonString[j]+"  digis";
      mehDtMuonn[j] = dbe->book1D(hcharname,hchartitle,250, 0., 500.);
      mehDtMuonn[j]->setAxisTitle("Number of Digis",1);
      mehDtMuonn[j]->setAxisTitle("Count",2);

      hcharname = "hDtLayer_"+MuonString[j];
      hchartitle= MuonString[j]+"  Layer";
      mehDtMuonLayer[j] = dbe->book1D(hcharname,hchartitle,12, 1., 13.);
      mehDtMuonLayer[j]->setAxisTitle("4 * (SuperLayer - 1) + Layer",1);
      mehDtMuonLayer[j]->setAxisTitle("Count",2);

      hcharname = "hDtMuonTime_"+MuonString[j];
      hchartitle= MuonString[j]+"  Time";
      mehDtMuonTime[j] = dbe->book1D(hcharname,hchartitle,300, 400., 1000.);
      mehDtMuonTime[j]->setAxisTitle("Time",1);
      mehDtMuonTime[j]->setAxisTitle("Count",2);

      hcharname = "hDtMuonTimevLayer_"+MuonString[j];
      hchartitle= MuonString[j]+"  Time vs. Layer";
      mehDtMuonTimevLayer[j] = dbe->bookProfile(hcharname,hchartitle,12, 1., 13., 300, 
						400., 1000.,"");
      mehDtMuonTimevLayer[j]->setAxisTitle("4 * (SuperLayer - 1) + Layer",1);
      mehDtMuonTimevLayer[j]->setAxisTitle("Time",2);
    }

    //CSC 
    mehCSCStripn = 0;
    hcharname = "hCSCStripn";
    hchartitle = "CSC Strip digis";
    mehCSCStripn = dbe->book1D(hcharname,hchartitle,250, 0., 500.);
    mehCSCStripn->setAxisTitle("Number of Digis",1);
    mehCSCStripn->setAxisTitle("Count",2);
    
    mehCSCStripADC = 0;
    hcharname = "hCSCStripADC";
    hchartitle = "CSC Strip ADC";
    mehCSCStripADC = dbe->book1D(hcharname,hchartitle, 110, 0., 1100.);
    mehCSCStripADC->setAxisTitle("ADC",1);
    mehCSCStripADC->setAxisTitle("Count",2);
    
    mehCSCWiren = 0;
    hcharname = "hCSCWiren";
    hchartitle = "CSC Wire digis";
    mehCSCWiren = dbe->book1D(hcharname,hchartitle,250, 0., 500.);
    mehCSCWiren->setAxisTitle("Number of Digis",1);
    mehCSCWiren->setAxisTitle("Count",2);
    
    mehCSCWireTime = 0;
    hcharname = "hCSCWireTime";
    hchartitle = "CSC Wire Time";
    mehCSCWireTime = dbe->book1D(hcharname,hchartitle,10, 0., 10.);
    mehCSCWireTime->setAxisTitle("Time",1);
    mehCSCWireTime->setAxisTitle("Count",2);
    
    // RPC 
    mehRPCMuonn = 0;
    hcharname = "hRPCMuonn";
    hchartitle = "RPC digis";
    mehCSCStripn = dbe->book1D(hcharname,hchartitle,250, 0., 500.);
    mehCSCStripn->setAxisTitle("Number of Digis",1);
    mehCSCStripn->setAxisTitle("Count",2);

    std::string MuonRPCString[5] = {"Wmin2", "Wmin1", "W0", "Wpu1", "Wpu2"};
    for(int i =0; i < 5; ++i) {
      mehRPCRes[i] = 0;
    }

    for(int j = 0; j < 5; ++j) {    
      hcharname = "hRPCRes_"+MuonRPCString[j];
      hchartitle= MuonRPCString[j]+"  Digi - Sim";   
      mehRPCRes[j] = dbe->book1D(hcharname,hchartitle,200, -8., 8.);
      mehRPCRes[j]->setAxisTitle("Digi - Sim center of strip x",1);
      mehRPCRes[j]->setAxisTitle("Count",2);
    }
  }

  // set default constants
  // ECal
  
  ECalgainConv_[0] = 0.;
  ECalgainConv_[1] = 1.;
  ECalgainConv_[2] = 2.;
  ECalgainConv_[3] = 12.;  
  ECalbarrelADCtoGeV_ = 0.035;
  ECalendcapADCtoGeV_ = 0.06;
}
 
GlobalDigisAnalyzer::~GlobalDigisAnalyzer() {}

void GlobalDigisAnalyzer::beginJob( void )
{
  std::string MsgLoggerCat = "GlobalDigisAnalyzer_beginJob";
  
  EcalMGPAGainRatio * defaultRatios = new EcalMGPAGainRatio();
  
  ECalgainConv_[0] = 0.;
  ECalgainConv_[1] = 1.;
  ECalgainConv_[2] = defaultRatios->gain12Over6() ;
  ECalgainConv_[3] = ECalgainConv_[2]*(defaultRatios->gain6Over1()) ;
  
  delete defaultRatios;
  
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "Modified Calorimeter gain constants: g0 = " << ECalgainConv_[0]
      << ", g1 = " << ECalgainConv_[1] << ", g2 = " << ECalgainConv_[2]
      << ", g3 = " << ECalgainConv_[3];
  }
  
  return;
}


void GlobalDigisAnalyzer::endJob()
{
  std::string MsgLoggerCat = "GlobalDigisAnalyzer_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " events.";
  return;
}

void GlobalDigisAnalyzer::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalDigisAnalyzer_analyze";
  
  // keep track of number of events processed
  ++count;


  
  // THIS BLOCK MIGRATED HERE FROM beginJob:
  // setup calorimeter constants from service
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  iSetup.get<EcalADCToGeVConstantRcd>().get(pAgc);
  const EcalADCToGeVConstant* agc = pAgc.product();
  ECalbarrelADCtoGeV_ = agc->getEBValue();
  ECalendcapADCtoGeV_ = agc->getEEValue();
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat)
      << "Modified Calorimeter ADCtoGeV constants: barrel = " 
      << ECalbarrelADCtoGeV_ << ", endcap = " << ECalendcapADCtoGeV_;
  }
  


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
  fillECal(iEvent,  iSetup);
  // gather Hcal information from event
  fillHCal(iEvent, iSetup);
  // gather Track information from event
  fillTrk(iEvent,  iSetup);
  // gather Muon information from event
  fillMuon(iEvent,  iSetup);
  
  if (verbosity > 0)
    edm::LogInfo (MsgLoggerCat)
      << "Done gathering data from event.";
  
  if (verbosity > 2)
    edm::LogInfo (MsgLoggerCat)
      << "Saving event contents:";
    
  return;
}

void GlobalDigisAnalyzer::fillECal(const edm::Event& iEvent, 
				   const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalDigisAnalyzer_fillECal";
  
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  
  
  // extract crossing frame from event
  edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;
  
  ////////////////////////
  //extract EB information
  ////////////////////////
  bool isBarrel = true;
  edm::Handle<EBDigiCollection> EcalDigiEB;  
  iEvent.getByLabel(ECalEBSrc_, EcalDigiEB);
  bool validDigiEB = true;
  if (!EcalDigiEB.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find EcalDigiEB in event!";
    validDigiEB = false;
  }  
  if (validDigiEB) {
    if ( EcalDigiEB->size() == 0) isBarrel = false;
    
    if (isBarrel) {
      
      // loop over simhits
      MapType ebSimMap;
      const std::string barrelHitsName(hitsProducer+"EcalHitsEB");
      iEvent.getByLabel("mix",barrelHitsName,crossingFrame);
      bool validXFrame = true;
      if (!crossingFrame.isValid()) {
	LogDebug(MsgLoggerCat)
	  << "Unable to find cal barrel crossingFrame in event!";
	validXFrame = false;
      }
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

      // loop over digis
      const EBDigiCollection *barrelDigi = EcalDigiEB.product();
      
      std::vector<double> ebAnalogSignal;
      std::vector<double> ebADCCounts;
      std::vector<double> ebADCGains;
      ebAnalogSignal.reserve(EBDataFrame::MAXSAMPLES);
      ebADCCounts.reserve(EBDataFrame::MAXSAMPLES);
      ebADCGains.reserve(EBDataFrame::MAXSAMPLES);
      
      int i = 0;
      for (unsigned int digis=0; digis<EcalDigiEB->size(); ++digis) {
	
	++i;
	
	EBDataFrame ebdf = (*barrelDigi)[digis];
	int nrSamples = ebdf.size();
	
	EBDetId ebid = ebdf.id () ;
	
	double Emax = 0;
	int Pmax = 0;
	double pedestalPreSample = 0.;
	double pedestalPreSampleAnalog = 0.;
	
	for (int sample = 0 ; sample < nrSamples; ++sample) {
	  ebAnalogSignal[sample] = 0.;
	  ebADCCounts[sample] = 0.;
	  ebADCGains[sample] = -1.;
	}
	
	// calculate maximum energy and pedestal
	for (int sample = 0 ; sample < nrSamples; ++sample) {
	  
	  EcalMGPASample thisSample = ebdf[sample];
	  ebADCCounts[sample] = (thisSample.adc());
	  ebADCGains[sample]  = (thisSample.gainId());
	  ebAnalogSignal[sample] = 
	    (ebADCCounts[sample] * ECalgainConv_[(int)ebADCGains[sample]]
	     * ECalbarrelADCtoGeV_);
	  if (Emax < ebAnalogSignal[sample]) {
	    Emax = ebAnalogSignal[sample];
	    Pmax = sample;
	  }
	  if ( sample < 3 ) {
	    pedestalPreSample += ebADCCounts[sample] ;
	    pedestalPreSampleAnalog += 
	      ebADCCounts[sample] * ECalgainConv_[(int)ebADCGains[sample]]
	      * ECalbarrelADCtoGeV_ ;
	  }
	  
	}
	pedestalPreSample /= 3. ; 
	pedestalPreSampleAnalog /= 3. ; 
	
	// calculate pedestal subtracted digi energy in the crystal
	double Erec = Emax - pedestalPreSampleAnalog
	  * ECalgainConv_[(int)ebADCGains[Pmax]];
	
	// gather necessary information
	mehEcalMaxPos[0]->Fill(Pmax);
	mehEcalSHE[0]->Fill(ebSimMap[ebid.rawId()]);
	mehEcalAEE[0]->Fill(Erec);
	//Adding protection against FPE
	if (ebSimMap[ebid.rawId()]!=0) {
	  mehEcalSHEvAEESHE[0]->Fill(Erec/ebSimMap[ebid.rawId()],
				     ebSimMap[ebid.rawId()]);
	}
	//else {
	//  std::cout<<"Would have been an FPE! with ebSimMap[ebid.rawId()]==0\n";
	//}

	mehEcalMultvAEE[0]->Fill(Pmax,(float)i);
      }
      
      if (verbosity > 1) {
	eventout += "\n          Number of EBDigis collected:.............. ";
	eventout += i;
      }
      mehEcaln[0]->Fill((float)i);
    }
  }
  
  /////////////////////////
  //extract EE information
  ////////////////////////
  bool isEndCap = true;
  edm::Handle<EEDigiCollection> EcalDigiEE;  
  iEvent.getByLabel(ECalEESrc_, EcalDigiEE);
  bool validDigiEE = true;
  if (!EcalDigiEE.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find EcalDigiEE in event!";
    validDigiEE = false;
  }  
  if (validDigiEE) {
    if (EcalDigiEE->size() == 0) isEndCap = false;
    
    if (isEndCap) {
      
      // loop over simhits
      MapType eeSimMap;
      const std::string endcapHitsName(hitsProducer+"EcalHitsEE");
      iEvent.getByLabel("mix",endcapHitsName,crossingFrame);
      bool validXFrame = true;
      if (!crossingFrame.isValid()) {
	LogDebug(MsgLoggerCat)
	  << "Unable to find cal endcap crossingFrame in event!";
	validXFrame = false;
      }
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

      // loop over digis
      const EEDigiCollection *endcapDigi = EcalDigiEE.product();
      
      std::vector<double> eeAnalogSignal;
      std::vector<double> eeADCCounts;
      std::vector<double> eeADCGains;
      eeAnalogSignal.reserve(EEDataFrame::MAXSAMPLES);
      eeADCCounts.reserve(EEDataFrame::MAXSAMPLES);
      eeADCGains.reserve(EEDataFrame::MAXSAMPLES);
      
      int inc = 0;
      for (unsigned int digis=0; digis<EcalDigiEE->size(); ++digis){ 
	
	++inc;
	
	EEDataFrame eedf = (*endcapDigi)[digis];
	int nrSamples = eedf.size();
	
	EEDetId eeid = eedf.id () ;
	
	double Emax = 0;
	int Pmax = 0;
	double pedestalPreSample = 0.;
	double pedestalPreSampleAnalog = 0.;
	
	for (int sample = 0 ; sample < nrSamples; ++sample) {
	  eeAnalogSignal[sample] = 0.;
	  eeADCCounts[sample] = 0.;
	  eeADCGains[sample] = -1.;
	}
	
	// calculate maximum enery and pedestal
	for (int sample = 0 ; sample < nrSamples; ++sample) {
	  
	  EcalMGPASample thisSample = eedf[sample];
	  
	  eeADCCounts[sample] = (thisSample.adc());
	  eeADCGains[sample]  = (thisSample.gainId());
	  eeAnalogSignal[sample] = 
	    (eeADCCounts[sample] * ECalgainConv_[(int)eeADCGains[sample]]
	     * ECalbarrelADCtoGeV_);
	  if (Emax < eeAnalogSignal[sample]) {
	    Emax = eeAnalogSignal[sample];
	    Pmax = sample;
	  }
	  if ( sample < 3 ) {
	    pedestalPreSample += eeADCCounts[sample] ;
	    pedestalPreSampleAnalog += 
	      eeADCCounts[sample] * ECalgainConv_[(int)eeADCGains[sample]]
	      * ECalbarrelADCtoGeV_ ;
	  }
	  
	}
	pedestalPreSample /= 3. ; 
	pedestalPreSampleAnalog /= 3. ; 
	
	// calculate pedestal subtracted digi energy in the crystal
	double Erec = Emax - pedestalPreSampleAnalog
	  * ECalgainConv_[(int)eeADCGains[Pmax]];
	
	// gather necessary information
	mehEcalMaxPos[1]->Fill(Pmax);
	mehEcalSHE[1]->Fill(eeSimMap[eeid.rawId()]);
	mehEcalAEE[1]->Fill(Erec);
	//Adding protection against FPE
        if (eeSimMap[eeid.rawId()]!=0){
	  mehEcalSHEvAEESHE[1]->Fill(Erec/eeSimMap[eeid.rawId()],
				     eeSimMap[eeid.rawId()]);
	}
	//else{
	//  std::cout<<"Would have been an FPE! with eeSimMap[eeid.rawId()]==0\n"; 
	//}
	mehEcalMultvAEE[1]->Fill(Pmax,(float)inc);
	
      }
      
      if (verbosity > 1) {
	eventout += "\n          Number of EEDigis collected:.............. ";
	eventout += inc;
      }
      
      mehEcaln[1]->Fill((float)inc);
    }
  }

  /////////////////////////
  //extract ES information
  ////////////////////////
  bool isPreshower = true;
  edm::Handle<ESDigiCollection> EcalDigiES;  
  iEvent.getByLabel(ECalESSrc_, EcalDigiES);
  bool validDigiES = true;
  if (!EcalDigiES.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find EcalDigiES in event!";
    validDigiES = false;
  } 
  
  // ONLY WHILE GEOMETRY IS REMOVED
  validDigiES = false;
 
  if (validDigiES) {
    if (EcalDigiES->size() == 0) isPreshower = false;
    
    if (isPreshower) {
      
      // loop over simhits
      const std::string preshowerHitsName(hitsProducer+"EcalHitsES");
      iEvent.getByLabel("mix",preshowerHitsName,crossingFrame);
      bool validXFrame = true;
      if (!crossingFrame.isValid()) {
	LogDebug(MsgLoggerCat)
	  << "Unable to find cal preshower crossingFrame in event!";
	validXFrame = false;
      }
      if (validXFrame) {
	std::auto_ptr<MixCollection<PCaloHit> >
	  preshowerHits(new MixCollection<PCaloHit>(crossingFrame.product()));
	
	// keep track of sum of simhit energy in each crystal
	MapType esSimMap;
	for (MixCollection<PCaloHit>::MixItr hitItr 
	       = preshowerHits->begin();
	     hitItr != preshowerHits->end();
	     ++hitItr) {
	  
	  ESDetId esid = ESDetId(hitItr->id());
	  
	  uint32_t crystid = esid.rawId();
	  esSimMap[crystid] += hitItr->energy();
	}
      }

      // loop over digis
      const ESDigiCollection *preshowerDigi = EcalDigiES.product();
      
      std::vector<double> esADCCounts;
      esADCCounts.reserve(ESDataFrame::MAXSAMPLES);
      
      int i = 0;
      for (unsigned int digis=0; digis<EcalDigiES->size(); ++digis) {
	
	++i;
	
	
	ESDataFrame esdf = (*preshowerDigi)[digis];
	int nrSamples = esdf.size();
	
	for (int sample = 0 ; sample < nrSamples; ++sample) {
	  esADCCounts[sample] = 0.;
	}
	
	// gether ADC counts
	for (int sample = 0 ; sample < nrSamples; ++sample) {
	  
	  ESSample thisSample = esdf[sample];
	  esADCCounts[sample] = (thisSample.adc());
	}
	
	mehEScalADC[0]->Fill(esADCCounts[0]);
	mehEScalADC[1]->Fill(esADCCounts[1]);
	mehEScalADC[2]->Fill(esADCCounts[2]);
	
      }
      
      if (verbosity > 1) {
	eventout += "\n          Number of ESDigis collected:.............. ";
	eventout += i;
      }
      
      mehEScaln->Fill((float)i);
    }
  }
  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  
  return;
}


void GlobalDigisAnalyzer::fillHCal(const edm::Event& iEvent, 
				   const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalDigisAnalyzer_fillHCal";
  
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  
  
  // get calibration info
  edm::ESHandle<HcalDbService> HCalconditions;
  iSetup.get<HcalDbRecord>().get(HCalconditions);
  if (!HCalconditions.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find HCalconditions in event!";
    return;
  } 
  //HcalCalibrations calibrations;
  CaloSamples tool;
  
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

  ////////////////////////
  // get HBHE information
  ///////////////////////
  edm::Handle<edm::SortedCollection<HBHEDataFrame> > hbhe;
  iEvent.getByLabel(HCalDigi_,hbhe);
  bool validHBHE = true;
  if (!hbhe.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find HBHEDataFrame in event!";
    validHBHE = false;
  }    

  if (validHBHE) {
    edm::SortedCollection<HBHEDataFrame>::const_iterator ihbhe;
    
    int iHB = 0;
    int iHE = 0; 
    for (ihbhe = hbhe->begin(); ihbhe != hbhe->end(); ++ihbhe) {
      HcalDetId cell(ihbhe->id()); 
      
      if ((cell.subdet() == sdHcalBrl) || (cell.subdet() == sdHcalEC)) {
	
	//HCalconditions->makeHcalCalibration(cell, &calibrations);
	const HcalCalibrations& calibrations = 
	  HCalconditions->getHcalCalibrations(cell);
	const HcalQIECoder *channelCoder = HCalconditions->getHcalCoder(cell);
	const HcalQIEShape *shape = HCalconditions->getHcalShape(channelCoder);
	HcalCoderDb coder(*channelCoder, *shape);
	coder.adc2fC(*ihbhe, tool);
	
	// get HB info
	if (cell.subdet() == sdHcalBrl) {
	  
	  ++iHB;
	  float fDigiSum = 0.0;
	  for  (int ii = 0; ii < tool.size(); ++ii) {
	    // default ped is 4.5
	    int capid = (*ihbhe)[ii].capid();
	    fDigiSum += (tool[ii] - calibrations.pedestal(capid));
	  }
	  
	  mehHcalSHE[0]->Fill(fHFEnergySimHits[cell.rawId()]);
	  mehHcalAEE[0]->Fill(fDigiSum);
	  //Adding protection against FPE
	  if (fHFEnergySimHits[cell.rawId()]!=0){
	    mehHcalAEESHE[0]->Fill(fDigiSum/fHFEnergySimHits[cell.rawId()]);
	  }
	  //else {
	  //  std::cout<<"It would have been an FPE! with fHFEnergySimHits[cell.rawId()]==0!\n";
	  //}
	  mehHcalSHEvAEE[0]->Fill(fDigiSum, fHFEnergySimHits[cell.rawId()]);
	}
	
	// get HE info
	if (cell.subdet() == sdHcalEC) {
	  
	  ++iHE;
	  float fDigiSum = 0.0;
	  for  (int ii = 0; ii < tool.size(); ++ii) {
	    int capid = (*ihbhe)[ii].capid();
	    fDigiSum += (tool[ii]-calibrations.pedestal(capid));
	  }
	  
	  mehHcalSHE[1]->Fill(fHFEnergySimHits[cell.rawId()]);
	  mehHcalAEE[1]->Fill(fDigiSum);
	  //Adding protection against FPE
	  if (fHFEnergySimHits[cell.rawId()]!=0){
	    mehHcalAEESHE[1]->Fill(fDigiSum/fHFEnergySimHits[cell.rawId()]);
	  }
	  //else{
	  //  std::cout<<"It would have been an FPE! with fHFEnergySimHits[cell.rawId()]==0!\n";
	  //}
	  mehHcalSHEvAEE[1]->Fill(fDigiSum, fHFEnergySimHits[cell.rawId()]);
	}
      }
    }
  
    if (verbosity > 1) {
      eventout += "\n          Number of HBDigis collected:.............. ";
      eventout += iHB;
    }
    mehHcaln[0]->Fill((float)iHB);
    
    if (verbosity > 1) {
      eventout += "\n          Number of HEDigis collected:.............. ";
      eventout += iHE;
    }
    mehHcaln[1]->Fill((float)iHE);
  }

  ////////////////////////
  // get HO information
  ///////////////////////
  edm::Handle<edm::SortedCollection<HODataFrame> > ho;
  iEvent.getByLabel(HCalDigi_,ho);
  bool validHO = true;
  if (!ho.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find HODataFrame in event!";
    validHO = false;
  }    
  if (validHO) {
    edm::SortedCollection<HODataFrame>::const_iterator iho;
    
    int iHO = 0; 
    for (iho = ho->begin(); iho != ho->end(); ++iho) {
      HcalDetId cell(iho->id()); 
      
      if (cell.subdet() == sdHcalOut) {
	
	//HCalconditions->makeHcalCalibration(cell, &calibrations);
	const HcalCalibrations& calibrations = 
	  HCalconditions->getHcalCalibrations(cell);
	const HcalQIECoder *channelCoder = HCalconditions->getHcalCoder(cell);
	const HcalQIEShape *shape = HCalconditions->getHcalShape(channelCoder);
	HcalCoderDb coder (*channelCoder, *shape);
	coder.adc2fC(*iho, tool);
	
	++iHO;
	float fDigiSum = 0.0;
	for  (int ii = 0; ii < tool.size(); ++ii) {
	  // default ped is 4.5
	  int capid = (*iho)[ii].capid();
	  fDigiSum += (tool[ii] - calibrations.pedestal(capid));
	}
	
	mehHcalSHE[2]->Fill(fHFEnergySimHits[cell.rawId()]);
	mehHcalAEE[2]->Fill(fDigiSum);
	//Adding protection against FPE
	if (fHFEnergySimHits[cell.rawId()]!=0){
	  mehHcalAEESHE[2]->Fill(fDigiSum/fHFEnergySimHits[cell.rawId()]);
	}
	//else{
	//  std::cout<<"It would have been an FPE! with fHFEnergySimHits[cell.rawId()]==0!\n";
	//}
	mehHcalSHEvAEE[2]->Fill(fDigiSum, fHFEnergySimHits[cell.rawId()]);
      }
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of HODigis collected:.............. ";
      eventout += iHO;
    }
    mehHcaln[2]->Fill((float)iHO);
  }  

  ////////////////////////
  // get HF information
  ///////////////////////
  edm::Handle<edm::SortedCollection<HFDataFrame> > hf;
  iEvent.getByLabel(HCalDigi_,hf);
  bool validHF = true;
  if (!hf.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find HFDataFrame in event!";
    validHF = false;
  }    
  if (validHF) {
    edm::SortedCollection<HFDataFrame>::const_iterator ihf;
    
    int iHF = 0; 
    for (ihf = hf->begin(); ihf != hf->end(); ++ihf) {
      HcalDetId cell(ihf->id()); 
      
      if (cell.subdet() == sdHcalFwd) {
	
	//HCalconditions->makeHcalCalibration(cell, &calibrations);
	const HcalCalibrations& calibrations = 
	  HCalconditions->getHcalCalibrations(cell);
	const HcalQIECoder *channelCoder = HCalconditions->getHcalCoder(cell);
	const HcalQIEShape *shape = HCalconditions->getHcalShape(channelCoder);
	HcalCoderDb coder (*channelCoder, *shape);
	coder.adc2fC(*ihf, tool);
	
	++iHF;
	float fDigiSum = 0.0;
	for  (int ii = 0; ii < tool.size(); ++ii) {
	  // default ped is 1.73077
	  int capid = (*ihf)[ii].capid();
	  fDigiSum += (tool[ii] - calibrations.pedestal(capid));
	}
	
	mehHcalSHE[3]->Fill(fHFEnergySimHits[cell.rawId()]);
	mehHcalAEE[3]->Fill(fDigiSum);
	//Adding protection against FPE
	if (fHFEnergySimHits[cell.rawId()]!=0){
	  mehHcalAEESHE[3]->Fill(fDigiSum/fHFEnergySimHits[cell.rawId()]);
	}
	//else{
	//  std::cout<<"It would have been an FPE! with fHFEnergySimHits[cell.rawId()]==0!\n";
	//}
	mehHcalSHEvAEE[3]->Fill(fDigiSum, fHFEnergySimHits[cell.rawId()]);
      }
    }
  
    if (verbosity > 1) {
      eventout += "\n          Number of HFDigis collected:.............. ";
      eventout += iHF;
    }
    mehHcaln[3]->Fill((float)iHF);
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  
  return;
}

void GlobalDigisAnalyzer::fillTrk(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();


  std::string MsgLoggerCat = "GlobalDigisAnalyzer_fillTrk";
  
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  
  
  // get strip information
  edm::Handle<edm::DetSetVector<SiStripDigi> > stripDigis;  
  iEvent.getByLabel(SiStripSrc_, stripDigis);
  bool validstripDigis = true;
  if (!stripDigis.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find stripDigis in event!";
    validstripDigis = false;
  }  
  
  if (validstripDigis) {
    int nStripBrl = 0, nStripFwd = 0;
    edm::DetSetVector<SiStripDigi>::const_iterator DSViter;
    for (DSViter = stripDigis->begin(); DSViter != stripDigis->end(); 
	 ++DSViter) {
      unsigned int id = DSViter->id;
      DetId detId(id);
      edm::DetSet<SiStripDigi>::const_iterator begin = DSViter->data.begin();
      edm::DetSet<SiStripDigi>::const_iterator end = DSViter->data.end();
      edm::DetSet<SiStripDigi>::const_iterator iter;
      
      // get TIB
      if (detId.subdetId() == sdSiTIB) {
	
	for (iter = begin; iter != end; ++iter) {
	  ++nStripBrl;
	  if (tTopo->tibLayer(id) == 1) {
	    mehSiStripADC[0]->Fill((*iter).adc());
	    mehSiStripStrip[0]->Fill((*iter).strip());
	  }
	  if (tTopo->tibLayer(id) == 2) {
	    mehSiStripADC[1]->Fill((*iter).adc());
	    mehSiStripStrip[1]->Fill((*iter).strip());
	  }	
	  if (tTopo->tibLayer(id) == 3) {
	    mehSiStripADC[2]->Fill((*iter).adc());
	    mehSiStripStrip[2]->Fill((*iter).strip());
	  }
	  if (tTopo->tibLayer(id) == 4) {
	    mehSiStripADC[3]->Fill((*iter).adc());
	    mehSiStripStrip[3]->Fill((*iter).strip());
	  }
	}
      }
      
      // get TOB
      if (detId.subdetId() == sdSiTOB) {
	
	for (iter = begin; iter != end; ++iter) {
	  ++nStripBrl;
	  if (tTopo->tobLayer(id) == 1) {
	    mehSiStripADC[4]->Fill((*iter).adc());
	    mehSiStripStrip[4]->Fill((*iter).strip());
	  }
	  if (tTopo->tobLayer(id) == 2) {
	    mehSiStripADC[5]->Fill((*iter).adc());
	    mehSiStripStrip[5]->Fill((*iter).strip());
	  }	
	  if (tTopo->tobLayer(id) == 3) {
	    mehSiStripADC[6]->Fill((*iter).adc());
	    mehSiStripStrip[6]->Fill((*iter).strip());
	  }
	  if (tTopo->tobLayer(id) == 4) {
	    mehSiStripADC[7]->Fill((*iter).adc());
	    mehSiStripStrip[7]->Fill((*iter).strip());
	  }
	}
      }    
      
      // get TID
      if (detId.subdetId() == sdSiTID) {
	
	for (iter = begin; iter != end; ++iter) {
	  ++nStripFwd;
	  if (tTopo->tidWheel(id) == 1) {
	    mehSiStripADC[8]->Fill((*iter).adc());
	    mehSiStripStrip[8]->Fill((*iter).strip());
	  }
	  if (tTopo->tidWheel(id) == 2) {
	    mehSiStripADC[9]->Fill((*iter).adc());
	    mehSiStripStrip[9]->Fill((*iter).strip());
	  }
	  if (tTopo->tidWheel(id) == 3) {
	    mehSiStripADC[10]->Fill((*iter).adc());
	    mehSiStripStrip[10]->Fill((*iter).strip());
	  }
	}
      }   
      
      // get TEC
      if (detId.subdetId() == sdSiTEC) {
	
	for (iter = begin; iter != end; ++iter) {
	  ++nStripFwd;
	  if (tTopo->tecWheel(id) == 1) {
	    mehSiStripADC[11]->Fill((*iter).adc());
	    mehSiStripStrip[11]->Fill((*iter).strip());
	  }
	  if (tTopo->tecWheel(id) == 2) {
	    mehSiStripADC[12]->Fill((*iter).adc());
	    mehSiStripStrip[12]->Fill((*iter).strip());
	  }
	  if (tTopo->tecWheel(id) == 3) {
	    mehSiStripADC[13]->Fill((*iter).adc());
	    mehSiStripStrip[13]->Fill((*iter).strip());
	  }
	  if (tTopo->tecWheel(id) == 4) {
	    mehSiStripADC[14]->Fill((*iter).adc());
	    mehSiStripStrip[14]->Fill((*iter).strip());
	  }
	  if (tTopo->tecWheel(id) == 5) {
	    mehSiStripADC[15]->Fill((*iter).adc());
	    mehSiStripStrip[15]->Fill((*iter).strip());
	  }
	  if (tTopo->tecWheel(id) == 6) {
	    mehSiStripADC[16]->Fill((*iter).adc());
	    mehSiStripStrip[16]->Fill((*iter).strip());
	  }
	  if (tTopo->tecWheel(id) == 7) {
	    mehSiStripADC[17]->Fill((*iter).adc());
	    mehSiStripStrip[17]->Fill((*iter).strip());
	  }
	  if (tTopo->tecWheel(id) == 8) {
	    mehSiStripADC[18]->Fill((*iter).adc());
	    mehSiStripStrip[18]->Fill((*iter).strip());
	  }
	}
      }     
    } // end loop over DataSetVector
    
    if (verbosity > 1) {
      eventout += "\n          Number of BrlStripDigis collected:........ ";
      eventout += nStripBrl;
    }
    for(int i = 0; i < 8; ++i) {
      mehSiStripn[i]->Fill((float)nStripBrl);
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of FrwdStripDigis collected:....... ";
      eventout += nStripFwd;
    }
    for(int i = 8; i < 19; ++i) {
      mehSiStripn[i]->Fill((float)nStripFwd);
    }
  }

  // get pixel information
  edm::Handle<edm::DetSetVector<PixelDigi> > pixelDigis;  
  iEvent.getByLabel(SiPxlSrc_, pixelDigis);
  bool validpixelDigis = true;
  if (!pixelDigis.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find pixelDigis in event!";
    validpixelDigis = false;
  }  
  if (validpixelDigis) {
    int nPxlBrl = 0, nPxlFwd = 0;
    edm::DetSetVector<PixelDigi>::const_iterator DPViter;
    for (DPViter = pixelDigis->begin(); DPViter != pixelDigis->end(); 
	 ++DPViter) {
      unsigned int id = DPViter->id;
      DetId detId(id);
      edm::DetSet<PixelDigi>::const_iterator begin = DPViter->data.begin();
      edm::DetSet<PixelDigi>::const_iterator end = DPViter->data.end();
      edm::DetSet<PixelDigi>::const_iterator iter;
      
      // get Barrel pixels
      if (detId.subdetId() == sdPxlBrl) {
	
	for (iter = begin; iter != end; ++iter) {
	  ++nPxlBrl;
	  if (tTopo->pxbLayer(id) == 1) {
	    mehSiPixelADC[0]->Fill((*iter).adc());
	    mehSiPixelRow[0]->Fill((*iter).row());
	    mehSiPixelCol[0]->Fill((*iter).column());
	  }
	  if (tTopo->pxbLayer(id) == 2) {
	    mehSiPixelADC[1]->Fill((*iter).adc());
	    mehSiPixelRow[1]->Fill((*iter).row());
	    mehSiPixelCol[1]->Fill((*iter).column());
	  }
	  if (tTopo->pxbLayer(id) == 3) {
	    mehSiPixelADC[2]->Fill((*iter).adc());
	    mehSiPixelRow[2]->Fill((*iter).row());
	    mehSiPixelCol[2]->Fill((*iter).column());
	  }
	}
      }
      
      // get Forward pixels
      if (detId.subdetId() == sdPxlFwd) {
	
	for (iter = begin; iter != end; ++iter) {
	  ++nPxlFwd;
	  if (tTopo->pxfDisk(id) == 1) {
	    if (tTopo->pxfSide(id) == 1) {
	      mehSiPixelADC[4]->Fill((*iter).adc());
	      mehSiPixelRow[4]->Fill((*iter).row());
	      mehSiPixelCol[4]->Fill((*iter).column());
	    }
	    if (tTopo->pxfSide(id) == 2) {
	      mehSiPixelADC[3]->Fill((*iter).adc());
	      mehSiPixelRow[3]->Fill((*iter).row());
	      mehSiPixelCol[3]->Fill((*iter).column());
	    }
	  }
	  if (tTopo->pxfDisk(id) == 2) {
	    if (tTopo->pxfSide(id) == 1) {
	      
	      mehSiPixelADC[6]->Fill((*iter).adc());
	      mehSiPixelRow[6]->Fill((*iter).row());
	      mehSiPixelCol[6]->Fill((*iter).column());
	    }
	    if (tTopo->pxfSide(id) == 2) {
	      mehSiPixelADC[5]->Fill((*iter).adc());
	      mehSiPixelRow[5]->Fill((*iter).row());
	      mehSiPixelCol[5]->Fill((*iter).column());
	    }
	  }
	}
      }
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of BrlPixelDigis collected:........ ";
      eventout += nPxlBrl;
    }
    for(int i = 0; i < 3; ++i) {
      mehSiPixeln[i]->Fill((float)nPxlBrl);
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of FrwdPixelDigis collected:....... ";
      eventout += nPxlFwd;
    }
    
    for(int i = 3; i < 7; ++i) {
      mehSiPixeln[i]->Fill((float)nPxlFwd);
    }
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  
  return;
}

void GlobalDigisAnalyzer::fillMuon(const edm::Event& iEvent, 
				   const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalDigisAnalyzer_fillMuon";
  
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  
  
  // get DT information
  edm::Handle<DTDigiCollection> dtDigis;  
  iEvent.getByLabel(MuDTSrc_, dtDigis);
  bool validdtDigis = true;
  if (!dtDigis.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find dtDigis in event!";
    validdtDigis = false;
  }  
  if (validdtDigis) {
    int nDt0 = 0; int nDt1 = 0; int nDt2 = 0; int nDt3 = 0;
    int nDt = 0;
    DTDigiCollection::DigiRangeIterator detUnitIt;
    for (detUnitIt = dtDigis->begin(); detUnitIt != dtDigis->end(); 
	 ++detUnitIt) {
      
      const DTLayerId& id = (*detUnitIt).first;
      const DTDigiCollection::Range& range = (*detUnitIt).second;
      for (DTDigiCollection::const_iterator digiIt = range.first;
	   digiIt != range.second;
	   ++digiIt) {
	
	++nDt;
	
	DTWireId wireId(id,(*digiIt).wire());
	if (wireId.station() == 1) {
	  mehDtMuonLayer[0]->Fill(id.layer());
	  mehDtMuonTime[0]->Fill((*digiIt).time());
	  mehDtMuonTimevLayer[0]->Fill(id.layer(),(*digiIt).time());
	  ++nDt0;
	}
	if (wireId.station() == 2) {
	  mehDtMuonLayer[1]->Fill(id.layer());
	  mehDtMuonTime[1]->Fill((*digiIt).time());
	  mehDtMuonTimevLayer[1]->Fill(id.layer(),(*digiIt).time());
	  ++nDt1;
	}
	if (wireId.station() == 3) {
	  mehDtMuonLayer[2]->Fill(id.layer());
	  mehDtMuonTime[2]->Fill((*digiIt).time());
	  mehDtMuonTimevLayer[2]->Fill(id.layer(),(*digiIt).time());
	  ++nDt2;
	}
	if (wireId.station() == 4) {
	  mehDtMuonLayer[3]->Fill(id.layer());
	  mehDtMuonTime[3]->Fill((*digiIt).time());
	  mehDtMuonTimevLayer[3]->Fill(id.layer(),(*digiIt).time());
	  ++nDt3;
	}
      }
    }
    mehDtMuonn[0]->Fill((float)nDt0);
    mehDtMuonn[1]->Fill((float)nDt1);
    mehDtMuonn[2]->Fill((float)nDt2);
    mehDtMuonn[3]->Fill((float)nDt3);
    
    
    if (verbosity > 1) {
      eventout += "\n          Number of DtMuonDigis collected:.......... ";
      eventout += nDt;
    }
  }

  // get CSC Strip information
  edm::Handle<CSCStripDigiCollection> strips;  
  iEvent.getByLabel(MuCSCStripSrc_, strips);
  bool validstrips = true;
  if (!strips.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find muon strips in event!";
    validstrips = false;
  }
  
  if (validstrips) {
    int nStrips = 0;
    for (CSCStripDigiCollection::DigiRangeIterator j = strips->begin();
	 j != strips->end();
	 ++j) {
      
      std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
      std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
      
      for ( ; digiItr != last; ++digiItr) {
	++nStrips;
	
	// average pedestals
	std::vector<int> adcCounts = digiItr->getADCCounts();
	theCSCStripPedestalSum += adcCounts[0];
	theCSCStripPedestalSum += adcCounts[1];
	theCSCStripPedestalCount += 2;
	
	// if there are enough pedestal statistics
	if (theCSCStripPedestalCount > 100) {
	  float pedestal = theCSCStripPedestalSum / theCSCStripPedestalCount;
	  if (adcCounts[5] > (pedestal + 100)) 
	    mehCSCStripADC->Fill(adcCounts[4] - pedestal);
	}
      }
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of CSCStripDigis collected:........ ";
      eventout += nStrips;
    }
    mehCSCStripn->Fill((float)nStrips);
  }

  // get CSC Wire information
  edm::Handle<CSCWireDigiCollection> wires;  
  iEvent.getByLabel(MuCSCWireSrc_, wires);
  bool validwires = true;
  if (!wires.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find muon wires in event!";
    validwires = false;
  }  
  
  if (validwires) {
    int nWires = 0;
    for (CSCWireDigiCollection::DigiRangeIterator j = wires->begin();
	 j != wires->end();
	 ++j) {
      
      std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
      std::vector<CSCWireDigi>::const_iterator endDigi = (*j).second.second;
      
      for ( ; digiItr != endDigi; ++digiItr) {
	++nWires;
	mehCSCWireTime->Fill(digiItr->getTimeBin());
      }
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of CSCWireDigis collected:......... ";
      eventout += nWires;
    }
    mehCSCWiren->Fill((float)nWires); 
  }

  // get RPC information
  // Get the RPC Geometry
  edm::ESHandle<RPCGeometry> rpcGeom;
  iSetup.get<MuonGeometryRecord>().get(rpcGeom);
  if (!rpcGeom.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find RPCGeometryRecord in event!";
    return;
  }
  
  edm::Handle<edm::PSimHitContainer> rpcsimHit;
  iEvent.getByLabel("g4SimHits", "MuonRPCHits", rpcsimHit);
  bool validrpcsim = true;
  if (!rpcsimHit.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find rpcsimHit in event!";
    validrpcsim = false;
  }   
  
  edm::Handle<RPCDigiCollection> rpcDigis;  
  iEvent.getByLabel(MuRPCSrc_, rpcDigis);
  bool validrpcdigi = true;
  if (!rpcDigis.isValid()) {
    LogDebug(MsgLoggerCat)
      << "Unable to find rpcDigis in event!";
    validrpcdigi = false;
  } 

  // ONLY UNTIL PROBLEM WITH RPC DIGIS IS FIGURED OUT
  validrpcdigi = false;

  // Loop on simhits
  edm::PSimHitContainer::const_iterator rpcsimIt;
  std::map<RPCDetId, std::vector<double> > allsims;

  if (validrpcsim) {
    for (rpcsimIt = rpcsimHit->begin(); rpcsimIt != rpcsimHit->end(); 
	 rpcsimIt++) {
      RPCDetId Rsid = (RPCDetId)(*rpcsimIt).detUnitId();
      int ptype = rpcsimIt->particleType();
      
      if (ptype == 13 || ptype == -13) {
	std::vector<double> buff;
	if (allsims.find(Rsid) != allsims.end() ){
	  buff= allsims[Rsid];
	}
	buff.push_back(rpcsimIt->localPosition().x());
	allsims[Rsid]=buff;
      }
    }
  }

  // CRASH HAPPENS SOMEWHERE HERE IN FOR DECLARATION
  // WHAT IS WRONG WITH rpcDigis?????
  if (validrpcdigi) {
    int nRPC = 0;
    RPCDigiCollection::DigiRangeIterator rpcdetUnitIt;
    for (rpcdetUnitIt = rpcDigis->begin(); rpcdetUnitIt != rpcDigis->end(); 
	 ++rpcdetUnitIt) {
      
      const RPCDetId Rsid = (*rpcdetUnitIt).first;      
      const RPCRoll* roll = 
	dynamic_cast<const RPCRoll* >( rpcGeom->roll(Rsid));   
      const RPCDigiCollection::Range& range = (*rpcdetUnitIt).second;
      
      std::vector<double> sims;
      if (allsims.find(Rsid) != allsims.end() ){
	sims = allsims[Rsid];
      }
      
      int ndigi = 0;
      for (RPCDigiCollection::const_iterator rpcdigiIt = range.first;
	   rpcdigiIt != range.second;
	   ++rpcdigiIt) {
	
	++ndigi;
	++nRPC;
      }
      
      if (sims.size() == 1 && ndigi == 1){
	double dis = roll->centreOfStrip(range.first->strip()).x()-sims[0];
	
	if (Rsid.region() == 0 ){
	  if (Rsid.ring() == -2)
	    mehRPCRes[0]->Fill((float)dis);
	  else if (Rsid.ring() == -1)
	    mehRPCRes[1]->Fill((float)dis);
	  else if (Rsid.ring() == 0)
	    mehRPCRes[2]->Fill((float)dis);
	  else if (Rsid.ring() == 1)
	    mehRPCRes[3]->Fill((float)dis);
	  else if (Rsid.ring() == 2)
	    mehRPCRes[4]->Fill((float)dis);
	}  
      }
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of RPCDigis collected:.............. ";
      eventout += nRPC;
    }
    mehRPCMuonn->Fill(float(nRPC));
  }
  
  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  
  return;
}
