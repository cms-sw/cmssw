/** \file GlobalRecHitsAnalyzer.cc
 *  
 *  See header file for description of class
 *
 *  \author M. Strang SUNY-Buffalo
 *  Testing by Ken Smith
 */
using namespace std;
#include "Validation/GlobalRecHits/interface/GlobalRecHitsHistogrammer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

GlobalRecHitsHistogrammer::GlobalRecHitsHistogrammer(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), frequency(0), label(""), getAllProvenances(false),
  printProvenanceInfo(false), count(0)
{
  std::string MsgLoggerCat = "GlobalRecHitsAnalyzer_GlobalRecHitsAnalyzer";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  outputfile = iPSet.getParameter<std::string>("outputFile");
  doOutput = iPSet.getParameter<bool>("DoOutput");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

  //get Labels to use to extract information
  GlobalRecHitSrc_ = iPSet.getParameter<edm::InputTag>("GlobalRecHitSrc");
  GlobalRecHitSrc_Token_ = consumes<PGlobalRecHit>(iPSet.getParameter<edm::InputTag>("GlobalRecHitSrc"));
  // ECalEBSrc_ = iPSet.getParameter<edm::InputTag>("ECalEBSrc");
  //ECalUncalEBSrc_ = iPSet.getParameter<edm::InputTag>("ECalUncalEBSrc");
  //ECalEESrc_ = iPSet.getParameter<edm::InputTag>("ECalEESrc");
  //ECalUncalEESrc_ = iPSet.getParameter<edm::InputTag>("ECalUncalEESrc");
  //ECalESSrc_ = iPSet.getParameter<edm::InputTag>("ECalESSrc");
  //HCalSrc_ = iPSet.getParameter<edm::InputTag>("HCalSrc");
  //SiStripSrc_ = iPSet.getParameter<edm::InputTag>("SiStripSrc"); 
  //SiPxlSrc_ = iPSet.getParameter<edm::InputTag>("SiPxlSrc");
  //MuDTSrc_ = iPSet.getParameter<edm::InputTag>("MuDTSrc");
  //MuDTSimSrc_ = iPSet.getParameter<edm::InputTag>("MuDTSimSrc");
  //MuCSCSrc_ = iPSet.getParameter<edm::InputTag>("MuCSCSrc");
  //MuRPCSrc_ = iPSet.getParameter<edm::InputTag>("MuRPCSrc");
  //MuRPCSimSrc_ = iPSet.getParameter<edm::InputTag>("MuRPCSimSrc");

  //conf_ = iPSet;

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
      << "    OutputFile     = " << outputfile << "\n"
      << "    DoOutput      = " << doOutput << "\n"
      << "    GetProv        = " << getAllProvenances << "\n"
      << "    PrintProv      = " << printProvenanceInfo << "\n"
      << "    Global Src     = " << GlobalRecHitSrc_ << "\n"
      << "===============================\n";
      
  }

}


GlobalRecHitsHistogrammer::~GlobalRecHitsHistogrammer()
{
}

void GlobalRecHitsHistogrammer::bookHistograms(DQMStore::IBooker & ibooker,
  edm::Run const &, edm::EventSetup const & ){

//monitor elements

//Si Strip
  string SiStripString[19] = {"TECW1", "TECW2", "TECW3", "TECW4", "TECW5", "TECW6",
     "TECW7", "TECW8", "TIBL1", "TIBL2", "TIBL3", "TIBL4", "TIDW1", "TIDW2", "TIDW3",
     "TOBL1", "TOBL2", "TOBL3", "TOBL4"};

  for (int i = 0; i < 19; ++i) {
    mehSiStripn[i] = 0;
    mehSiStripResX[i] = 0;
    mehSiStripResY[i] = 0;
  }

  string hcharname, hchartitle;
  ibooker.setCurrentFolder("GlobalRecHitsV/SiStrips");
  for (int amend = 0; amend < 19; ++amend) {
    hcharname = "hSiStripn_" + SiStripString[amend];
    hchartitle= SiStripString[amend] + "  rechits";
    mehSiStripn[amend] = ibooker.book1D(hcharname, hchartitle, 20, 0., 20.);
    mehSiStripn[amend]->setAxisTitle("Number of hits in " + SiStripString[amend], 1);
    mehSiStripn[amend]->setAxisTitle("Count", 2);

    hcharname = "hSiStripResX_" + SiStripString[amend];
    hchartitle = SiStripString[amend] + " rechit x resolution";
    mehSiStripResX[amend] = ibooker.book1D(hcharname, hchartitle, 200, -0.02, .02);
    mehSiStripResX[amend]->setAxisTitle("X-resolution in " + SiStripString[amend], 1);
    mehSiStripResX[amend]->setAxisTitle("Count", 2);

    hcharname = "hSiStripResY_" + SiStripString[amend];
    hchartitle = SiStripString[amend] + " rechit y resolution";
    mehSiStripResY[amend] = ibooker.book1D(hcharname, hchartitle, 200, -0.02, .02);
    mehSiStripResY[amend]->setAxisTitle("Y-resolution in " + SiStripString[amend], 1);
    mehSiStripResY[amend]->setAxisTitle("Count", 2);
  }


  //HCal
  //string hcharname, hchartitle;
  string HCalString[4] = {"HB", "HE", "HF", "HO"};
  float HCalnUpper[4] = {3000., 3000., 3000., 2000.};
  float HCalnLower[4] = {2000., 2000., 2000., 1000.};
  for (int j =0; j <4; ++j) {
    mehHcaln[j] = 0;
    mehHcalRes[j] = 0;
  }

  ibooker.setCurrentFolder("GlobalRecHitsV/HCals");
  for (int amend = 0; amend < 4; ++amend) {
    hcharname = "hHcaln_" + HCalString[amend];
    hchartitle = HCalString[amend]+"  rechits";
    mehHcaln[amend] = ibooker.book1D(hcharname, hchartitle, 500, HCalnLower[amend],
        HCalnUpper[amend]);

    mehHcaln[amend]->setAxisTitle("Number of RecHits", 1);
    mehHcaln[amend]->setAxisTitle("Count", 2);

    hcharname = "hHcalRes_" + HCalString[amend];
    hchartitle = HCalString[amend] + "  rechit resolution";
    mehHcalRes[amend] = ibooker.book1D(hcharname, hchartitle, 25, -2., 2.);
    mehHcalRes[amend]->setAxisTitle("RecHit E - SimHit E", 1);
    mehHcalRes[amend]->setAxisTitle("Count", 2);
  }


  //Ecal
  string ECalString[3] = {"EB", "EE", "ES"};
  int ECalnBins[3] = {700, 100, 50};
  float ECalnUpper[3] = {20000., 62000., 300.};
  float ECalnLower[3] = {6000., 60000., 100.};
  int ECalResBins[3] = {200, 200, 200};
  float ECalResUpper[3] = {1., 0.3, .0002};
  float ECalResLower[3] = {-1., -0.3, -.0002};
  for (int i = 0; i < 3; ++i) {
    mehEcaln[i] = 0;
    mehEcalRes[i] = 0;
  }

  ibooker.setCurrentFolder("GlobalRecHitsV/ECals");
  for (int amend = 0; amend < 3; ++amend) {
    hcharname = "hEcaln_" + ECalString[amend];
    hchartitle = ECalString[amend] + "  rechits";
    mehEcaln[amend] = ibooker.book1D(hcharname, hchartitle, ECalnBins[amend],
        ECalnLower[amend], ECalnUpper[amend]);

    mehEcaln[amend]->setAxisTitle("Number of RecHits", 1);
    mehEcaln[amend]->setAxisTitle("Count", 2);

    hcharname = "hEcalRes_" + ECalString[amend];
    hchartitle = ECalString[amend] + "  rechit resolution";
    mehEcalRes[amend] = ibooker.book1D(hcharname, hchartitle, ECalResBins[amend],
        ECalResLower[amend], ECalResUpper[amend]);

    mehEcalRes[amend]->setAxisTitle("RecHit E - SimHit E", 1);
    mehEcalRes[amend]->setAxisTitle("Count", 2);
  }


  //Si Pixels
  string SiPixelString[7] = {"BRL1", "BRL2", "BRL3", "FWD1n", "FWD1p", "FWD2n", "FWD2p"};
  for (int j =0; j < 7; ++j) {
    mehSiPixeln[j] = 0;
    mehSiPixelResX[j] = 0;
    mehSiPixelResY[j] = 0;
  }

  ibooker.setCurrentFolder("GlobalRecHitsV/SiPixels");
  for (int amend = 0; amend < 7; ++amend) {
    hcharname = "hSiPixeln_" + SiPixelString[amend];
    hchartitle= SiPixelString[amend] + " rechits";
    mehSiPixeln[amend] = ibooker.book1D(hcharname, hchartitle, 20, 0., 20.);
    mehSiPixeln[amend]->setAxisTitle("Number of hits in " + SiPixelString[amend], 1);
    mehSiPixeln[amend]->setAxisTitle("Count", 2);

    hcharname = "hSiPixelResX_" + SiPixelString[amend];
    hchartitle= SiPixelString[amend] + " rechit x resolution";
    mehSiPixelResX[amend] = ibooker.book1D(hcharname, hchartitle, 200, -0.02, .02);
    mehSiPixelResX[amend]->setAxisTitle("X-resolution in " + SiPixelString[amend], 1);
    mehSiPixelResX[amend]->setAxisTitle("Count", 2);

    hcharname = "hSiPixelResY_" + SiPixelString[amend];
    hchartitle= SiPixelString[amend] + " rechit y resolution";
    mehSiPixelResY[amend] = ibooker.book1D(hcharname, hchartitle, 200, -0.02, .02);
    mehSiPixelResY[amend]->setAxisTitle("Y-resolution in "+SiPixelString[amend], 1);
    mehSiPixelResY[amend]->setAxisTitle("Count", 2);
  }

  //Muons
  ibooker.setCurrentFolder("GlobalRecHitsV/Muons");
  mehDtMuonn = 0;
  mehCSCn = 0;
  mehRPCn = 0;

  //std::vector<MonitorElement *> me_List = {mehDtMuonn, mehCSCn, mehRPCn};
  string n_List[3] = {"hDtMuonn", "hCSCn", "hRPCn"};
  //float hist_prop[3] = [25., 0., 50.];
  string hist_string[3] = {"Dt", "CSC", "RPC"};

  for (int amend = 0; amend < 3; ++amend) {
    hchartitle = hist_string[amend] + " rechits";
    if (amend == 0) {
      mehDtMuonn = ibooker.book1D(n_List[amend], hchartitle, 25, 0., 50.);
      mehDtMuonn->setAxisTitle("Number of Rechits", 1);
      mehDtMuonn->setAxisTitle("Count", 2);
    }
    if (amend == 1) {
      mehCSCn = ibooker.book1D(n_List[amend], hchartitle, 25, 0., 50.);
      mehCSCn->setAxisTitle("Number of Rechits", 1);
      mehCSCn->setAxisTitle("Count", 2);
    }
    if (amend == 2) {
      mehRPCn = ibooker.book1D(n_List[amend], hchartitle, 25, 0., 50.);
      mehRPCn->setAxisTitle("Number of Rechits", 1);
      mehRPCn->setAxisTitle("Count", 2);
    }
  }

  mehDtMuonRes = 0;
  mehCSCResRDPhi = 0;
  mehRPCResX = 0;

  hcharname = "hDtMuonRes";
  hchartitle = "DT wire distance resolution";
  mehDtMuonRes = ibooker.book1D(hcharname, hchartitle, 200, -0.2, 0.2);

  hcharname = "CSCResRDPhi";
  hchartitle = "CSC perp*dphi resolution";
  mehCSCResRDPhi = ibooker.book1D(hcharname, hchartitle, 200, -0.2, 0.2);

  hcharname = "hRPCResX";
  hchartitle = "RPC rechits x resolution";
  mehRPCResX = ibooker.book1D(hcharname, hchartitle, 50, -5., 5.);

}

void GlobalRecHitsHistogrammer::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsHistogrammer_analyze";

  // keep track of number of events processed
  ++count;

  // get event id information
  edm::RunNumber_t nrun = iEvent.id().run();
  edm::EventNumber_t nevt = iEvent.id().event();

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

  // clear event holders
  //clear();  Not in example I'm using, thus I comment it out.
 
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
	//eventout += (AllProv[i]->product).moduleLabel();
	eventout += AllProv[i]->moduleLabel();
	eventout += "\n       ProductID    : ";
	//eventout += (AllProv[i]->product).productID_.id_;
	eventout += AllProv[i]->productID().id();
	eventout += "\n       ClassName    : ";
	//eventout += (AllProv[i]->product).fullClassName_;
	eventout += AllProv[i]->className();
	eventout += "\n       InstanceName : ";
	//eventout += (AllProv[i]->product).productInstanceName_;
	eventout += AllProv[i]->productInstanceName();
	eventout += "\n       BranchName   : ";
	//eventout += (AllProv[i]->product).branchName_;
	eventout += AllProv[i]->branchName();
      }
      eventout += "\n       ******************************\n";
      edm::LogInfo(MsgLoggerCat) << eventout << "\n";
      printProvenanceInfo = false;
    }
    getAllProvenances = false;
  }

edm::Handle<PGlobalRecHit> srcGlobalRecHits;
  iEvent.getByToken(GlobalRecHitSrc_Token_,srcGlobalRecHits);
  if (!srcGlobalRecHits.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find PGlobalRecHit in event!";
    return;
  }
    
    int nEBCalRecHits = srcGlobalRecHits->getnEBCalRecHits();
    int nEECalRecHits = srcGlobalRecHits->getnEECalRecHits();
    int nESCalRecHits = srcGlobalRecHits->getnESCalRecHits();

    int nHBCalRecHits = srcGlobalRecHits->getnHBCalRecHits();
    int nHECalRecHits = srcGlobalRecHits->getnHECalRecHits();
    int nHOCalRecHits = srcGlobalRecHits->getnHOCalRecHits();
    int nHFCalRecHits = srcGlobalRecHits->getnHFCalRecHits();        

    int nTIBL1RecHits = srcGlobalRecHits->getnTIBL1RecHits();    
    int nTIBL2RecHits = srcGlobalRecHits->getnTIBL2RecHits();    
    int nTIBL3RecHits = srcGlobalRecHits->getnTIBL3RecHits();    
    int nTIBL4RecHits = srcGlobalRecHits->getnTIBL4RecHits();    
    int nTOBL1RecHits = srcGlobalRecHits->getnTOBL1RecHits();    
    int nTOBL2RecHits = srcGlobalRecHits->getnTOBL2RecHits();    
    int nTOBL3RecHits = srcGlobalRecHits->getnTOBL3RecHits();    
    int nTOBL4RecHits = srcGlobalRecHits->getnTOBL4RecHits();    
    int nTIDW1RecHits = srcGlobalRecHits->getnTIDW1RecHits();    
    int nTIDW2RecHits = srcGlobalRecHits->getnTIDW2RecHits();    
    int nTIDW3RecHits = srcGlobalRecHits->getnTIDW3RecHits();    
    int nTECW1RecHits = srcGlobalRecHits->getnTECW1RecHits();    
    int nTECW2RecHits = srcGlobalRecHits->getnTECW2RecHits();    
    int nTECW3RecHits = srcGlobalRecHits->getnTECW3RecHits();  
    int nTECW4RecHits = srcGlobalRecHits->getnTECW4RecHits();    
    int nTECW5RecHits = srcGlobalRecHits->getnTECW5RecHits();    
    int nTECW6RecHits = srcGlobalRecHits->getnTECW6RecHits();  
    int nTECW7RecHits = srcGlobalRecHits->getnTECW7RecHits();    
    int nTECW8RecHits = srcGlobalRecHits->getnTECW8RecHits();

    int nBRL1RecHits = srcGlobalRecHits->getnBRL1RecHits();    
    int nBRL2RecHits = srcGlobalRecHits->getnBRL2RecHits();    
    int nBRL3RecHits = srcGlobalRecHits->getnBRL3RecHits();       
    int nFWD1nRecHits = srcGlobalRecHits->getnFWD1nRecHits();
    int nFWD1pRecHits = srcGlobalRecHits->getnFWD1pRecHits();    
    int nFWD2nRecHits = srcGlobalRecHits->getnFWD2nRecHits();    
    int nFWD2pRecHits = srcGlobalRecHits->getnFWD2pRecHits(); 
  
    int nDTRecHits = srcGlobalRecHits->getnDTRecHits();  

    int nCSCRecHits = srcGlobalRecHits->getnCSCRecHits();

    int nRPCRecHits = srcGlobalRecHits->getnRPCRecHits();

    // get Ecal info
    std::vector<PGlobalRecHit::ECalRecHit> EECalRecHits = 
      srcGlobalRecHits->getEECalRecHits();
    mehEcaln[0]->Fill((float)nEECalRecHits);
    for (unsigned int i = 0; i < EECalRecHits.size(); ++i) {
      mehEcalRes[0]->Fill(EECalRecHits[i].RE - EECalRecHits[i].SHE);
    }
    
    std::vector<PGlobalRecHit::ECalRecHit> EBCalRecHits = 
      srcGlobalRecHits->getEBCalRecHits();
    mehEcaln[1]->Fill((float)nEBCalRecHits);
    for (unsigned int i = 0; i < EBCalRecHits.size(); ++i) {
      mehEcalRes[1]->Fill(EBCalRecHits[i].RE - EBCalRecHits[i].SHE);
    }

    std::vector<PGlobalRecHit::ECalRecHit> ESCalRecHits = 
      srcGlobalRecHits->getESCalRecHits();
    mehEcaln[2]->Fill((float)nESCalRecHits);
    for (unsigned int i = 0; i < ESCalRecHits.size(); ++i) {
      mehEcalRes[2]->Fill(ESCalRecHits[i].RE - ESCalRecHits[i].SHE);
    }

    // Get HCal info
    std::vector<PGlobalRecHit::HCalRecHit> HBCalRecHits = 
      srcGlobalRecHits->getHBCalRecHits();
    mehHcaln[0]->Fill((float)nHBCalRecHits);
    for (unsigned int i = 0; i < HBCalRecHits.size(); ++i) {
      mehHcalRes[0]->Fill(HBCalRecHits[i].REC - HBCalRecHits[i].SHE); 
    }

    std::vector<PGlobalRecHit::HCalRecHit> HECalRecHits = 
      srcGlobalRecHits->getHECalRecHits();
    mehHcaln[1]->Fill((float)nHECalRecHits);
    for (unsigned int i = 0; i < HECalRecHits.size(); ++i) {
      mehHcalRes[1]->Fill(HECalRecHits[i].REC - HECalRecHits[i].SHE); 
    }

    std::vector<PGlobalRecHit::HCalRecHit> HOCalRecHits = 
      srcGlobalRecHits->getHOCalRecHits();
    mehHcaln[2]->Fill((float)nHOCalRecHits);
    for (unsigned int i = 0; i < HOCalRecHits.size(); ++i) {
      mehHcalRes[2]->Fill(HOCalRecHits[i].REC - HOCalRecHits[i].SHE); 
    }

    std::vector<PGlobalRecHit::HCalRecHit> HFCalRecHits = 
      srcGlobalRecHits->getHFCalRecHits();
    mehHcaln[3]->Fill((float)nHFCalRecHits);
    for (unsigned int i = 0; i < HFCalRecHits.size(); ++i) {
      mehHcalRes[3]->Fill(HFCalRecHits[i].REC - HFCalRecHits[i].SHE); 
    }

    // get SiStrip info
    std::vector<PGlobalRecHit::SiStripRecHit> TIBL1RecHits =
      srcGlobalRecHits->getTIBL1RecHits();      
    mehSiStripn[0]->Fill((float)nTIBL1RecHits);
    for (unsigned int i = 0; i < TIBL1RecHits.size(); ++i) {
      mehSiStripResX[0]->Fill(TIBL1RecHits[i].RX - TIBL1RecHits[i].SX);
      mehSiStripResY[0]->Fill(TIBL1RecHits[i].RY - TIBL1RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TIBL2RecHits =
      srcGlobalRecHits->getTIBL2RecHits();      
    mehSiStripn[1]->Fill((float)nTIBL2RecHits);
    for (unsigned int i = 0; i < TIBL2RecHits.size(); ++i) {
      mehSiStripResX[1]->Fill(TIBL2RecHits[i].RX - TIBL2RecHits[i].SX);
      mehSiStripResY[1]->Fill(TIBL2RecHits[i].RY - TIBL2RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TIBL3RecHits =
      srcGlobalRecHits->getTIBL3RecHits();      
    mehSiStripn[2]->Fill((float)nTIBL3RecHits);
    for (unsigned int i = 0; i < TIBL3RecHits.size(); ++i) {
      mehSiStripResX[2]->Fill(TIBL3RecHits[i].RX - TIBL3RecHits[i].SX);
      mehSiStripResY[2]->Fill(TIBL3RecHits[i].RY - TIBL3RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TIBL4RecHits =
      srcGlobalRecHits->getTIBL4RecHits();      
    mehSiStripn[3]->Fill((float)nTIBL4RecHits);
    for (unsigned int i = 0; i < TIBL4RecHits.size(); ++i) {
      mehSiStripResX[3]->Fill(TIBL4RecHits[i].RX - TIBL4RecHits[i].SX);
      mehSiStripResY[3]->Fill(TIBL4RecHits[i].RY - TIBL4RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TOBL1RecHits =
      srcGlobalRecHits->getTOBL1RecHits();      
    mehSiStripn[4]->Fill((float)nTOBL1RecHits);
    for (unsigned int i = 0; i < TOBL1RecHits.size(); ++i) {
      mehSiStripResX[4]->Fill(TOBL1RecHits[i].RX - TOBL1RecHits[i].SX);
      mehSiStripResY[4]->Fill(TOBL1RecHits[i].RY - TOBL1RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TOBL2RecHits =
      srcGlobalRecHits->getTOBL2RecHits();      
    mehSiStripn[5]->Fill((float)nTOBL2RecHits);
    for (unsigned int i = 0; i < TOBL2RecHits.size(); ++i) {
      mehSiStripResX[5]->Fill(TOBL2RecHits[i].RX - TOBL2RecHits[i].SX);
      mehSiStripResY[5]->Fill(TOBL2RecHits[i].RY - TOBL2RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TOBL3RecHits =
      srcGlobalRecHits->getTOBL3RecHits();      
    mehSiStripn[6]->Fill((float)nTOBL3RecHits);
    for (unsigned int i = 0; i < TOBL3RecHits.size(); ++i) {
      mehSiStripResX[6]->Fill(TOBL3RecHits[i].RX - TOBL3RecHits[i].SX);
      mehSiStripResY[6]->Fill(TOBL3RecHits[i].RY - TOBL3RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TOBL4RecHits =
      srcGlobalRecHits->getTOBL4RecHits();      
    mehSiStripn[7]->Fill((float)nTOBL4RecHits);
    for (unsigned int i = 0; i < TOBL4RecHits.size(); ++i) {
      mehSiStripResX[7]->Fill(TOBL4RecHits[i].RX - TOBL4RecHits[i].SX);
      mehSiStripResY[7]->Fill(TOBL4RecHits[i].RY - TOBL4RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TIDW1RecHits =
      srcGlobalRecHits->getTIDW1RecHits();      
    mehSiStripn[8]->Fill((float)nTIDW1RecHits);
    for (unsigned int i = 0; i < TIDW1RecHits.size(); ++i) {
      mehSiStripResX[8]->Fill(TIDW1RecHits[i].RX - TIDW1RecHits[i].SX);
      mehSiStripResY[8]->Fill(TIDW1RecHits[i].RY - TIDW1RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TIDW2RecHits =
      srcGlobalRecHits->getTIDW2RecHits();      
    mehSiStripn[9]->Fill((float)nTIDW2RecHits);
    for (unsigned int i = 0; i < TIDW2RecHits.size(); ++i) {
      mehSiStripResX[9]->Fill(TIDW2RecHits[i].RX - TIDW2RecHits[i].SX);
      mehSiStripResY[9]->Fill(TIDW2RecHits[i].RY - TIDW2RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TIDW3RecHits =
      srcGlobalRecHits->getTIDW3RecHits();      
    mehSiStripn[10]->Fill((float)nTIDW3RecHits);
    for (unsigned int i = 0; i < TIDW3RecHits.size(); ++i) {
      mehSiStripResX[10]->Fill(TIDW3RecHits[i].RX - TIDW3RecHits[i].SX);
      mehSiStripResY[10]->Fill(TIDW3RecHits[i].RY - TIDW3RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TECW1RecHits =
      srcGlobalRecHits->getTECW1RecHits();      
    mehSiStripn[11]->Fill((float)nTECW1RecHits);
    for (unsigned int i = 0; i < TECW1RecHits.size(); ++i) {
      mehSiStripResX[11]->Fill(TECW1RecHits[i].RX - TECW1RecHits[i].SX);
      mehSiStripResY[11]->Fill(TECW1RecHits[i].RY - TECW1RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TECW2RecHits =
      srcGlobalRecHits->getTECW2RecHits();      
    mehSiStripn[12]->Fill((float)nTECW2RecHits);
    for (unsigned int i = 0; i < TECW2RecHits.size(); ++i) {
      mehSiStripResX[12]->Fill(TECW2RecHits[i].RX - TECW2RecHits[i].SX);
      mehSiStripResY[12]->Fill(TECW2RecHits[i].RY - TECW2RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TECW3RecHits =
      srcGlobalRecHits->getTECW3RecHits();      
    mehSiStripn[13]->Fill((float)nTECW3RecHits);
    for (unsigned int i = 0; i < TECW3RecHits.size(); ++i) {
      mehSiStripResX[13]->Fill(TECW3RecHits[i].RX - TECW3RecHits[i].SX);
      mehSiStripResY[13]->Fill(TECW3RecHits[i].RY - TECW3RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TECW4RecHits =
      srcGlobalRecHits->getTECW4RecHits();      
    mehSiStripn[14]->Fill((float)nTECW4RecHits);
    for (unsigned int i = 0; i < TECW4RecHits.size(); ++i) {
      mehSiStripResX[14]->Fill(TECW4RecHits[i].RX - TECW4RecHits[i].SX);
      mehSiStripResY[14]->Fill(TECW4RecHits[i].RY - TECW4RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TECW5RecHits =
      srcGlobalRecHits->getTECW5RecHits();      
    mehSiStripn[15]->Fill((float)nTECW5RecHits);
    for (unsigned int i = 0; i < TECW5RecHits.size(); ++i) {
      mehSiStripResX[15]->Fill(TECW5RecHits[i].RX - TECW5RecHits[i].SX);
      mehSiStripResY[15]->Fill(TECW5RecHits[i].RY - TECW5RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TECW6RecHits =
      srcGlobalRecHits->getTECW6RecHits();      
    mehSiStripn[16]->Fill((float)nTECW6RecHits);
    for (unsigned int i = 0; i < TECW6RecHits.size(); ++i) {
      mehSiStripResX[16]->Fill(TECW6RecHits[i].RX - TECW6RecHits[i].SX);
      mehSiStripResY[16]->Fill(TECW6RecHits[i].RY - TECW6RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TECW7RecHits =
      srcGlobalRecHits->getTECW7RecHits();      
    mehSiStripn[17]->Fill((float)nTECW7RecHits);
    for (unsigned int i = 0; i < TECW7RecHits.size(); ++i) {
      mehSiStripResX[17]->Fill(TECW7RecHits[i].RX - TECW7RecHits[i].SX);
      mehSiStripResY[17]->Fill(TECW7RecHits[i].RY - TECW7RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiStripRecHit> TECW8RecHits =
      srcGlobalRecHits->getTECW8RecHits();      
    mehSiStripn[18]->Fill((float)nTECW8RecHits);
    for (unsigned int i = 0; i < TECW8RecHits.size(); ++i) {
      mehSiStripResX[18]->Fill(TECW8RecHits[i].RX - TECW8RecHits[i].SX);
      mehSiStripResY[18]->Fill(TECW8RecHits[i].RY - TECW8RecHits[i].SY);
    }

    // get SiPixel info
    std::vector<PGlobalRecHit::SiPixelRecHit> BRL1RecHits =
      srcGlobalRecHits->getBRL1RecHits();      
    mehSiPixeln[0]->Fill((float)nBRL1RecHits);
    for (unsigned int i = 0; i < BRL1RecHits.size(); ++i) {
      mehSiPixelResX[0]->Fill(BRL1RecHits[i].RX - BRL1RecHits[i].SX);
      mehSiPixelResY[0]->Fill(BRL1RecHits[i].RY - BRL1RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiPixelRecHit> BRL2RecHits =
      srcGlobalRecHits->getBRL2RecHits();      
    mehSiPixeln[1]->Fill((float)nBRL2RecHits);
    for (unsigned int i = 0; i < BRL2RecHits.size(); ++i) {
      mehSiPixelResX[1]->Fill(BRL2RecHits[i].RX - BRL2RecHits[i].SX);
      mehSiPixelResY[1]->Fill(BRL2RecHits[i].RY - BRL2RecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiPixelRecHit> BRL3RecHits =
      srcGlobalRecHits->getBRL3RecHits();      
    mehSiPixeln[2]->Fill((float)nBRL3RecHits);
    for (unsigned int i = 0; i < BRL3RecHits.size(); ++i) {
      mehSiPixelResX[2]->Fill(BRL3RecHits[i].RX - BRL3RecHits[i].SX);
      mehSiPixelResY[2]->Fill(BRL3RecHits[i].RY - BRL3RecHits[i].SY);
   }

    std::vector<PGlobalRecHit::SiPixelRecHit> FWD1pRecHits =
      srcGlobalRecHits->getFWD1pRecHits();      
    mehSiPixeln[3]->Fill((float)nFWD1pRecHits);
    for (unsigned int i = 0; i < FWD1pRecHits.size(); ++i) {
      mehSiPixelResX[3]->Fill(FWD1pRecHits[i].RX - FWD1pRecHits[i].SX);
      mehSiPixelResY[3]->Fill(FWD1pRecHits[i].RY - FWD1pRecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiPixelRecHit> FWD1nRecHits =
      srcGlobalRecHits->getFWD1nRecHits();      
    mehSiPixeln[4]->Fill((float)nFWD1nRecHits);
    for (unsigned int i = 0; i < FWD1nRecHits.size(); ++i) {
      mehSiPixelResX[4]->Fill(FWD1nRecHits[i].RX - FWD1nRecHits[i].SX);
      mehSiPixelResY[4]->Fill(FWD1nRecHits[i].RY - FWD1nRecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiPixelRecHit> FWD2pRecHits =
      srcGlobalRecHits->getFWD2pRecHits();      
    mehSiPixeln[5]->Fill((float)nFWD2pRecHits);
    for (unsigned int i = 0; i < FWD2pRecHits.size(); ++i) {
      mehSiPixelResX[5]->Fill(FWD2pRecHits[i].RX - FWD2pRecHits[i].SX);
      mehSiPixelResY[5]->Fill(FWD2pRecHits[i].RY - FWD2pRecHits[i].SY);
    }

    std::vector<PGlobalRecHit::SiPixelRecHit> FWD2nRecHits =
      srcGlobalRecHits->getFWD2nRecHits();      
    mehSiPixeln[6]->Fill((float)nFWD2nRecHits);
    for (unsigned int i = 0; i < FWD2nRecHits.size(); ++i) {
      mehSiPixelResX[6]->Fill(FWD2nRecHits[i].RX - FWD2nRecHits[i].SX);
      mehSiPixelResY[6]->Fill(FWD2nRecHits[i].RY - FWD2nRecHits[i].SY);
    }

    // get DtMuon info
    std::vector<PGlobalRecHit::DTRecHit> DTRecHits =
      srcGlobalRecHits->getDTRecHits();      
    mehDtMuonn->Fill((float)nDTRecHits);
    for (unsigned int i = 0; i < DTRecHits.size(); ++i) {
      mehDtMuonRes->Fill(DTRecHits[i].RHD - DTRecHits[i].SHD);
    }

    // get CSC info
    std::vector<PGlobalRecHit::CSCRecHit> CSCRecHits =
      srcGlobalRecHits->getCSCRecHits();      
    mehCSCn->Fill((float)nCSCRecHits);
    for (unsigned int i = 0; i < CSCRecHits.size(); ++i) {
      mehCSCResRDPhi->Fill(CSCRecHits[i].RHPERP * 
			 (CSCRecHits[i].RHPHI - CSCRecHits[i].SHPHI));
    }

    // get RPC info
    std::vector<PGlobalRecHit::RPCRecHit> RPCRecHits =
      srcGlobalRecHits->getRPCRecHits();      
    mehRPCn->Fill((float)nRPCRecHits);
    for (unsigned int i = 0; i < RPCRecHits.size(); ++i) {
      mehRPCResX->Fill(RPCRecHits[i].RHX - RPCRecHits[i].SHX);
    }

  if (verbosity > 0)
    edm::LogInfo (MsgLoggerCat)
      << "Done gathering data from event.";

}

