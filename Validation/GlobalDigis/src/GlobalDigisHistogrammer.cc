/** \file GlobalDigisHistogrammer.cc
 *
 *  See header file for description of class
 *
 *  \author M. Strang SUNY-Buffalo
 */

#include "Validation/GlobalDigis/interface/GlobalDigisHistogrammer.h"

GlobalDigisHistogrammer::GlobalDigisHistogrammer(const edm::ParameterSet &iPSet)
    : fName(""),
      verbosity(0),
      frequency(0),
      label(""),
      getAllProvenances(false),
      printProvenanceInfo(false),
      theCSCStripPedestalSum(0),
      theCSCStripPedestalCount(0),
      count(0) {
  std::string MsgLoggerCat = "GlobalDigisHistogrammer_GlobalDigisHistogrammer";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  outputfile = iPSet.getParameter<std::string>("outputFile");
  doOutput = iPSet.getParameter<bool>("DoOutput");
  edm::ParameterSet m_Prov = iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

  // get Labels to use to extract information
  GlobalDigisSrc_ = iPSet.getParameter<edm::InputTag>("GlobalDigisSrc");
  // ECalEBSrc_ = iPSet.getParameter<edm::InputTag>("ECalEBSrc");
  // ECalEESrc_ = iPSet.getParameter<edm::InputTag>("ECalEESrc");
  // ECalESSrc_ = iPSet.getParameter<edm::InputTag>("ECalESSrc");
  // HCalSrc_ = iPSet.getParameter<edm::InputTag>("HCalSrc");
  // SiStripSrc_ = iPSet.getParameter<edm::InputTag>("SiStripSrc");
  // SiPxlSrc_ = iPSet.getParameter<edm::InputTag>("SiPxlSrc");
  // MuDTSrc_ = iPSet.getParameter<edm::InputTag>("MuDTSrc");
  // MuCSCStripSrc_ = iPSet.getParameter<edm::InputTag>("MuCSCStripSrc");
  // MuCSCWireSrc_ = iPSet.getParameter<edm::InputTag>("MuCSCWireSrc");

  // fix for consumes
  GlobalDigisSrc_Token_ = consumes<PGlobalDigi>(iPSet.getParameter<edm::InputTag>("GlobalDigisSrc"));
  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // create persistent object
  // produces<PGlobalDigi>(label);

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) << "\n===============================\n"
                               << "Initialized as EDHistogrammer with parameter values:\n"
                               << "    Name          = " << fName << "\n"
                               << "    Verbosity     = " << verbosity << "\n"
                               << "    Frequency     = " << frequency << "\n"
                               << "    OutputFile    = " << outputfile << "\n"
                               << "    DoOutput      = " << doOutput << "\n"
                               << "    GetProv       = " << getAllProvenances << "\n"
                               << "    PrintProv     = " << printProvenanceInfo << "\n"
                               << "    Global Src    = " << GlobalDigisSrc_ << "\n"

                               << "===============================\n";
  }
}

// set default constants
// ECal

// ECalgainConv_[0] = 0.;
// ECalgainConv_[1] = 1.;
// ECalgainConv_[2] = 2.;
// ECalgainConv_[3] = 12.;
// ECalbarrelADCtoGeV_ = 0.035;
// ECalendcapADCtoGeV_ = 0.06;

void GlobalDigisHistogrammer::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &) {
  // monitor elements

  // Si Strip  ***Done***
  std::string SiStripString[19] = {"TECW1",
                                   "TECW2",
                                   "TECW3",
                                   "TECW4",
                                   "TECW5",
                                   "TECW6",
                                   "TECW7",
                                   "TECW8",
                                   "TIBL1",
                                   "TIBL2",
                                   "TIBL3",
                                   "TIBL4",
                                   "TIDW1",
                                   "TIDW2",
                                   "TIDW3",
                                   "TOBL1",
                                   "TOBL2",
                                   "TOBL3",
                                   "TOBL4"};

  for (int i = 0; i < 19; ++i) {
    mehSiStripn[i] = nullptr;
    mehSiStripADC[i] = nullptr;
    mehSiStripStrip[i] = nullptr;
  }

  ibooker.setCurrentFolder("GlobalDigisV/SiStrips");
  for (int amend = 0; amend < 19; ++amend) {
    mehSiStripn[amend] =
        ibooker.book1D("hSiStripn_" + SiStripString[amend], SiStripString[amend] + "  Digis", 500, 0., 1000.);

    mehSiStripn[amend]->setAxisTitle("Number of Digis", 1);
    mehSiStripn[amend]->setAxisTitle("Count", 2);
    mehSiStripADC[amend] =
        ibooker.book1D("hSiStripADC_" + SiStripString[amend], SiStripString[amend] + " ADC", 150, 0.0, 300.);

    mehSiStripADC[amend]->setAxisTitle("ADC", 1);
    mehSiStripADC[amend]->setAxisTitle("Count", 2);
    mehSiStripStrip[amend] =
        ibooker.book1D("hSiStripStripADC_" + SiStripString[amend], SiStripString[amend] + " Strip", 200, 0.0, 800.);
    mehSiStripStrip[amend]->setAxisTitle("Strip Number", 1);
    mehSiStripStrip[amend]->setAxisTitle("Count", 2);
  }

  // HCal  **DONE**
  std::string HCalString[4] = {"HB", "HE", "HO", "HF"};
  float calnUpper[4] = {3000., 3000., 3000., 2000.};
  float calnLower[4] = {2000., 2000., 2000., 1000.};
  float SHEUpper[4] = {0.05, .05, 0.05, 20};
  float SHEvAEEUpper[4] = {5000, 5000, 5000, 20};
  float SHEvAEELower[4] = {-5000, -5000, -5000, -20};
  int SHEvAEEnBins[4] = {200, 200, 200, 40};
  double ProfileUpper[4] = {1., 1., 1., 20.};

  for (int i = 0; i < 4; ++i) {
    mehHcaln[i] = nullptr;
    mehHcalAEE[i] = nullptr;
    mehHcalSHE[i] = nullptr;
    mehHcalAEESHE[i] = nullptr;
    mehHcalSHEvAEE[i] = nullptr;
  }

  ibooker.setCurrentFolder("GlobalDigisV/HCals");
  for (int amend = 0; amend < 4; ++amend) {
    mehHcaln[amend] = ibooker.book1D(
        "hHcaln_" + HCalString[amend], HCalString[amend] + "  digis", 1000, calnLower[amend], calnUpper[amend]);

    mehHcaln[amend]->setAxisTitle("Number of Digis", 1);
    mehHcaln[amend]->setAxisTitle("Count", 2);
    mehHcalAEE[amend] = ibooker.book1D("hHcalAEE_" + HCalString[amend], HCalString[amend] + "Cal AEE", 60, -10., 50.);

    mehHcalAEE[amend]->setAxisTitle("Analog Equivalent Energy", 1);
    mehHcalAEE[amend]->setAxisTitle("Count", 2);
    mehHcalSHE[amend] =
        ibooker.book1D("hHcalSHE_" + HCalString[amend], HCalString[amend] + "Cal SHE", 100, 0.0, SHEUpper[amend]);

    mehHcalSHE[amend]->setAxisTitle("Simulated Hit Energy", 1);
    mehHcalSHE[amend]->setAxisTitle("Count", 2);
    mehHcalAEESHE[amend] = ibooker.book1D("hHcalAEESHE_" + HCalString[amend],
                                          HCalString[amend] + "Cal AEE/SHE",
                                          SHEvAEEnBins[amend],
                                          SHEvAEELower[amend],
                                          SHEvAEEUpper[amend]);

    mehHcalAEESHE[amend]->setAxisTitle("ADC / SHE", 1);
    mehHcalAEESHE[amend]->setAxisTitle("Count", 2);

    //************  Not sure how to do Profile ME **************
    mehHcalSHEvAEE[amend] = ibooker.bookProfile("hHcalSHEvAEE_" + HCalString[amend],
                                                HCalString[amend] + "Cal SHE vs. AEE",
                                                60,
                                                (float)-10.,
                                                (float)50.,
                                                100,
                                                (float)0.,
                                                (float)ProfileUpper[amend],
                                                "");

    mehHcalSHEvAEE[amend]->setAxisTitle("AEE / SHE", 1);
    mehHcalSHEvAEE[amend]->setAxisTitle("SHE", 2);
  }

  // Ecal **Done **
  std::string ECalString[2] = {"EB", "EE"};

  for (int i = 0; i < 2; ++i) {
    mehEcaln[i] = nullptr;
    mehEcalAEE[i] = nullptr;
    mehEcalSHE[i] = nullptr;
    mehEcalMaxPos[i] = nullptr;
    mehEcalMultvAEE[i] = nullptr;
    mehEcalSHEvAEESHE[i] = nullptr;
  }

  ibooker.setCurrentFolder("GlobalDigisV/ECals");
  for (int amend = 0; amend < 2; ++amend) {
    mehEcaln[amend] = ibooker.book1D("hEcaln_" + ECalString[amend], ECalString[amend] + "  digis", 300, 1000., 4000.);

    mehEcaln[amend]->setAxisTitle("Number of Digis", 1);
    mehEcaln[amend]->setAxisTitle("Count", 2);
    mehEcalAEE[amend] = ibooker.book1D("hEcalAEE_" + ECalString[amend], ECalString[amend] + "Cal AEE", 100, 0., 1.);

    mehEcalAEE[amend]->setAxisTitle("Analog Equivalent Energy", 1);
    mehEcalAEE[amend]->setAxisTitle("Count", 2);
    mehEcalSHE[amend] = ibooker.book1D("hEcalSHE_" + ECalString[amend], ECalString[amend] + "Cal SHE", 50, 0., 5.);

    mehEcalSHE[amend]->setAxisTitle("Simulated Hit Energy", 1);
    mehEcalSHE[amend]->setAxisTitle("Count", 2);
    mehEcalMaxPos[amend] =
        ibooker.book1D("hEcalMaxPos_" + ECalString[amend], ECalString[amend] + "Cal MaxPos", 10, 0., 10.);

    mehEcalMaxPos[amend]->setAxisTitle("Maximum Position", 1);
    mehEcalMaxPos[amend]->setAxisTitle("Count", 2);

    //************  Not sure how to do Profile ME **************
    mehEcalSHEvAEESHE[amend] = ibooker.bookProfile("hEcalSHEvAEESHE_" + ECalString[amend],
                                                   ECalString[amend] + "Cal SHE vs. AEE/SHE",
                                                   100,
                                                   (float)0.,
                                                   (float)10.,
                                                   50,
                                                   (float)0.,
                                                   (float)5.,
                                                   "");

    mehEcalSHEvAEESHE[amend]->setAxisTitle("AEE / SHE", 1);
    mehEcalSHEvAEESHE[amend]->setAxisTitle("SHE", 2);
    mehEcalMultvAEE[amend] = ibooker.bookProfile("hEcalMultvAEE_" + ECalString[amend],
                                                 ECalString[amend] + "Cal Multi vs. AEE",
                                                 100,
                                                 (float)0.,
                                                 (float)10.,
                                                 400,
                                                 (float)0.,
                                                 (float)4000.,
                                                 "");
    mehEcalMultvAEE[amend]->setAxisTitle("Analog Equivalent Energy", 1);
    mehEcalMultvAEE[amend]->setAxisTitle("Number of Digis", 2);
  }

  mehEcaln[2] = nullptr;
  mehEcaln[2] = ibooker.book1D("hEcaln_ES", "ESCAL  digis", 100, 0., 500.);
  mehEcaln[2]->setAxisTitle("Number of Digis", 1);
  mehEcaln[2]->setAxisTitle("Count", 2);
  std::string ADCNumber[3] = {"0", "1", "2"};
  for (int i = 0; i < 3; ++i) {
    mehEScalADC[i] = nullptr;
    mehEScalADC[i] = ibooker.book1D("hEcalADC" + ADCNumber[i] + "_ES", "ESCAL  ADC" + ADCNumber[i], 150, 950., 1500.);

    mehEScalADC[i]->setAxisTitle("ADC" + ADCNumber[i], 1);
    mehEScalADC[i]->setAxisTitle("Count", 2);
  }

  // Si Pixels ***DONE***
  std::string SiPixelString[7] = {"BRL1", "BRL2", "BRL3", "FWD1n", "FWD1p", "FWD2n", "FWD2p"};

  for (int j = 0; j < 7; ++j) {
    mehSiPixeln[j] = nullptr;
    mehSiPixelADC[j] = nullptr;
    mehSiPixelRow[j] = nullptr;
    mehSiPixelCol[j] = nullptr;
  }

  ibooker.setCurrentFolder("GlobalDigisV/SiPixels");
  for (int amend = 0; amend < 7; ++amend) {
    if (amend < 3) {
      mehSiPixeln[amend] =
          ibooker.book1D("hSiPixeln_" + SiPixelString[amend], SiPixelString[amend] + " Digis", 50, 0., 100.);
    } else {
      mehSiPixeln[amend] =
          ibooker.book1D("hSiPixeln_" + SiPixelString[amend], SiPixelString[amend] + " Digis", 25, 0., 50.);
    }

    mehSiPixeln[amend]->setAxisTitle("Number of Digis", 1);
    mehSiPixeln[amend]->setAxisTitle("Count", 2);
    mehSiPixelADC[amend] =
        ibooker.book1D("hSiPixelADC_" + SiPixelString[amend], SiPixelString[amend] + " ADC", 150, 0.0, 300.);

    mehSiPixelADC[amend]->setAxisTitle("ADC", 1);
    mehSiPixelADC[amend]->setAxisTitle("Count", 2);
    mehSiPixelRow[amend] =
        ibooker.book1D("hSiPixelRow_" + SiPixelString[amend], SiPixelString[amend] + " Row", 100, 0.0, 100.);

    mehSiPixelRow[amend]->setAxisTitle("Row Number", 1);
    mehSiPixelRow[amend]->setAxisTitle("Count", 2);
    mehSiPixelCol[amend] =
        ibooker.book1D("hSiPixelColumn_" + SiPixelString[amend], SiPixelString[amend] + " Column", 200, 0.0, 500.);

    mehSiPixelCol[amend]->setAxisTitle("Column Number", 1);
    mehSiPixelCol[amend]->setAxisTitle("Count", 2);
  }

  // Muons ***DONE****
  ibooker.setCurrentFolder("GlobalDigisV/Muons");
  std::string MuonString[4] = {"MB1", "MB2", "MB3", "MB4"};

  for (int i = 0; i < 4; ++i) {
    mehDtMuonn[i] = nullptr;
    mehDtMuonLayer[i] = nullptr;
    mehDtMuonTime[i] = nullptr;
    mehDtMuonTimevLayer[i] = nullptr;
  }

  for (int j = 0; j < 4; ++j) {
    mehDtMuonn[j] = ibooker.book1D("hDtMuonn_" + MuonString[j], MuonString[j] + "  digis", 25, 0., 50.);

    mehDtMuonn[j]->setAxisTitle("Number of Digis", 1);
    mehDtMuonn[j]->setAxisTitle("Count", 2);
    mehDtMuonLayer[j] = ibooker.book1D("hDtLayer_" + MuonString[j], MuonString[j] + "  Layer", 12, 1., 13.);

    mehDtMuonLayer[j]->setAxisTitle("4 * (SuperLayer - 1) + Layer", 1);
    mehDtMuonLayer[j]->setAxisTitle("Count", 2);
    mehDtMuonTime[j] = ibooker.book1D("hDtMuonTime_" + MuonString[j], MuonString[j] + "  Time", 300, 400., 1000.);

    mehDtMuonTime[j]->setAxisTitle("Time", 1);
    mehDtMuonTime[j]->setAxisTitle("Count", 2);
    mehDtMuonTimevLayer[j] = ibooker.bookProfile(
        "hDtMuonTimevLayer_" + MuonString[j], MuonString[j] + "  Time vs. Layer", 12, 1., 13., 300, 400., 1000., "");

    mehDtMuonTimevLayer[j]->setAxisTitle("4 * (SuperLayer - 1) + Layer", 1);
    mehDtMuonTimevLayer[j]->setAxisTitle("Time", 2);
  }

  //  ****  Have to do CSC and RPC now *****
  // CSC
  mehCSCStripn = nullptr;
  mehCSCStripn = ibooker.book1D("hCSCStripn", "CSC Strip digis", 25, 0., 50.);
  mehCSCStripn->setAxisTitle("Number of Digis", 1);
  mehCSCStripn->setAxisTitle("Count", 2);

  mehCSCStripADC = nullptr;
  mehCSCStripADC = ibooker.book1D("hCSCStripADC", "CSC Strip ADC", 110, 0., 1100.);
  mehCSCStripADC->setAxisTitle("ADC", 1);
  mehCSCStripADC->setAxisTitle("Count", 2);

  mehCSCWiren = nullptr;
  mehCSCWiren = ibooker.book1D("hCSCWiren", "CSC Wire digis", 25, 0., 50.);
  mehCSCWiren->setAxisTitle("Number of Digis", 1);
  mehCSCWiren->setAxisTitle("Count", 2);

  mehCSCWireTime = nullptr;
  mehCSCWiren = ibooker.book1D("hCSCWireTime", "CSC Wire Time", 10, 0., 10.);
  mehCSCWiren->setAxisTitle("Time", 1);
  mehCSCWiren->setAxisTitle("Count", 2);

}  // close bookHistograms

GlobalDigisHistogrammer::~GlobalDigisHistogrammer() {}

void GlobalDigisHistogrammer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::string MsgLoggerCat = "GlobalDigisHistogrammer_analyze";

  // keep track of number of events processed
  ++count;

  // get event id information
  edm::RunNumber_t nrun = iEvent.id().run();
  edm::EventNumber_t nevt = iEvent.id().event();

  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat) << "Processing run " << nrun << ", event " << nevt << " (" << count << " events total)";
  } else if (verbosity == 0) {
    if (nevt % frequency == 0 || nevt == 1) {
      edm::LogInfo(MsgLoggerCat) << "Processing run " << nrun << ", event " << nevt << " (" << count
                                 << " events total)";
    }
  }

  // clear event holders
  // clear();

  // look at information available in the event
  if (getAllProvenances) {
    std::vector<const edm::StableProvenance *> AllProv;
    iEvent.getAllStableProvenance(AllProv);

    if (verbosity >= 0)
      edm::LogInfo(MsgLoggerCat) << "Number of Provenances = " << AllProv.size();

    if (printProvenanceInfo && (verbosity >= 0)) {
      TString eventout("\nProvenance info:\n");

      for (auto &i : AllProv) {
        eventout += "\n       ******************************";
        eventout += "\n       Module       : ";
        // eventout += (AllProv[i]->product).moduleLabel();
        eventout += i->moduleLabel();
        eventout += "\n       ProductID    : ";
        // eventout += (AllProv[i]->product).productID_.id_;
        eventout += i->productID().id();
        eventout += "\n       ClassName    : ";
        // eventout += (AllProv[i]->product).fullClassName_;
        eventout += i->className();
        eventout += "\n       InstanceName : ";
        // eventout += (AllProv[i]->product).productInstanceName_;
        eventout += i->productInstanceName();
        eventout += "\n       BranchName   : ";
        // eventout += (AllProv[i]->product).branchName_;
        eventout += i->branchName();
      }
      eventout += "\n       ******************************\n";
      edm::LogInfo(MsgLoggerCat) << eventout << "\n";
      printProvenanceInfo = false;
      getAllProvenances = false;
    }
    edm::Handle<PGlobalDigi> srcGlobalDigis;
    iEvent.getByToken(GlobalDigisSrc_Token_, srcGlobalDigis);
    if (!srcGlobalDigis.isValid()) {
      edm::LogWarning(MsgLoggerCat) << "Unable to find PGlobalDigis in event!";
      return;
    }

    int nEBCalDigis = srcGlobalDigis->getnEBCalDigis();
    int nEECalDigis = srcGlobalDigis->getnEECalDigis();
    int nESCalDigis = srcGlobalDigis->getnESCalDigis();

    int nHBCalDigis = srcGlobalDigis->getnHBCalDigis();
    int nHECalDigis = srcGlobalDigis->getnHECalDigis();
    int nHOCalDigis = srcGlobalDigis->getnHOCalDigis();
    int nHFCalDigis = srcGlobalDigis->getnHFCalDigis();

    int nTIBL1Digis = srcGlobalDigis->getnTIBL1Digis();
    int nTIBL2Digis = srcGlobalDigis->getnTIBL2Digis();
    int nTIBL3Digis = srcGlobalDigis->getnTIBL3Digis();
    int nTIBL4Digis = srcGlobalDigis->getnTIBL4Digis();
    int nTOBL1Digis = srcGlobalDigis->getnTOBL1Digis();
    int nTOBL2Digis = srcGlobalDigis->getnTOBL2Digis();
    int nTOBL3Digis = srcGlobalDigis->getnTOBL3Digis();
    int nTOBL4Digis = srcGlobalDigis->getnTOBL4Digis();
    int nTIDW1Digis = srcGlobalDigis->getnTIDW1Digis();
    int nTIDW2Digis = srcGlobalDigis->getnTIDW2Digis();
    int nTIDW3Digis = srcGlobalDigis->getnTIDW3Digis();
    int nTECW1Digis = srcGlobalDigis->getnTECW1Digis();
    int nTECW2Digis = srcGlobalDigis->getnTECW2Digis();
    int nTECW3Digis = srcGlobalDigis->getnTECW3Digis();
    int nTECW4Digis = srcGlobalDigis->getnTECW4Digis();
    int nTECW5Digis = srcGlobalDigis->getnTECW5Digis();
    int nTECW6Digis = srcGlobalDigis->getnTECW6Digis();
    int nTECW7Digis = srcGlobalDigis->getnTECW7Digis();
    int nTECW8Digis = srcGlobalDigis->getnTECW8Digis();

    int nBRL1Digis = srcGlobalDigis->getnBRL1Digis();
    int nBRL2Digis = srcGlobalDigis->getnBRL2Digis();
    int nBRL3Digis = srcGlobalDigis->getnBRL3Digis();
    int nFWD1nDigis = srcGlobalDigis->getnFWD1nDigis();
    int nFWD1pDigis = srcGlobalDigis->getnFWD1pDigis();
    int nFWD2nDigis = srcGlobalDigis->getnFWD2nDigis();
    int nFWD2pDigis = srcGlobalDigis->getnFWD2pDigis();

    int nMB1Digis = srcGlobalDigis->getnMB1Digis();
    int nMB2Digis = srcGlobalDigis->getnMB2Digis();
    int nMB3Digis = srcGlobalDigis->getnMB3Digis();
    int nMB4Digis = srcGlobalDigis->getnMB4Digis();

    int nCSCstripDigis = srcGlobalDigis->getnCSCstripDigis();

    int nCSCwireDigis = srcGlobalDigis->getnCSCwireDigis();

    // get Ecal info
    std::vector<PGlobalDigi::ECalDigi> EECalDigis = srcGlobalDigis->getEECalDigis();
    mehEcaln[0]->Fill((float)nEECalDigis);
    for (auto &EECalDigi : EECalDigis) {
      mehEcalAEE[0]->Fill(EECalDigi.AEE);
      mehEcalMaxPos[0]->Fill(EECalDigi.maxPos);
      mehEcalMultvAEE[0]->Fill(EECalDigi.AEE, (float)nEECalDigis, 1);
      if (EECalDigi.SHE != 0.) {
        mehEcalSHE[0]->Fill(EECalDigi.SHE);
        mehEcalSHEvAEESHE[0]->Fill(EECalDigi.AEE / EECalDigi.SHE, EECalDigi.SHE, 1);
      }
    }

    std::vector<PGlobalDigi::ECalDigi> EBCalDigis = srcGlobalDigis->getEBCalDigis();
    mehEcaln[1]->Fill((float)nEBCalDigis);
    for (auto &EBCalDigi : EBCalDigis) {
      mehEcalAEE[1]->Fill(EBCalDigi.AEE);
      mehEcalMaxPos[1]->Fill(EBCalDigi.maxPos);
      mehEcalMultvAEE[1]->Fill(EBCalDigi.AEE, (float)nEBCalDigis, 1);
      if (EBCalDigi.SHE != 0.) {
        mehEcalSHE[1]->Fill(EBCalDigi.SHE);
        mehEcalSHEvAEESHE[1]->Fill(EBCalDigi.AEE / EBCalDigi.SHE, EBCalDigi.SHE, 1);
      }
    }

    std::vector<PGlobalDigi::ESCalDigi> ESCalDigis = srcGlobalDigis->getESCalDigis();
    mehEcaln[2]->Fill((float)nESCalDigis);
    for (auto &ESCalDigi : ESCalDigis) {
      mehEScalADC[0]->Fill(ESCalDigi.ADC0);
      mehEScalADC[1]->Fill(ESCalDigi.ADC1);
      mehEScalADC[2]->Fill(ESCalDigi.ADC2);
    }

    // Get HCal info
    std::vector<PGlobalDigi::HCalDigi> HBCalDigis = srcGlobalDigis->getHBCalDigis();
    mehHcaln[0]->Fill((float)nHBCalDigis);
    for (auto &HBCalDigi : HBCalDigis) {
      mehHcalAEE[0]->Fill(HBCalDigi.AEE);
      if (HBCalDigi.SHE != 0.) {
        mehHcalSHE[0]->Fill(HBCalDigi.SHE);
        mehHcalAEESHE[0]->Fill(HBCalDigi.AEE / HBCalDigi.SHE);
        mehHcalSHEvAEE[0]->Fill(HBCalDigi.AEE, HBCalDigi.SHE, 1);
      }
    }
    std::vector<PGlobalDigi::HCalDigi> HECalDigis = srcGlobalDigis->getHECalDigis();
    mehHcaln[1]->Fill((float)nHECalDigis);
    for (auto &HECalDigi : HECalDigis) {
      mehHcalAEE[1]->Fill(HECalDigi.AEE);
      if (HECalDigi.SHE != 0.) {
        mehHcalSHE[1]->Fill(HECalDigi.SHE);
        mehHcalAEESHE[1]->Fill(HECalDigi.AEE / HECalDigi.SHE);
        mehHcalSHEvAEE[1]->Fill(HECalDigi.AEE, HECalDigi.SHE, 1);
      }
    }

    std::vector<PGlobalDigi::HCalDigi> HOCalDigis = srcGlobalDigis->getHOCalDigis();
    mehHcaln[2]->Fill((float)nHOCalDigis);
    for (auto &HOCalDigi : HOCalDigis) {
      mehHcalAEE[2]->Fill(HOCalDigi.AEE);
      if (HOCalDigi.SHE != 0.) {
        mehHcalSHE[2]->Fill(HOCalDigi.SHE);
        mehHcalAEESHE[2]->Fill(HOCalDigi.AEE / HOCalDigi.SHE);
        mehHcalSHEvAEE[2]->Fill(HOCalDigi.AEE, HOCalDigi.SHE, 1);
      }
    }

    std::vector<PGlobalDigi::HCalDigi> HFCalDigis = srcGlobalDigis->getHFCalDigis();
    mehHcaln[3]->Fill((float)nHFCalDigis);
    for (auto &HFCalDigi : HFCalDigis) {
      mehHcalAEE[3]->Fill(HFCalDigi.AEE);
      if (HFCalDigi.SHE != 0.) {
        mehHcalSHE[3]->Fill(HFCalDigi.SHE);
        mehHcalAEESHE[3]->Fill(HFCalDigi.AEE / HFCalDigi.SHE);
        mehHcalSHEvAEE[3]->Fill(HFCalDigi.AEE, HFCalDigi.SHE, 1);
      }
    }

    // get SiStrip info
    std::vector<PGlobalDigi::SiStripDigi> TIBL1Digis = srcGlobalDigis->getTIBL1Digis();
    mehSiStripn[0]->Fill((float)nTIBL1Digis);
    for (auto &TIBL1Digi : TIBL1Digis) {
      mehSiStripADC[0]->Fill(TIBL1Digi.ADC);
      mehSiStripStrip[0]->Fill(TIBL1Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIBL2Digis = srcGlobalDigis->getTIBL2Digis();
    mehSiStripn[1]->Fill((float)nTIBL2Digis);
    for (auto &TIBL2Digi : TIBL2Digis) {
      mehSiStripADC[1]->Fill(TIBL2Digi.ADC);
      mehSiStripStrip[1]->Fill(TIBL2Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIBL3Digis = srcGlobalDigis->getTIBL3Digis();
    mehSiStripn[2]->Fill((float)nTIBL3Digis);
    for (auto &TIBL3Digi : TIBL3Digis) {
      mehSiStripADC[2]->Fill(TIBL3Digi.ADC);
      mehSiStripStrip[2]->Fill(TIBL3Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIBL4Digis = srcGlobalDigis->getTIBL4Digis();
    mehSiStripn[3]->Fill((float)nTIBL4Digis);
    for (auto &TIBL4Digi : TIBL4Digis) {
      mehSiStripADC[3]->Fill(TIBL4Digi.ADC);
      mehSiStripStrip[3]->Fill(TIBL4Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TOBL1Digis = srcGlobalDigis->getTOBL1Digis();
    mehSiStripn[4]->Fill((float)nTOBL1Digis);
    for (auto &TOBL1Digi : TOBL1Digis) {
      mehSiStripADC[4]->Fill(TOBL1Digi.ADC);
      mehSiStripStrip[4]->Fill(TOBL1Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TOBL2Digis = srcGlobalDigis->getTOBL2Digis();
    mehSiStripn[5]->Fill((float)nTOBL2Digis);
    for (auto &TOBL2Digi : TOBL2Digis) {
      mehSiStripADC[5]->Fill(TOBL2Digi.ADC);
      mehSiStripStrip[5]->Fill(TOBL2Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TOBL3Digis = srcGlobalDigis->getTOBL3Digis();
    mehSiStripn[6]->Fill((float)nTOBL3Digis);
    for (auto &TOBL3Digi : TOBL3Digis) {
      mehSiStripADC[6]->Fill(TOBL3Digi.ADC);
      mehSiStripStrip[6]->Fill(TOBL3Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TOBL4Digis = srcGlobalDigis->getTOBL4Digis();
    mehSiStripn[7]->Fill((float)nTOBL4Digis);
    for (auto &TOBL4Digi : TOBL4Digis) {
      mehSiStripADC[7]->Fill(TOBL4Digi.ADC);
      mehSiStripStrip[7]->Fill(TOBL4Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIDW1Digis = srcGlobalDigis->getTIDW1Digis();
    mehSiStripn[8]->Fill((float)nTIDW1Digis);
    for (auto &TIDW1Digi : TIDW1Digis) {
      mehSiStripADC[8]->Fill(TIDW1Digi.ADC);
      mehSiStripStrip[8]->Fill(TIDW1Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIDW2Digis = srcGlobalDigis->getTIDW2Digis();
    mehSiStripn[9]->Fill((float)nTIDW2Digis);
    for (auto &TIDW2Digi : TIDW2Digis) {
      mehSiStripADC[9]->Fill(TIDW2Digi.ADC);
      mehSiStripStrip[9]->Fill(TIDW2Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIDW3Digis = srcGlobalDigis->getTIDW3Digis();
    mehSiStripn[10]->Fill((float)nTIDW3Digis);
    for (auto &TIDW3Digi : TIDW3Digis) {
      mehSiStripADC[10]->Fill(TIDW3Digi.ADC);
      mehSiStripStrip[10]->Fill(TIDW3Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW1Digis = srcGlobalDigis->getTECW1Digis();
    mehSiStripn[11]->Fill((float)nTECW1Digis);
    for (auto &TECW1Digi : TECW1Digis) {
      mehSiStripADC[11]->Fill(TECW1Digi.ADC);
      mehSiStripStrip[11]->Fill(TECW1Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW2Digis = srcGlobalDigis->getTECW2Digis();
    mehSiStripn[12]->Fill((float)nTECW2Digis);
    for (auto &TECW2Digi : TECW2Digis) {
      mehSiStripADC[12]->Fill(TECW2Digi.ADC);
      mehSiStripStrip[12]->Fill(TECW2Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW3Digis = srcGlobalDigis->getTECW3Digis();
    mehSiStripn[13]->Fill((float)nTECW3Digis);
    for (auto &TECW3Digi : TECW3Digis) {
      mehSiStripADC[13]->Fill(TECW3Digi.ADC);
      mehSiStripStrip[13]->Fill(TECW3Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW4Digis = srcGlobalDigis->getTECW4Digis();
    mehSiStripn[14]->Fill((float)nTECW4Digis);
    for (auto &TECW4Digi : TECW4Digis) {
      mehSiStripADC[14]->Fill(TECW4Digi.ADC);
      mehSiStripStrip[14]->Fill(TECW4Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW5Digis = srcGlobalDigis->getTECW5Digis();
    mehSiStripn[15]->Fill((float)nTECW5Digis);
    for (auto &TECW5Digi : TECW5Digis) {
      mehSiStripADC[15]->Fill(TECW5Digi.ADC);
      mehSiStripStrip[15]->Fill(TECW5Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW6Digis = srcGlobalDigis->getTECW6Digis();
    mehSiStripn[16]->Fill((float)nTECW6Digis);
    for (auto &TECW6Digi : TECW6Digis) {
      mehSiStripADC[16]->Fill(TECW6Digi.ADC);
      mehSiStripStrip[16]->Fill(TECW6Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW7Digis = srcGlobalDigis->getTECW7Digis();
    mehSiStripn[17]->Fill((float)nTECW7Digis);
    for (auto &TECW7Digi : TECW7Digis) {
      mehSiStripADC[17]->Fill(TECW7Digi.ADC);
      mehSiStripStrip[17]->Fill(TECW7Digi.STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW8Digis = srcGlobalDigis->getTECW8Digis();
    mehSiStripn[18]->Fill((float)nTECW8Digis);
    for (auto &TECW8Digi : TECW8Digis) {
      mehSiStripADC[18]->Fill(TECW8Digi.ADC);
      mehSiStripStrip[18]->Fill(TECW8Digi.STRIP);
    }

    // get SiPixel info
    std::vector<PGlobalDigi::SiPixelDigi> BRL1Digis = srcGlobalDigis->getBRL1Digis();
    mehSiPixeln[0]->Fill((float)nBRL1Digis);
    for (auto &BRL1Digi : BRL1Digis) {
      mehSiPixelADC[0]->Fill(BRL1Digi.ADC);
      mehSiPixelRow[0]->Fill(BRL1Digi.ROW);
      mehSiPixelCol[0]->Fill(BRL1Digi.COLUMN);
    }

    std::vector<PGlobalDigi::SiPixelDigi> BRL2Digis = srcGlobalDigis->getBRL2Digis();
    mehSiPixeln[1]->Fill((float)nBRL2Digis);
    for (auto &BRL2Digi : BRL2Digis) {
      mehSiPixelADC[1]->Fill(BRL2Digi.ADC);
      mehSiPixelRow[1]->Fill(BRL2Digi.ROW);
      mehSiPixelCol[1]->Fill(BRL2Digi.COLUMN);
    }

    std::vector<PGlobalDigi::SiPixelDigi> BRL3Digis = srcGlobalDigis->getBRL3Digis();
    mehSiPixeln[2]->Fill((float)nBRL3Digis);
    for (auto &BRL3Digi : BRL3Digis) {
      mehSiPixelADC[2]->Fill(BRL3Digi.ADC);
      mehSiPixelRow[2]->Fill(BRL3Digi.ROW);
      mehSiPixelCol[2]->Fill(BRL3Digi.COLUMN);
    }

    std::vector<PGlobalDigi::SiPixelDigi> FWD1pDigis = srcGlobalDigis->getFWD1pDigis();
    mehSiPixeln[3]->Fill((float)nFWD1pDigis);
    for (auto &FWD1pDigi : FWD1pDigis) {
      mehSiPixelADC[3]->Fill(FWD1pDigi.ADC);
      mehSiPixelRow[3]->Fill(FWD1pDigi.ROW);
      mehSiPixelCol[3]->Fill(FWD1pDigi.COLUMN);
    }

    std::vector<PGlobalDigi::SiPixelDigi> FWD1nDigis = srcGlobalDigis->getFWD1nDigis();
    mehSiPixeln[4]->Fill((float)nFWD1nDigis);
    for (auto &FWD1nDigi : FWD1nDigis) {
      mehSiPixelADC[4]->Fill(FWD1nDigi.ADC);
      mehSiPixelRow[4]->Fill(FWD1nDigi.ROW);
      mehSiPixelCol[4]->Fill(FWD1nDigi.COLUMN);
    }

    std::vector<PGlobalDigi::SiPixelDigi> FWD2pDigis = srcGlobalDigis->getFWD2pDigis();
    mehSiPixeln[5]->Fill((float)nFWD2pDigis);
    for (auto &FWD2pDigi : FWD2pDigis) {
      mehSiPixelADC[5]->Fill(FWD2pDigi.ADC);
      mehSiPixelRow[5]->Fill(FWD2pDigi.ROW);
      mehSiPixelCol[5]->Fill(FWD2pDigi.COLUMN);
    }

    std::vector<PGlobalDigi::SiPixelDigi> FWD2nDigis = srcGlobalDigis->getFWD2nDigis();
    mehSiPixeln[6]->Fill((float)nFWD2nDigis);
    for (auto &FWD2nDigi : FWD2nDigis) {
      mehSiPixelADC[6]->Fill(FWD2nDigi.ADC);
      mehSiPixelRow[6]->Fill(FWD2nDigi.ROW);
      mehSiPixelCol[6]->Fill(FWD2nDigi.COLUMN);
    }

    // get DtMuon info
    std::vector<PGlobalDigi::DTDigi> MB1Digis = srcGlobalDigis->getMB1Digis();
    mehDtMuonn[0]->Fill((float)nMB1Digis);
    for (auto &MB1Digi : MB1Digis) {
      float layer = 4.0 * (MB1Digi.SLAYER - 1.0) + MB1Digi.LAYER;
      mehDtMuonLayer[0]->Fill(layer);
      mehDtMuonTime[0]->Fill(MB1Digi.TIME);
      mehDtMuonTimevLayer[0]->Fill(layer, MB1Digi.TIME, 1);
    }

    std::vector<PGlobalDigi::DTDigi> MB2Digis = srcGlobalDigis->getMB2Digis();
    mehDtMuonn[1]->Fill((float)nMB2Digis);
    for (auto &MB2Digi : MB2Digis) {
      float layer = 4.0 * (MB2Digi.SLAYER - 1.0) + MB2Digi.LAYER;
      mehDtMuonLayer[1]->Fill(layer);
      mehDtMuonTime[1]->Fill(MB2Digi.TIME);
      mehDtMuonTimevLayer[1]->Fill(layer, MB2Digi.TIME, 1);
    }

    std::vector<PGlobalDigi::DTDigi> MB3Digis = srcGlobalDigis->getMB3Digis();
    mehDtMuonn[2]->Fill((float)nMB3Digis);
    for (auto &MB3Digi : MB3Digis) {
      float layer = 4.0 * (MB3Digi.SLAYER - 1.0) + MB3Digi.LAYER;
      mehDtMuonLayer[2]->Fill(layer);
      mehDtMuonTime[2]->Fill(MB3Digi.TIME);
      mehDtMuonTimevLayer[2]->Fill(layer, MB3Digi.TIME, 1);
    }

    std::vector<PGlobalDigi::DTDigi> MB4Digis = srcGlobalDigis->getMB4Digis();
    mehDtMuonn[3]->Fill((float)nMB4Digis);
    for (auto &MB4Digi : MB4Digis) {
      float layer = 4.0 * (MB4Digi.SLAYER - 1.0) + MB4Digi.LAYER;
      mehDtMuonLayer[3]->Fill(layer);
      mehDtMuonTime[3]->Fill(MB4Digi.TIME);
      mehDtMuonTimevLayer[3]->Fill(layer, MB4Digi.TIME, 1);
    }

    // get CSC Strip info
    std::vector<PGlobalDigi::CSCstripDigi> CSCstripDigis = srcGlobalDigis->getCSCstripDigis();
    mehCSCStripn->Fill((float)nCSCstripDigis);
    for (auto &CSCstripDigi : CSCstripDigis) {
      mehCSCStripADC->Fill(CSCstripDigi.ADC);
    }

    // get CSC Wire info
    std::vector<PGlobalDigi::CSCwireDigi> CSCwireDigis = srcGlobalDigis->getCSCwireDigis();
    mehCSCWiren->Fill((float)nCSCwireDigis);
    for (auto &CSCwireDigi : CSCwireDigis) {
      mehCSCWireTime->Fill(CSCwireDigi.TIME);
    }
    if (verbosity > 0)
      edm::LogInfo(MsgLoggerCat) << "Done gathering data from event.";

  }  // end loop through events
}

// define this as a plug-in
// DEFINE_FWK_MODULE(GlobalDigisHistogrammer);
