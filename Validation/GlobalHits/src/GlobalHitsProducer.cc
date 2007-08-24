/** \file GlobalHitsProducer.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2007/04/30 19:38:26 $
 *  $Revision: 1.9 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "Validation/GlobalHits/interface/GlobalHitsProducer.h"

GlobalHitsProducer::GlobalHitsProducer(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), frequency(0), vtxunit(0), label(""), 
  getAllProvenances(false), printProvenanceInfo(false), //nRawGenPart(0), 
  //ECalEBSrc_(""), ECalEESrc_(""), ECalESSrc_(""), HCalSrc_(""),
  //PxlBrlLowSrc_(""), PxlBrlHighSrc_(""), PxlFwdLowSrc_(""),
  //PxlFwdHighSrc_(""), SiTIBLowSrc_(""), SiTIBHighSrc_(""),
  //SiTOBLowSrc_(""), SiTOBHighSrc_(""), SiTIDLowSrc_(""), 
  //SiTIDHighSrc_(""), SiTECLowSrc_(""), SiTECHighSrc_(""),
  //MuonDtSrc_(""), MuonCscSrc_(""), MuonRpcSrc_(""), 
  count(0)
{
  std::string MsgLoggerCat = "GlobalHitsProducer_GlobalHitsProducer";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  vtxunit = iPSet.getUntrackedParameter<int>("VtxUnit");
  //label = iPSet.getParameter<std::string>("Label");
  outputfile = iPSet.getParameter<std::string>("OutputFile");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

  //get Labels to use to extract information
  PxlBrlLowSrc_ = iPSet.getParameter<edm::InputTag>("PxlBrlLowSrc");
  PxlBrlHighSrc_ = iPSet.getParameter<edm::InputTag>("PxlBrlHighSrc");
  PxlFwdLowSrc_ = iPSet.getParameter<edm::InputTag>("PxlFwdLowSrc");
  PxlFwdHighSrc_ = iPSet.getParameter<edm::InputTag>("PxlFwdHighSrc");

  SiTIBLowSrc_ = iPSet.getParameter<edm::InputTag>("SiTIBLowSrc");
  SiTIBHighSrc_ = iPSet.getParameter<edm::InputTag>("SiTIBHighSrc");
  SiTOBLowSrc_ = iPSet.getParameter<edm::InputTag>("SiTOBLowSrc");
  SiTOBHighSrc_ = iPSet.getParameter<edm::InputTag>("SiTOBHighSrc");
  SiTIDLowSrc_ = iPSet.getParameter<edm::InputTag>("SiTIDLowSrc");
  SiTIDHighSrc_ = iPSet.getParameter<edm::InputTag>("SiTIDHighSrc");
  SiTECLowSrc_ = iPSet.getParameter<edm::InputTag>("SiTECLowSrc");
  SiTECHighSrc_ = iPSet.getParameter<edm::InputTag>("SiTECHighSrc");

  MuonCscSrc_ = iPSet.getParameter<edm::InputTag>("MuonCscSrc");
  MuonDtSrc_ = iPSet.getParameter<edm::InputTag>("MuonDtSrc");
  MuonRpcSrc_ = iPSet.getParameter<edm::InputTag>("MuonRpcSrc");

  ECalEBSrc_ = iPSet.getParameter<edm::InputTag>("ECalEBSrc");
  ECalEESrc_ = iPSet.getParameter<edm::InputTag>("ECalEESrc");
  ECalESSrc_ = iPSet.getParameter<edm::InputTag>("ECalESSrc");

  HCalSrc_ = iPSet.getParameter<edm::InputTag>("HCalSrc");

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // create persistent object
  //produces<PGlobalSimHit>(label);

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "    VtxUnit       = " << vtxunit << "\n"
      //<< "    Label         = " << label << "\n"
      << "    OutputFile    = " << outputfile << "\n"
      << "    GetProv       = " << getAllProvenances << "\n"
      << "    PrintProv     = " << printProvenanceInfo << "\n"
      << "    PxlBrlLowSrc  = " << PxlBrlLowSrc_.label() 
      << ":" << PxlBrlLowSrc_.instance() << "\n"
      << "    PxlBrlHighSrc = " << PxlBrlHighSrc_.label() 
      << ":" << PxlBrlHighSrc_.instance() << "\n"
      << "    PxlFwdLowSrc  = " << PxlFwdLowSrc_.label() 
      << ":" << PxlBrlLowSrc_.instance() << "\n"
      << "    PxlFwdHighSrc = " << PxlFwdHighSrc_.label() 
      << ":" << PxlBrlHighSrc_.instance() << "\n"
      << "    SiTIBLowSrc   = " << SiTIBLowSrc_.label() 
      << ":" << SiTIBLowSrc_.instance() << "\n"
      << "    SiTIBHighSrc  = " << SiTIBHighSrc_.label() 
      << ":" << SiTIBHighSrc_.instance() << "\n"
      << "    SiTOBLowSrc   = " << SiTOBLowSrc_.label() 
      << ":" << SiTOBLowSrc_.instance() << "\n"
      << "    SiTOBHighSrc  = " << SiTOBHighSrc_.label() 
      << ":" << SiTOBHighSrc_.instance() << "\n"
      << "    SiTIDLowSrc   = " << SiTIDLowSrc_.label() 
      << ":" << SiTIDLowSrc_.instance() << "\n"
      << "    SiTIDHighSrc  = " << SiTIDHighSrc_.label() 
      << ":" << SiTIDHighSrc_.instance() << "\n"
      << "    SiTECLowSrc   = " << SiTECLowSrc_.label() 
      << ":" << SiTECLowSrc_.instance() << "\n"
      << "    SiTECHighSrc  = " << SiTECHighSrc_.label() 
      << ":" << SiTECHighSrc_.instance() << "\n"
      << "    MuonCscSrc    = " << MuonCscSrc_.label() 
      << ":" << MuonCscSrc_.instance() << "\n"
      << "    MuonDtSrc     = " << MuonDtSrc_.label() 
      << ":" << MuonDtSrc_.instance() << "\n"
      << "    MuonRpcSrc    = " << MuonRpcSrc_.label() 
      << ":" << MuonRpcSrc_.instance() << "\n"
      << "    ECalEBSrc     = " << ECalEBSrc_.label() 
      << ":" << ECalEBSrc_.instance() << "\n"
      << "    ECalEESrc     = " << ECalEESrc_.label() 
      << ":" << ECalEESrc_.instance() << "\n"
      << "    ECalESSrc     = " << ECalESSrc_.label() 
      << ":" << ECalESSrc_.instance() << "\n"
      << "    HCalSrc       = " << HCalSrc_.label() 
      << ":" << HCalSrc_.instance() << "\n"
      << "===============================\n";
  }

  // get dqm info
  dbe = 0;
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  if (dbe) {
    if (verbosity >= 0 ) {
      dbe->setVerbose(1);
    } else {
      dbe->setVerbose(0);
    }
  }

  if (dbe) {
    if (verbosity >= 0 ) dbe->showDirStructure();
  }

  // initialize monitor elements
  for (Int_t i = 0; i < 2; ++i) {
    meMCRGP[i] = 0;
    meMCG4Vtx[i] = 0;
    meGeantVtxX[i] = 0;
    meGeantVtxY[i] = 0;
    meGeantVtxZ[i] = 0; 
    meMCG4Trk[i] = 0;
    meCaloEcal[i] = 0;
    meCaloEcalE[i] = 0;
    meCaloEcalToF[i] = 0;
    meCaloPreSh[i] = 0;
    meCaloPreShE[i] = 0;
    meCaloPreShToF[i] = 0;
    meCaloHcal[i] = 0;
    meCaloHcalE[i] = 0;
    meCaloHcalToF[i] = 0;
    meTrackerPx[i] = 0;
    meTrackerSi[i] = 0;
    meMuon[i] = 0;
    meMuonDtToF[i] = 0;
    meMuonCscToF[i] = 0;
    meMuonRpcFToF[i] = 0;
    meMuonRpcBToF[i] = 0;
  }
  meGeantTrkPt = 0;
  meGeantTrkE = 0;
  meCaloEcalPhi = 0;
  meCaloEcalEta = 0;
  meCaloPreShPhi = 0;
  meCaloPreShEta = 0;
  meCaloHcalPhi = 0;
  meCaloHcalEta = 0;
  meTrackerPxPhi = 0;
  meTrackerPxEta = 0;
  meTrackerPxBToF = 0;
  meTrackerPxBR = 0;
  meTrackerPxFToF = 0;
  meTrackerPxFZ = 0;
  meTrackerSiPhi = 0;
  meTrackerSiEta = 0;
  meTrackerSiBToF = 0;
  meTrackerSiBR = 0;
  meTrackerSiFToF = 0;
  meTrackerSiFZ = 0;
  meMuonPhi = 0;
  meMuonEta = 0;
  meMuonDtR = 0;
  meMuonCscZ = 0;
  meMuonRpcBR = 0;
  meMuonRpcFZ = 0;

  //create histograms
  Char_t hname[200];
  Char_t htitle[200];
  if (dbe) {

    // MCGeant
    dbe->setCurrentFolder("MCGeant");
    sprintf(hname,"hMCRGP_1003");
    sprintf(htitle,"RawGenParticles");
    meMCRGP[0] = dbe->book1D(hname,htitle,100,0.,5000.);
    sprintf(hname,"hMCRGP_1013");
    meMCRGP[1] = dbe->book1D(hname,htitle,100,0.,500.);  

    sprintf(hname,"hMCG4Vtx_1001");
    sprintf(htitle,"G4 Vertices");
    meMCG4Vtx[0] = dbe->book1D(hname,htitle,100,0.,50000.);
    sprintf(hname,"hMCG4Vtx2_1011");
    meMCG4Vtx[1] = dbe->book1D(hname,htitle,100,-0.5,99.5); 

    sprintf(hname,"hMCG4Trk_1002");
    sprintf(htitle,"G4 Tracks");
    meMCG4Trk[0] = dbe->book1D(hname,htitle,150,0.,15000.);
    sprintf(hname,"hMCG4Trk2_1012");
    meMCG4Trk[1] = dbe->book1D(hname,htitle,150,-0.5,99.5);    

    sprintf(hname,"hGeantVtxX2_1104");
    sprintf(htitle,"Geant vertex x/micrometer");
    meGeantVtxX[0] = dbe->book1D(hname,htitle,100,-8000000.,8000000.);
    sprintf(hname,"hGeantVtxX_1101");
    meGeantVtxX[1] = dbe->book1D(hname,htitle,100,-50.,50.); 

    sprintf(hname,"hGeantVtxY_1105");
    sprintf(htitle,"Geant vertex y/micrometer");
    meGeantVtxY[0] = dbe->book1D(hname,htitle,100,-8000000,8000000.);
    sprintf(hname,"hGeantVtxY_1102");
    meGeantVtxY[1] = dbe->book1D(hname,htitle,100,-50.,50.); 

    sprintf(hname,"hGeantVtxZ_1106");
    sprintf(htitle,"Geant vertex z/millimeter");
    meGeantVtxZ[0] = dbe->book1D(hname,htitle,100,-11000.,11000.);
    sprintf(hname,"hGeantVtxZ_1103");
    meGeantVtxZ[1] = dbe->book1D(hname,htitle,100,-250.,250.);

    sprintf(hname,"hGeantTrkPt_1201");
    sprintf(htitle,"Geant track pt/GeV");
    meGeantTrkPt = dbe->book1D(hname,htitle,100,0.,200.);

    sprintf(hname,"hGeantTrkE_1202");
    sprintf(htitle,"Geant track E/GeV");
    meGeantTrkE = dbe->book1D(hname,htitle,100,0.,5000.);

    // ECal
    dbe->setCurrentFolder("ECal");
    sprintf(hname,"hCaloEcal_2101");
    sprintf(htitle,"Ecal hits");
    meCaloEcal[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hCaloEcal2_2111");
    meCaloEcal[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);

    sprintf(hname,"hCaloEcalE_2102");
    sprintf(htitle,"Ecal hits, energy/GeV");
    meCaloEcalE[0] = dbe->book1D(hname,htitle,100,0.,10.);
    sprintf(hname,"hCaloEcalE2_2112");
    meCaloEcalE[1] = dbe->book1D(hname,htitle,100,0.,0.1);

    sprintf(hname,"hCaloEcalToF_2103");
    sprintf(htitle,"Ecal hits, ToF/ns");
    meCaloEcalToF[0] = dbe->book1D(hname,htitle,100,0.,1000.);
    sprintf(hname,"hCaloEcalToF2_2113");
    meCaloEcalToF[1] = dbe->book1D(hname,htitle,100,0.,100.);

    sprintf(hname,"hCaloEcalPhi_2104");
    sprintf(htitle,"Ecal hits, phi/rad");
    meCaloEcalPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);

    sprintf(hname,"hCaloEcalEta_2105");
    sprintf(htitle,"Ecal hits, eta");
    meCaloEcalEta = dbe->book1D(hname,htitle,100,-5.5,5.5);

    sprintf(hname,"hCaloPreSh_2201");
    sprintf(htitle,"PreSh hits");
    meCaloPreSh[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hCaloPreSh2_2211");
    meCaloPreSh[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);

    sprintf(hname,"hCaloPreShE_2202");
    sprintf(htitle,"PreSh hits, energy/GeV");
    meCaloPreShE[0] = dbe->book1D(hname,htitle,100,0.,10.);
    sprintf(hname,"hCaloPreShE2_2212");
    meCaloPreShE[1] = dbe->book1D(hname,htitle,100,0.,0.1);

    sprintf(hname,"hCaloPreShToF_2203");
    sprintf(htitle,"PreSh hits, ToF/ns");
    meCaloPreShToF[0] = dbe->book1D(hname,htitle,100,0.,1000.);
    sprintf(hname,"hCaloPreShToF2_2213");
    meCaloPreShToF[1] = dbe->book1D(hname,htitle,100,0.,100.);

    sprintf(hname,"hCaloPreShPhi_2204");
    sprintf(htitle,"PreSh hits, phi/rad");
    meCaloPreShPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);

    sprintf(hname,"hCaloPreShEta_2205");
    sprintf(htitle,"PreSh hits, eta");
    meCaloPreShEta = dbe->book1D(hname,htitle,100,-5.5,5.5);

    // Hcal
    dbe->setCurrentFolder("HCal");
    sprintf(hname,"hCaloHcal_2301");
    sprintf(htitle,"Hcal hits");
    meCaloHcal[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hCaloHcal2_2311");
    meCaloHcal[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);

    sprintf(hname,"hCaloHcalE_2302");
    sprintf(htitle,"Hcal hits, energy/GeV");
    meCaloHcalE[0] = dbe->book1D(hname,htitle,100,0.,10.);
    sprintf(hname,"hCaloHcalE2_2312");
    meCaloHcalE[1] = dbe->book1D(hname,htitle,100,0.,0.1);

    sprintf(hname,"hCaloHcalToF_2303");
    sprintf(htitle,"Hcal hits, ToF/ns");
    meCaloHcalToF[0] = dbe->book1D(hname,htitle,100,0.,1000.);
    sprintf(hname,"hCaloHcalToF2_2313");
    meCaloHcalToF[1] = dbe->book1D(hname,htitle,100,0.,100.);

    sprintf(hname,"hCaloHcalPhi_2304");
    sprintf(htitle,"Hcal hits, phi/rad");
    meCaloHcalPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);

    sprintf(hname,"hCaloHcalEta_2305");
    sprintf(htitle,"Hcal hits, eta");
    meCaloHcalEta = dbe->book1D(hname,htitle,100,-5.5,5.5);

    dbe->setCurrentFolder("Tracker");
    sprintf(hname,"hTrackerPx_3101");
    sprintf(htitle,"Pixel hits");
    meTrackerPx[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hTrackerPx2_3111");
    meTrackerPx[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);

    sprintf(hname,"hTrackerPxPhi_3102");
    sprintf(htitle,"Pixel hits phi/rad");
    meTrackerPxPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);

    sprintf(hname,"hTrackerPxEta_3103");
    sprintf(htitle,"Pixel hits eta");
    meTrackerPxEta = dbe->book1D(hname,htitle,100,-3.5,3.5);

    sprintf(hname,"hTrackerPxBToF_3104");
    sprintf(htitle,"Pixel barrel hits, ToF/ns");
    meTrackerPxBToF = dbe->book1D(hname,htitle,100,0.,40.);

    sprintf(hname,"hTrackerPxBR_3106");
    sprintf(htitle,"Pixel barrel hits, R/cm");
    meTrackerPxBR = dbe->book1D(hname,htitle,100,0.,50.);

    sprintf(hname,"hTrackerPxFToF_3105");
    sprintf(htitle,"Pixel forward hits, ToF/ns");
    meTrackerPxFToF = dbe->book1D(hname,htitle,100,0.,50.);

    sprintf(hname,"hTrackerPxFZ_3107");
    sprintf(htitle,"Pixel forward hits, Z/cm");
    meTrackerPxFZ = dbe->book1D(hname,htitle,200,-100.,100.);

    sprintf(hname,"hTrackerSi_3201");
    sprintf(htitle,"Silicon hits");
    meTrackerSi[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hTrackerSi2_3211");
    meTrackerSi[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);

    sprintf(hname,"hTrackerSiPhi_3202");
    sprintf(htitle,"Silicon hits phi/rad");
    meTrackerSiPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);

    sprintf(hname,"hTrackerSiEta_3203");
    sprintf(htitle,"Silicon hits eta");
    meTrackerSiEta = dbe->book1D(hname,htitle,100,-3.5,3.5);

    sprintf(hname,"hTrackerSiBToF_3204");
    sprintf(htitle,"Silicon barrel hits, ToF/ns");
    meTrackerSiBToF = dbe->book1D(hname,htitle,100,0.,50.);

    sprintf(hname,"hTrackerSiBR_3206");
    sprintf(htitle,"Silicon barrel hits, R/cm");
    meTrackerSiBR = dbe->book1D(hname,htitle,100,0.,200.);

    sprintf(hname,"hTrackerSiFToF_3205");
    sprintf(htitle,"Silicon forward hits, ToF/ns");
    meTrackerSiFToF = dbe->book1D(hname,htitle,100,0.,75.);

    sprintf(hname,"hTrackerSiFZ_3207");
    sprintf(htitle,"Silicon forward hits, Z/cm");
    meTrackerSiFZ = dbe->book1D(hname,htitle,200,-300.,300.);

    dbe->setCurrentFolder("Muon");
    sprintf(hname,"hMuon_4001");
    sprintf(htitle,"Muon hits");
    meMuon[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hMuon2_4011");
    meMuon[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);

    sprintf(hname,"hMuonPhi_4002");
    sprintf(htitle,"Muon hits phi/rad");
    meMuonPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);

    sprintf(hname,"hMuonEta_4003");
    sprintf(htitle,"Muon hits eta");
    meMuonEta = dbe->book1D(hname,htitle,100,-3.5,3.5);

    sprintf(hname,"hMuonCscToF_4201");
    sprintf(htitle,"Muon CSC hits, ToF/ns");
    meMuonCscToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonCscToF2_4202");
    meMuonCscToF[1] = dbe->book1D(hname,htitle,100,0.,50.);

    sprintf(hname,"hMuonCscZ_4203");
    sprintf(htitle,"Muon CSC hits, Z/cm");
    meMuonCscZ = dbe->book1D(hname,htitle,200,-1500.,1500.);

    sprintf(hname,"hMuonDtToF_4101");
    sprintf(htitle,"Muon DT hits, ToF/ns");
    meMuonDtToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonDtToF2_4102");
    meMuonDtToF[1] = dbe->book1D(hname,htitle,100,0.,50.);

    sprintf(hname,"hMuonDtR_4103");
    sprintf(htitle,"Muon DT hits, R/cm");
    meMuonDtR = dbe->book1D(hname,htitle,100,0.,1500.); 

    sprintf(hname,"hMuonRpcFToF_4304");
    sprintf(htitle,"Muon RPC forward hits, ToF/ns");
    meMuonRpcFToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonRpcFToF2_4305");
    meMuonRpcFToF[1] = dbe->book1D(hname,htitle,100,0.,50.);

    sprintf(hname,"hMuonRpcFZ_4306");
    sprintf(htitle,"Muon RPC forward hits, Z/cm");
    meMuonRpcFZ = dbe->book1D(hname,htitle,201,-1500.,1500.);

    sprintf(hname,"hMuonRpcBToF_4101");
    sprintf(htitle,"Muon RPC barrel hits, ToF/ns");
    meMuonRpcBToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonRpcBToF2_4102");
    meMuonRpcBToF[1] = dbe->book1D(hname,htitle,100,0.,50.);

    sprintf(hname,"hMuonRpcBR_4103");
    sprintf(htitle,"Muon RPC barrel hits, R/cm");
    meMuonRpcBR = dbe->book1D(hname,htitle,100,0.,1500.);   
  }
}

GlobalHitsProducer::~GlobalHitsProducer() 
{
  if (outputfile.size() != 0 && dbe) dbe->save(outputfile);
}

void GlobalHitsProducer::beginJob(const edm::EventSetup& iSetup)
{
  //clear();
  return;
}

void GlobalHitsProducer::endJob()
{
  std::string MsgLoggerCat = "GlobalHitsProducer_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " events.";
  return;
}

void GlobalHitsProducer::analyze(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalHitsProducer_produce";

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

  // clear event holders
  //clear();

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
  //gather G4MC information from event
  fillG4MC(iEvent);
  // gather Tracker information from event
  fillTrk(iEvent,iSetup);
  // gather muon information from event
  fillMuon(iEvent, iSetup);
  // gather Ecal information from event
  fillECal(iEvent, iSetup);
  // gather Hcal information from event
  fillHCal(iEvent, iSetup);

  if (verbosity > 0)
    edm::LogInfo (MsgLoggerCat)
      << "Done gathering data from event.";

  // produce object to put into event
  //std::auto_ptr<PGlobalSimHit> pOut(new PGlobalSimHit);

  //if (verbosity > 2)
  //  edm::LogInfo (MsgLoggerCat)
  //    << "Saving event contents:";

  // call store functions
  // store G4MC information in product
  //storeG4MC(*pOut);
  // store Tracker information in produce
  //storeTrk(*pOut);
  // store Muon information in produce
  //storeMuon(*pOut);
  // store ECal information in produce
  //storeECal(*pOut);
  // store HCal information in produce
  //storeHCal(*pOut);

  // store information in event
  //iEvent.put(pOut,label);

  return;
}

//==================fill and store functions================================
void GlobalHitsProducer::fillG4MC(const edm::Event& iEvent)
{

  std::string MsgLoggerCat = "GlobalHitsProducer_fillG4MC";
 
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  //////////////////////
  // get MC information
  /////////////////////
  edm::Handle<edm::HepMCProduct> HepMCEvt;
  std::vector<edm::Handle<edm::HepMCProduct> > AllHepMCEvt;
  iEvent.getManyByType(AllHepMCEvt);

  // loop through products and extract VtxSmearing if available. Any of them
  // should have the information needed
  for (unsigned int i = 0; i < AllHepMCEvt.size(); ++i) {
    HepMCEvt = AllHepMCEvt[i];
    if ((HepMCEvt.provenance()->product()).moduleLabel() == "VtxSmeared")
      break;
  }

  if (!HepMCEvt.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find HepMCProduct in event!";
    return;
  } else {
    eventout += "\n          Using HepMCProduct: ";
    eventout += (HepMCEvt.provenance()->product()).moduleLabel();
  }
  const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();
  nRawGenPart = MCEvt->particles_size();

  if (verbosity > 1) {
    eventout += "\n          Number of Raw Particles collected:......... ";
    eventout += nRawGenPart;
  }  

  if (meMCRGP[0]) meMCRGP[0]->Fill((float)nRawGenPart);
  if (meMCRGP[1]) meMCRGP[1]->Fill((float)nRawGenPart);  

  ////////////////////////////
  // get G4Vertex information
  ////////////////////////////
  // convert unit stored in SimVertex to mm
  float unit = 0.;
  if (vtxunit == 0) unit = 1.;  // already in mm
  if (vtxunit == 1) unit = 10.; // stored in cm, convert to mm

  edm::Handle<edm::SimVertexContainer> G4VtxContainer;
  iEvent.getByType(G4VtxContainer);
  if (!G4VtxContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find SimVertex in event!";
    return;
  }
  int i = 0;
  edm::SimVertexContainer::const_iterator itVtx;
  for (itVtx = G4VtxContainer->begin(); itVtx != G4VtxContainer->end(); 
       ++itVtx) {
    
    ++i;

    const HepLorentzVector& G4Vtx = itVtx->position();
    //G4VtxX.push_back((G4Vtx[0]*unit)/micrometer);
    //G4VtxY.push_back((G4Vtx[1]*unit)/micrometer);
    //G4VtxZ.push_back((G4Vtx[2]*unit)/millimeter);

    if (meGeantVtxX[0]) meGeantVtxX[0]->Fill((G4Vtx[0]*unit)/micrometer);
    if (meGeantVtxX[1]) meGeantVtxX[1]->Fill((G4Vtx[0]*unit)/micrometer);
    
    if (meGeantVtxY[0]) meGeantVtxY[0]->Fill((G4Vtx[1]*unit)/micrometer);
    if (meGeantVtxY[1]) meGeantVtxY[1]->Fill((G4Vtx[1]*unit)/micrometer);
    
    if (meGeantVtxZ[0]) meGeantVtxZ[0]->Fill((G4Vtx[2]*unit)/millimeter);
    if (meGeantVtxZ[1]) meGeantVtxZ[1]->Fill((G4Vtx[2]*unit)/millimeter); 
    
  }

  if (verbosity > 1) {
    eventout += "\n          Number of G4Vertices collected:............ ";
    eventout += i;
  }  

  if (meMCG4Vtx[0]) meMCG4Vtx[0]->Fill((float)i);
  if (meMCG4Vtx[1]) meMCG4Vtx[1]->Fill((float)i);  

  ///////////////////////////
  // get G4Track information
  ///////////////////////////
  edm::Handle<edm::SimTrackContainer> G4TrkContainer;
  iEvent.getByType(G4TrkContainer);
  if (!G4TrkContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find SimTrack in event!";
    return;
  }
  i = 0;
  edm::SimTrackContainer::const_iterator itTrk;
  for (itTrk = G4TrkContainer->begin(); itTrk != G4TrkContainer->end(); 
       ++itTrk) {

    ++i;

    const HepLorentzVector& G4Trk = itTrk->momentum();
    //G4TrkPt.push_back(sqrt(G4Trk[0]*G4Trk[0]+G4Trk[1]*G4Trk[1])); //GeV
    //G4TrkE.push_back(G4Trk[3]);                                   //GeV

    if (meGeantTrkPt) meGeantTrkPt->
			Fill(sqrt(G4Trk[0]*G4Trk[0]+G4Trk[1]*G4Trk[1]));
    if (meGeantTrkE) meGeantTrkE->Fill(G4Trk[3]);
  } 

  if (verbosity > 1) {
    eventout += "\n          Number of G4Tracks collected:.............. ";
    eventout += i;
  }  

  if (meMCG4Trk[0]) meMCG4Trk[0]->Fill((float)i);
  if (meMCG4Trk[1]) meMCG4Trk[1]->Fill((float)i); 

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
    
  return;
}

/*
void GlobalHitsProducer::storeG4MC(PGlobalSimHit& product)
{
  std::string MsgLoggerCat = "GlobalHitsProducer_storeG4MC";

  if (verbosity > 2) {
    TString eventout("\n       nRawGenPart        = ");
    eventout += nRawGenPart;
    eventout += "\n       nG4Vtx             = ";
    eventout += G4VtxX.size();
    for (unsigned int i = 0; i < G4VtxX.size(); ++i) {
      eventout += "\n          (x,y,z)         = (";
      eventout += G4VtxX[i];
      eventout += ", ";
      eventout += G4VtxY[i];
      eventout += ", ";
      eventout += G4VtxZ[i];
      eventout += ")";      
    }
    eventout += "\n       nG4Trk             = ";
    eventout += G4TrkPt.size();
    for (unsigned int i = 0; i < G4TrkPt.size(); ++i) {
      eventout += "\n          (pt,e)          = (";
      eventout += G4TrkPt[i];
      eventout += ", ";
      eventout += G4TrkE[i];
      eventout += ")";
    }    
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  } // end verbose output

  product.putRawGenPart(nRawGenPart);
  product.putG4Vtx(G4VtxX, G4VtxY, G4VtxZ);
  product.putG4Trk(G4TrkPt, G4TrkE);

  return;
}
*/

void GlobalHitsProducer::fillTrk(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{

  nPxlHits = 0;
  std::string MsgLoggerCat = "GlobalHitsProducer_fillTrk";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  
  
  // access the tracker geometry
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
  if (!theTrackerGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerDigiGeometryRecord in event!";
    return;
  }
  const TrackerGeometry& theTracker(*theTrackerGeometry);
    
  // iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;

  ///////////////////////////////
  // get Pixel Barrel information
  ///////////////////////////////
  edm::PSimHitContainer thePxlBrlHits;
  // extract low container
  edm::Handle<edm::PSimHitContainer> PxlBrlLowContainer;
  iEvent.getByLabel(PxlBrlLowSrc_,PxlBrlLowContainer);
  if (!PxlBrlLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerHitsPixelBarrelLowTof in event!";
    return;
  }
  // extract high container
  edm::Handle<edm::PSimHitContainer> PxlBrlHighContainer;
  iEvent.getByLabel(PxlBrlHighSrc_,PxlBrlHighContainer);
  if (!PxlBrlHighContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerHitsPixelBarrelHighTof in event!";
    return;
  }
  // place both containers into new container
  thePxlBrlHits.insert(thePxlBrlHits.end(),PxlBrlLowContainer->begin(),
		       PxlBrlLowContainer->end());
  thePxlBrlHits.insert(thePxlBrlHits.end(),PxlBrlHighContainer->begin(),
		       PxlBrlHighContainer->end());

  // cycle through new container
  int i = 0, j = 0;
  for (itHit = thePxlBrlHits.begin(); itHit != thePxlBrlHits.end(); ++itHit) {

    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dTrk) && (subdetector == sdPxlBrl)) {

      // get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theTracker.idToDetUnit(theDetUnitId);

      if (!theDet) {
	edm::LogWarning(MsgLoggerCat)
	  << "Unable to get GeomDetUnit from PxlBrlHits for Hit " << i;
	continue;
      }

      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bSurface = theDet->surface();

      // gather necessary information
      //PxlBrlToF.push_back(itHit->tof());
      //PxlBrlR.push_back(bSurface.toGlobal(itHit->localPosition()).perp());
      //PxlBrlPhi.push_back(bSurface.toGlobal(itHit->localPosition()).phi());
      //PxlBrlEta.push_back(bSurface.toGlobal(itHit->localPosition()).eta());

      if(meTrackerPxBToF) meTrackerPxBToF->Fill(itHit->tof());
      if(meTrackerPxBR) 
	meTrackerPxBR->Fill(bSurface.toGlobal(itHit->localPosition()).perp());
      if(meTrackerPxPhi) 
	meTrackerPxPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if(meTrackerPxEta) 
	meTrackerPxEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat)
	<< "PxlBrl PSimHit " << i 
	<< " is expected to be (det,subdet) = (" 
	<< dTrk << "," << sdPxlBrl
	<< "); value returned is: ("
	<< detector << "," << subdetector << ")";
      continue;
    } // end detector type check
  } // end loop through PxlBrl Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Pixel Barrel Hits collected:..... ";
    eventout += j;
  }  
  
  nPxlHits += j;

  /////////////////////////////////
  // get Pixel Forward information
  ////////////////////////////////
  edm::PSimHitContainer thePxlFwdHits;
  // extract low container
  edm::Handle<edm::PSimHitContainer> PxlFwdLowContainer;
  iEvent.getByLabel(PxlFwdLowSrc_,PxlFwdLowContainer);
  if (!PxlFwdLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerHitsPixelEndcapLowTof in event!";
    return;
  }
  // extract high container
  edm::Handle<edm::PSimHitContainer> PxlFwdHighContainer;
  iEvent.getByLabel(PxlFwdHighSrc_,PxlFwdHighContainer);
  if (!PxlFwdHighContainer.isValid()) {
    edm::LogWarning("GlobalHitsProducer_fillTrk")
      << "Unable to find TrackerHitsPixelEndcapHighTof in event!";
    return;
  }
  // place both containers into new container
  thePxlFwdHits.insert(thePxlFwdHits.end(),PxlFwdLowContainer->begin(),
		       PxlFwdLowContainer->end());
  thePxlFwdHits.insert(thePxlFwdHits.end(),PxlFwdHighContainer->begin(),
		       PxlFwdHighContainer->end());

  // cycle through new container
  i = 0; j = 0;
  for (itHit = thePxlFwdHits.begin(); itHit != thePxlFwdHits.end(); ++itHit) {

    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dTrk) && (subdetector == sdPxlFwd)) {

      // get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theTracker.idToDetUnit(theDetUnitId);

      if (!theDet) {
	edm::LogWarning(MsgLoggerCat)
	  << "Unable to get GeomDetUnit from PxlFwdHits for Hit " << i;;
	continue;
      }

      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bSurface = theDet->surface();

      // gather necessary information
      //PxlFwdToF.push_back(itHit->tof());
      //PxlFwdZ.push_back(bSurface.toGlobal(itHit->localPosition()).z());
      //PxlFwdPhi.push_back(bSurface.toGlobal(itHit->localPosition()).phi());
      //PxlFwdEta.push_back(bSurface.toGlobal(itHit->localPosition()).eta());

      if(meTrackerPxFToF) meTrackerPxFToF->Fill(itHit->tof());
      if(meTrackerPxFZ) 
	meTrackerPxFZ->Fill(bSurface.toGlobal(itHit->localPosition()).z());
      if(meTrackerPxPhi) 
	meTrackerPxPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if(meTrackerPxEta) 
	meTrackerPxEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat)
	<< "PxlFwd PSimHit " << i 
	<< " is expected to be (det,subdet) = (" 
	<< dTrk << "," << sdPxlFwd
	<< "); value returned is: ("
	<< detector << "," << subdetector << ")";
      continue;
    } // end detector type check
  } // end loop through PxlFwd Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Pixel Forward Hits collected:.... ";
    eventout += j;
  }  

  nPxlHits += j;

  if (meTrackerPx[0]) meTrackerPx[0]->Fill((float)nPxlHits);
  if (meTrackerPx[1]) meTrackerPx[1]->Fill((float)nPxlHits); 

  ///////////////////////////////////
  // get Silicon Barrel information
  //////////////////////////////////
  nSiHits = 0;
  edm::PSimHitContainer theSiBrlHits;
  // extract TIB low container
  edm::Handle<edm::PSimHitContainer> SiTIBLowContainer;
  iEvent.getByLabel(SiTIBLowSrc_,SiTIBLowContainer);
  if (!SiTIBLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerHitsTIBLowTof in event!";
    return;
  }
  // extract TIB high container
  edm::Handle<edm::PSimHitContainer> SiTIBHighContainer;
  iEvent.getByLabel(SiTIBHighSrc_,SiTIBHighContainer);
  if (!SiTIBHighContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerHitsTIBHighTof in event!";
    return;
  }
  // extract TOB low container
  edm::Handle<edm::PSimHitContainer> SiTOBLowContainer;
  iEvent.getByLabel(SiTOBLowSrc_,SiTOBLowContainer);
  if (!SiTOBLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerHitsTOBLowTof in event!";
    return;
  }
  // extract TOB high container
  edm::Handle<edm::PSimHitContainer> SiTOBHighContainer;
  iEvent.getByLabel(SiTOBHighSrc_,SiTOBHighContainer);
  if (!SiTOBHighContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerHitsTOBHighTof in event!";
    return;
  }
  // place all containers into new container
  theSiBrlHits.insert(theSiBrlHits.end(),SiTIBLowContainer->begin(),
		       SiTIBLowContainer->end());
  theSiBrlHits.insert(theSiBrlHits.end(),SiTIBHighContainer->begin(),
		       SiTIBHighContainer->end());
  theSiBrlHits.insert(theSiBrlHits.end(),SiTOBLowContainer->begin(),
		       SiTOBLowContainer->end());
  theSiBrlHits.insert(theSiBrlHits.end(),SiTOBHighContainer->begin(),
		       SiTOBHighContainer->end());

  // cycle through new container
  i = 0; j = 0;
  for (itHit = theSiBrlHits.begin(); itHit != theSiBrlHits.end(); ++itHit) {

    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dTrk) && 
	((subdetector == sdSiTIB) ||
	 (subdetector == sdSiTOB))) {

      // get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theTracker.idToDetUnit(theDetUnitId);

      if (!theDet) {
	edm::LogWarning(MsgLoggerCat)
	  << "Unable to get GeomDetUnit from SiBrlHits for Hit " << i;
	continue;
      }

      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bSurface = theDet->surface();

      // gather necessary information
      //SiBrlToF.push_back(itHit->tof());
      //SiBrlR.push_back(bSurface.toGlobal(itHit->localPosition()).perp());
      //SiBrlPhi.push_back(bSurface.toGlobal(itHit->localPosition()).phi());
      //SiBrlEta.push_back(bSurface.toGlobal(itHit->localPosition()).eta());

      if(meTrackerSiBToF) meTrackerSiBToF->Fill(itHit->tof());
      if(meTrackerSiBR) 
	meTrackerSiBR->Fill(bSurface.toGlobal(itHit->localPosition()).perp());
      if(meTrackerSiPhi) 
	meTrackerSiPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if(meTrackerSiEta) 
	meTrackerSiEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat)
	<< "SiBrl PSimHit " << i 
	<< " is expected to be (det,subdet) = (" 
	<< dTrk << "," << sdSiTIB
	<< " || " << sdSiTOB << "); value returned is: ("
	<< detector << "," << subdetector << ")";
      continue;
    } // end detector type check
  } // end loop through SiBrl Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Silicon Barrel Hits collected:... ";
    eventout += j;
  }  

  nSiHits += j;

  ///////////////////////////////////
  // get Silicon Forward information
  ///////////////////////////////////
  edm::PSimHitContainer theSiFwdHits;
  // extract TID low container
  edm::Handle<edm::PSimHitContainer> SiTIDLowContainer;
  iEvent.getByLabel(SiTIDLowSrc_,SiTIDLowContainer);
  if (!SiTIDLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerHitsTIDLowTof in event!";
    return;
  }
  // extract TID high container
  edm::Handle<edm::PSimHitContainer> SiTIDHighContainer;
  iEvent.getByLabel(SiTIDHighSrc_,SiTIDHighContainer);
  if (!SiTIDHighContainer.isValid()) {
    edm::LogWarning("GlobalHitsProducer_fillTrk")
      << "Unable to find TrackerHitsTIDHighTof in event!";
    return;
  }
  // extract TEC low container
  edm::Handle<edm::PSimHitContainer> SiTECLowContainer;
  iEvent.getByLabel(SiTECLowSrc_,SiTECLowContainer);
  if (!SiTECLowContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerHitsTECLowTof in event!";
    return;
  }
  // extract TEC high container
  edm::Handle<edm::PSimHitContainer> SiTECHighContainer;
  iEvent.getByLabel(SiTECHighSrc_,SiTECHighContainer);
  if (!SiTECHighContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find TrackerHitsTECHighTof in event!";
    return;
  }
  // place all containers into new container
  theSiFwdHits.insert(theSiFwdHits.end(),SiTIDLowContainer->begin(),
		       SiTIDLowContainer->end());
  theSiFwdHits.insert(theSiFwdHits.end(),SiTIDHighContainer->begin(),
		       SiTIDHighContainer->end());
  theSiFwdHits.insert(theSiFwdHits.end(),SiTECLowContainer->begin(),
		       SiTECLowContainer->end());
  theSiFwdHits.insert(theSiFwdHits.end(),SiTECHighContainer->begin(),
		       SiTECHighContainer->end());

  // cycle through container
  i = 0; j = 0;
  for (itHit = theSiFwdHits.begin(); itHit != theSiFwdHits.end(); ++itHit) {

    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned 
    if ((detector == dTrk) && 
	((subdetector == sdSiTID) ||
	 (subdetector == sdSiTEC))) {
      
      // get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theTracker.idToDetUnit(theDetUnitId);
      
      if (!theDet) {
	edm::LogWarning(MsgLoggerCat)
	  << "Unable to get GeomDetUnit from SiFwdHits Hit " << i;
	return;
      }
      
      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bSurface = theDet->surface();
      
      // gather necessary information
      //SiFwdToF.push_back(itHit->tof());
      //SiFwdZ.push_back(bSurface.toGlobal(itHit->localPosition()).z());
      //SiFwdPhi.push_back(bSurface.toGlobal(itHit->localPosition()).phi());
      //SiFwdEta.push_back(bSurface.toGlobal(itHit->localPosition()).eta());

      if(meTrackerSiFToF) meTrackerSiFToF->Fill(itHit->tof());
      if(meTrackerSiFZ) 
	meTrackerSiFZ->Fill(bSurface.toGlobal(itHit->localPosition()).z());
      if(meTrackerSiPhi) 
	meTrackerSiPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if(meTrackerSiEta) 
	meTrackerSiEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat)
	<< "SiFwd PSimHit " << i 
	<< " is expected to be (det,subdet) = (" 
	<< dTrk << "," << sdSiTOB
	<< " || " << sdSiTEC << "); value returned is: ("
	<< detector << "," << subdetector << ")";
      continue;
    } // end check detector type
  } // end loop through SiFwd Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Silicon Forward Hits collected:.. ";
    eventout += j;
  }  

  nSiHits +=j;

  if (meTrackerSi[0]) meTrackerSi[0]->Fill((float)nSiHits);
  if (meTrackerSi[1]) meTrackerSi[1]->Fill((float)nSiHits); 

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

/*
void GlobalHitsProducer::storeTrk(PGlobalSimHit& product)
{
  std::string MsgLoggerCat = "GlobalHitsProducer_storeTrk";

  if (verbosity > 2) {
    TString eventout("\n       nPxlBrlHits        = ");
    eventout += PxlBrlToF.size();
    for (unsigned int i = 0; i < PxlBrlToF.size(); ++i) {
      eventout += "\n          (tof,r,phi,eta) = (";
      eventout += PxlBrlToF[i];
      eventout += ", ";
      eventout += PxlBrlR[i];
      eventout += ", ";
      eventout += PxlBrlPhi[i];
      eventout += ", ";
      eventout += PxlBrlEta[i];
      eventout += ")";      
    } // end PxlBrl output
    eventout += "\n       nPxlFwdHits        = ";
    eventout += PxlFwdToF.size();
    for (unsigned int i = 0; i < PxlFwdToF.size(); ++i) {
      eventout += "\n          (tof,z,phi,eta) = (";
      eventout += PxlFwdToF[i];
      eventout += ", ";
      eventout += PxlFwdZ[i];
      eventout += ", ";
      eventout += PxlFwdPhi[i];
      eventout += ", ";
      eventout += PxlFwdEta[i];
      eventout += ")";      
    } // end PxlFwd output
    eventout += "\n       nSiBrlHits         = ";
    eventout += SiBrlToF.size();
    for (unsigned int i = 0; i < SiBrlToF.size(); ++i) {
      eventout += "\n          (tof,r,phi,eta) = (";
      eventout += SiBrlToF[i];
      eventout += ", ";
      eventout += SiBrlR[i];
      eventout += ", ";
      eventout += SiBrlPhi[i];
      eventout += ", ";
      eventout += SiBrlEta[i];
      eventout += ")";      
    } // end SiBrl output
    eventout += "\n       nSiFwdHits         = ";
    eventout += SiFwdToF.size();
    for (unsigned int i = 0; i < SiFwdToF.size(); ++i) {
      eventout += "\n          (tof,z,phi,eta) = (";
      eventout += SiFwdToF[i];
      eventout += ", ";
      eventout += SiFwdZ[i];
      eventout += ", ";
      eventout += SiFwdPhi[i];
      eventout += ", ";
      eventout += SiFwdEta[i];
      eventout += ")";      
    } // end SiFwd output
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  } // end verbose output

  product.putPxlBrlHits(PxlBrlToF,PxlBrlR,PxlBrlPhi,PxlBrlEta);
  product.putPxlFwdHits(PxlFwdToF,PxlFwdZ,PxlFwdPhi,PxlFwdEta);
  product.putSiBrlHits(SiBrlToF,SiBrlR,SiBrlPhi,SiBrlEta);
  product.putSiFwdHits(SiFwdToF,SiFwdZ,SiFwdPhi,SiFwdEta);

  return;
}
*/

void GlobalHitsProducer::fillMuon(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{
  nMuonHits = 0;
  std::string MsgLoggerCat = "GlobalHitsProducer_fillMuon";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  

  // iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;

  //int i = 0, j = 0;
  ///////////////////////
  // access the CSC Muon
  ///////////////////////
  // access the CSC Muon geometry
  edm::ESHandle<CSCGeometry> theCSCGeometry;
  iSetup.get<MuonGeometryRecord>().get(theCSCGeometry);
  if (!theCSCGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find MuonGeometryRecord for the CSCGeometry in event!";
    return;
  }
  const CSCGeometry& theCSCMuon(*theCSCGeometry);

  // get Muon CSC information
  edm::Handle<edm::PSimHitContainer> MuonCSCContainer;
  iEvent.getByLabel(MuonCscSrc_,MuonCSCContainer);
  if (!MuonCSCContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find MuonCSCHits in event!";
    return;
  }

  // cycle through container
  int i = 0, j = 0;
  for (itHit = MuonCSCContainer->begin(); itHit != MuonCSCContainer->end(); 
       ++itHit) {

    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dMuon) && 
        (subdetector == sdMuonCSC)) {

      // get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theCSCMuon.idToDetUnit(theDetUnitId);
    
      if (!theDet) {
	edm::LogWarning(MsgLoggerCat)
	  << "Unable to get GeomDetUnit from theCSCMuon for hit " << i;
	continue;
      }
     
      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bSurface = theDet->surface();
    
      // gather necessary information
      //MuonCscToF.push_back(itHit->tof());
      //MuonCscZ.push_back(bSurface.toGlobal(itHit->localPosition()).z());
      //MuonCscPhi.push_back(bSurface.toGlobal(itHit->localPosition()).phi());
      //MuonCscEta.push_back(bSurface.toGlobal(itHit->localPosition()).eta());

      if (meMuonCscToF[0]) meMuonCscToF[0]->Fill(itHit->tof());
      if (meMuonCscToF[1]) meMuonCscToF[1]->Fill(itHit->tof());
      if (meMuonCscZ) 
	meMuonCscZ->Fill(bSurface.toGlobal(itHit->localPosition()).z());
      if (meMuonPhi)
	meMuonPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if (meMuonEta)
	meMuonEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat)
        << "MuonCsc PSimHit " << i 
        << " is expected to be (det,subdet) = (" 
        << dMuon << "," << sdMuonCSC
        << "); value returned is: ("
        << detector << "," << subdetector << ")";
      continue;
    } // end detector type check
  } // end loop through CSC Hits

  if (verbosity > 1) {
    eventout += "\n          Number of CSC muon Hits collected:......... ";
    eventout += j;
  }  

  nMuonHits += j;

  //i = 0, j = 0;
  /////////////////////
  // access the DT Muon
  /////////////////////
  // access the DT Muon geometry
  edm::ESHandle<DTGeometry> theDTGeometry;
  iSetup.get<MuonGeometryRecord>().get(theDTGeometry);
  if (!theDTGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find MuonGeometryRecord for the DTGeometry in event!";
    return;
  }
  const DTGeometry& theDTMuon(*theDTGeometry);

  // get Muon DT information
  edm::Handle<edm::PSimHitContainer> MuonDtContainer;
  iEvent.getByLabel(MuonDtSrc_,MuonDtContainer);
  if (!MuonDtContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find MuonDTHits in event!";
    return;
  }

  // cycle through container
  i = 0, j = 0;
  for (itHit = MuonDtContainer->begin(); itHit != MuonDtContainer->end(); 
       ++itHit) {

    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dMuon) && 
        (subdetector == sdMuonDT)) {

      // CSC uses wires and layers rather than the full detID
      // get the wireId
      DTWireId wireId(itHit->detUnitId());

      // get the DTLayer from the geometry using the wireID
      const DTLayer *theDet = theDTMuon.layer(wireId.layerId());

      if (!theDet) {
	edm::LogWarning(MsgLoggerCat)
	  << "Unable to get GeomDetUnit from theDtMuon for hit " << i;
	continue;
      }
     
      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bSurface = theDet->surface();
    
      // gather necessary information
      //MuonDtToF.push_back(itHit->tof());
      //MuonDtR.push_back(bSurface.toGlobal(itHit->localPosition()).perp());
      //MuonDtPhi.push_back(bSurface.toGlobal(itHit->localPosition()).phi());
      //MuonDtEta.push_back(bSurface.toGlobal(itHit->localPosition()).eta());

      if (meMuonDtToF[0]) meMuonDtToF[0]->Fill(itHit->tof());
      if (meMuonDtToF[1]) meMuonDtToF[1]->Fill(itHit->tof());
      if (meMuonDtR) 
	meMuonDtR->Fill(bSurface.toGlobal(itHit->localPosition()).perp());
      if (meMuonPhi)
	meMuonPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
      if (meMuonEta)
	meMuonEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

    } else {
      edm::LogWarning(MsgLoggerCat)
        << "MuonDt PSimHit " << i 
        << " is expected to be (det,subdet) = (" 
        << dMuon << "," << sdMuonDT
        << "); value returned is: ("
        << detector << "," << subdetector << ")";
      continue;
    } // end detector type check
  } // end loop through DT Hits

  if (verbosity > 1) {
    eventout += "\n          Number of DT muon Hits collected:.......... ";
    eventout += j;
  } 

  nMuonHits += j;

  //i = 0, j = 0;
  //int RPCBrl = 0, RPCFwd = 0;
  ///////////////////////
  // access the RPC Muon
  ///////////////////////
  // access the RPC Muon geometry
  edm::ESHandle<RPCGeometry> theRPCGeometry;
  iSetup.get<MuonGeometryRecord>().get(theRPCGeometry);
  if (!theRPCGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find MuonGeometryRecord for the RPCGeometry in event!";
    return;
  }
  const RPCGeometry& theRPCMuon(*theRPCGeometry);

  // get Muon RPC information
  edm::Handle<edm::PSimHitContainer> MuonRPCContainer;
  iEvent.getByLabel(MuonRpcSrc_,MuonRPCContainer);
  if (!MuonRPCContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find MuonRPCHits in event!";
    return;
  }

  // cycle through container
  i = 0, j = 0;
  int RPCBrl =0, RPCFwd = 0;
  for (itHit = MuonRPCContainer->begin(); itHit != MuonRPCContainer->end(); 
       ++itHit) {

    ++i;

    // create a DetID from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dMuon) && 
        (subdetector == sdMuonRPC)) {
      
      // get an RPCDetID from the detUnitID
      RPCDetId RPCId(itHit->detUnitId());      

      // find the region of the RPC hit
      int region = RPCId.region();

      // get the GeomDetUnit from the geometry using the RPCDetId
      const GeomDetUnit *theDet = theRPCMuon.idToDetUnit(theDetUnitId);

      if (!theDet) {
	edm::LogWarning(MsgLoggerCat)
	  << "Unable to get GeomDetUnit from theRPCMuon for hit " << i;
	continue;
      }

      ++j;

      // get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bSurface = theDet->surface();
    
      // gather necessary information
      if ((region == sdMuonRPCRgnFwdp) || (region == sdMuonRPCRgnFwdn)) {
	++RPCFwd;

	//MuonRpcFwdToF.push_back(itHit->tof());
	//MuonRpcFwdZ.push_back(bSurface.toGlobal(itHit->localPosition()).z());
	//MuonRpcFwdPhi.
	//  push_back(bSurface.toGlobal(itHit->localPosition()).phi());
	//MuonRpcFwdEta.
	//  push_back(bSurface.toGlobal(itHit->localPosition()).eta());

	if (meMuonRpcFToF[0]) meMuonRpcFToF[0]->Fill(itHit->tof());
	if (meMuonRpcFToF[1]) meMuonRpcFToF[1]->Fill(itHit->tof());
	if (meMuonRpcFZ) 
	  meMuonRpcFZ->Fill(bSurface.toGlobal(itHit->localPosition()).z());
	if (meMuonPhi)
	  meMuonPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
	if (meMuonEta)
	  meMuonEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());

      } else if (region == sdMuonRPCRgnBrl) {
	++RPCBrl;

	//MuonRpcBrlToF.push_back(itHit->tof());
	//MuonRpcBrlR.
	//  push_back(bSurface.toGlobal(itHit->localPosition()).perp());
	//MuonRpcBrlPhi.
	//  push_back(bSurface.toGlobal(itHit->localPosition()).phi());
	//MuonRpcBrlEta.
	//  push_back(bSurface.toGlobal(itHit->localPosition()).eta());

	if (meMuonRpcBToF[0]) meMuonRpcBToF[0]->Fill(itHit->tof());
	if (meMuonRpcBToF[1]) meMuonRpcBToF[1]->Fill(itHit->tof());
	if (meMuonRpcBR) 
	  meMuonRpcBR->Fill(bSurface.toGlobal(itHit->localPosition()).perp());
	if (meMuonPhi)
	  meMuonPhi->Fill(bSurface.toGlobal(itHit->localPosition()).phi());
	if (meMuonEta)
	  meMuonEta->Fill(bSurface.toGlobal(itHit->localPosition()).eta());
	
      } else {
	edm::LogWarning(MsgLoggerCat)
	  << "Invalid region for RPC Muon hit" << i;
	continue;
      } // end check of region
    } else {
      edm::LogWarning(MsgLoggerCat)
        << "MuonRpc PSimHit " << i 
        << " is expected to be (det,subdet) = (" 
        << dMuon << "," << sdMuonRPC
        << "); value returned is: ("
        << detector << "," << subdetector << ")";
      continue;
    } // end detector type check
  } // end loop through RPC Hits

  if (verbosity > 1) {
    eventout += "\n          Number of RPC muon Hits collected:......... ";
    eventout += j;
    eventout += "\n                    RPC Barrel muon Hits:............ ";
    eventout += RPCBrl;
    eventout += "\n                    RPC Forward muon Hits:........... ";
    eventout += RPCFwd;    
  }  

  nMuonHits += j;

  if (meMuon[0]) meMuon[0]->Fill((float)nMuonHits);
  if (meMuon[1]) meMuon[1]->Fill((float)nMuonHits); 

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

/*
void GlobalHitsProducer::storeMuon(PGlobalSimHit& product)
{
  std::string MsgLoggerCat = "GlobalHitsProducer_storeMuon";

  if (verbosity > 2) {
    TString eventout("\n       nMuonCSCHits       = ");
    eventout += MuonCscToF.size();
    for (unsigned int i = 0; i < MuonCscToF.size(); ++i) {
      eventout += "\n          (tof,z,phi,eta) = (";
      eventout += MuonCscToF[i];
      eventout += ", ";
      eventout += MuonCscZ[i];
      eventout += ", ";
      eventout += MuonCscPhi[i];
      eventout += ", ";
      eventout += MuonCscEta[i];
      eventout += ")";      
    } // end MuonCsc output
    eventout += "\n       nMuonDtHits        = ";
    eventout += MuonDtToF.size();
    for (unsigned int i = 0; i < MuonDtToF.size(); ++i) {
      eventout += "\n          (tof,r,phi,eta) = (";
      eventout += MuonDtToF[i];
      eventout += ", ";
      eventout += MuonDtR[i];
      eventout += ", ";
      eventout += MuonDtPhi[i];
      eventout += ", ";
      eventout += MuonDtEta[i];
      eventout += ")";      
    } // end MuonDt output
    eventout += "\n       nMuonRpcBrlHits    = ";
    eventout += MuonRpcBrlToF.size();
    for (unsigned int i = 0; i < MuonRpcBrlToF.size(); ++i) {
      eventout += "\n          (tof,r,phi,eta) = (";
      eventout += MuonRpcBrlToF[i];
      eventout += ", ";
      eventout += MuonRpcBrlR[i];
      eventout += ", ";
      eventout += MuonRpcBrlPhi[i];
      eventout += ", ";
      eventout += MuonRpcBrlEta[i];
      eventout += ")";      
    } // end MuonRpcBrl output
    eventout += "\n       nMuonRpcFwdHits    = ";
    eventout += MuonRpcFwdToF.size();
    for (unsigned int i = 0; i < MuonRpcFwdToF.size(); ++i) {
      eventout += "\n          (tof,z,phi,eta) = (";
      eventout += MuonRpcFwdToF[i];
      eventout += ", ";
      eventout += MuonRpcFwdZ[i];
      eventout += ", ";
      eventout += MuonRpcFwdPhi[i];
      eventout += ", ";
      eventout += MuonRpcFwdEta[i]; 
      eventout += ")";      
    } // end MuonRpcFwd output
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  } // end verbose output

  product.putMuonCscHits(MuonCscToF,MuonCscZ,MuonCscPhi,MuonCscEta);
  product.putMuonDtHits(MuonDtToF,MuonDtR,MuonDtPhi,MuonDtEta);
  product.putMuonRpcBrlHits(MuonRpcBrlToF,MuonRpcBrlR,MuonRpcBrlPhi,
			     MuonRpcBrlEta);
  product.putMuonRpcFwdHits(MuonRpcFwdToF,MuonRpcFwdZ,MuonRpcFwdPhi,
			     MuonRpcFwdEta);

  return;
}
*/

void GlobalHitsProducer::fillECal(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalHitsProducer_fillECal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  
  
  // access the calorimeter geometry
  edm::ESHandle<CaloGeometry> theCaloGeometry;
  iSetup.get<IdealGeometryRecord>().get(theCaloGeometry);
  if (!theCaloGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find IdealGeometryRecord in event!";
    return;
  }
  const CaloGeometry& theCalo(*theCaloGeometry);
    
  // iterator to access containers
  edm::PCaloHitContainer::const_iterator itHit;

  ///////////////////////////////
  // get  ECal information
  ///////////////////////////////
  edm::PCaloHitContainer theECalHits;
  // extract EB container
  edm::Handle<edm::PCaloHitContainer> EBContainer;
  iEvent.getByLabel(ECalEBSrc_,EBContainer);
  if (!EBContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find EcalHitsEB in event!";
    return;
  }
  // extract EE container
  edm::Handle<edm::PCaloHitContainer> EEContainer;
  iEvent.getByLabel(ECalEESrc_,EEContainer);
  if (!EEContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find EcalHitsEE in event!";
    return;
  }
  // place both containers into new container
  theECalHits.insert(theECalHits.end(),EBContainer->begin(),
		       EBContainer->end());
  theECalHits.insert(theECalHits.end(),EEContainer->begin(),
		       EEContainer->end());

  // cycle through new container
  int i = 0, j = 0;
  for (itHit = theECalHits.begin(); itHit != theECalHits.end(); ++itHit) {

    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->id());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dEcal) && 
	((subdetector == sdEcalBrl) ||
	 (subdetector == sdEcalFwd))) {

      // get the Cell geometry
      const CaloCellGeometry *theDet = theCalo.
	getSubdetectorGeometry(theDetUnitId)->getGeometry(theDetUnitId);

      if (!theDet) {
	edm::LogWarning(MsgLoggerCat)
	  << "Unable to get CaloCellGeometry from ECalHits for Hit " << i;
	continue;
      }

      ++j;

      // get the global position of the cell
      const GlobalPoint& globalposition = theDet->getPosition();

      // gather necessary information
      //ECalE.push_back(itHit->energy());
      //ECalToF.push_back(itHit->time());
      //ECalPhi.push_back(globalposition.phi());
      //ECalEta.push_back(globalposition.eta());

      if (meCaloEcalE[0]) meCaloEcalE[0]->Fill(itHit->energy());
      if (meCaloEcalE[1]) meCaloEcalE[1]->Fill(itHit->energy());
      if (meCaloEcalToF[0]) meCaloEcalToF[0]->Fill(itHit->time());
      if (meCaloEcalToF[1]) meCaloEcalToF[1]->Fill(itHit->time());
      if (meCaloEcalPhi) meCaloEcalPhi->Fill(globalposition.phi());
      if (meCaloEcalEta) meCaloEcalEta->Fill(globalposition.eta());

    } else {
      edm::LogWarning(MsgLoggerCat)
	<< "ECal PCaloHit " << i 
	<< " is expected to be (det,subdet) = (" 
	<< dEcal << "," << sdEcalBrl
	<< " || " << sdEcalFwd << "); value returned is: ("
	<< detector << "," << subdetector << ")";
      continue;
    } // end detector type check
  } // end loop through ECal Hits

  if (verbosity > 1) {
    eventout += "\n          Number of ECal Hits collected:............. ";
    eventout += j;
  }  

  if (meCaloEcal[0]) meCaloEcal[0]->Fill((float)j);
  if (meCaloEcal[1]) meCaloEcal[1]->Fill((float)j); 

  ////////////////////////////
  // Get Preshower information
  ////////////////////////////
  // extract PreShower container
  edm::Handle<edm::PCaloHitContainer> PreShContainer;
  iEvent.getByLabel(ECalESSrc_,PreShContainer);
  if (!PreShContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find EcalHitsES in event!";
    return;
  }

  // cycle through container
  i = 0, j = 0;
  for (itHit = PreShContainer->begin(); 
       itHit != PreShContainer->end(); ++itHit) {

    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->id());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dEcal) && 
	(subdetector == sdEcalPS)) {

      // get the Cell geometry
      const CaloCellGeometry *theDet = theCalo.
	getSubdetectorGeometry(theDetUnitId)->getGeometry(theDetUnitId);

      if (!theDet) {
	edm::LogWarning(MsgLoggerCat)
	  << "Unable to get CaloCellGeometry from PreShContainer for Hit " 
	  << i;
	continue;
      }

      ++j;

      // get the global position of the cell
      const GlobalPoint& globalposition = theDet->getPosition();

      // gather necessary information
      //PreShE.push_back(itHit->energy());
      //PreShToF.push_back(itHit->time());
      //PreShPhi.push_back(globalposition.phi());
      //PreShEta.push_back(globalposition.eta());

      if (meCaloPreShE[0]) meCaloPreShE[0]->Fill(itHit->energy());
      if (meCaloPreShE[1]) meCaloPreShE[1]->Fill(itHit->energy());
      if (meCaloPreShToF[0]) meCaloPreShToF[0]->Fill(itHit->time());
      if (meCaloPreShToF[1]) meCaloPreShToF[1]->Fill(itHit->time());
      if (meCaloPreShPhi) meCaloPreShPhi->Fill(globalposition.phi());
      if (meCaloPreShEta) meCaloPreShEta->Fill(globalposition.eta());

    } else {
      edm::LogWarning(MsgLoggerCat)
	<< "PreSh PCaloHit " << i 
	<< " is expected to be (det,subdet) = (" 
	<< dEcal << "," << sdEcalPS
	<< "); value returned is: ("
	<< detector << "," << subdetector << ")";
      continue;
    } // end detector type check
  } // end loop through PreShower Hits

  if (verbosity > 1) {
    eventout += "\n          Number of PreSh Hits collected:............ ";
    eventout += j;
  }  

  if (meCaloPreSh[0]) meCaloPreSh[0]->Fill((float)j);
  if (meCaloPreSh[1]) meCaloPreSh[1]->Fill((float)j); 

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

/*
void GlobalHitsProducer::storeECal(PGlobalSimHit& product)
{
  std::string MsgLoggerCat = "GlobalHitsProducer_storeECal";

  if (verbosity > 2) {
    TString eventout("\n       nECalHits          = ");
    eventout += ECalE.size();
    for (unsigned int i = 0; i < ECalE.size(); ++i) {
      eventout += "\n          (e,tof,phi,eta) = (";
      eventout += ECalE[i];
      eventout += ", ";
      eventout += ECalToF[i];
      eventout += ", ";
      eventout += ECalPhi[i];
      eventout += ", ";
      eventout += ECalEta[i];  
      eventout += ")";   
    } // end ECal output
    eventout += "\n       nPreShHits         = ";
    eventout += PreShE.size();
    for (unsigned int i = 0; i < PreShE.size(); ++i) {
      eventout += "\n          (e,tof,phi,eta) = (";
      eventout += PreShE[i];
      eventout += ", ";
      eventout += PreShToF[i];
      eventout += ", ";
      eventout += PreShPhi[i];
      eventout += ", ";
      eventout += PreShEta[i]; 
      eventout += ")";    
    } // end PreShower output
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  } // end verbose output

  product.putECalHits(ECalE,ECalToF,ECalPhi,ECalEta);
  product.putPreShHits(PreShE,PreShToF,PreShPhi,PreShEta);

  return;
}
*/

void GlobalHitsProducer::fillHCal(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalHitsProducer_fillHCal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  
  
  // access the calorimeter geometry
  edm::ESHandle<CaloGeometry> theCaloGeometry;
  iSetup.get<IdealGeometryRecord>().get(theCaloGeometry);
  if (!theCaloGeometry.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find IdealGeometryRecord in event!";
    return;
  }
  const CaloGeometry& theCalo(*theCaloGeometry);
    
  // iterator to access containers
  edm::PCaloHitContainer::const_iterator itHit;

  ///////////////////////////////
  // get  HCal information
  ///////////////////////////////
  // extract HCal container
  edm::Handle<edm::PCaloHitContainer> HCalContainer;
  iEvent.getByLabel(HCalSrc_,HCalContainer);
  if (!HCalContainer.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find HCalHits in event!";
    return;
  }

  // cycle through container
  int i = 0, j = 0;
  for (itHit = HCalContainer->begin(); 
       itHit != HCalContainer->end(); ++itHit) {

    ++i;

    // create a DetId from the detUnitId
    DetId theDetUnitId(itHit->id());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    // check that expected detector is returned
    if ((detector == dHcal) && 
	((subdetector == sdHcalBrl) ||
	 (subdetector == sdHcalEC) ||
	 (subdetector == sdHcalOut) ||
	 (subdetector == sdHcalFwd))) {

      // get the Cell geometry
      const CaloCellGeometry *theDet = theCalo.
	getSubdetectorGeometry(theDetUnitId)->getGeometry(theDetUnitId);

      if (!theDet) {
	edm::LogWarning(MsgLoggerCat)
	  << "Unable to get CaloCellGeometry from HCalContainer for Hit " << i;
	continue;
      }

      ++j;

      // get the global position of the cell
      const GlobalPoint& globalposition = theDet->getPosition();

      // gather necessary information
      //HCalE.push_back(itHit->energy());
      //HCalToF.push_back(itHit->time());
      //HCalPhi.push_back(globalposition.phi());
      //HCalEta.push_back(globalposition.eta());

      if (meCaloHcalE[0]) meCaloHcalE[0]->Fill(itHit->energy());
      if (meCaloHcalE[1]) meCaloHcalE[1]->Fill(itHit->energy());
      if (meCaloHcalToF[0]) meCaloHcalToF[0]->Fill(itHit->time());
      if (meCaloHcalToF[1]) meCaloHcalToF[1]->Fill(itHit->time());
      if (meCaloHcalPhi) meCaloHcalPhi->Fill(globalposition.phi());
      if (meCaloHcalEta) meCaloHcalEta->Fill(globalposition.eta());

    } else {
      edm::LogWarning(MsgLoggerCat)
	<< "HCal PCaloHit " << i 
	<< " is expected to be (det,subdet) = (" 
	<< dHcal << "," << sdHcalBrl
	<< " || " << sdHcalEC << " || " << sdHcalOut << " || " << sdHcalFwd
	<< "); value returned is: ("
	<< detector << "," << subdetector << ")";
      continue;
    } // end detector type check
  } // end loop through HCal Hits

  if (verbosity > 1) {
    eventout += "\n          Number of HCal Hits collected:............. ";
    eventout += j;
  }  

  if (meCaloHcal[0]) meCaloHcal[0]->Fill((float)j);
  if (meCaloHcal[1]) meCaloHcal[1]->Fill((float)j); 

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

/*
void GlobalHitsProducer::storeHCal(PGlobalSimHit& product)
{
  std::string MsgLoggerCat = "GlobalHitsProducer_storeHCal";

  if (verbosity > 2) {
    TString eventout("\n       nHCalHits          = ");
    eventout += HCalE.size();
    for (unsigned int i = 0; i < HCalE.size(); ++i) {
      eventout += "\n          (e,tof,phi,eta) = (";
      eventout += HCalE[i];
      eventout += ", ";
      eventout += HCalToF[i];
      eventout += ", ";
      eventout += HCalPhi[i];
      eventout += ", ";
      eventout += HCalEta[i];  
      eventout += ")";      
    } // end HCal output
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  } // end verbose output

  product.putHCalHits(HCalE,HCalToF,HCalPhi,HCalEta);

  return;
}
*/

/*
void GlobalHitsProducer::clear()
{
  std::string MsgLoggerCat = "GlobalHitsProducer_clear";

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat)
      << "Clearing event holders"; 

  // reset G4MC info
  nRawGenPart = 0;
  G4VtxX.clear();
  G4VtxY.clear();
  G4VtxZ.clear();
  G4TrkPt.clear();
  G4TrkE.clear();

  // reset electromagnetic info
  // reset ECal info
  ECalE.clear();
  ECalToF.clear();
  ECalPhi.clear();
  ECalEta.clear();
  // reset Preshower info
  PreShE.clear();
  PreShToF.clear();
  PreShPhi.clear();
  PreShEta.clear();

  // reset hadronic info
  // reset HCal info
  HCalE.clear();
  HCalToF.clear();
  HCalPhi.clear();
  HCalEta.clear();

  // reset tracker info
  // reset Pixel info
  PxlBrlToF.clear(); 
  PxlBrlR.clear(); 
  PxlBrlPhi.clear(); 
  PxlBrlEta.clear(); 
  PxlFwdToF.clear(); 
  PxlFwdZ.clear();
  PxlFwdPhi.clear(); 
  PxlFwdEta.clear();
  // reset strip info
  SiBrlToF.clear(); 
  SiBrlR.clear(); 
  SiBrlPhi.clear(); 
  SiBrlEta.clear(); 
  SiFwdToF.clear(); 
  SiFwdZ.clear();
  SiFwdPhi.clear(); 
  SiFwdEta.clear();

  // reset muon info
  // reset muon DT info
  MuonDtToF.clear(); 
  MuonDtR.clear();
  MuonDtPhi.clear();
  MuonDtEta.clear();
  // reset muon CSC info
  MuonCscToF.clear(); 
  MuonCscZ.clear();
  MuonCscPhi.clear();
  MuonCscEta.clear();
  // rest muon RPC info
  MuonRpcBrlToF.clear(); 
  MuonRpcBrlR.clear();
  MuonRpcBrlPhi.clear();
  MuonRpcBrlEta.clear(); 
  MuonRpcFwdToF.clear(); 
  MuonRpcFwdZ.clear();
  MuonRpcFwdPhi.clear();
  MuonRpcFwdEta.clear();

  return;
}
*/

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalHitsProducer);
//DEFINE_SEAL_MODULE();
//DEFINE_ANOTHER_FWK_MODULE(GlobalHitsProducer);
