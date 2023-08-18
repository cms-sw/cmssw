/** \file GlobalRecHitsProducer.cc
 *  
 *  See header file for description of class
 *
 *  \author M. Strang SUNY-Buffalo
 */

#include "FWCore/Utilities/interface/Exception.h"
#include "Validation/GlobalRecHits/interface/GlobalRecHitsProducer.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

GlobalRecHitsProducer::GlobalRecHitsProducer(const edm::ParameterSet& iPSet)
    : fName(""),
      verbosity(0),
      frequency(0),
      label(""),
      getAllProvenances(false),
      printProvenanceInfo(false),
      trackerHitAssociatorConfig_(iPSet, consumesCollector()),
      caloGeomToken_(esConsumes()),
      tTopoToken_(esConsumes()),
      tGeomToken_(esConsumes()),
      dtGeomToken_(esConsumes()),
      cscGeomToken_(esConsumes()),
      rpcGeomToken_(esConsumes()),
      count(0) {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_GlobalRecHitsProducer";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  label = iPSet.getParameter<std::string>("Label");
  edm::ParameterSet m_Prov = iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

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

  // fix for consumes
  ECalUncalEBSrc_Token_ = consumes<EBUncalibratedRecHitCollection>(iPSet.getParameter<edm::InputTag>("ECalUncalEBSrc"));
  ECalUncalEESrc_Token_ = consumes<EEUncalibratedRecHitCollection>(iPSet.getParameter<edm::InputTag>("ECalUncalEESrc"));
  ECalEBSrc_Token_ = consumes<EBRecHitCollection>(iPSet.getParameter<edm::InputTag>("ECalEBSrc"));
  ECalEESrc_Token_ = consumes<EERecHitCollection>(iPSet.getParameter<edm::InputTag>("ECalEESrc"));
  ECalESSrc_Token_ = consumes<ESRecHitCollection>(iPSet.getParameter<edm::InputTag>("ECalESSrc"));
  HCalSrc_Token_ = consumes<edm::PCaloHitContainer>(iPSet.getParameter<edm::InputTag>("HCalSrc"));
  SiStripSrc_Token_ = consumes<SiStripMatchedRecHit2DCollection>(iPSet.getParameter<edm::InputTag>("SiStripSrc"));
  SiPxlSrc_Token_ = consumes<SiPixelRecHitCollection>(iPSet.getParameter<edm::InputTag>("SiPxlSrc"));

  MuDTSrc_Token_ = consumes<DTRecHitCollection>(iPSet.getParameter<edm::InputTag>("MuDTSrc"));
  MuDTSimSrc_Token_ = consumes<edm::PSimHitContainer>(iPSet.getParameter<edm::InputTag>("MuDTSimSrc"));

  MuCSCSrc_Token_ = consumes<CSCRecHit2DCollection>(iPSet.getParameter<edm::InputTag>("MuCSCSrc"));
  MuCSCHits_Token_ = consumes<CrossingFrame<PSimHit>>(
      edm::InputTag(std::string("mix"), iPSet.getParameter<std::string>("hitsProducer") + std::string("MuonCSCHits")));

  MuRPCSrc_Token_ = consumes<RPCRecHitCollection>(iPSet.getParameter<edm::InputTag>("MuRPCSrc"));
  MuRPCSimSrc_Token_ = consumes<edm::PSimHitContainer>(iPSet.getParameter<edm::InputTag>("MuRPCSimSrc"));

  EBHits_Token_ = consumes<CrossingFrame<PCaloHit>>(
      edm::InputTag(std::string("mix"), iPSet.getParameter<std::string>("hitsProducer") + std::string("EcalHitsEB")));
  EEHits_Token_ =
      consumes<CrossingFrame<PCaloHit>>(edm::InputTag(std::string("mix"), iPSet.getParameter<std::string>("hitsProduc\
er") + std::string("EcalHitsEE")));

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // create persistent object
  produces<PGlobalRecHit>(label);

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) << "\n===============================\n"
                               << "Initialized as EDProducer with parameter values:\n"
                               << "    Name           = " << fName << "\n"
                               << "    Verbosity      = " << verbosity << "\n"
                               << "    Frequency      = " << frequency << "\n"
                               << "    Label          = " << label << "\n"
                               << "    GetProv        = " << getAllProvenances << "\n"
                               << "    PrintProv      = " << printProvenanceInfo << "\n"
                               << "    ECalEBSrc      = " << ECalEBSrc_.label() << ":" << ECalEBSrc_.instance() << "\n"
                               << "    ECalUncalEBSrc = " << ECalUncalEBSrc_.label() << ":"
                               << ECalUncalEBSrc_.instance() << "\n"
                               << "    ECalEESrc      = " << ECalEESrc_.label() << ":" << ECalUncalEESrc_.instance()
                               << "\n"
                               << "    ECalUncalEESrc = " << ECalUncalEESrc_.label() << ":" << ECalEESrc_.instance()
                               << "\n"
                               << "    ECalESSrc      = " << ECalESSrc_.label() << ":" << ECalESSrc_.instance() << "\n"
                               << "    HCalSrc        = " << HCalSrc_.label() << ":" << HCalSrc_.instance() << "\n"
                               << "    SiStripSrc     = " << SiStripSrc_.label() << ":" << SiStripSrc_.instance()
                               << "\n"
                               << "    SiPixelSrc     = " << SiPxlSrc_.label() << ":" << SiPxlSrc_.instance() << "\n"
                               << "    MuDTSrc        = " << MuDTSrc_.label() << ":" << MuDTSrc_.instance() << "\n"
                               << "    MuDTSimSrc     = " << MuDTSimSrc_.label() << ":" << MuDTSimSrc_.instance()
                               << "\n"
                               << "    MuCSCSrc       = " << MuCSCSrc_.label() << ":" << MuCSCSrc_.instance() << "\n"
                               << "    MuRPCSrc       = " << MuRPCSrc_.label() << ":" << MuRPCSrc_.instance() << "\n"
                               << "    MuRPCSimSrc    = " << MuRPCSimSrc_.label() << ":" << MuRPCSimSrc_.instance()
                               << "\n"
                               << "===============================\n";
  }
}

GlobalRecHitsProducer::~GlobalRecHitsProducer() {}

void GlobalRecHitsProducer::beginJob() {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_beginJob";

  // clear storage vectors
  clear();
  return;
}

void GlobalRecHitsProducer::endJob() {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) << "Terminating having processed " << count << " events.";
  return;
}

void GlobalRecHitsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_produce";

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
  clear();

  // look at information available in the event
  if (getAllProvenances) {
    std::vector<const edm::StableProvenance*> AllProv;
    iEvent.getAllStableProvenance(AllProv);

    if (verbosity >= 0)
      edm::LogInfo(MsgLoggerCat) << "Number of Provenances = " << AllProv.size();

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
    edm::LogInfo(MsgLoggerCat) << "Done gathering data from event.";

  // produce object to put into event
  std::unique_ptr<PGlobalRecHit> pOut(new PGlobalRecHit);

  if (verbosity > 2)
    edm::LogInfo(MsgLoggerCat) << "Saving event contents:";

  // call store functions
  // store ECal information in produce
  storeECal(*pOut);
  // store HCal information in produce
  storeHCal(*pOut);
  // store Track information in produce
  storeTrk(*pOut);
  // store Muon information in produce
  storeMuon(*pOut);

  // store information in event
  iEvent.put(std::move(pOut), label);

  return;
}

void GlobalRecHitsProducer::fillECal(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_fillECal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  // extract crossing frame from event
  //edm::Handle<CrossingFrame> crossingFrame;
  edm::Handle<CrossingFrame<PCaloHit>> crossingFrame;
  //iEvent.getByType(crossingFrame);
  //if (!crossingFrame.isValid()) {
  //  edm::LogWarning(MsgLoggerCat)
  //    << "Unable to find crossingFrame in event!";
  //  return;
  //}

  ////////////////////////
  //extract EB information
  ////////////////////////
  edm::Handle<EBUncalibratedRecHitCollection> EcalUncalibRecHitEB;
  iEvent.getByToken(ECalUncalEBSrc_Token_, EcalUncalibRecHitEB);
  if (!EcalUncalibRecHitEB.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find EcalUncalRecHitEB in event!";
    return;
  }

  edm::Handle<EBRecHitCollection> EcalRecHitEB;
  iEvent.getByToken(ECalEBSrc_Token_, EcalRecHitEB);
  if (!EcalRecHitEB.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find EcalRecHitEB in event!";
    return;
  }

  // loop over simhits
  iEvent.getByToken(EBHits_Token_, crossingFrame);
  if (!crossingFrame.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find cal barrel crossingFrame in event!";
    return;
  }
  //std::unique_ptr<MixCollection<PCaloHit> >
  //  barrelHits(new MixCollection<PCaloHit>
  //	       (crossingFrame.product(), barrelHitsName));
  std::unique_ptr<MixCollection<PCaloHit>> barrelHits(new MixCollection<PCaloHit>(crossingFrame.product()));

  // keep track of sum of simhit energy in each crystal
  MapType ebSimMap;
  for (MixCollection<PCaloHit>::MixItr hitItr = barrelHits->begin(); hitItr != barrelHits->end(); ++hitItr) {
    EBDetId ebid = EBDetId(hitItr->id());

    uint32_t crystid = ebid.rawId();
    ebSimMap[crystid] += hitItr->energy();
  }

  int nEBRecHits = 0;
  // loop over RecHits
  const EBUncalibratedRecHitCollection* EBUncalibRecHit = EcalUncalibRecHitEB.product();
  const EBRecHitCollection* EBRecHit = EcalRecHitEB.product();

  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EBUncalibRecHit->begin();
       uncalibRecHit != EBUncalibRecHit->end();
       ++uncalibRecHit) {
    EBDetId EBid = EBDetId(uncalibRecHit->id());

    EcalRecHitCollection::const_iterator myRecHit = EBRecHit->find(EBid);

    if (myRecHit != EBRecHit->end()) {
      ++nEBRecHits;
      EBRE.push_back(myRecHit->energy());
      EBSHE.push_back(ebSimMap[EBid.rawId()]);
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of EBRecHits collected:............ ";
    eventout += nEBRecHits;
  }

  ////////////////////////
  //extract EE information
  ////////////////////////
  edm::Handle<EEUncalibratedRecHitCollection> EcalUncalibRecHitEE;
  iEvent.getByToken(ECalUncalEESrc_Token_, EcalUncalibRecHitEE);
  if (!EcalUncalibRecHitEE.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find EcalUncalRecHitEE in event!";
    return;
  }

  edm::Handle<EERecHitCollection> EcalRecHitEE;
  iEvent.getByToken(ECalEESrc_Token_, EcalRecHitEE);
  if (!EcalRecHitEE.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find EcalRecHitEE in event!";
    return;
  }

  // loop over simhits
  iEvent.getByToken(EEHits_Token_, crossingFrame);
  if (!crossingFrame.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find cal endcap crossingFrame in event!";
    return;
  }
  //std::unique_ptr<MixCollection<PCaloHit> >
  //  endcapHits(new MixCollection<PCaloHit>
  //	       (crossingFrame.product(), endcapHitsName));
  std::unique_ptr<MixCollection<PCaloHit>> endcapHits(new MixCollection<PCaloHit>(crossingFrame.product()));

  // keep track of sum of simhit energy in each crystal
  MapType eeSimMap;
  for (MixCollection<PCaloHit>::MixItr hitItr = endcapHits->begin(); hitItr != endcapHits->end(); ++hitItr) {
    EEDetId eeid = EEDetId(hitItr->id());

    uint32_t crystid = eeid.rawId();
    eeSimMap[crystid] += hitItr->energy();
  }

  int nEERecHits = 0;
  // loop over RecHits
  const EEUncalibratedRecHitCollection* EEUncalibRecHit = EcalUncalibRecHitEE.product();
  const EERecHitCollection* EERecHit = EcalRecHitEE.product();

  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EEUncalibRecHit->begin();
       uncalibRecHit != EEUncalibRecHit->end();
       ++uncalibRecHit) {
    EEDetId EEid = EEDetId(uncalibRecHit->id());

    EcalRecHitCollection::const_iterator myRecHit = EERecHit->find(EEid);

    if (myRecHit != EERecHit->end()) {
      ++nEERecHits;
      EERE.push_back(myRecHit->energy());
      EESHE.push_back(eeSimMap[EEid.rawId()]);
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of EERecHits collected:............ ";
    eventout += nEERecHits;
  }

  ////////////////////////
  //extract ES information
  ////////////////////////
  edm::Handle<ESRecHitCollection> EcalRecHitES;
  iEvent.getByToken(ECalESSrc_Token_, EcalRecHitES);
  if (!EcalRecHitES.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find EcalRecHitES in event!";
    return;
  }

  // loop over simhits
  iEvent.getByToken(ESHits_Token_, crossingFrame);
  if (!crossingFrame.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find cal preshower crossingFrame in event!";
    return;
  }
  //std::unique_ptr<MixCollection<PCaloHit> >
  //  preshowerHits(new MixCollection<PCaloHit>
  //	       (crossingFrame.product(), preshowerHitsName));
  std::unique_ptr<MixCollection<PCaloHit>> preshowerHits(new MixCollection<PCaloHit>(crossingFrame.product()));

  // keep track of sum of simhit energy in each crystal
  MapType esSimMap;
  for (MixCollection<PCaloHit>::MixItr hitItr = preshowerHits->begin(); hitItr != preshowerHits->end(); ++hitItr) {
    ESDetId esid = ESDetId(hitItr->id());

    uint32_t crystid = esid.rawId();
    esSimMap[crystid] += hitItr->energy();
  }

  int nESRecHits = 0;
  // loop over RecHits
  const ESRecHitCollection* ESRecHit = EcalRecHitES.product();
  for (EcalRecHitCollection::const_iterator recHit = ESRecHit->begin(); recHit != ESRecHit->end(); ++recHit) {
    ESDetId ESid = ESDetId(recHit->id());

    ++nESRecHits;
    ESRE.push_back(recHit->energy());
    ESSHE.push_back(esSimMap[ESid.rawId()]);
  }

  if (verbosity > 1) {
    eventout += "\n          Number of ESRecHits collected:............ ";
    eventout += nESRecHits;
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalRecHitsProducer::storeECal(PGlobalRecHit& product) {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_storeECal";

  if (verbosity > 2) {
    TString eventout("\n         nEBRecHits     = ");
    eventout += EBRE.size();
    for (unsigned int i = 0; i < EBRE.size(); ++i) {
      eventout += "\n      (RE, SHE) = (";
      eventout += EBRE[i];
      eventout += ", ";
      eventout += EBSHE[i];
      eventout += ")";
    }
    eventout += "\n         nEERecHits     = ";
    eventout += EERE.size();
    for (unsigned int i = 0; i < EERE.size(); ++i) {
      eventout += "\n      (RE, SHE) = (";
      eventout += EERE[i];
      eventout += ", ";
      eventout += EESHE[i];
      eventout += ")";
    }
    eventout += "\n         nESRecHits     = ";
    eventout += ESRE.size();
    for (unsigned int i = 0; i < ESRE.size(); ++i) {
      eventout += "\n      (RE, SHE) = (";
      eventout += ESRE[i];
      eventout += ", ";
      eventout += ESSHE[i];
      eventout += ")";
    }
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  }

  product.putEBCalRecHits(EBRE, EBSHE);
  product.putEECalRecHits(EERE, EESHE);
  product.putESCalRecHits(ESRE, ESSHE);

  return;
}

void GlobalRecHitsProducer::fillHCal(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_fillHCal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  // get geometry
  const auto& geometry = iSetup.getHandle(caloGeomToken_);
  if (!geometry.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find CaloGeometry in event!";
    return;
  }

  ///////////////////////
  // extract simhit info
  //////////////////////
  edm::Handle<edm::PCaloHitContainer> hcalHits;
  iEvent.getByToken(HCalSrc_Token_, hcalHits);
  if (!hcalHits.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find hcalHits in event!";
    return;
  }
  const edm::PCaloHitContainer* simhitResult = hcalHits.product();

  MapType fHBEnergySimHits;
  MapType fHEEnergySimHits;
  MapType fHOEnergySimHits;
  MapType fHFEnergySimHits;
  for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin(); simhits != simhitResult->end();
       ++simhits) {
    HcalDetId detId(simhits->id());
    uint32_t cellid = detId.rawId();

    if (detId.subdet() == sdHcalBrl) {
      fHBEnergySimHits[cellid] += simhits->energy();
    }
    if (detId.subdet() == sdHcalEC) {
      fHEEnergySimHits[cellid] += simhits->energy();
    }
    if (detId.subdet() == sdHcalOut) {
      fHOEnergySimHits[cellid] += simhits->energy();
    }
    if (detId.subdet() == sdHcalFwd) {
      fHFEnergySimHits[cellid] += simhits->energy();
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

  Double_t maxHBEta = -1000.;
  Double_t maxHEEta = -1000.;
  Double_t maxHOEta = -1000.;
  Double_t maxHFEta = -1000.;

  Double_t PI = 3.141592653589;

  ////////////////////////
  // get HBHE information
  ///////////////////////
  std::vector<edm::Handle<HBHERecHitCollection>> hbhe;

  //iEvent.getManyByType(hbhe);
  throw cms::Exception("UnsupportedFunction") << "GlobalRecHitsProducer::fillHCal: "
                                              << "getManyByType has not been supported by the Framework since 2015. "
                                              << "This module has been broken since then. Maybe it should be deleted. "
                                              << "Another possibility is to upgrade to use GetterOfProducts instead.";

  if (!hbhe[0].isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find any HBHERecHitCollections in event!";
    return;
  }
  std::vector<edm::Handle<HBHERecHitCollection>>::iterator ihbhe;
  const CaloGeometry* geo = geometry.product();

  int iHB = 0;
  int iHE = 0;
  for (ihbhe = hbhe.begin(); ihbhe != hbhe.end(); ++ihbhe) {
    // find max values
    for (HBHERecHitCollection::const_iterator jhbhe = (*ihbhe)->begin(); jhbhe != (*ihbhe)->end(); ++jhbhe) {
      HcalDetId cell(jhbhe->id());

      if (cell.subdet() == sdHcalBrl) {
        const HcalGeometry* cellGeometry =
            dynamic_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, cell.subdet()));
        double fEta = cellGeometry->getPosition(cell).eta();
        double fPhi = cellGeometry->getPosition(cell).phi();
        if ((jhbhe->energy()) > maxHBEnergy) {
          maxHBEnergy = jhbhe->energy();
          maxHBPhi = fPhi;
          maxHOPhi = maxHBPhi;
          maxHBEta = fEta;
          maxHOEta = maxHBEta;
        }
      }

      if (cell.subdet() == sdHcalEC) {
        const HcalGeometry* cellGeometry =
            dynamic_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, cell.subdet()));
        double fEta = cellGeometry->getPosition(cell).eta();
        double fPhi = cellGeometry->getPosition(cell).phi();
        if ((jhbhe->energy()) > maxHEEnergy) {
          maxHEEnergy = jhbhe->energy();
          maxHEPhi = fPhi;
          maxHEEta = fEta;
        }
      }
    }  // end find max values

    for (HBHERecHitCollection::const_iterator jhbhe = (*ihbhe)->begin(); jhbhe != (*ihbhe)->end(); ++jhbhe) {
      HcalDetId cell(jhbhe->id());

      if (cell.subdet() == sdHcalBrl) {
        ++iHB;

        const HcalGeometry* cellGeometry =
            dynamic_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, cell.subdet()));
        double fEta = cellGeometry->getPosition(cell).eta();
        double fPhi = cellGeometry->getPosition(cell).phi();

        float deltaphi = maxHBPhi - fPhi;
        if (fPhi > maxHBPhi) {
          deltaphi = fPhi - maxHBPhi;
        }
        if (deltaphi > PI) {
          deltaphi = 2.0 * PI - deltaphi;
        }
        float deltaeta = fEta - maxHBEta;
        Double_t r = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);

        HBCalREC.push_back(jhbhe->energy());
        HBCalR.push_back(r);
        HBCalSHE.push_back(fHBEnergySimHits[cell.rawId()]);
      }

      if (cell.subdet() == sdHcalEC) {
        ++iHE;

        const HcalGeometry* cellGeometry =
            dynamic_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, cell.subdet()));
        double fEta = cellGeometry->getPosition(cell).eta();
        double fPhi = cellGeometry->getPosition(cell).phi();

        float deltaphi = maxHEPhi - fPhi;
        if (fPhi > maxHEPhi) {
          deltaphi = fPhi - maxHEPhi;
        }
        if (deltaphi > PI) {
          deltaphi = 2.0 * PI - deltaphi;
        }
        float deltaeta = fEta - maxHEEta;
        Double_t r = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);

        HECalREC.push_back(jhbhe->energy());
        HECalR.push_back(r);
        HECalSHE.push_back(fHEEnergySimHits[cell.rawId()]);
      }
    }
  }  // end loop through collection

  if (verbosity > 1) {
    eventout += "\n          Number of HBRecHits collected:............ ";
    eventout += iHB;
  }

  if (verbosity > 1) {
    eventout += "\n          Number of HERecHits collected:............ ";
    eventout += iHE;
  }

  ////////////////////////
  // get HF information
  ///////////////////////
  std::vector<edm::Handle<HFRecHitCollection>> hf;

  //iEvent.getManyByType(hf);
  throw cms::Exception("UnsupportedFunction") << "GlobalRecHitsProducer::fillHCal: "
                                              << "getManyByType has not been supported by the Framework since 2015. "
                                              << "This module has been broken since then. Maybe it should be deleted. "
                                              << "Another possibility is to upgrade to use GetterOfProducts instead.";

  if (!hf[0].isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find any HFRecHitCollections in event!";
    return;
  }
  std::vector<edm::Handle<HFRecHitCollection>>::iterator ihf;

  int iHF = 0;
  for (ihf = hf.begin(); ihf != hf.end(); ++ihf) {
    // find max values
    for (HFRecHitCollection::const_iterator jhf = (*ihf)->begin(); jhf != (*ihf)->end(); ++jhf) {
      HcalDetId cell(jhf->id());

      if (cell.subdet() == sdHcalFwd) {
        auto cellGeometry = geometry->getSubdetectorGeometry(cell)->getGeometry(cell);
        double fEta = cellGeometry->getPosition().eta();
        double fPhi = cellGeometry->getPosition().phi();
        if ((jhf->energy()) > maxHFEnergy) {
          maxHFEnergy = jhf->energy();
          maxHFPhi = fPhi;
          maxHFEta = fEta;
        }
      }
    }  // end find max values

    for (HFRecHitCollection::const_iterator jhf = (*ihf)->begin(); jhf != (*ihf)->end(); ++jhf) {
      HcalDetId cell(jhf->id());

      if (cell.subdet() == sdHcalFwd) {
        ++iHF;

        auto cellGeometry = geometry->getSubdetectorGeometry(cell)->getGeometry(cell);
        double fEta = cellGeometry->getPosition().eta();
        double fPhi = cellGeometry->getPosition().phi();

        float deltaphi = maxHBPhi - fPhi;
        if (fPhi > maxHFPhi) {
          deltaphi = fPhi - maxHFPhi;
        }
        if (deltaphi > PI) {
          deltaphi = 2.0 * PI - deltaphi;
        }
        float deltaeta = fEta - maxHFEta;
        Double_t r = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);

        HFCalREC.push_back(jhf->energy());
        HFCalR.push_back(r);
        HFCalSHE.push_back(fHFEnergySimHits[cell.rawId()]);
      }
    }
  }  // end loop through collection

  if (verbosity > 1) {
    eventout += "\n          Number of HFDigis collected:.............. ";
    eventout += iHF;
  }

  ////////////////////////
  // get HO information
  ///////////////////////
  std::vector<edm::Handle<HORecHitCollection>> ho;

  //iEvent.getManyByType(ho);
  throw cms::Exception("UnsupportedFunction") << "GlobalRecHitsProducer::fillHCal: "
                                              << "getManyByType has not been supported by the Framework since 2015. "
                                              << "This module has been broken since then. Maybe it should be deleted. "
                                              << "Another possibility is to upgrade to use GetterOfProducts instead.";

  if (!ho[0].isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find any HORecHitCollections in event!";
    return;
  }
  std::vector<edm::Handle<HORecHitCollection>>::iterator iho;

  int iHO = 0;
  for (iho = ho.begin(); iho != ho.end(); ++iho) {
    for (HORecHitCollection::const_iterator jho = (*iho)->begin(); jho != (*iho)->end(); ++jho) {
      HcalDetId cell(jho->id());

      if (cell.subdet() == sdHcalOut) {
        ++iHO;

        auto cellGeometry = geometry->getSubdetectorGeometry(cell)->getGeometry(cell);
        double fEta = cellGeometry->getPosition().eta();
        double fPhi = cellGeometry->getPosition().phi();

        float deltaphi = maxHOPhi - fPhi;
        if (fPhi > maxHOPhi) {
          deltaphi = fPhi - maxHOPhi;
        }
        if (deltaphi > PI) {
          deltaphi = 2.0 * PI - deltaphi;
        }
        float deltaeta = fEta - maxHOEta;
        Double_t r = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);

        HOCalREC.push_back(jho->energy());
        HOCalR.push_back(r);
        HOCalSHE.push_back(fHOEnergySimHits[cell.rawId()]);
      }
    }
  }  // end loop through collection

  if (verbosity > 1) {
    eventout += "\n          Number of HODigis collected:.............. ";
    eventout += iHO;
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalRecHitsProducer::storeHCal(PGlobalRecHit& product) {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_storeHCal";

  if (verbosity > 2) {
    TString eventout("\n         nHBRecHits     = ");
    eventout += HBCalREC.size();
    for (unsigned int i = 0; i < HBCalREC.size(); ++i) {
      eventout += "\n      (REC, R, SHE) = (";
      eventout += HBCalREC[i];
      eventout += ", ";
      eventout += HBCalR[i];
      eventout += ", ";
      eventout += HBCalSHE[i];
      eventout += ")";
    }
    eventout += "\n         nHERecHits     = ";
    eventout += HECalREC.size();
    for (unsigned int i = 0; i < HECalREC.size(); ++i) {
      eventout += "\n      (REC, R, SHE) = (";
      eventout += HECalREC[i];
      eventout += ", ";
      eventout += HECalR[i];
      eventout += ", ";
      eventout += HECalSHE[i];
      eventout += ")";
    }
    eventout += "\n         nHFRecHits     = ";
    eventout += HFCalREC.size();
    for (unsigned int i = 0; i < HFCalREC.size(); ++i) {
      eventout += "\n      (REC, R, SHE) = (";
      eventout += HFCalREC[i];
      eventout += ", ";
      eventout += HFCalR[i];
      eventout += ", ";
      eventout += HFCalSHE[i];
      eventout += ")";
    }
    eventout += "\n         nHORecHits     = ";
    eventout += HOCalREC.size();
    for (unsigned int i = 0; i < HOCalREC.size(); ++i) {
      eventout += "\n      (REC, R, SHE) = (";
      eventout += HOCalREC[i];
      eventout += ", ";
      eventout += HOCalR[i];
      eventout += ", ";
      eventout += HOCalSHE[i];
      eventout += ")";
    }

    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  }

  product.putHBCalRecHits(HBCalREC, HBCalR, HBCalSHE);
  product.putHECalRecHits(HECalREC, HECalR, HECalSHE);
  product.putHOCalRecHits(HOCalREC, HOCalR, HOCalSHE);
  product.putHFCalRecHits(HFCalREC, HFCalR, HFCalSHE);

  return;
}

void GlobalRecHitsProducer::fillTrk(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &iSetup.getData(tTopoToken_);
  ;
  std::string MsgLoggerCat = "GlobalRecHitsProducer_fillTrk";
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  // get strip information
  edm::Handle<SiStripMatchedRecHit2DCollection> rechitsmatched;
  iEvent.getByToken(SiStripSrc_Token_, rechitsmatched);
  if (!rechitsmatched.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find stripmatchedrechits in event!";
    return;
  }

  TrackerHitAssociator associate(iEvent, trackerHitAssociatorConfig_);

  const auto& tGeomHandle = iSetup.getHandle(tGeomToken_);
  if (!tGeomHandle.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find TrackerDigiGeometry in event!";
    return;
  }
  const TrackerGeometry& tracker(*tGeomHandle);

  int nStripBrl = 0, nStripFwd = 0;

  // loop over det units
  for (TrackerGeometry::DetContainer::const_iterator it = tGeomHandle->dets().begin(); it != tGeomHandle->dets().end();
       ++it) {
    uint32_t myid = ((*it)->geographicalId()).rawId();
    DetId detid = ((*it)->geographicalId());

    //loop over rechits-matched in the same subdetector
    SiStripMatchedRecHit2DCollection::const_iterator rechitmatchedMatch = rechitsmatched->find(detid);

    if (rechitmatchedMatch != rechitsmatched->end()) {
      SiStripMatchedRecHit2DCollection::DetSet rechitmatchedRange = *rechitmatchedMatch;
      SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitmatchedRangeIteratorBegin =
          rechitmatchedRange.begin();
      SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitmatchedRangeIteratorEnd = rechitmatchedRange.end();
      SiStripMatchedRecHit2DCollection::DetSet::const_iterator itermatched;

      for (itermatched = rechitmatchedRangeIteratorBegin; itermatched != rechitmatchedRangeIteratorEnd; ++itermatched) {
        SiStripMatchedRecHit2D const rechit = *itermatched;
        LocalPoint position = rechit.localPosition();

        float mindist = 999999.;
        float distx = 999999.;
        float disty = 999999.;
        float dist = 999999.;
        std::pair<LocalPoint, LocalVector> closestPair;
        matched.clear();

        float rechitmatchedx = position.x();
        float rechitmatchedy = position.y();

        matched = associate.associateHit(rechit);

        if (!matched.empty()) {
          //project simhit;
          const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker.idToDet(rechit.geographicalId());
          const StripGeomDetUnit* partnerstripdet = (StripGeomDetUnit*)gluedDet->stereoDet();
          std::pair<LocalPoint, LocalVector> hitPair;

          for (std::vector<PSimHit>::const_iterator m = matched.begin(); m != matched.end(); m++) {
            //project simhit;
            hitPair = projectHit((*m), partnerstripdet, gluedDet->surface());
            distx = fabs(rechitmatchedx - hitPair.first.x());
            disty = fabs(rechitmatchedy - hitPair.first.y());
            dist = sqrt(distx * distx + disty * disty);

            if (dist < mindist) {
              mindist = dist;
              closestPair = hitPair;
            }
          }

          // get TIB
          if (detid.subdetId() == sdSiTIB) {
            ++nStripBrl;

            if (tTopo->tibLayer(myid) == 1) {
              TIBL1RX.push_back(rechitmatchedx);
              TIBL1RY.push_back(rechitmatchedy);
              TIBL1SX.push_back(closestPair.first.x());
              TIBL1SY.push_back(closestPair.first.y());
            }
            if (tTopo->tibLayer(myid) == 2) {
              TIBL2RX.push_back(rechitmatchedx);
              TIBL2RY.push_back(rechitmatchedy);
              TIBL2SX.push_back(closestPair.first.x());
              TIBL2SY.push_back(closestPair.first.y());
            }
            if (tTopo->tibLayer(myid) == 3) {
              TIBL3RX.push_back(rechitmatchedx);
              TIBL3RY.push_back(rechitmatchedy);
              TIBL3SX.push_back(closestPair.first.x());
              TIBL3SY.push_back(closestPair.first.y());
            }
            if (tTopo->tibLayer(myid) == 4) {
              TIBL4RX.push_back(rechitmatchedx);
              TIBL4RY.push_back(rechitmatchedy);
              TIBL4SX.push_back(closestPair.first.x());
              TIBL4SY.push_back(closestPair.first.y());
            }
          }

          // get TOB
          if (detid.subdetId() == sdSiTOB) {
            ++nStripBrl;

            if (tTopo->tobLayer(myid) == 1) {
              TOBL1RX.push_back(rechitmatchedx);
              TOBL1RY.push_back(rechitmatchedy);
              TOBL1SX.push_back(closestPair.first.x());
              TOBL1SY.push_back(closestPair.first.y());
            }
            if (tTopo->tobLayer(myid) == 2) {
              TOBL2RX.push_back(rechitmatchedx);
              TOBL2RY.push_back(rechitmatchedy);
              TOBL2SX.push_back(closestPair.first.x());
              TOBL2SY.push_back(closestPair.first.y());
            }
            if (tTopo->tobLayer(myid) == 3) {
              TOBL3RX.push_back(rechitmatchedx);
              TOBL3RY.push_back(rechitmatchedy);
              TOBL3SX.push_back(closestPair.first.x());
              TOBL3SY.push_back(closestPair.first.y());
            }
            if (tTopo->tobLayer(myid) == 4) {
              TOBL4RX.push_back(rechitmatchedx);
              TOBL4RY.push_back(rechitmatchedy);
              TOBL4SX.push_back(closestPair.first.x());
              TOBL4SY.push_back(closestPair.first.y());
            }
          }

          // get TID
          if (detid.subdetId() == sdSiTID) {
            ++nStripFwd;

            if (tTopo->tidWheel(myid) == 1) {
              TIDW1RX.push_back(rechitmatchedx);
              TIDW1RY.push_back(rechitmatchedy);
              TIDW1SX.push_back(closestPair.first.x());
              TIDW1SY.push_back(closestPair.first.y());
            }
            if (tTopo->tidWheel(myid) == 2) {
              TIDW2RX.push_back(rechitmatchedx);
              TIDW2RY.push_back(rechitmatchedy);
              TIDW2SX.push_back(closestPair.first.x());
              TIDW2SY.push_back(closestPair.first.y());
            }
            if (tTopo->tidWheel(myid) == 3) {
              TIDW3RX.push_back(rechitmatchedx);
              TIDW3RY.push_back(rechitmatchedy);
              TIDW3SX.push_back(closestPair.first.x());
              TIDW3SY.push_back(closestPair.first.y());
            }
          }

          // get TEC
          if (detid.subdetId() == sdSiTEC) {
            ++nStripFwd;

            if (tTopo->tecWheel(myid) == 1) {
              TECW1RX.push_back(rechitmatchedx);
              TECW1RY.push_back(rechitmatchedy);
              TECW1SX.push_back(closestPair.first.x());
              TECW1SY.push_back(closestPair.first.y());
            }
            if (tTopo->tecWheel(myid) == 2) {
              TECW2RX.push_back(rechitmatchedx);
              TECW2RY.push_back(rechitmatchedy);
              TECW2SX.push_back(closestPair.first.x());
              TECW2SY.push_back(closestPair.first.y());
            }
            if (tTopo->tecWheel(myid) == 3) {
              TECW3RX.push_back(rechitmatchedx);
              TECW3RY.push_back(rechitmatchedy);
              TECW3SX.push_back(closestPair.first.x());
              TECW3SY.push_back(closestPair.first.y());
            }
            if (tTopo->tecWheel(myid) == 4) {
              TECW4RX.push_back(rechitmatchedx);
              TECW4RY.push_back(rechitmatchedy);
              TECW4SX.push_back(closestPair.first.x());
              TECW4SY.push_back(closestPair.first.y());
            }
            if (tTopo->tecWheel(myid) == 5) {
              TECW5RX.push_back(rechitmatchedx);
              TECW5RY.push_back(rechitmatchedy);
              TECW5SX.push_back(closestPair.first.x());
              TECW5SY.push_back(closestPair.first.y());
            }
            if (tTopo->tecWheel(myid) == 6) {
              TECW6RX.push_back(rechitmatchedx);
              TECW6RY.push_back(rechitmatchedy);
              TECW6SX.push_back(closestPair.first.x());
              TECW6SY.push_back(closestPair.first.y());
            }
            if (tTopo->tecWheel(myid) == 7) {
              TECW7RX.push_back(rechitmatchedx);
              TECW7RY.push_back(rechitmatchedy);
              TECW7SX.push_back(closestPair.first.x());
              TECW7SY.push_back(closestPair.first.y());
            }
            if (tTopo->tecWheel(myid) == 8) {
              TECW8RX.push_back(rechitmatchedx);
              TECW8RY.push_back(rechitmatchedy);
              TECW8SX.push_back(closestPair.first.x());
              TECW8SY.push_back(closestPair.first.y());
            }
          }

        }  // end if matched empty
      }
    }
  }  // end loop over det units

  if (verbosity > 1) {
    eventout += "\n          Number of BrlStripRecHits collected:...... ";
    eventout += nStripBrl;
  }

  if (verbosity > 1) {
    eventout += "\n          Number of FrwdStripRecHits collected:..... ";
    eventout += nStripFwd;
  }

  // get pixel information
  //Get RecHits
  edm::Handle<SiPixelRecHitCollection> recHitColl;
  iEvent.getByToken(SiPxlSrc_Token_, recHitColl);
  if (!recHitColl.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find SiPixelRecHitCollection in event!";
    return;
  }

  int nPxlBrl = 0, nPxlFwd = 0;
  //iterate over detunits
  for (TrackerGeometry::DetContainer::const_iterator it = tGeomHandle->dets().begin(); it != tGeomHandle->dets().end();
       ++it) {
    uint32_t myid = ((*it)->geographicalId()).rawId();
    DetId detId = ((*it)->geographicalId());
    int subid = detId.subdetId();

    if (!((subid == sdPxlBrl) || (subid == sdPxlFwd)))
      continue;

    //const PixelGeomDetUnit * theGeomDet =
    //  dynamic_cast<const PixelGeomDetUnit*>(theTracker.idToDet(detId) );

    SiPixelRecHitCollection::const_iterator pixeldet = recHitColl->find(detId);
    if (pixeldet == recHitColl->end())
      continue;
    SiPixelRecHitCollection::DetSet pixelrechitRange = *pixeldet;
    SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.begin();
    SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorEnd = pixelrechitRange.end();
    SiPixelRecHitCollection::DetSet::const_iterator pixeliter = pixelrechitRangeIteratorBegin;
    std::vector<PSimHit> matched;

    //----Loop over rechits for this detId
    for (; pixeliter != pixelrechitRangeIteratorEnd; ++pixeliter) {
      matched.clear();
      matched = associate.associateHit(*pixeliter);

      if (!matched.empty()) {
        float closest = 9999.9;
        //std::vector<PSimHit>::const_iterator closestit = matched.begin();
        LocalPoint lp = pixeliter->localPosition();
        float rechit_x = lp.x();
        float rechit_y = lp.y();

        float sim_x = 0.;
        float sim_y = 0.;

        //loop over sim hits and fill closet
        for (std::vector<PSimHit>::const_iterator m = matched.begin(); m != matched.end(); ++m) {
          float sim_x1 = (*m).entryPoint().x();
          float sim_x2 = (*m).exitPoint().x();
          float sim_xpos = 0.5 * (sim_x1 + sim_x2);

          float sim_y1 = (*m).entryPoint().y();
          float sim_y2 = (*m).exitPoint().y();
          float sim_ypos = 0.5 * (sim_y1 + sim_y2);

          float x_res = fabs(sim_xpos - rechit_x);
          float y_res = fabs(sim_ypos - rechit_y);

          float dist = sqrt(x_res * x_res + y_res * y_res);

          if (dist < closest) {
            closest = dist;
            sim_x = sim_xpos;
            sim_y = sim_ypos;
          }
        }  // end sim hit loop

        // get Barrel pixels
        if (subid == sdPxlBrl) {
          ++nPxlBrl;

          if (tTopo->pxbLayer(myid) == 1) {
            BRL1RX.push_back(rechit_x);
            BRL1RY.push_back(rechit_y);
            BRL1SX.push_back(sim_x);
            BRL1SY.push_back(sim_y);
          }
          if (tTopo->pxbLayer(myid) == 2) {
            BRL2RX.push_back(rechit_x);
            BRL2RY.push_back(rechit_y);
            BRL2SX.push_back(sim_x);
            BRL2SY.push_back(sim_y);
          }
          if (tTopo->pxbLayer(myid) == 3) {
            BRL3RX.push_back(rechit_x);
            BRL3RY.push_back(rechit_y);
            BRL3SX.push_back(sim_x);
            BRL3SY.push_back(sim_y);
          }
        }

        // get Forward pixels
        if (subid == sdPxlFwd) {
          ++nPxlFwd;

          if (tTopo->pxfDisk(myid) == 1) {
            if (tTopo->pxfSide(myid) == 1) {
              FWD1nRX.push_back(rechit_x);
              FWD1nRY.push_back(rechit_y);
              FWD1nSX.push_back(sim_x);
              FWD1nSY.push_back(sim_y);
            }
            if (tTopo->pxfSide(myid) == 2) {
              FWD1pRX.push_back(rechit_x);
              FWD1pRY.push_back(rechit_y);
              FWD1pSX.push_back(sim_x);
              FWD1pSY.push_back(sim_y);
            }
          }
          if (tTopo->pxfDisk(myid) == 2) {
            if (tTopo->pxfSide(myid) == 1) {
              FWD2nRX.push_back(rechit_x);
              FWD2nRY.push_back(rechit_y);
              FWD2nSX.push_back(sim_x);
              FWD2nSY.push_back(sim_y);
            }
            if (tTopo->pxfSide(myid) == 2) {
              FWD2pRX.push_back(rechit_x);
              FWD2pRY.push_back(rechit_y);
              FWD2pSX.push_back(sim_x);
              FWD2pSY.push_back(sim_y);
            }
          }
        }
      }  // end matched emtpy
    }    // <-----end rechit loop
  }      // <------ end detunit loop

  if (verbosity > 1) {
    eventout += "\n          Number of BrlPixelRecHits collected:...... ";
    eventout += nPxlBrl;
  }

  if (verbosity > 1) {
    eventout += "\n          Number of FrwdPixelRecHits collected:..... ";
    eventout += nPxlFwd;
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalRecHitsProducer::storeTrk(PGlobalRecHit& product) {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_storeTrk";

  if (verbosity > 2) {
    // strip output
    TString eventout("\n         nTIBL1     = ");
    eventout += TIBL1RX.size();
    for (unsigned int i = 0; i < TIBL1RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TIBL1RX[i];
      eventout += ", ";
      eventout += TIBL1RY[i];
      eventout += ", ";
      eventout += TIBL1SX[i];
      eventout += ", ";
      eventout += TIBL1SY[i];
      eventout += ")";
    }
    eventout += "\n         nTIBL2     = ";
    eventout += TIBL2RX.size();
    for (unsigned int i = 0; i < TIBL2RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TIBL2RX[i];
      eventout += ", ";
      eventout += TIBL2RY[i];
      eventout += ", ";
      eventout += TIBL2SX[i];
      eventout += ", ";
      eventout += TIBL2SY[i];
      eventout += ")";
    }
    eventout += "\n         nTIBL3     = ";
    eventout += TIBL3RX.size();
    for (unsigned int i = 0; i < TIBL3RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TIBL3RX[i];
      eventout += ", ";
      eventout += TIBL3RY[i];
      eventout += ", ";
      eventout += TIBL3SX[i];
      eventout += ", ";
      eventout += TIBL3SY[i];
      eventout += ")";
    }
    eventout += "\n         nTIBL4     = ";
    eventout += TIBL4RX.size();
    for (unsigned int i = 0; i < TIBL4RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TIBL4RX[i];
      eventout += ", ";
      eventout += TIBL4RY[i];
      eventout += ", ";
      eventout += TIBL4SX[i];
      eventout += ", ";
      eventout += TIBL4SY[i];
      eventout += ")";
    }
    eventout += "\n         nTOBL1     = ";
    eventout += TOBL1RX.size();
    for (unsigned int i = 0; i < TOBL1RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TOBL1RX[i];
      eventout += ", ";
      eventout += TOBL1RY[i];
      eventout += ", ";
      eventout += TOBL1SX[i];
      eventout += ", ";
      eventout += TOBL1SY[i];
      eventout += ")";
    }
    eventout += "\n         nTOBL2     = ";
    eventout += TOBL2RX.size();
    for (unsigned int i = 0; i < TOBL2RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TOBL2RX[i];
      eventout += ", ";
      eventout += TOBL2RY[i];
      eventout += ", ";
      eventout += TOBL2SX[i];
      eventout += ", ";
      eventout += TOBL2SY[i];
      eventout += ")";
    }
    eventout += "\n         nTOBL3     = ";
    eventout += TOBL3RX.size();
    for (unsigned int i = 0; i < TOBL3RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TOBL3RX[i];
      eventout += ", ";
      eventout += TOBL3RY[i];
      eventout += ", ";
      eventout += TOBL3SX[i];
      eventout += ", ";
      eventout += TOBL3SY[i];
      eventout += ")";
    }
    eventout += "\n         nTOBL4     = ";
    eventout += TOBL4RX.size();
    for (unsigned int i = 0; i < TOBL4RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TOBL4RX[i];
      eventout += ", ";
      eventout += TOBL4RY[i];
      eventout += ", ";
      eventout += TOBL4SX[i];
      eventout += ", ";
      eventout += TOBL4SY[i];
      eventout += ")";
    }
    eventout += "\n         nTIDW1     = ";
    eventout += TIDW1RX.size();
    for (unsigned int i = 0; i < TIDW1RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TIDW1RX[i];
      eventout += ", ";
      eventout += TIDW1RY[i];
      eventout += ", ";
      eventout += TIDW1SX[i];
      eventout += ", ";
      eventout += TIDW1SY[i];
      eventout += ")";
    }
    eventout += "\n         nTIDW2     = ";
    eventout += TIDW2RX.size();
    for (unsigned int i = 0; i < TIDW2RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TIDW2RX[i];
      eventout += ", ";
      eventout += TIDW2RY[i];
      eventout += ", ";
      eventout += TIDW2SX[i];
      eventout += ", ";
      eventout += TIDW2SY[i];
      eventout += ")";
    }
    eventout += "\n         nTIDW3     = ";
    eventout += TIDW3RX.size();
    for (unsigned int i = 0; i < TIDW3RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TIDW3RX[i];
      eventout += ", ";
      eventout += TIDW3RY[i];
      eventout += ", ";
      eventout += TIDW3SX[i];
      eventout += ", ";
      eventout += TIDW3SY[i];
      eventout += ")";
    }
    eventout += "\n         nTECW1     = ";
    eventout += TECW1RX.size();
    for (unsigned int i = 0; i < TECW1RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TECW1RX[i];
      eventout += ", ";
      eventout += TECW1RY[i];
      eventout += ", ";
      eventout += TECW1SX[i];
      eventout += ", ";
      eventout += TECW1SY[i];
      eventout += ")";
    }
    eventout += "\n         nTECW2     = ";
    eventout += TECW2RX.size();
    for (unsigned int i = 0; i < TECW2RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TECW2RX[i];
      eventout += ", ";
      eventout += TECW2RY[i];
      eventout += ", ";
      eventout += TECW2SX[i];
      eventout += ", ";
      eventout += TECW2SY[i];
      eventout += ")";
    }
    eventout += "\n         nTECW3     = ";
    eventout += TECW3RX.size();
    for (unsigned int i = 0; i < TECW3RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TECW3RX[i];
      eventout += ", ";
      eventout += TECW3RY[i];
      eventout += ", ";
      eventout += TECW3SX[i];
      eventout += ", ";
      eventout += TECW3SY[i];
      eventout += ")";
    }
    eventout += "\n         nTECW4     = ";
    eventout += TECW4RX.size();
    for (unsigned int i = 0; i < TECW4RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TECW4RX[i];
      eventout += ", ";
      eventout += TECW4RY[i];
      eventout += ", ";
      eventout += TECW4SX[i];
      eventout += ", ";
      eventout += TECW4SY[i];
      eventout += ")";
    }
    eventout += "\n         nTECW5     = ";
    eventout += TECW5RX.size();
    for (unsigned int i = 0; i < TECW5RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TECW5RX[i];
      eventout += ", ";
      eventout += TECW5RY[i];
      eventout += ", ";
      eventout += TECW5SX[i];
      eventout += ", ";
      eventout += TECW5SY[i];
      eventout += ")";
    }
    eventout += "\n         nTECW6     = ";
    eventout += TECW6RX.size();
    for (unsigned int i = 0; i < TECW6RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TECW6RX[i];
      eventout += ", ";
      eventout += TECW6RY[i];
      eventout += ", ";
      eventout += TECW6SX[i];
      eventout += ", ";
      eventout += TECW6SY[i];
      eventout += ")";
    }
    eventout += "\n         nTECW7     = ";
    eventout += TECW7RX.size();
    for (unsigned int i = 0; i < TECW7RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TECW7RX[i];
      eventout += ", ";
      eventout += TECW7RY[i];
      eventout += ", ";
      eventout += TECW7SX[i];
      eventout += ", ";
      eventout += TECW7SY[i];
      eventout += ")";
    }
    eventout += "\n         nTECW8     = ";
    eventout += TECW8RX.size();
    for (unsigned int i = 0; i < TECW8RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += TECW8RX[i];
      eventout += ", ";
      eventout += TECW8RY[i];
      eventout += ", ";
      eventout += TECW8SX[i];
      eventout += ", ";
      eventout += TECW8SY[i];
      eventout += ")";
    }

    // pixel output
    eventout += "\n         nBRL1     = ";
    eventout += BRL1RX.size();
    for (unsigned int i = 0; i < BRL1RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += BRL1RX[i];
      eventout += ", ";
      eventout += BRL1RY[i];
      eventout += ", ";
      eventout += BRL1SX[i];
      eventout += ", ";
      eventout += BRL1SY[i];
      eventout += ")";
    }
    eventout += "\n         nBRL2     = ";
    eventout += BRL2RX.size();
    for (unsigned int i = 0; i < BRL2RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += BRL2RX[i];
      eventout += ", ";
      eventout += BRL2RY[i];
      eventout += ", ";
      eventout += BRL2SX[i];
      eventout += ", ";
      eventout += BRL2SY[i];
      eventout += ")";
    }
    eventout += "\n         nBRL3     = ";
    eventout += BRL3RX.size();
    for (unsigned int i = 0; i < BRL3RX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += BRL3RX[i];
      eventout += ", ";
      eventout += BRL3RY[i];
      eventout += ", ";
      eventout += BRL3SX[i];
      eventout += ", ";
      eventout += BRL3SY[i];
      eventout += ")";
    }
    eventout += "\n         nFWD1p     = ";
    eventout += FWD1pRX.size();
    for (unsigned int i = 0; i < FWD1pRX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += FWD1pRX[i];
      eventout += ", ";
      eventout += FWD1pRY[i];
      eventout += ", ";
      eventout += FWD1pSX[i];
      eventout += ", ";
      eventout += FWD1pSY[i];
      eventout += ")";
    }
    eventout += "\n         nFWD1n     = ";
    eventout += FWD1nRX.size();
    for (unsigned int i = 0; i < FWD1nRX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += FWD1nRX[i];
      eventout += ", ";
      eventout += FWD1nRY[i];
      eventout += ", ";
      eventout += FWD1nSX[i];
      eventout += ", ";
      eventout += FWD1nSY[i];
      eventout += ")";
    }
    eventout += "\n         nFWD2p     = ";
    eventout += FWD2pRX.size();
    for (unsigned int i = 0; i < FWD2pRX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += FWD2pRX[i];
      eventout += ", ";
      eventout += FWD2pRY[i];
      eventout += ", ";
      eventout += FWD2pSX[i];
      eventout += ", ";
      eventout += FWD2pSY[i];
      eventout += ")";
    }
    eventout += "\n         nFWD2p     = ";
    eventout += FWD2nRX.size();
    for (unsigned int i = 0; i < FWD2nRX.size(); ++i) {
      eventout += "\n      (RX, RY, SX, SY) = (";
      eventout += FWD2nRX[i];
      eventout += ", ";
      eventout += FWD2nRY[i];
      eventout += ", ";
      eventout += FWD2nSX[i];
      eventout += ", ";
      eventout += FWD2nSY[i];
      eventout += ")";
    }

    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  }

  // strip output
  product.putTIBL1RecHits(TIBL1RX, TIBL1RY, TIBL1SX, TIBL1SY);
  product.putTIBL2RecHits(TIBL2RX, TIBL2RY, TIBL2SX, TIBL2SY);
  product.putTIBL3RecHits(TIBL3RX, TIBL3RY, TIBL3SX, TIBL3SY);
  product.putTIBL4RecHits(TIBL4RX, TIBL4RY, TIBL4SX, TIBL4SY);
  product.putTOBL1RecHits(TOBL1RX, TOBL1RY, TOBL1SX, TOBL1SY);
  product.putTOBL2RecHits(TOBL2RX, TOBL2RY, TOBL2SX, TOBL2SY);
  product.putTOBL3RecHits(TOBL3RX, TOBL3RY, TOBL3SX, TOBL3SY);
  product.putTOBL4RecHits(TOBL4RX, TOBL4RY, TOBL4SX, TOBL4SY);
  product.putTIDW1RecHits(TIDW1RX, TIDW1RY, TIDW1SX, TIDW1SY);
  product.putTIDW2RecHits(TIDW2RX, TIDW2RY, TIDW2SX, TIDW2SY);
  product.putTIDW3RecHits(TIDW3RX, TIDW3RY, TIDW3SX, TIDW3SY);
  product.putTECW1RecHits(TECW1RX, TECW1RY, TECW1SX, TECW1SY);
  product.putTECW2RecHits(TECW2RX, TECW2RY, TECW2SX, TECW2SY);
  product.putTECW3RecHits(TECW3RX, TECW3RY, TECW3SX, TECW3SY);
  product.putTECW4RecHits(TECW4RX, TECW4RY, TECW4SX, TECW4SY);
  product.putTECW5RecHits(TECW5RX, TECW5RY, TECW5SX, TECW5SY);
  product.putTECW6RecHits(TECW6RX, TECW6RY, TECW6SX, TECW6SY);
  product.putTECW7RecHits(TECW7RX, TECW7RY, TECW7SX, TECW7SY);
  product.putTECW8RecHits(TECW8RX, TECW8RY, TECW8SX, TECW8SY);

  // pixel output
  product.putBRL1RecHits(BRL1RX, BRL1RY, BRL1SX, BRL1SY);
  product.putBRL2RecHits(BRL2RX, BRL2RY, BRL2SX, BRL2SY);
  product.putBRL3RecHits(BRL3RX, BRL3RY, BRL3SX, BRL3SY);
  product.putFWD1pRecHits(FWD1pRX, FWD1pRY, FWD1pSX, FWD1pSY);
  product.putFWD1nRecHits(FWD1nRX, FWD1nRY, FWD1nSX, FWD1nSY);
  product.putFWD2pRecHits(FWD2pRX, FWD2pRY, FWD2pSX, FWD2pSY);
  product.putFWD2nRecHits(FWD2nRX, FWD2nRY, FWD2nSX, FWD2nSY);

  return;
}

void GlobalRecHitsProducer::fillMuon(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_fillMuon";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  // get DT information
  const auto& dtGeom = iSetup.getHandle(dtGeomToken_);
  if (!dtGeom.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find DTMuonGeometryRecord in event!";
    return;
  }

  edm::Handle<edm::PSimHitContainer> dtsimHits;
  iEvent.getByToken(MuDTSimSrc_Token_, dtsimHits);
  if (!dtsimHits.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find dtsimHits in event!";
    return;
  }

  std::map<DTWireId, edm::PSimHitContainer> simHitsPerWire =
      DTHitQualityUtils::mapSimHitsPerWire(*(dtsimHits.product()));

  edm::Handle<DTRecHitCollection> dtRecHits;
  iEvent.getByToken(MuDTSrc_Token_, dtRecHits);
  if (!dtRecHits.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find dtRecHits in event!";
    return;
  }

  std::map<DTWireId, std::vector<DTRecHit1DPair>> recHitsPerWire = map1DRecHitsPerWire(dtRecHits.product());

  int nDt = compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, 1);

  if (verbosity > 1) {
    eventout += "\n          Number of DtMuonRecHits collected:........ ";
    eventout += nDt;
  }

  // get CSC Strip information
  // get map of sim hits
  theMap.clear();
  //edm::Handle<CrossingFrame> cf;
  edm::Handle<CrossingFrame<PSimHit>> cf;
  //iEvent.getByType(cf);
  //if (!cf.isValid()) {
  //  edm::LogWarning(MsgLoggerCat)
  //    << "Unable to find CrossingFrame in event!";
  //  return;
  //}
  //MixCollection<PSimHit> simHits(cf.product(), "MuonCSCHits");
  iEvent.getByToken(MuCSCHits_Token_, cf);
  if (!cf.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find muo CSC  crossingFrame in event!";
    return;
  }
  MixCollection<PSimHit> simHits(cf.product());

  // arrange the hits by detUnit
  for (MixCollection<PSimHit>::MixItr hitItr = simHits.begin(); hitItr != simHits.end(); ++hitItr) {
    theMap[hitItr->detUnitId()].push_back(*hitItr);
  }

  // get geometry
  const auto& hGeom = iSetup.getHandle(cscGeomToken_);
  if (!hGeom.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find CSCMuonGeometryRecord in event!";
    return;
  }
  const CSCGeometry* theCSCGeometry = &*hGeom;

  // get rechits
  edm::Handle<CSCRecHit2DCollection> hRecHits;
  iEvent.getByToken(MuCSCSrc_Token_, hRecHits);
  if (!hRecHits.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find CSC RecHits in event!";
    return;
  }
  const CSCRecHit2DCollection* cscRecHits = hRecHits.product();

  int nCSC = 0;
  for (CSCRecHit2DCollection::const_iterator recHitItr = cscRecHits->begin(); recHitItr != cscRecHits->end();
       ++recHitItr) {
    int detId = (*recHitItr).cscDetId().rawId();

    edm::PSimHitContainer simHits;
    std::map<int, edm::PSimHitContainer>::const_iterator mapItr = theMap.find(detId);
    if (mapItr != theMap.end()) {
      simHits = mapItr->second;
    }

    if (simHits.size() == 1) {
      ++nCSC;

      const GeomDetUnit* detUnit = theCSCGeometry->idToDetUnit(CSCDetId(detId));
      const CSCLayer* layer = dynamic_cast<const CSCLayer*>(detUnit);

      int chamberType = layer->chamber()->specs()->chamberType();
      plotResolution(simHits[0], *recHitItr, layer, chamberType);
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of CSCRecHits collected:........... ";
    eventout += nCSC;
  }

  // get RPC information
  std::map<double, int> mapsim, maprec;
  std::map<int, double> nmapsim, nmaprec;

  const auto& rpcGeom = iSetup.getHandle(rpcGeomToken_);
  if (!rpcGeom.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find RPCMuonGeometryRecord in event!";
    return;
  }

  edm::Handle<edm::PSimHitContainer> simHit;
  iEvent.getByToken(MuRPCSimSrc_Token_, simHit);
  if (!simHit.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find RPCSimHit in event!";
    return;
  }

  edm::Handle<RPCRecHitCollection> recHit;
  iEvent.getByToken(MuRPCSrc_Token_, recHit);
  if (!simHit.isValid()) {
    edm::LogWarning(MsgLoggerCat) << "Unable to find RPCRecHit in event!";
    return;
  }

  int nRPC = 0;
  RPCRecHitCollection::const_iterator recIt;
  int nrec = 0;
  for (recIt = recHit->begin(); recIt != recHit->end(); ++recIt) {
    RPCDetId Rid = (RPCDetId)(*recIt).rpcId();
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(rpcGeom->roll(Rid));
    if (roll->isForward()) {
      if (verbosity > 1) {
        eventout += "\n          Number of RPCRecHits collected:........... ";
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
  for (std::map<double, int>::iterator iter = maprec.begin(); iter != maprec.end(); ++iter) {
    i = i + 1;
    nmaprec[i] = (*iter).first;
  }

  edm::PSimHitContainer::const_iterator simIt;
  int nsim = 0;
  for (simIt = simHit->begin(); simIt != simHit->end(); simIt++) {
    int ptype = (*simIt).particleType();
    //RPCDetId Rsid = (RPCDetId)(*simIt).detUnitId();
    if (ptype == 13 || ptype == -13) {
      nsim = nsim + 1;
      LocalPoint shitlocal = (*simIt).localPosition();
      double shitlocalx = shitlocal.x();
      mapsim[shitlocalx] = nsim;
    }
  }

  i = 0;
  for (std::map<double, int>::iterator iter = mapsim.begin(); iter != mapsim.end(); ++iter) {
    i = i + 1;
    nmapsim[i] = (*iter).first;
  }

  if (nsim == nrec) {
    for (int r = 0; r < nsim; r++) {
      ++nRPC;
      RPCRHX.push_back(nmaprec[r + 1]);
      RPCSHX.push_back(nmapsim[r + 1]);
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of RPCRecHits collected:........... ";
    eventout += nRPC;
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalRecHitsProducer::storeMuon(PGlobalRecHit& product) {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_storeMuon";

  if (verbosity > 2) {
    // dt output
    TString eventout("\n         nDT     = ");
    eventout += DTRHD.size();
    for (unsigned int i = 0; i < DTRHD.size(); ++i) {
      eventout += "\n      (RHD, SHD) = (";
      eventout += DTRHD[i];
      eventout += ", ";
      eventout += DTSHD[i];
      eventout += ")";
    }

    // CSC Strip
    eventout += "\n         nCSC     = ";
    eventout += CSCRHPHI.size();
    for (unsigned int i = 0; i < CSCRHPHI.size(); ++i) {
      eventout += "\n      (rhphi, rhperp, shphi) = (";
      eventout += CSCRHPHI[i];
      eventout += ", ";
      eventout += CSCRHPERP[i];
      eventout += ", ";
      eventout += CSCSHPHI[i];
      eventout += ")";
    }

    // RPC
    eventout += "\n         nRPC     = ";
    eventout += RPCRHX.size();
    for (unsigned int i = 0; i < RPCRHX.size(); ++i) {
      eventout += "\n      (rhx, shx) = (";
      eventout += RPCRHX[i];
      eventout += ", ";
      eventout += RPCSHX[i];
      eventout += ")";
    }

    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  }

  product.putDTRecHits(DTRHD, DTSHD);

  product.putCSCRecHits(CSCRHPHI, CSCRHPERP, CSCSHPHI);

  product.putRPCRecHits(RPCRHX, RPCSHX);

  return;
}

void GlobalRecHitsProducer::clear() {
  std::string MsgLoggerCat = "GlobalRecHitsProducer_clear";

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << "Clearing event holders";

  // reset electromagnetic info
  // EE info
  EERE.clear();
  EESHE.clear();
  // EB info
  EBRE.clear();
  EBSHE.clear();
  // ES info
  ESRE.clear();
  ESSHE.clear();

  // reset HCal Info
  HBCalREC.clear();
  HBCalR.clear();
  HBCalSHE.clear();
  HECalREC.clear();
  HECalR.clear();
  HECalSHE.clear();
  HOCalREC.clear();
  HOCalR.clear();
  HOCalSHE.clear();
  HFCalREC.clear();
  HFCalR.clear();
  HFCalSHE.clear();

  // reset Track Info
  TIBL1RX.clear();
  TIBL2RX.clear();
  TIBL3RX.clear();
  TIBL4RX.clear();
  TIBL1RY.clear();
  TIBL2RY.clear();
  TIBL3RY.clear();
  TIBL4RY.clear();
  TIBL1SX.clear();
  TIBL2SX.clear();
  TIBL3SX.clear();
  TIBL4SX.clear();
  TIBL1SY.clear();
  TIBL2SY.clear();
  TIBL3SY.clear();
  TIBL4SY.clear();

  TOBL1RX.clear();
  TOBL2RX.clear();
  TOBL3RX.clear();
  TOBL4RX.clear();
  TOBL1RY.clear();
  TOBL2RY.clear();
  TOBL3RY.clear();
  TOBL4RY.clear();
  TOBL1SX.clear();
  TOBL2SX.clear();
  TOBL3SX.clear();
  TOBL4SX.clear();
  TOBL1SY.clear();
  TOBL2SY.clear();
  TOBL3SY.clear();
  TOBL4SY.clear();

  TIDW1RX.clear();
  TIDW2RX.clear();
  TIDW3RX.clear();
  TIDW1RY.clear();
  TIDW2RY.clear();
  TIDW3RY.clear();
  TIDW1SX.clear();
  TIDW2SX.clear();
  TIDW3SX.clear();
  TIDW1SY.clear();
  TIDW2SY.clear();
  TIDW3SY.clear();

  TECW1RX.clear();
  TECW2RX.clear();
  TECW3RX.clear();
  TECW4RX.clear();
  TECW5RX.clear();
  TECW6RX.clear();
  TECW7RX.clear();
  TECW8RX.clear();
  TECW1RY.clear();
  TECW2RY.clear();
  TECW3RY.clear();
  TECW4RY.clear();
  TECW5RY.clear();
  TECW6RY.clear();
  TECW7RY.clear();
  TECW8RY.clear();
  TECW1SX.clear();
  TECW2SX.clear();
  TECW3SX.clear();
  TECW4SX.clear();
  TECW5SX.clear();
  TECW6SX.clear();
  TECW7SX.clear();
  TECW8SX.clear();
  TECW1SY.clear();
  TECW2SY.clear();
  TECW3SY.clear();
  TECW4SY.clear();
  TECW5SY.clear();
  TECW6SY.clear();
  TECW7SY.clear();
  TECW8SY.clear();

  BRL1RX.clear();
  BRL1RY.clear();
  BRL1SX.clear();
  BRL1SY.clear();
  BRL2RX.clear();
  BRL2RY.clear();
  BRL2SX.clear();
  BRL2SY.clear();
  BRL3RX.clear();
  BRL3RY.clear();
  BRL3SX.clear();
  BRL3SY.clear();

  FWD1pRX.clear();
  FWD1pRY.clear();
  FWD1pSX.clear();
  FWD1pSY.clear();
  FWD1nRX.clear();
  FWD1nRY.clear();
  FWD1nSX.clear();
  FWD1nSY.clear();
  FWD2pRX.clear();
  FWD2pRY.clear();
  FWD2pSX.clear();
  FWD2pSY.clear();
  FWD2nRX.clear();
  FWD2nRY.clear();
  FWD2nSX.clear();
  FWD2nSY.clear();

  //muon clear
  DTRHD.clear();
  DTSHD.clear();

  CSCRHPHI.clear();
  CSCRHPERP.clear();
  CSCSHPHI.clear();

  RPCRHX.clear();
  RPCSHX.clear();

  return;
}

//needed by to do the residual for matched hits in SiStrip
std::pair<LocalPoint, LocalVector> GlobalRecHitsProducer::projectHit(const PSimHit& hit,
                                                                     const StripGeomDetUnit* stripDet,
                                                                     const BoundPlane& plane) {
  const StripTopology& topol = stripDet->specificTopology();
  GlobalPoint globalpos = stripDet->surface().toGlobal(hit.localPosition());
  LocalPoint localHit = plane.toLocal(globalpos);
  //track direction
  LocalVector locdir = hit.localDirection();
  //rotate track in new frame

  GlobalVector globaldir = stripDet->surface().toGlobal(locdir);
  LocalVector dir = plane.toLocal(globaldir);
  float scale = -localHit.z() / dir.z();

  LocalPoint projectedPos = localHit + scale * dir;

  float selfAngle = topol.stripAngle(topol.strip(hit.localPosition()));

  // vector along strip in hit frame
  LocalVector stripDir(sin(selfAngle), cos(selfAngle), 0);

  LocalVector localStripDir(plane.toLocal(stripDet->surface().toGlobal(stripDir)));

  return std::pair<LocalPoint, LocalVector>(projectedPos, localStripDir);
}

// Return a map between DTRecHit1DPair and wireId
std::map<DTWireId, std::vector<DTRecHit1DPair>> GlobalRecHitsProducer::map1DRecHitsPerWire(
    const DTRecHitCollection* dt1DRecHitPairs) {
  std::map<DTWireId, std::vector<DTRecHit1DPair>> ret;

  for (DTRecHitCollection::const_iterator rechit = dt1DRecHitPairs->begin(); rechit != dt1DRecHitPairs->end();
       rechit++) {
    ret[(*rechit).wireId()].push_back(*rechit);
  }

  return ret;
}

// Compute SimHit distance from wire (cm)
float GlobalRecHitsProducer::simHitDistFromWire(const DTLayer* layer, DTWireId wireId, const PSimHit& hit) {
  float xwire = layer->specificTopology().wirePosition(wireId.wire());
  LocalPoint entryP = hit.entryPoint();
  LocalPoint exitP = hit.exitPoint();
  float xEntry = entryP.x() - xwire;
  float xExit = exitP.x() - xwire;

  //FIXME: check...
  return fabs(xEntry - (entryP.z() * (xExit - xEntry)) / (exitP.z() - entryP.z()));
}

// Find the RecHit closest to the muon SimHit
template <typename type>
const type* GlobalRecHitsProducer::findBestRecHit(const DTLayer* layer,
                                                  DTWireId wireId,
                                                  const std::vector<type>& recHits,
                                                  const float simHitDist) {
  float res = 99999;
  const type* theBestRecHit = nullptr;
  // Loop over RecHits within the cell
  for (typename std::vector<type>::const_iterator recHit = recHits.begin(); recHit != recHits.end(); recHit++) {
    float distTmp = recHitDistFromWire(*recHit, layer);
    if (fabs(distTmp - simHitDist) < res) {
      res = fabs(distTmp - simHitDist);
      theBestRecHit = &(*recHit);
    }
  }  // End of loop over RecHits within the cell

  return theBestRecHit;
}

// Compute the distance from wire (cm) of a hits in a DTRecHit1DPair
float GlobalRecHitsProducer::recHitDistFromWire(const DTRecHit1DPair& hitPair, const DTLayer* layer) {
  // Compute the rechit distance from wire
  return fabs(hitPair.localPosition(DTEnums::Left).x() - hitPair.localPosition(DTEnums::Right).x()) / 2.;
}

// Compute the distance from wire (cm) of a hits in a DTRecHit1D
float GlobalRecHitsProducer::recHitDistFromWire(const DTRecHit1D& recHit, const DTLayer* layer) {
  return fabs(recHit.localPosition().x() - layer->specificTopology().wirePosition(recHit.wireId().wire()));
}

template <typename type>
int GlobalRecHitsProducer::compute(const DTGeometry* dtGeom,
                                   const std::map<DTWireId, std::vector<PSimHit>>& _simHitsPerWire,
                                   const std::map<DTWireId, std::vector<type>>& _recHitsPerWire,
                                   int step) {
  std::map<DTWireId, std::vector<PSimHit>> simHitsPerWire = _simHitsPerWire;
  std::map<DTWireId, std::vector<type>> recHitsPerWire = _recHitsPerWire;
  int nDt = 0;
  // Loop over cells with a muon SimHit
  for (std::map<DTWireId, std::vector<PSimHit>>::const_iterator wireAndSHits = simHitsPerWire.begin();
       wireAndSHits != simHitsPerWire.end();
       wireAndSHits++) {
    DTWireId wireId = (*wireAndSHits).first;
    std::vector<PSimHit> simHitsInCell = (*wireAndSHits).second;

    // Get the layer
    const DTLayer* layer = dtGeom->layer(wireId);

    // Look for a mu hit in the cell
    const PSimHit* muSimHit = DTHitQualityUtils::findMuSimHit(simHitsInCell);
    if (muSimHit == nullptr) {
      continue;  // Skip this cell
    }

    // Find the distance of the simhit from the wire
    float simHitWireDist = simHitDistFromWire(layer, wireId, *muSimHit);
    // Skip simhits out of the cell
    if (simHitWireDist > 2.1) {
      continue;  // Skip this cell
    }
    //GlobalPoint simHitGlobalPos = layer->toGlobal(muSimHit->localPosition());

    // Look for RecHits in the same cell
    if (recHitsPerWire.find(wireId) == recHitsPerWire.end()) {
      continue;  // No RecHit found in this cell
    } else {
      // vector<type> recHits = (*wireAndRecHits).second;
      std::vector<type> recHits = recHitsPerWire[wireId];

      // Find the best RecHit
      const type* theBestRecHit = findBestRecHit(layer, wireId, recHits, simHitWireDist);

      float recHitWireDist = recHitDistFromWire(*theBestRecHit, layer);

      ++nDt;

      DTRHD.push_back(recHitWireDist);
      DTSHD.push_back(simHitWireDist);

    }  // find rechits
  }    // loop over simhits

  return nDt;
}

void GlobalRecHitsProducer::plotResolution(const PSimHit& simHit,
                                           const CSCRecHit2D& recHit,
                                           const CSCLayer* layer,
                                           int chamberType) {
  GlobalPoint simHitPos = layer->toGlobal(simHit.localPosition());
  GlobalPoint recHitPos = layer->toGlobal(recHit.localPosition());

  CSCRHPHI.push_back(recHitPos.phi());
  CSCRHPERP.push_back(recHitPos.perp());
  CSCSHPHI.push_back(simHitPos.phi());
}

//define this as a plug-in
//DEFINE_FWK_MODULE(GlobalRecHitsProducer);
