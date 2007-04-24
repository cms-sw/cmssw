#include "Validation/GlobalRecHits/interface/GlobalRecHitsProducer.h"

GlobalRecHitsProducer::GlobalRecHitsProducer(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), frequency(0), label(""), getAllProvenances(false),
  printProvenanceInfo(false), count(0)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_GlobalRecHitsProducer";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  label = iPSet.getParameter<std::string>("Label");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

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
  MuCSCSrc_ = iPSet.getParameter<edm::InputTag>("MuCSCSrc");
  MuRPCSrc_ = iPSet.getParameter<edm::InputTag>("MuRPCSrc");

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // create persistent object
  produces<PGlobalRecHit>(label);

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDProducer with parameter values:\n"
      << "    Name           = " << fName << "\n"
      << "    Verbosity      = " << verbosity << "\n"
      << "    Frequency      = " << frequency << "\n"
      << "    Label          = " << label << "\n"
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
      << "    MuCSCSrc       = " << MuCSCSrc_.label()
      << ":" << MuCSCSrc_.instance() << "\n"
      << "    MuRPCSrc       = " << MuRPCSrc_.label()
      << ":" << MuRPCSrc_.instance() << "\n"
      << "===============================\n";
  }
}

GlobalRecHitsProducer::~GlobalRecHitsProducer() 
{
}

void GlobalRecHitsProducer::beginJob(const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_beginJob";

  // clear storage vectors
  clear();
  return;
}

void GlobalRecHitsProducer::endJob()
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " events.";
  return;
}

void GlobalRecHitsProducer::produce(edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_produce";

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
  clear();

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
	eventout += (AllProv[i]->product).moduleLabel();
	//eventout += AllProv[i]->moduleLabel();
	eventout += "\n       ProductID    : ";
	eventout += (AllProv[i]->product).productID_.id_;
	//eventout += AllProv[i]->productID().id();
	eventout += "\n       ClassName    : ";
	eventout += (AllProv[i]->product).fullClassName_;
	//eventout += AllProv[i]->className();
	eventout += "\n       InstanceName : ";
	eventout += (AllProv[i]->product).productInstanceName_;
	//eventout += AllProv[i]->productInstanceName();
	eventout += "\n       BranchName   : ";
	eventout += (AllProv[i]->product).branchName_;
	//eventout += AllProv[i]->branchName();
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

  // produce object to put into event
  std::auto_ptr<PGlobalRecHit> pOut(new PGlobalRecHit);

  if (verbosity > 2)
    edm::LogInfo (MsgLoggerCat)
      << "Saving event contents:";

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
  iEvent.put(pOut,label);

  return;
}

void GlobalRecHitsProducer::fillECal(edm::Event& iEvent, 
				     const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_fillECal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  

  // extract crossing frame from event
  edm::Handle<CrossingFrame> crossingFrame;
  iEvent.getByType(crossingFrame);
  if (!crossingFrame.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find crossingFrame in event!";
    return;
  }

  ////////////////////////
  //extract EB information
  ////////////////////////
  edm::Handle<EBUncalibratedRecHitCollection> EcalUncalibRecHitEB;
  iEvent.getByLabel(ECalUncalEBSrc_, EcalUncalibRecHitEB);
  if (!EcalUncalibRecHitEB.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find EcalUncalRecHitEB in event!";
    return;
  }  

  edm::Handle<EBRecHitCollection> EcalRecHitEB;
  iEvent.getByLabel(ECalEBSrc_, EcalRecHitEB);
  if (!EcalRecHitEB.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find EcalRecHitEB in event!";
    return;
  }  

  // loop over simhits
  const std::string barrelHitsName("EcalHitsEB");
  std::auto_ptr<MixCollection<PCaloHit> >
    barrelHits(new MixCollection<PCaloHit>
	       (crossingFrame.product(), barrelHitsName));
  
  // keep track of sum of simhit energy in each crystal
  MapType ebSimMap;
  for (MixCollection<PCaloHit>::MixItr hitItr 
	 = barrelHits->begin();
       hitItr != barrelHits->end();
       ++hitItr) {
    
    EBDetId ebid = EBDetId(hitItr->id());
    
    uint32_t crystid = ebid.rawId();
    ebSimMap[crystid] += hitItr->energy();
  }
  
  int nEBRecHits = 0;
  // loop over RecHits
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
  iEvent.getByLabel(ECalUncalEESrc_, EcalUncalibRecHitEE);
  if (!EcalUncalibRecHitEE.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find EcalUncalRecHitEE in event!";
    return;
  }  

  edm::Handle<EERecHitCollection> EcalRecHitEE;
  iEvent.getByLabel(ECalEESrc_, EcalRecHitEE);
  if (!EcalRecHitEE.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find EcalRecHitEE in event!";
    return;
  }  

  // loop over simhits
  const std::string endcapHitsName("EcalHitsEE");
  std::auto_ptr<MixCollection<PCaloHit> >
    endcapHits(new MixCollection<PCaloHit>
	       (crossingFrame.product(), endcapHitsName));
  
  // keep track of sum of simhit energy in each crystal
  MapType eeSimMap;
  for (MixCollection<PCaloHit>::MixItr hitItr 
	 = endcapHits->begin();
       hitItr != endcapHits->end();
       ++hitItr) {
    
    EEDetId eeid = EEDetId(hitItr->id());
    
    uint32_t crystid = eeid.rawId();
    eeSimMap[crystid] += hitItr->energy();
  }
  
  int nEERecHits = 0;
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
  iEvent.getByLabel(ECalESSrc_, EcalRecHitES);
  if (!EcalRecHitES.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find EcalRecHitES in event!";
    return;
  }  

  // loop over simhits
  const std::string preshowerHitsName("EcalHitsES");
  std::auto_ptr<MixCollection<PCaloHit> >
    preshowerHits(new MixCollection<PCaloHit>
	       (crossingFrame.product(), preshowerHitsName));
  
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
  
  int nESRecHits = 0;
  // loop over RecHits
  const ESRecHitCollection *ESRecHit = EcalRecHitES.product();
  for (EcalRecHitCollection::const_iterator recHit =
	 ESRecHit->begin();
       recHit != ESRecHit->end();
       ++recHit) {

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

void GlobalRecHitsProducer::storeECal(PGlobalRecHit& product)
{
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

  product.putEBCalRecHits(EBRE,EBSHE);
  product.putEECalRecHits(EBRE,EBSHE);
  product.putESCalRecHits(EBRE,EBSHE);

  return;
}

void GlobalRecHitsProducer::fillHCal(edm::Event& iEvent, 
				   const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_fillHCal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  

  // get geometry
  edm::ESHandle<CaloGeometry> geometry;
  iSetup.get<IdealGeometryRecord>().get(geometry);
  if (!geometry.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find CaloGeometry in event!";
    return;
  }

  ///////////////////////
  // extract simhit info
  //////////////////////
  edm::Handle<edm::PCaloHitContainer> hcalHits;
  iEvent.getByLabel(HCalSrc_,hcalHits);
  if (!hcalHits.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find hcalHits in event!";
    return;
  }  
  const edm::PCaloHitContainer *simhitResult = hcalHits.product();
  
  MapType fHBEnergySimHits;
  MapType fHEEnergySimHits;
  MapType fHOEnergySimHits;
  MapType fHFEnergySimHits;
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

  // max values to be used (HO is found in HB)
  Double_t maxHBEnergy = 0.;
  Double_t maxHEEnergy = 0.;
  Double_t maxHOEnergy = 0.;
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
  std::vector<edm::Handle<HBHERecHitCollection> > hbhe;
  iEvent.getManyByType(hbhe);
  if (!hbhe[0].isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find any HBHERecHitCollections in event!";
    return;
  } 
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
	double fEta = cellGeometry->getPosition().eta () ;
	double fPhi = cellGeometry->getPosition().phi () ;
	if ( (jhbhe->energy()) > maxHBEnergy ) {
	  maxHBEnergy = jhbhe->energy();
	  maxHOEnergy = maxHBEnergy;
	  maxHBPhi = fPhi;
	  maxHOPhi = maxHBPhi;
	  maxHBEta = fEta;
	  maxHOEta = maxHBEta;
	}	  
      }
	
      if (cell.subdet() == sdHcalEC) {
	
	const CaloCellGeometry* cellGeometry =
	  geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	double fEta = cellGeometry->getPosition().eta () ;
	double fPhi = cellGeometry->getPosition().phi () ;
	if ( (jhbhe->energy()) > maxHEEnergy ) {
	  maxHEEnergy = jhbhe->energy();
	  maxHEPhi = fPhi;
	  maxHEEta = fEta;
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
	double fEta = cellGeometry->getPosition().eta () ;
	double fPhi = cellGeometry->getPosition().phi () ;

	float deltaphi = maxHBPhi - fPhi;
	if (fPhi > maxHBPhi) { deltaphi = fPhi - maxHBPhi;}
	if (deltaphi > PI) { deltaphi = 2.0 * PI - deltaphi;}
	float deltaeta = fEta - maxHBEta;
	Double_t r = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);

	HBCalREC.push_back(jhbhe->energy());
	HBCalR.push_back(r);
	HBCalSHE.push_back(fHBEnergySimHits[cell.rawId()]);
      }

      if (cell.subdet() == sdHcalEC) {

	++iHE;

	const CaloCellGeometry* cellGeometry =
	  geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	double fEta = cellGeometry->getPosition().eta () ;
	double fPhi = cellGeometry->getPosition().phi () ;

	float deltaphi = maxHEPhi - fPhi;
	if (fPhi > maxHEPhi) { deltaphi = fPhi - maxHEPhi;}
	if (deltaphi > PI) { deltaphi = 2.0 * PI - deltaphi;}
	float deltaeta = fEta - maxHEEta;
	Double_t r = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);

	HECalREC.push_back(jhbhe->energy());
	HECalR.push_back(r);
	HECalSHE.push_back(fHEEnergySimHits[cell.rawId()]);
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

  ////////////////////////
  // get HF information
  ///////////////////////
  std::vector<edm::Handle<HFRecHitCollection> > hf;
  iEvent.getManyByType(hf);
  if (!hf[0].isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find any HFRecHitCollections in event!";
    return;
  } 
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
	double fEta = cellGeometry->getPosition().eta () ;
	double fPhi = cellGeometry->getPosition().phi () ;
	if ( (jhf->energy()) > maxHFEnergy ) {
	  maxHFEnergy = jhf->energy();
	  maxHFPhi = fPhi;
	  maxHFEta = fEta;
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
	double fEta = cellGeometry->getPosition().eta () ;
	double fPhi = cellGeometry->getPosition().phi () ;

	float deltaphi = maxHBPhi - fPhi;
	if (fPhi > maxHFPhi) { deltaphi = fPhi - maxHFPhi;}
	if (deltaphi > PI) { deltaphi = 2.0 * PI - deltaphi;}
	float deltaeta = fEta - maxHFEta;
	Double_t r = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);

	HFCalREC.push_back(jhf->energy());
	HFCalR.push_back(r);
	HFCalSHE.push_back(fHFEnergySimHits[cell.rawId()]);
      }
    }
  } // end loop through collection

  if (verbosity > 1) {
    eventout += "\n          Number of HFDigis collected:.............. ";
    eventout += iHF;
  }

  ////////////////////////
  // get HO information
  ///////////////////////
  std::vector<edm::Handle<HORecHitCollection> > ho;
  iEvent.getManyByType(ho);
  if (!ho[0].isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find any HORecHitCollections in event!";
    return;
  } 
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
	double fEta = cellGeometry->getPosition().eta () ;
	double fPhi = cellGeometry->getPosition().phi () ;

	float deltaphi = maxHOPhi - fPhi;
	if (fPhi > maxHOPhi) { deltaphi = fPhi - maxHOPhi;}
	if (deltaphi > PI) { deltaphi = 2.0 * PI - deltaphi;}
	float deltaeta = fEta - maxHOEta;
	Double_t r = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);

	HOCalREC.push_back(jho->energy());
	HOCalR.push_back(r);
	HOCalSHE.push_back(fHOEnergySimHits[cell.rawId()]);
      }
    }
  } // end loop through collection

  if (verbosity > 1) {
    eventout += "\n          Number of HODigis collected:.............. ";
    eventout += iHO;
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalRecHitsProducer::storeHCal(PGlobalRecHit& product)
{
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

  product.putHBCalRecHits(HBCalREC,HBCalR,HBCalSHE);
  product.putHECalRecHits(HECalREC,HECalR,HECalSHE);
  product.putHOCalRecHits(HOCalREC,HOCalR,HOCalSHE);
  product.putHFCalRecHits(HFCalREC,HFCalR,HFCalSHE);

  return;
}

void GlobalRecHitsProducer::fillTrk(edm::Event& iEvent, 
				   const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_fillTrk";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  

  /*
  // get strip information
  edm::Handle<edm::DetSetVector<SiStripDigi> > stripDigis;  
  iEvent.getByLabel(SiStripSrc_, stripDigis);
  if (!stripDigis.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find stripDigis in event!";
    return;
  }  

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
      TIBDetId tibid(id);
      for (iter = begin; iter != end; ++iter) {
	++nStripBrl;
	if (tibid.layer() == 1) {
	  TIBL1ADC.push_back((*iter).adc());
	  TIBL1Strip.push_back((*iter).strip());
	}
	if (tibid.layer() == 2) {
	  TIBL2ADC.push_back((*iter).adc());
	  TIBL2Strip.push_back((*iter).strip());
	}	
	if (tibid.layer() == 3) {
	  TIBL3ADC.push_back((*iter).adc());
	  TIBL3Strip.push_back((*iter).strip());
	}
	if (tibid.layer() == 4) {
	  TIBL4ADC.push_back((*iter).adc());
	  TIBL4Strip.push_back((*iter).strip());
	}
      }
    }
    
    // get TOB
    if (detId.subdetId() == sdSiTOB) {
      TOBDetId tobid(id);
      for (iter = begin; iter != end; ++iter) {
	++nStripBrl;
	if (tobid.layer() == 1) {
	  TOBL1ADC.push_back((*iter).adc());
	  TOBL1Strip.push_back((*iter).strip());
	}
	if (tobid.layer() == 2) {
	  TOBL2ADC.push_back((*iter).adc());
	  TOBL2Strip.push_back((*iter).strip());
	}	
	if (tobid.layer() == 3) {
	  TOBL3ADC.push_back((*iter).adc());
	  TOBL3Strip.push_back((*iter).strip());
	}
	if (tobid.layer() == 4) {
	  TOBL4ADC.push_back((*iter).adc());
	  TOBL4Strip.push_back((*iter).strip());
	}
      }
    }    
    
    // get TID
    if (detId.subdetId() == sdSiTID) {
      TIDDetId tidid(id);
      for (iter = begin; iter != end; ++iter) {
	++nStripFwd;
	if (tidid.wheel() == 1) {
	  TIDW1ADC.push_back((*iter).adc());
	  TIDW1Strip.push_back((*iter).strip());
	}
	if (tidid.wheel() == 2) {
	  TIDW2ADC.push_back((*iter).adc());
	  TIDW2Strip.push_back((*iter).strip());
	}
	if (tidid.wheel() == 3) {
	  TIDW3ADC.push_back((*iter).adc());
	  TIDW3Strip.push_back((*iter).strip());
	}
      }
    }   

    // get TEC
    if (detId.subdetId() == sdSiTEC) {
      TECDetId tecid(id);
      for (iter = begin; iter != end; ++iter) {
	++nStripFwd;
	if (tecid.wheel() == 1) {
	  TECW1ADC.push_back((*iter).adc());
	  TECW1Strip.push_back((*iter).strip());
	}
	if (tecid.wheel() == 2) {
	  TECW2ADC.push_back((*iter).adc());
	  TECW2Strip.push_back((*iter).strip());
	}
	if (tecid.wheel() == 3) {
	  TECW3ADC.push_back((*iter).adc());
	  TECW3Strip.push_back((*iter).strip());
	}
	if (tecid.wheel() == 4) {
	  TECW4ADC.push_back((*iter).adc());
	  TECW4Strip.push_back((*iter).strip());
	}
	if (tecid.wheel() == 5) {
	  TECW5ADC.push_back((*iter).adc());
	  TECW5Strip.push_back((*iter).strip());
	}
	if (tecid.wheel() == 6) {
	  TECW6ADC.push_back((*iter).adc());
	  TECW6Strip.push_back((*iter).strip());
	}
	if (tecid.wheel() == 7) {
	  TECW7ADC.push_back((*iter).adc());
	  TECW7Strip.push_back((*iter).strip());
	}
	if (tecid.wheel() == 8) {
	  TECW8ADC.push_back((*iter).adc());
	  TECW8Strip.push_back((*iter).strip());
	}
      }
    }     
  } // end loop over DataSetVector

  if (verbosity > 1) {
    eventout += "\n          Number of BrlStripDigis collected:........ ";
    eventout += nStripBrl;
  }

  if (verbosity > 1) {
    eventout += "\n          Number of FrwdStripDigis collected:....... ";
    eventout += nStripFwd;
  }

  // get pixel information
  edm::Handle<edm::DetSetVector<PixelDigi> > pixelDigis;  
  iEvent.getByLabel(SiPxlSrc_, pixelDigis);
  if (!pixelDigis.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find pixelDigis in event!";
    return;
  }  

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
      PXBDetId bdetid(id);
      for (iter = begin; iter != end; ++iter) {
	++nPxlBrl;
	if (bdetid.layer() == 1) {
	  BRL1ADC.push_back((*iter).adc());
	  BRL1Row.push_back((*iter).row());
	  BRL1Col.push_back((*iter).column());	  
	}
	if (bdetid.layer() == 2) {
	  BRL2ADC.push_back((*iter).adc());
	  BRL2Row.push_back((*iter).row());
	  BRL2Col.push_back((*iter).column());	  
	}
	if (bdetid.layer() == 3) {
	  BRL3ADC.push_back((*iter).adc());
	  BRL3Row.push_back((*iter).row());
	  BRL3Col.push_back((*iter).column());	  
	}
      }
    }

    // get Forward pixels
    if (detId.subdetId() == sdPxlFwd) {
      PXFDetId fdetid(id);
      for (iter = begin; iter != end; ++iter) {
	++nPxlFwd;
	if (fdetid.disk() == 1) {
	  if (fdetid.side() == 1) {
	    FWD1nADC.push_back((*iter).adc());
	    FWD1nRow.push_back((*iter).row());
	    FWD1nCol.push_back((*iter).column());
	  }
	  if (fdetid.side() == 2) {
	    FWD1pADC.push_back((*iter).adc());
	    FWD1pRow.push_back((*iter).row());
	    FWD1pCol.push_back((*iter).column());
	  }
	}
	if (fdetid.disk() == 2) {
	  if (fdetid.side() == 1) {
	    FWD2nADC.push_back((*iter).adc());
	    FWD2nRow.push_back((*iter).row());
	    FWD2nCol.push_back((*iter).column());
	  }
	  if (fdetid.side() == 2) {
	    FWD2pADC.push_back((*iter).adc());
	    FWD2pRow.push_back((*iter).row());
	    FWD2pCol.push_back((*iter).column());
	  }
	}
      }
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of BrlPixelDigis collected:........ ";
    eventout += nPxlBrl;
  }

  if (verbosity > 1) {
    eventout += "\n          Number of FrwdPixelDigis collected:....... ";
    eventout += nPxlFwd;
  }
  */

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalRecHitsProducer::storeTrk(PGlobalRecHit& product)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_storeTrk";

  /*
  if (verbosity > 2) {

    // strip output
    TString eventout("\n         nTIBL1     = ");
    eventout += TIBL1ADC.size();
    for (unsigned int i = 0; i < TIBL1ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TIBL1ADC[i];
      eventout += ", ";
      eventout += TIBL1Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTIBL2     = ";
    eventout += TIBL2ADC.size();
    for (unsigned int i = 0; i < TIBL2ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TIBL2ADC[i];
      eventout += ", ";
      eventout += TIBL2Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTIBL3     = ";
    eventout += TIBL3ADC.size();
    for (unsigned int i = 0; i < TIBL3ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TIBL3ADC[i];
      eventout += ", ";
      eventout += TIBL3Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTIBL4     = ";
    eventout += TIBL4ADC.size();
    for (unsigned int i = 0; i < TIBL4ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TIBL4ADC[i];
      eventout += ", ";
      eventout += TIBL4Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTOBL1     = ";
    eventout += TOBL1ADC.size();
    for (unsigned int i = 0; i < TOBL1ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TOBL1ADC[i];
      eventout += ", ";
      eventout += TOBL1Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTOBL2     = ";
    eventout += TOBL2ADC.size();
    for (unsigned int i = 0; i < TOBL2ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TOBL2ADC[i];
      eventout += ", ";
      eventout += TOBL2Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTOBL3     = ";
    eventout += TOBL3ADC.size();
    for (unsigned int i = 0; i < TOBL3ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TOBL3ADC[i];
      eventout += ", ";
      eventout += TOBL3Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTOBL4     = ";
    eventout += TOBL4ADC.size();
    for (unsigned int i = 0; i < TOBL4ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TOBL4ADC[i];
      eventout += ", ";
      eventout += TOBL4Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTIDW1     = ";
    eventout += TIDW1ADC.size();
    for (unsigned int i = 0; i < TIDW1ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TIDW1ADC[i];
      eventout += ", ";
      eventout += TIDW1Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTIDW2     = ";
    eventout += TIDW2ADC.size();
    for (unsigned int i = 0; i < TIDW2ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TIDW2ADC[i];
      eventout += ", ";
      eventout += TIDW2Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTIDW3     = ";
    eventout += TIDW3ADC.size();
    for (unsigned int i = 0; i < TIDW3ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TIDW3ADC[i];
      eventout += ", ";
      eventout += TIDW3Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTECW1     = ";
    eventout += TECW1ADC.size();
    for (unsigned int i = 0; i < TECW1ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TECW1ADC[i];
      eventout += ", ";
      eventout += TECW1Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTECW2     = ";
    eventout += TECW2ADC.size();
    for (unsigned int i = 0; i < TECW2ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TECW2ADC[i];
      eventout += ", ";
      eventout += TECW2Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTECW3     = ";
    eventout += TECW3ADC.size();
    for (unsigned int i = 0; i < TECW3ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TECW3ADC[i];
      eventout += ", ";
      eventout += TECW3Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTECW4     = ";
    eventout += TECW4ADC.size();
    for (unsigned int i = 0; i < TECW4ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TECW4ADC[i];
      eventout += ", ";
      eventout += TECW4Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTECW5     = ";
    eventout += TECW5ADC.size();
    for (unsigned int i = 0; i < TECW5ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TECW5ADC[i];
      eventout += ", ";
      eventout += TECW5Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTECW6     = ";
    eventout += TECW6ADC.size();
    for (unsigned int i = 0; i < TECW6ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TECW6ADC[i];
      eventout += ", ";
      eventout += TECW6Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTECW7     = ";
    eventout += TECW7ADC.size();
    for (unsigned int i = 0; i < TECW7ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TECW7ADC[i];
      eventout += ", ";
      eventout += TECW7Strip[i];
      eventout += ")";
    }
    eventout += "\n         nTECW8     = ";
    eventout += TECW8ADC.size();
    for (unsigned int i = 0; i < TECW8ADC.size(); ++i) {
      eventout += "\n      (ADC, strip) = (";
      eventout += TECW8ADC[i];
      eventout += ", ";
      eventout += TECW8Strip[i];
      eventout += ")";
    }

    // pixel output
    eventout += "\n         nBRL1     = ";
    eventout += BRL1ADC.size();
    for (unsigned int i = 0; i < BRL1ADC.size(); ++i) {
      eventout += "\n      (ADC, row, column) = (";
      eventout += BRL1ADC[i];
      eventout += ", ";
      eventout += BRL1Row[i];
      eventout += ", ";
      eventout += BRL1Col[i];
      eventout += ")";
    } 
    eventout += "\n         nBRL2     = ";
    eventout += BRL2ADC.size();
    for (unsigned int i = 0; i < BRL2ADC.size(); ++i) {
      eventout += "\n      (ADC, row, column) = (";
      eventout += BRL2ADC[i];
      eventout += ", ";
      eventout += BRL2Row[i];
      eventout += ", ";
      eventout += BRL2Col[i];
      eventout += ")";
    } 
    eventout += "\n         nBRL3     = ";
    eventout += BRL3ADC.size();
    for (unsigned int i = 0; i < BRL3ADC.size(); ++i) {
      eventout += "\n      (ADC, row, column) = (";
      eventout += BRL3ADC[i];
      eventout += ", ";
      eventout += BRL3Row[i];
      eventout += ", ";
      eventout += BRL3Col[i];
      eventout += ")";
    }    
    eventout += "\n         nFWD1p     = ";
    eventout += FWD1pADC.size();
    for (unsigned int i = 0; i < FWD1pADC.size(); ++i) {
      eventout += "\n      (ADC, row, column) = (";
      eventout += FWD1pADC[i];
      eventout += ", ";
      eventout += FWD1pRow[i];
      eventout += ", ";
      eventout += FWD1pCol[i];
      eventout += ")";
    } 
    eventout += "\n         nFWD1p     = ";
    eventout += FWD1nADC.size();
    for (unsigned int i = 0; i < FWD1nADC.size(); ++i) {
      eventout += "\n      (ADC, row, column) = (";
      eventout += FWD1nADC[i];
      eventout += ", ";
      eventout += FWD1nRow[i];
      eventout += ", ";
      eventout += FWD1nCol[i];
      eventout += ")";
    } 
    eventout += "\n         nFWD1p     = ";
    eventout += FWD2pADC.size();
    for (unsigned int i = 0; i < FWD2pADC.size(); ++i) {
      eventout += "\n      (ADC, row, column) = (";
      eventout += FWD2pADC[i];
      eventout += ", ";
      eventout += FWD2pRow[i];
      eventout += ", ";
      eventout += FWD2pCol[i];
      eventout += ")";
    } 
    eventout += "\n         nFWD2p     = ";
    eventout += FWD2nADC.size();
    for (unsigned int i = 0; i < FWD2nADC.size(); ++i) {
      eventout += "\n      (ADC, row, column) = (";
      eventout += FWD2nADC[i];
      eventout += ", ";
      eventout += FWD2nRow[i];
      eventout += ", ";
      eventout += FWD2nCol[i];
      eventout += ")";
    } 

    edm::LogInfo(MsgLoggerCat) << eventout << "\n";  
  }

  // strip output
  product.putTIBL1Digis(TIBL1ADC,TIBL1Strip);
  product.putTIBL2Digis(TIBL2ADC,TIBL2Strip);
  product.putTIBL3Digis(TIBL3ADC,TIBL3Strip);
  product.putTIBL4Digis(TIBL4ADC,TIBL4Strip);
  product.putTOBL1Digis(TOBL1ADC,TOBL1Strip);
  product.putTOBL2Digis(TOBL2ADC,TOBL2Strip);
  product.putTOBL3Digis(TOBL3ADC,TOBL3Strip);
  product.putTOBL4Digis(TOBL4ADC,TOBL4Strip);
  product.putTIDW1Digis(TIDW1ADC,TIDW1Strip);
  product.putTIDW2Digis(TIDW2ADC,TIDW2Strip);
  product.putTIDW3Digis(TIDW3ADC,TIDW3Strip);
  product.putTECW1Digis(TECW1ADC,TECW1Strip);
  product.putTECW2Digis(TECW2ADC,TECW2Strip);
  product.putTECW3Digis(TECW3ADC,TECW3Strip);
  product.putTECW4Digis(TECW4ADC,TECW4Strip);
  product.putTECW5Digis(TECW5ADC,TECW5Strip);
  product.putTECW6Digis(TECW6ADC,TECW6Strip);  
  product.putTECW7Digis(TECW7ADC,TECW7Strip);
  product.putTECW8Digis(TECW8ADC,TECW8Strip);  

  // pixel output
  product.putBRL1Digis(BRL1ADC, BRL1Row, BRL1Col);
  product.putBRL2Digis(BRL2ADC, BRL2Row, BRL2Col);
  product.putBRL3Digis(BRL3ADC, BRL3Row, BRL3Col);
  product.putFWD1pDigis(FWD1pADC, FWD1pRow, FWD1pCol);
  product.putFWD1nDigis(FWD1nADC, FWD1nRow, FWD1nCol);
  product.putFWD2pDigis(FWD2pADC, FWD2pRow, FWD2pCol);
  product.putFWD2nDigis(FWD2nADC, FWD2nRow, FWD2nCol);

  */

  return;
}

void GlobalRecHitsProducer::fillMuon(edm::Event& iEvent, 
				   const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_fillMuon";
  
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  

  /*
  // get DT information
  edm::Handle<DTDigiCollection> dtDigis;  
  iEvent.getByLabel(MuDTSrc_, dtDigis);
  if (!dtDigis.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find dtDigis in event!";
    return;
  }  

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
	MB1SLayer.push_back(id.superlayer());
	MB1Time.push_back((*digiIt).time());
	MB1Layer.push_back(id.layer());
      }
      if (wireId.station() == 2) {
	MB2SLayer.push_back(id.superlayer());
	MB2Time.push_back((*digiIt).time());
	MB2Layer.push_back(id.layer());
      }
      if (wireId.station() == 3) {
	MB3SLayer.push_back(id.superlayer());
	MB3Time.push_back((*digiIt).time());
	MB3Layer.push_back(id.layer());
      }
      if (wireId.station() == 4) {
	MB4SLayer.push_back(id.superlayer());
	MB4Time.push_back((*digiIt).time());
	MB4Layer.push_back(id.layer());
      }
    }
  }
                                                                     
  if (verbosity > 1) {
    eventout += "\n          Number of DtMuonDigis collected:.......... ";
    eventout += nDt;
  }

  // get CSC Strip information
  edm::Handle<CSCStripDigiCollection> strips;  
  iEvent.getByLabel(MuCSCStripSrc_, strips);
  if (!strips.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find muon strips in event!";
    return;
  }  

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
	  CSCStripADC.push_back(adcCounts[4] - pedestal);	  
      }
    }
  }
                                                        
  if (verbosity > 1) {
    eventout += "\n          Number of CSCStripDigis collected:........ ";
    eventout += nStrips;
  }

  // get CSC Wire information
  edm::Handle<CSCWireDigiCollection> wires;  
  iEvent.getByLabel(MuCSCWireSrc_, wires);
  if (!wires.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find muon wires in event!";
    return;
  }  

  int nWires = 0;
  for (CSCWireDigiCollection::DigiRangeIterator j = wires->begin();
       j != wires->end();
       ++j) {

    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator endDigi = (*j).second.second;

    for ( ; digiItr != endDigi; ++digiItr) {
      ++nWires;

      CSCWireTime.push_back(digiItr->getTimeBin());	  
    }
  }
                                                        
  if (verbosity > 1) {
    eventout += "\n          Number of CSCWireDigis collected:......... ";
    eventout += nWires;
  }

  */

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  
  return;
}

void GlobalRecHitsProducer::storeMuon(PGlobalRecHit& product)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_storeMuon";

  /*
  if (verbosity > 2) {

    // dt output
    TString eventout("\n         nMB1     = ");
    eventout += MB1SLayer.size();
    for (unsigned int i = 0; i < MB1SLayer.size(); ++i) {
      eventout += "\n      (slayer, time, layer) = (";
      eventout += MB1SLayer[i];
      eventout += ", ";
      eventout += MB1Time[i];
      eventout += ", ";
      eventout += MB1Layer[i];
      eventout += ")";
    }
    eventout += "\n         nMB2     = ";
    eventout += MB2SLayer.size();
    for (unsigned int i = 0; i < MB2SLayer.size(); ++i) {
      eventout += "\n      (slayer, time, layer) = (";
      eventout += MB2SLayer[i];
      eventout += ", ";
      eventout += MB2Time[i];
      eventout += ", ";
      eventout += MB2Layer[i];
      eventout += ")";
    }
    eventout += "\n         nMB3     = ";
    eventout += MB3SLayer.size();
    for (unsigned int i = 0; i < MB3SLayer.size(); ++i) {
      eventout += "\n      (slayer, time, layer) = (";
      eventout += MB3SLayer[i];
      eventout += ", ";
      eventout += MB3Time[i];
      eventout += ", ";
      eventout += MB3Layer[i];
      eventout += ")";
    }
    eventout += "\n         nMB2     = ";
    eventout += MB4SLayer.size();
    for (unsigned int i = 0; i < MB4SLayer.size(); ++i) {
      eventout += "\n      (slayer, time, layer) = (";
      eventout += MB4SLayer[i];
      eventout += ", ";
      eventout += MB4Time[i];
      eventout += ", ";
      eventout += MB4Layer[i];
      eventout += ")";
    }    

    // CSC Strip
    eventout += "\n         nCSCStrip     = ";
    eventout += CSCStripADC.size();
    for (unsigned int i = 0; i < CSCStripADC.size(); ++i) {
      eventout += "\n      (adc) = (";
      eventout += CSCStripADC[i];
      eventout += ")";
    }    

    // CSC Wire
    eventout += "\n         nCSCWire     = ";
    eventout += CSCWireTime.size();
    for (unsigned int i = 0; i < CSCWireTime.size(); ++i) {
      eventout += "\n      (time) = (";
      eventout += CSCWireTime[i];
      eventout += ")";
    }    

    edm::LogInfo(MsgLoggerCat) << eventout << "\n";  
  }
  
  product.putMB1Digis(MB1SLayer,MB1Time,MB1Layer);
  product.putMB2Digis(MB2SLayer,MB2Time,MB2Layer);
  product.putMB3Digis(MB3SLayer,MB3Time,MB3Layer);
  product.putMB4Digis(MB4SLayer,MB4Time,MB4Layer);  

  product.putCSCstripDigis(CSCStripADC);

  product.putCSCwireDigis(CSCWireTime);
  */

  return;
}

void GlobalRecHitsProducer::clear()
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_clear";

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat)
      << "Clearing event holders"; 

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
  HBCalSHE.clear();
  HECalREC.clear();
  HECalSHE.clear();
  HOCalREC.clear();
  HOCalSHE.clear();
  HFCalREC.clear();
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

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalRecHitsProducer);
