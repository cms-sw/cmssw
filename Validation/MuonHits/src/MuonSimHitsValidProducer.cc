#include "Validation/MuonHits/interface/MuonSimHitsValidProducer.h"

MuonSimHitsValidProducer::MuonSimHitsValidProducer(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), label(""), getAllProvenances(false),
  printProvenanceInfo(false), nRawGenPart(0), count(0)

{
  /// get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  label = iPSet.getParameter<std::string>("Label");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

  /// get labels for input tags
   CSCHitsSrc_ = iPSet.getParameter<edm::InputTag>("CSCHitsSrc");
   DTHitsSrc_  = iPSet.getParameter<edm::InputTag>("DTHitsSrc");
   RPCHitsSrc_ = iPSet.getParameter<edm::InputTag>("RPCHitsSrc");

  /// use value of first digit to determine default output level (inclusive)
  /// 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  /// create persistent object
  produces<PMuonSimHit>(label);

  /// print out Parameter Set information being used
  if (verbosity > 0) {
    edm::LogInfo ("MuonSimHitsValidProducer::MuonSimHitsValidProducer") 
      << "\n===============================\n"
      << "Initialized as EDProducer with parameter values:\n"
      << "    Name      = " << fName << "\n"
      << "    Verbosity = " << verbosity << "\n"
      << "    Label     = " << label << "\n"
      << "    GetProv   = " << getAllProvenances << "\n"
      << "    PrintProv = " << printProvenanceInfo << "\n"
      << "    CSCHitsSrc=  " <<CSCHitsSrc_.label() 
      << ":" << CSCHitsSrc_.instance() << "\n"
      << "    DTHitsSrc =  " <<DTHitsSrc_.label()
      << ":" << DTHitsSrc_.instance() << "\n"
      << "    RPCHitsSrc=  " <<RPCHitsSrc_.label()
      << ":" << RPCHitsSrc_.instance() << "\n"
      << "===============================\n";
  }
}

MuonSimHitsValidProducer::~MuonSimHitsValidProducer() 
{
}

void MuonSimHitsValidProducer::beginJob(const edm::EventSetup& iSetup)
{
  clear();
  return;
}

void MuonSimHitsValidProducer::endJob()
{
  if (verbosity > 0)
    edm::LogInfo ("MuonSimHitsValidProducer::endJob") 
      << "Terminating having processed " << count << " events.";
  return;
}

void MuonSimHitsValidProducer::produce(edm::Event& iEvent, 
			       const edm::EventSetup& iSetup)
{
  /// keep track of number of events processed
  ++count;

  /// get event id information
  int nrun = iEvent.id().run();
  int nevt = iEvent.id().event();

  if (verbosity > 0) {
    edm::LogInfo ("MuonSimHitsValidProducer::produce")
      << "Processing run " << nrun << ", event " << nevt;
  }

  /// clear event holders
  clear();

  /// look at information available in the event
  if (getAllProvenances) {

    std::vector<const edm::Provenance*> AllProv;
    iEvent.getAllProvenance(AllProv);

    if (verbosity > 0)
      edm::LogInfo ("MuonSimHitsValidProducer::produce")
	<< "Number of Provenances = " << AllProv.size();

    if (printProvenanceInfo && (verbosity > 0)) {
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
      eventout += "       ******************************\n";
      edm::LogInfo("MuonSimHitsValidProducer::produce") << eventout << "\n";
    }
  }

  /// call fill functions
  /// gather G4MC information from event
  fillG4MC(iEvent);

  /// gather CSC, DT and RPC information from event
  fillCSC(iEvent, iSetup);
  fillDT(iEvent, iSetup);
  fillRPC(iEvent, iSetup);

  if (verbosity > 0)
    edm::LogInfo ("MuonSimHitsValidProducer::produce")
      << "Done gathering data from event.";

  /// produce object to put into event
  std::auto_ptr<PMuonSimHit> pOut(new PMuonSimHit);

  if (verbosity > 2)
    edm::LogInfo ("MuonSimHitsValidProducer::produce")
      << "Saving event contents:";

  /// call store functions
  /// store G4MC information in product
  storeG4MC(*pOut);

  /// store CSC, DT and RPC information in produce
  storeCSC(*pOut);
  storeDT(*pOut);
  storeRPC(*pOut);

  /// store information in event
  iEvent.put(pOut,label);

  return;
}

void MuonSimHitsValidProducer::fillG4MC(edm::Event& iEvent)
{
 
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  /// get MC information
  edm::Handle<edm::HepMCProduct> HepMCEvt;
  std::vector<edm::Handle<edm::HepMCProduct> > AllHepMCEvt;
  iEvent.getManyByType(AllHepMCEvt);

  /// loop through products and extract VtxSmearing if available. Any of them
  /// should have the information needed
  for (unsigned int i = 0; i < AllHepMCEvt.size(); ++i) {
    HepMCEvt = AllHepMCEvt[i];
    if ((HepMCEvt.provenance()->product()).moduleLabel() == "VtxSmeared")
      break;
  }

  if (!HepMCEvt.isValid()) {
    edm::LogWarning("MuonSimHitsValidProducer::fillG4MC")
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

  /// get G4Vertex information
  edm::Handle<edm::SimVertexContainer> G4VtxContainer;
  iEvent.getByType(G4VtxContainer);
  if (!G4VtxContainer.isValid()) {
    edm::LogWarning("MuonSimHitsValidProducer::fillG4MC")
      << "Unable to find SimVertex in event!";
    return;
  }
  int i = 0;
  edm::SimVertexContainer::const_iterator itVtx;
  for (itVtx = G4VtxContainer->begin(); itVtx != G4VtxContainer->end(); 
       ++itVtx) {    
    ++i;
    const HepLorentzVector& G4Vtx = itVtx->position();
    G4VtxX.push_back(G4Vtx[0]/micrometer);
    G4VtxY.push_back(G4Vtx[1]/micrometer);
    G4VtxZ.push_back(G4Vtx[2]/millimeter);
  }

  if (verbosity > 1) {
    eventout += "\n          Number of G4Vertices collected:............ ";
    eventout += i;
  }  

  /// get G4Track information
  edm::Handle<edm::SimTrackContainer> G4TrkContainer;
  iEvent.getByType(G4TrkContainer);
  if (!G4TrkContainer.isValid()) {
    edm::LogWarning("MuonSimHitsValidProducer::fillG4MC")
      << "Unable to find SimTrack in event!";
    return;
  }
  i = 0;
  edm::SimTrackContainer::const_iterator itTrk;
  for (itTrk = G4TrkContainer->begin(); itTrk != G4TrkContainer->end(); 
       ++itTrk) {
    ++i;
    double etaInit =0, phiInit =0, pInit =0;
    const HepLorentzVector& G4Trk = itTrk->momentum();
    pInit =sqrt(G4Trk[0]*G4Trk[0]+G4Trk[1]*G4Trk[1]+G4Trk[2]*G4Trk[2]);
    
	  if ( pInit == 0) 
		 edm::LogError("TrackerHitProducer::fillG4MC") 
		 << "TrackerTest::INFO: Primary has p = 0 ";
	  else {
		 double costheta  = G4Trk[2]/pInit;
	         double theta = acos(TMath::Min(TMath::Max(costheta, -1.),1.));
		 etaInit = -log(tan(theta/2));
		  if ( G4Trk[0] != 0 || G4Trk[1] != 0) 
			  phiInit = atan2(G4Trk[1],G4Trk[0]);
	  }
    G4TrkPt.push_back(sqrt(G4Trk[0]*G4Trk[0]+G4Trk[1]*G4Trk[1])); 
    G4TrkE.push_back(G4Trk[3]);
    G4TrkEta.push_back(etaInit);                                   
    G4TrkPhi.push_back(phiInit);  

  } 
  if (verbosity > 1) {
    eventout += "\n          Number of G4Tracks collected:.............. ";
    eventout += i;
  }  

  if (verbosity > 0)
    edm::LogInfo("MuonSimHitsValidProducer::fillG4MC") << eventout << "\n";

  return;
}

void MuonSimHitsValidProducer::storeG4MC(PMuonSimHit& product)
{
  if (verbosity > 2) {
    TString eventout("\nnRawGenPart        = ");
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
      eventout += "\n          (pt,e,eta,phi)          = (";
      eventout += G4TrkPt[i];
      eventout += ", ";
      eventout += G4TrkE[i];
      eventout += ")";
      eventout += G4TrkEta[i];
      eventout += ")";
      eventout += G4TrkPhi[i];
      eventout += ")";
    }    
    edm::LogInfo("MuonSimHitsValidProducer::storeG4MC") << eventout;
  }

  product.putRawGenPart(nRawGenPart);
  product.putG4Vtx(G4VtxX, G4VtxY, G4VtxZ);
  product.putG4Trk(G4TrkPt, G4TrkE, G4TrkEta, G4TrkPhi);

  return;
}

void MuonSimHitsValidProducer::fillCSC(edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering CSC info:";  

  /// iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;

  /// access the CSC
  /// access the CSC geometry
  edm::ESHandle<CSCGeometry> theCSCGeometry;
  iSetup.get<MuonGeometryRecord>().get(theCSCGeometry);
  if (!theCSCGeometry.isValid()) {
    edm::LogWarning("MuonSimHitsValidProducer::fillCSC")
      << "Unable to find MuonGeometryRecord for the CSCGeometry in event!";
    return;
  }
  const CSCGeometry& theCSCMuon(*theCSCGeometry);

  /// get  CSC information
  edm::Handle<edm::PSimHitContainer> MuonCSCContainer;
  iEvent.getByLabel(CSCHitsSrc_,MuonCSCContainer);
//  iEvent.getByLabel("g4SimHits","MuonCSCHits",MuonCSCContainer);
  if (!MuonCSCContainer.isValid()) {
    edm::LogWarning("MuonSimHitsValidProducer::fillCSC")
      << "Unable to find MuonCSCHits in event!";
    return;
  }

  /// cycle through container
  int i = 0, j = 0;
  for (itHit = MuonCSCContainer->begin(); itHit != MuonCSCContainer->end(); 
       ++itHit) {
    ++i;
    /// create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    /// check that expected detector is returned
    if ((detector == deMuon) && 
        (subdetector == sdeMuonCSC)) {

      /// get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theCSCMuon.idToDetUnit(theDetUnitId);
    
      if (!theDet) {
	edm::LogWarning("MuonSimHitsValidProducer::fillCSC")
	  << "Unable to get GeomDetUnit from theCSCMuon for hit " << i;
	continue;
      }
     
      ++j;

      /// get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bsurf = theDet->surface();

      /// gather necessary information

      const CSCDetId& id=CSCDetId(itHit->detUnitId());

      int cscid=id.endcap()*100000 + id.station()*10000 +
                id.ring()*1000     + id.chamber()*10 +id.layer(); 

      CSCHitsId.push_back(cscid);
      
      CSCHitsDetUnId.push_back(itHit->detUnitId());
      CSCHitsTrkId.push_back(itHit->trackId());
      CSCHitsProcType.push_back(itHit->processType());
      CSCHitsPartType.push_back(itHit->particleType());
      CSCHitsPabs.push_back(itHit->pabs());
      
      CSCHitsGlobPosZ.push_back(bsurf.toGlobal(itHit->localPosition()).z());
      CSCHitsGlobPosPhi.push_back(bsurf.toGlobal(itHit->localPosition()).phi());
      CSCHitsGlobPosEta.push_back(bsurf.toGlobal(itHit->localPosition()).eta());

      CSCHitsLocPosX.push_back(itHit->localPosition().x());
      CSCHitsLocPosY.push_back(itHit->localPosition().y());
      CSCHitsLocPosZ.push_back(itHit->localPosition().z());

      CSCHitsLocDirX.push_back(itHit->localDirection().x());
      CSCHitsLocDirY.push_back(itHit->localDirection().y());
      CSCHitsLocDirZ.push_back(itHit->localDirection().z());
      CSCHitsLocDirTheta.push_back(itHit->localDirection().theta());
      CSCHitsLocDirPhi.push_back(itHit->localDirection().phi());

      CSCHitsExitPointX.push_back(itHit->exitPoint().x());
      CSCHitsExitPointY.push_back(itHit->exitPoint().y());
      CSCHitsExitPointZ.push_back(itHit->exitPoint().z());

      CSCHitsEntryPointX.push_back(itHit->entryPoint().x());
      CSCHitsEntryPointY.push_back(itHit->entryPoint().y());
      CSCHitsEntryPointZ.push_back(itHit->entryPoint().z());

      CSCHitsEnLoss.push_back(itHit->energyLoss());
      CSCHitsTimeOfFlight.push_back(itHit->tof());

    } else {
      edm::LogWarning("MuonSimHitsValidProducer::fillCSC")
        << "MuonCsc PSimHit " << i 
        << " is expected to be (det,subdet) = (" 
        << deMuon << "," << sdeMuonCSC
        << "); value returned is: ("
        << detector << "," << subdetector << ")";
      continue;
    } 
  } 

  if (verbosity > 1) {
    eventout += "\n          Number of CSC muon Hits collected:......... ";
    eventout += j;
  }  

  if (verbosity > 0)
    edm::LogInfo("MuonSimHitsValidProducer::fillCSC") << eventout << "\n";

  return;
}

void MuonSimHitsValidProducer::fillDT(edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{
 TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering DT info:";  

  /// iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;

  /// access the DT
  /// access the DT geometry
  edm::ESHandle<DTGeometry> theDTGeometry;
  iSetup.get<MuonGeometryRecord>().get(theDTGeometry);
  if (!theDTGeometry.isValid()) {
    edm::LogWarning("MuonSimHitsValidProducer::fillDT")
      << "Unable to find MuonGeometryRecord for the DTGeometry in event!";
    return;
  }
  const DTGeometry& theDTMuon(*theDTGeometry);

  /// get DT information
  edm::Handle<edm::PSimHitContainer> MuonDTContainer;
  iEvent.getByLabel(DTHitsSrc_,MuonDTContainer);
//  iEvent.getByLabel("g4SimHits","MuonDTHits",MuonDTContainer);
  if (!MuonDTContainer.isValid()) {
    edm::LogWarning("MuonSimHitsValidProducer::fillDT")
      << "Unable to find MuonDTHits in event!";
    return;
  }

  /// cycle through container
  int i = 0, j = 0;
  for (itHit = MuonDTContainer->begin(); itHit != MuonDTContainer->end(); 
       ++itHit) {

    ++i;

    /// create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    /// check that expected detector is returned
    if ((detector == deMuon) && 
        (subdetector == sdeMuonDT)) {
       
      /// get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theDTMuon.idToDetUnit(theDetUnitId);
    
      if (!theDet) {
  	edm::LogWarning("MuonSimHitsValidProducer::fillDT") 
	  << "Unable to get GeomDetUnit from theDTMuon for hit " << i;
	continue;
      }
     
      ++j;

      /// get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bsurf = theDet->surface();
    
      /// gather necessary information

      DTHitsDetUnId.push_back(itHit->detUnitId());
      DTHitsTrkId.push_back(itHit->trackId());
      DTHitsProcType.push_back(itHit->processType());
      DTHitsPartType.push_back(itHit->particleType());
      DTHitsPabs.push_back(itHit->pabs());

      DTHitsGlobPosZ.push_back(bsurf.toGlobal(itHit->localPosition()).z());
      DTHitsGlobPosPhi.push_back(bsurf.toGlobal(itHit->localPosition()).phi());
      DTHitsGlobPosEta.push_back(bsurf.toGlobal(itHit->localPosition()).eta());
      
      DTHitsLocPosX.push_back(itHit->localPosition().x());
      DTHitsLocPosY.push_back(itHit->localPosition().y());
      DTHitsLocPosZ.push_back(itHit->localPosition().z());

      DTHitsLocDirX.push_back(itHit->localDirection().x());
      DTHitsLocDirY.push_back(itHit->localDirection().y());
      DTHitsLocDirZ.push_back(itHit->localDirection().z());
      DTHitsLocDirTheta.push_back(itHit->localDirection().theta());
      DTHitsLocDirPhi.push_back(itHit->localDirection().phi());

      DTHitsExitPointX.push_back(itHit->exitPoint().x());
      DTHitsExitPointY.push_back(itHit->exitPoint().y());
      DTHitsExitPointZ.push_back(itHit->exitPoint().z());

      DTHitsEntryPointX.push_back(itHit->entryPoint().x());
      DTHitsEntryPointY.push_back(itHit->entryPoint().y());
      DTHitsEntryPointZ.push_back(itHit->entryPoint().z());

      DTHitsEnLoss.push_back(itHit->energyLoss());
      DTHitsTimeOfFlight.push_back(itHit->tof());
      
    } else {
      edm::LogWarning("MuonSimHitsValidProducer::fillDT")
        << "MuonDT PSimHit " << i 
        << " is expected to be (det,subdet) = (" 
        << deMuon << "," << sdeMuonDT
        << "); value returned is: ("
        << detector << "," << subdetector << ")";
      continue;
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of DT muon Hits collected:......... ";
    eventout += j;
  }  

  if (verbosity > 0)
    edm::LogInfo("MuonSimHitsValidProducer::fillDT") << eventout << "\n";
return;
}

void MuonSimHitsValidProducer::fillRPC(edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering RPC info:";  

  /// iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;

  /// access the RPC 
  /// access the RPC geometry
  edm::ESHandle<RPCGeometry> theRPCGeometry;
  iSetup.get<MuonGeometryRecord>().get(theRPCGeometry);
  if (!theRPCGeometry.isValid()) {
    edm::LogWarning("MuonSimHitsValidProducer::fillRPC")
      << "Unable to find MuonGeometryRecord for the RPCGeometry in event!";
    return;
  }
  const RPCGeometry& theRPCMuon(*theRPCGeometry);

  // get Muon RPC information
  edm::Handle<edm::PSimHitContainer> MuonRPCContainer;
  iEvent.getByLabel(RPCHitsSrc_,MuonRPCContainer);
//  iEvent.getByLabel("g4SimHits","MuonRPCHits",MuonRPCContainer);
  if (!MuonRPCContainer.isValid()) {
    edm::LogWarning("MuonSimHitsValidProducer::fillRPC")
      << "Unable to find MuonRPCHits in event!";
    return;
  }

  /// cycle through container
  int i = 0, j = 0;
  for (itHit = MuonRPCContainer->begin(); itHit != MuonRPCContainer->end(); 
       ++itHit) {

    ++i;

    /// create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    /// check that expected detector is returned
    if ((detector == deMuon) && 
        (subdetector == sdeMuonRPC)) {

      /// get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theRPCMuon.idToDetUnit(theDetUnitId);
    
      if (!theDet) {
	edm::LogWarning("MuonSimHitsValidProducer::fillRPC")
	  << "Unable to get GeomDetUnit from theRPCMuon for hit " << i;
	continue;
      }
     
      ++j;

      /// get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bsurf = theDet->surface();
    
      /// gather necessary information

      RPCHitsDetUnId.push_back(itHit->detUnitId());
      RPCHitsTrkId.push_back(itHit->trackId());
      RPCHitsProcType.push_back(itHit->processType());
      RPCHitsPartType.push_back(itHit->particleType());
      RPCHitsPabs.push_back(itHit->pabs());
      
      RPCHitsGlobPosZ.push_back(bsurf.toGlobal(itHit->localPosition()).z());
      RPCHitsGlobPosPhi.push_back(bsurf.toGlobal(itHit->localPosition()).phi());
      RPCHitsGlobPosEta.push_back(bsurf.toGlobal(itHit->localPosition()).eta());

      RPCHitsLocPosX.push_back(itHit->localPosition().x());
      RPCHitsLocPosY.push_back(itHit->localPosition().y());
      RPCHitsLocPosZ.push_back(itHit->localPosition().z());

      RPCHitsLocDirX.push_back(itHit->localDirection().x());
      RPCHitsLocDirY.push_back(itHit->localDirection().y());
      RPCHitsLocDirZ.push_back(itHit->localDirection().z());
      RPCHitsLocDirTheta.push_back(itHit->localDirection().theta());
      RPCHitsLocDirPhi.push_back(itHit->localDirection().phi());

      RPCHitsExitPointX.push_back(itHit->exitPoint().x());
      RPCHitsExitPointY.push_back(itHit->exitPoint().y());
      RPCHitsExitPointZ.push_back(itHit->exitPoint().z());

      RPCHitsEntryPointX.push_back(itHit->entryPoint().x());
      RPCHitsEntryPointY.push_back(itHit->entryPoint().y());
      RPCHitsEntryPointZ.push_back(itHit->entryPoint().z());

      RPCHitsEnLoss.push_back(itHit->energyLoss());
      RPCHitsTimeOfFlight.push_back(itHit->tof());

    } else {
      edm::LogWarning("MuonSimHitsValidProducer::fillRPC")
        << "MuonRpc PSimHit " << i 
        << " is expected to be (det,subdet) = (" 
        << deMuon << "," << sdeMuonRPC
        << "); value returned is: ("
        << detector << "," << subdetector << ")";
      continue;
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of RPC muon Hits collected:......... ";
    eventout += j;
  }  

  if (verbosity > 0)
    edm::LogInfo("MuonSimHitsValidProducer::fillRPC") << eventout << "\n";

return;
}

void MuonSimHitsValidProducer::storeCSC(PMuonSimHit& product)
{
    product.putCSCHits(
                  CSCHitsId,
                  CSCHitsDetUnId,     CSCHitsTrkId,       CSCHitsProcType, 
                  CSCHitsPartType,    CSCHitsPabs,
                  CSCHitsGlobPosZ,    CSCHitsGlobPosPhi,  CSCHitsGlobPosEta,
                  CSCHitsLocPosX,     CSCHitsLocPosY,     CSCHitsLocPosZ, 
		  CSCHitsLocDirX,     CSCHitsLocDirY,     CSCHitsLocDirZ, 
                  CSCHitsLocDirTheta, CSCHitsLocDirPhi,
		  CSCHitsExitPointX,  CSCHitsExitPointY,  CSCHitsExitPointZ,
		  CSCHitsEntryPointX, CSCHitsEntryPointY, CSCHitsEntryPointZ,
		  CSCHitsEnLoss,      CSCHitsTimeOfFlight);
return;
}

void MuonSimHitsValidProducer::storeDT(PMuonSimHit& product)
{
    product.putDTHits(
                  DTHitsDetUnId,      DTHitsTrkId,        DTHitsProcType, 
                  DTHitsPartType,     DTHitsPabs,
                  DTHitsGlobPosZ,     DTHitsGlobPosPhi,   DTHitsGlobPosEta,
                  DTHitsLocPosX,      DTHitsLocPosY,      DTHitsLocPosZ, 
		  DTHitsLocDirX,      DTHitsLocDirY,      DTHitsLocDirZ, 
                  DTHitsLocDirTheta,  DTHitsLocDirPhi,
		  DTHitsExitPointX,   DTHitsExitPointY,   DTHitsExitPointZ,
		  DTHitsEntryPointX,  DTHitsEntryPointY,  DTHitsEntryPointZ,
		  DTHitsEnLoss,       DTHitsTimeOfFlight);
 
      return;
}

void MuonSimHitsValidProducer::storeRPC(PMuonSimHit& product)
{
    product.putRPCHits(
                  RPCHitsDetUnId,     RPCHitsTrkId,       RPCHitsProcType, 
                  RPCHitsPartType,    RPCHitsPabs,
                  RPCHitsGlobPosZ,    RPCHitsGlobPosPhi,  RPCHitsGlobPosEta,
                  RPCHitsLocPosX,     RPCHitsLocPosY,     RPCHitsLocPosZ, 
		  RPCHitsLocDirX,     RPCHitsLocDirY,     RPCHitsLocDirZ, 
                  RPCHitsLocDirTheta, RPCHitsLocDirPhi,
		  RPCHitsExitPointX,  RPCHitsExitPointY,  RPCHitsExitPointZ,
		  RPCHitsEntryPointX, RPCHitsEntryPointY, RPCHitsEntryPointZ,
		  RPCHitsEnLoss,      RPCHitsTimeOfFlight);
  return;
}

void MuonSimHitsValidProducer::clear()
{
  if (verbosity > 0)
    edm::LogInfo("MuonSimHitsValidProducer::clear")
      << "Clearing event holders"; 

  /// reset G4MC info
  nRawGenPart = 0;
  G4VtxX.clear();
  G4VtxY.clear();
  G4VtxZ.clear();
  G4TrkPt.clear();
  G4TrkE.clear();

  /// reset CSC info

  CSCHitsId.clear();
  CSCHitsDetUnId.clear();
  CSCHitsTrkId.clear(); 
  CSCHitsProcType.clear(); 
  CSCHitsPartType.clear(); 
  CSCHitsPabs.clear();
  CSCHitsGlobPosZ.clear(); 
  CSCHitsGlobPosPhi.clear(); 
  CSCHitsGlobPosEta.clear(); 
  CSCHitsLocPosX.clear(); 
  CSCHitsLocPosY.clear(); 
  CSCHitsLocPosZ.clear(); 
  CSCHitsLocDirX.clear(); 
  CSCHitsLocDirY.clear(); 
  CSCHitsLocDirZ.clear(); 
  CSCHitsLocDirTheta.clear(); 
  CSCHitsLocDirPhi.clear();
  CSCHitsExitPointX.clear(); 
  CSCHitsExitPointY.clear(); 
  CSCHitsExitPointZ.clear();
  CSCHitsEntryPointX.clear(); 
  CSCHitsEntryPointY.clear(); 
  CSCHitsEntryPointZ.clear();
  CSCHitsEnLoss.clear(); 
  CSCHitsTimeOfFlight.clear();

  /// reset DT info

  DTHitsDetUnId.clear();
  DTHitsTrkId.clear(); 
  DTHitsProcType.clear(); 
  DTHitsPartType.clear(); 
  DTHitsPabs.clear();
  DTHitsGlobPosZ.clear(); 
  DTHitsGlobPosPhi.clear(); 
  DTHitsGlobPosEta.clear(); 
  DTHitsLocPosX.clear(); 
  DTHitsLocPosY.clear(); 
  DTHitsLocPosZ.clear(); 
  DTHitsLocDirX.clear(); 
  DTHitsLocDirY.clear(); 
  DTHitsLocDirZ.clear(); 
  DTHitsLocDirTheta.clear(); 
  DTHitsLocDirPhi.clear();
  DTHitsExitPointX.clear(); 
  DTHitsExitPointY.clear(); 
  DTHitsExitPointZ.clear();
  DTHitsEntryPointX.clear(); 
  DTHitsEntryPointY.clear(); 
  DTHitsEntryPointZ.clear();
  DTHitsEnLoss.clear(); 
  DTHitsTimeOfFlight.clear();

  /// reset RPC info

  RPCHitsDetUnId.clear();
  RPCHitsTrkId.clear(); 
  RPCHitsProcType.clear(); 
  RPCHitsPartType.clear(); 
  RPCHitsPabs.clear();
  RPCHitsGlobPosZ.clear(); 
  RPCHitsGlobPosPhi.clear(); 
  RPCHitsGlobPosEta.clear(); 
  RPCHitsLocPosX.clear(); 
  RPCHitsLocPosY.clear(); 
  RPCHitsLocPosZ.clear(); 
  RPCHitsLocDirX.clear(); 
  RPCHitsLocDirY.clear(); 
  RPCHitsLocDirZ.clear(); 
  RPCHitsLocDirTheta.clear(); 
  RPCHitsLocDirPhi.clear();
  RPCHitsExitPointX.clear(); 
  RPCHitsExitPointY.clear(); 
  RPCHitsExitPointZ.clear();
  RPCHitsEntryPointX.clear(); 
  RPCHitsEntryPointY.clear(); 
  RPCHitsEntryPointZ.clear();
  RPCHitsEnLoss.clear(); 
  RPCHitsTimeOfFlight.clear();

  return;
}
