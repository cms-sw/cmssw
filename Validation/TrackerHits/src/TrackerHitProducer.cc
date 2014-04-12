#include "Validation/TrackerHits/interface/TrackerHitProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

// tracker info
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

// data in edm::event
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

// helper files
#include <CLHEP/Vector/LorentzVector.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "TString.h"

#include <cmath>
#include <memory>
#include <stdlib.h>

TrackerHitProducer::TrackerHitProducer(const edm::ParameterSet& iPSet)
  : getAllProvenances( iPSet.getParameter<edm::ParameterSet>( "ProvenanceLookup" ).getUntrackedParameter<bool>( "GetAllProvenances", false ) )
  , printProvenanceInfo( iPSet.getParameter<edm::ParameterSet>( "ProvenanceLookup" ).getUntrackedParameter<bool>( "PrintProvenanceInfo", false ) )
  , verbosity( iPSet.getUntrackedParameter<int>( "Verbosity", 0 ) )
  , count( 0 )
  , nRawGenPart( 0 )
  , config_( iPSet )
  , edmHepMCProductToken_( consumes<edm::HepMCProduct>( edm::InputTag( iPSet.getUntrackedParameter<std::string>( "HepMCProductLabel", "source" )
                                                                     , iPSet.getUntrackedParameter<std::string>( "HepMCInputInstance", "" )
								       )
							)
			   )
  , edmSimVertexContainerToken_( consumes<edm::SimVertexContainer>( iPSet.getParameter<edm::InputTag>("G4VtxSrc") ) )
  , edmSimTrackContainerToken_( consumes<edm::SimTrackContainer>( iPSet.getParameter<edm::InputTag>( "G4TrkSrc" ) ) )
  , edmPSimHitContainer_pxlBrlLow_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "PxlBrlLowSrc" ) ) )
  , edmPSimHitContainer_pxlBrlHigh_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "PxlBrlHighSrc" ) ) )
  , edmPSimHitContainer_pxlFwdLow_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "PxlFwdLowSrc" ) ) )
  , edmPSimHitContainer_pxlFwdHigh_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "PxlFwdHighSrc" ) ) )
  , edmPSimHitContainer_siTIBLow_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "SiTIBLowSrc" ) ) )
  , edmPSimHitContainer_siTIBHigh_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "SiTIBHighSrc" ) ) )
  , edmPSimHitContainer_siTOBLow_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "SiTOBLowSrc" ) ) )
  , edmPSimHitContainer_siTOBHigh_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "SiTOBHighSrc" ) ) )
  , edmPSimHitContainer_siTIDLow_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "SiTIDLowSrc" ) ) )
  , edmPSimHitContainer_siTIDHigh_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "SiTIDHighSrc" ) ) )
  , edmPSimHitContainer_siTECLow_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "SiTECLowSrc" ) ) )
  , edmPSimHitContainer_siTECHigh_Token_( consumes<edm::PSimHitContainer>( iPSet.getParameter<edm::InputTag>( "SiTECHighSrc" ) ) )
  , fName( iPSet.getUntrackedParameter<std::string>( "Name", "" ) )
  , label( iPSet.getParameter<std::string>( "Label" ) )
{
  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // create persistent object
  produces<PTrackerSimHit>(label);

  // print out Parameter Set information being used
  if (verbosity > 0) {
    edm::LogInfo ("TrackerHitProducer::TrackerHitProducer") 
      << "\n===============================\n"
      << "Initialized as EDProducer with parameter values:\n"
      << "    Name      =" << fName << "\n"
      << "    Verbosity =" << verbosity << "\n"
      << "    Label     =" << label << "\n"
      << "    GetProv   =" << getAllProvenances << "\n"
      << "    PrintProv =" << printProvenanceInfo << "\n"
      << "    PxlBrlLowSrc  = " << iPSet.getParameter<edm::InputTag>("PxlBrlLowSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("PxlBrlLowSrc").instance() << "\n"
      << "    PxlBrlHighSrc = " << iPSet.getParameter<edm::InputTag>("PxlBrlHighSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("PxlBrlHighSrc").instance() << "\n"
      << "    PxlFwdLowSrc  = " << iPSet.getParameter<edm::InputTag>("PxlFwdLowSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("PxlFwdLowSrc").instance() << "\n"
      << "    PxlFwdHighSrc = " << iPSet.getParameter<edm::InputTag>("PxlFwdHighSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("PxlFwdHighSrc").instance() << "\n"
      << "    SiTIBLowSrc   = " << iPSet.getParameter<edm::InputTag>("SiTIBLowSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("SiTIBLowSrc").instance() << "\n"
      << "    SiTIBHighSrc  = " << iPSet.getParameter<edm::InputTag>("SiTIBHighSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("SiTIBHighSrc").instance() << "\n"
      << "    SiTOBLowSrc   = " << iPSet.getParameter<edm::InputTag>("SiTOBLowSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("SiTOBLowSrc").instance() << "\n"
      << "    SiTOBHighSrc  = " << iPSet.getParameter<edm::InputTag>("SiTOBHighSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("SiTOBHighSrc").instance() << "\n"
      << "    SiTIDLowSrc   = " << iPSet.getParameter<edm::InputTag>("SiTIDLowSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("SiTIDLowSrc").instance() << "\n"
      << "    SiTIDHighSrc  = " << iPSet.getParameter<edm::InputTag>("SiTIDHighSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("SiTIDHighSrc").instance() << "\n"
      << "    SiTECLowSrc   = " << iPSet.getParameter<edm::InputTag>("SiTECLowSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("SiTECLowSrc").instance() << "\n"
      << "    SiTECHighSrc  = " << iPSet.getParameter<edm::InputTag>("SiTECHighSrc").label() 
      << ":" << iPSet.getParameter<edm::InputTag>("SiTECHighSrc").instance() << "\n"
      << "===============================\n";  }
}

TrackerHitProducer::~TrackerHitProducer() 
{
}

void TrackerHitProducer::beginJob()
{
  if (verbosity > 0)
    edm::LogInfo ("TrackerHitProducer::beginJob") 
      << "Starting the job...";
  clear();
  return;
}

void TrackerHitProducer::endJob()
{
  if (verbosity > 0)
    edm::LogInfo ("TrackerHitProducer::endJob") 
      << "Terminating having processed" << count << "events.";
  return;
}

void TrackerHitProducer::produce(edm::Event& iEvent, 
                   const edm::EventSetup& iSetup)
{
  // keep track of number of events processed
  ++count;

  // get event id information
  int nrun = iEvent.id().run();
  int nevt = iEvent.id().event();

  // get event setup information
  //edm::ESHandle<edm::SetupData> pSetup;
  //iSetup.get<edm::SetupRecord>().get(pSetup);

  if (verbosity > 0) {
    edm::LogInfo ("TrackerHitProducer::produce")
      << "Processing run" << nrun << "," << "event " << nevt;
  }

  // clear event holders
  clear();

  // look at information available in the event
  if (getAllProvenances) {

    std::vector<const edm::Provenance*> AllProv;
    iEvent.getAllProvenance(AllProv);

    if (verbosity > 0)
      edm::LogInfo ("TrackerHitProducer::produce")
    << "Number of Provenances =" << AllProv.size();

    if (printProvenanceInfo && (verbosity > 0)) {
      TString eventout("\nProvenance info:\n");
      
      for (unsigned int i = 0; i < AllProv.size(); ++i) {
    eventout += "\n       ******************************";
    eventout += "\n       Module       : ";
    eventout += AllProv[i]->moduleLabel();
    eventout += "\n       ProductID process index: ";
    eventout += AllProv[i]->productID().processIndex();
    eventout += "\n       ProductID product index: ";
    eventout += AllProv[i]->productID().productIndex();
    eventout += "\n       ClassName    : ";
    eventout += AllProv[i]->className();
    eventout += "\n       InstanceName : ";
    eventout += AllProv[i]->productInstanceName();
    eventout += "\n       BranchName   : ";
    eventout += AllProv[i]->branchName();
      }
      eventout += "       ******************************\n";
      edm::LogInfo("TrackerHitProducer::produce") << eventout;
    }
  }

  // call fill functions
  //gather G4MC information from event
  fillG4MC(iEvent);
  // gather Tracker information from event
  fillTrk(iEvent,iSetup);

  if (verbosity > 0)
    edm::LogInfo ("TrackerHitProducer::produce")
      << "Done gathering data from event.";

  // produce object to put into event
  std::auto_ptr<PTrackerSimHit> pOut(new PTrackerSimHit);

  if (verbosity > 2)
    edm::LogInfo ("TrackerHitProducer::produce")
      << "Saving event contents:";

  // call store functions
  // store G4MC information in product
  storeG4MC(*pOut);
  // store Tracker information in produce
  storeTrk(*pOut);

  // store information in event
  iEvent.put(pOut,label);

  return;
}

//==================fill and store functions================================
void TrackerHitProducer::fillG4MC(edm::Event& iEvent)
{
 
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";

  //////////////////////
  // get MC information
  /////////////////////
  edm::Handle<edm::HepMCProduct> HepMCEvt_;
  iEvent.getByToken( edmHepMCProductToken_, HepMCEvt_ );
  if (!HepMCEvt_.isValid()) {
    edm::LogError("TrackerHitProducer::fillG4MC")
      << "Unable to find HepMCProduct in event!";
    return;
  }
  const HepMC::GenEvent* MCEvt = HepMCEvt_->GetEvent();
  nRawGenPart = MCEvt->particles_size();

  if (verbosity > 1) {
    eventout += "\n          Number of Raw Particles collected:         ";
    eventout += nRawGenPart;
  }  

  ////////////////////////////
  // get G4Vertex information
  ////////////////////////////
  edm::Handle<edm::SimVertexContainer> G4VtxContainer;
  iEvent.getByToken( edmSimVertexContainerToken_, G4VtxContainer );
  if (!G4VtxContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillG4MC")
      << "Unable to find SimVertex in event!";
    return;
  }
  int i = 0;
  edm::SimVertexContainer::const_iterator itVtx;
  for (itVtx = G4VtxContainer->begin(); itVtx != G4VtxContainer->end(); 
       ++itVtx) {
    
    ++i;

    const CLHEP::HepLorentzVector& G4Vtx = CLHEP::HepLorentzVector(itVtx->position().x(),
                                                     itVtx->position().y(),
                                                     itVtx->position().z(),  
                                                     itVtx->position().e());
    G4VtxX.push_back(G4Vtx[0]/micrometer); //cm from code -> micrometer *10000
    G4VtxY.push_back(G4Vtx[1]/micrometer); //cm from code -> micrometer *10000
    G4VtxZ.push_back(G4Vtx[2]/millimeter); //cm from code -> millimeter *10
  }

  if (verbosity > 1) {
    eventout += "\n          Number of G4Vertices collected:            ";
    eventout += i;
  }  

  ///////////////////////////
  // get G4Track information
  ///////////////////////////
  edm::Handle<edm::SimTrackContainer> G4TrkContainer;
  iEvent.getByToken( edmSimTrackContainerToken_, G4TrkContainer );
  if (!G4TrkContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillG4MC")
      << "Unable to find SimTrack in event!";
    return;
  }
  i = 0;
  edm::SimTrackContainer::const_iterator itTrk;
  for (itTrk = G4TrkContainer->begin(); itTrk != G4TrkContainer->end(); 
       ++itTrk) {

    ++i;

    double etaInit =0, phiInit =0, pInit =0;
    const CLHEP::HepLorentzVector& G4Trk = CLHEP::HepLorentzVector(itTrk->momentum().x(),
                                                     itTrk->momentum().y(),
                                                     itTrk->momentum().z(),
                                                     itTrk->momentum().e());
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
    G4TrkPt.push_back(sqrt(G4Trk[0]*G4Trk[0]+G4Trk[1]*G4Trk[1])); //GeV
    G4TrkE.push_back(G4Trk[3]);                                   //GeV
    G4TrkEta.push_back(etaInit);                                   
    G4TrkPhi.push_back(phiInit);                                   
  } 

  if (verbosity > 1) {
    eventout += "\n          Number of G4Tracks collected:              ";
    eventout += i;
  }  

  if (verbosity > 0)
    edm::LogInfo("TrackerHitProducer::fillG4MC") << eventout;

  return;
}

void TrackerHitProducer::storeG4MC(PTrackerSimHit& product)
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
    edm::LogInfo("TrackerHitProducer::storeG4MC") << eventout;
  } // end verbose output

  product.putRawGenPart(nRawGenPart);
  product.putG4Vtx(G4VtxX, G4VtxY, G4VtxZ);
  product.putG4Trk(G4TrkPt, G4TrkE, G4TrkEta, G4TrkPhi);

  return;
}

void TrackerHitProducer::fillTrk(edm::Event& iEvent, 
                const edm::EventSetup& iSetup)
{
  TString eventout;
  int sysID = 0;
  if (verbosity > 0)
    eventout = "\nGathering info:";  
//////////////////    
  // iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;
//  edm::PSimHitContainer theHits;

  ///////////////////////////////
  // get Pixel Barrel information
  ///////////////////////////////
  
  // extract low container
  edm::Handle<edm::PSimHitContainer> PxlBrlLowContainer;
  iEvent.getByToken( edmPSimHitContainer_pxlBrlLow_Token_, PxlBrlLowContainer );
  if (!PxlBrlLowContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsPixelBarrelLowTof in event!";
    return;
  }
    
  // place both containers into new container
//  theHits.insert(theHits.end(),PxlBrlLowContainer->begin(),
//             PxlBrlLowContainer->end());
               
  sysID = 100;   // TrackerHitsPixelBarrelLowTof
  int j = 0;               
  for (itHit = PxlBrlLowContainer->begin(); itHit != PxlBrlLowContainer->end(); ++itHit) {
//  for (itHit = theHits.begin(); itHit != theHits.end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through PxlBrl Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Pixel Barrel Low TOF Hits collected:     ";
    eventout += j;
  }  
  
//  theHits.clear();
    
  // extract high container
  edm::Handle<edm::PSimHitContainer> PxlBrlHighContainer;
  iEvent.getByToken( edmPSimHitContainer_pxlBrlHigh_Token_, PxlBrlHighContainer );
  if (!PxlBrlHighContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsPixelBarrelHighTof in event!";
    return;
  }
  
  
  sysID = 200;   // TrackerHitsPixelBarrelHighTof
  j = 0;               
  for (itHit = PxlBrlHighContainer->begin(); itHit != PxlBrlHighContainer->end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through PxlBrl Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Pixel Barrel High TOF Hits collected:     ";
    eventout += j;
  }  

  
  /////////////////////////////////
  // get Pixel Forward information
  ////////////////////////////////
  // extract low container
  edm::Handle<edm::PSimHitContainer> PxlFwdLowContainer;
  iEvent.getByToken( edmPSimHitContainer_pxlFwdLow_Token_, PxlFwdLowContainer );
  if (!PxlFwdLowContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsPixelEndcapLowTof in event!";
    return;
  }
  
  sysID = 300;   // TrackerHitsPixelEndcapLowTof
  j = 0;               
  for (itHit = PxlFwdLowContainer->begin(); itHit != PxlFwdLowContainer->end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through PxlFwd Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Pixel Forward Low TOF Hits collected:     ";
    eventout += j;
  }  
  
  
  // extract high container
  edm::Handle<edm::PSimHitContainer> PxlFwdHighContainer;
  iEvent.getByToken( edmPSimHitContainer_pxlFwdHigh_Token_, PxlFwdHighContainer );
  if (!PxlFwdHighContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsPixelEndcapHighTof in event!";
    return;
  }
  
  sysID = 400;   // TrackerHitsPixelEndcapHighTof
  j = 0;               
  for (itHit = PxlFwdHighContainer->begin(); itHit != PxlFwdHighContainer->end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through PxlFwd Hits

  if (verbosity > 1) {
    eventout += "\n          Number of Pixel Forward High TOF Hits collected:     ";
    eventout += j;
  }  
          
 
  ///////////////////////////////////
  // get Silicon TIB information
  //////////////////////////////////
  // extract TIB low container
  edm::Handle<edm::PSimHitContainer> SiTIBLowContainer;
  iEvent.getByToken( edmPSimHitContainer_siTIBLow_Token_, SiTIBLowContainer );
  if (!SiTIBLowContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsTIBLowTof in event!";
    return;
  }
  
  sysID = 10;   // TrackerHitsTIBLowTof
  j = 0;               
  for (itHit = SiTIBLowContainer->begin(); itHit != SiTIBLowContainer->end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through PxlFwd Hits

  if (verbosity > 1) {
    eventout += "\n          Number of TIB low TOF Hits collected:     ";
    eventout += j;
  }  
                 
  
  // extract TIB high container
  edm::Handle<edm::PSimHitContainer> SiTIBHighContainer;
  iEvent.getByToken( edmPSimHitContainer_siTIBHigh_Token_, SiTIBHighContainer );
  if (!SiTIBHighContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsTIBHighTof in event!";
    return;
  }

  sysID = 20;   // TrackerHitsTIBHighTof
  j = 0;               
  for (itHit = SiTIBHighContainer->begin(); itHit != SiTIBHighContainer->end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through PxlFwd Hits

  if (verbosity > 1) {
    eventout += "\n          Number of TIB high TOF Hits collected:     ";
    eventout += j;
  }  
  
  ///////////////////////////////////
  // get Silicon TOB information
  //////////////////////////////////
  // extract TOB low container
  edm::Handle<edm::PSimHitContainer> SiTOBLowContainer;
  iEvent.getByToken( edmPSimHitContainer_siTOBLow_Token_, SiTOBLowContainer );
  if (!SiTOBLowContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsTOBLowTof in event!";
    return;
  }
  
  sysID = 30;   // TrackerHitsTOBLowTof
  j = 0;               
  for (itHit = SiTOBLowContainer->begin(); itHit != SiTOBLowContainer->end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through PxlFwd Hits

  if (verbosity > 1) {
    eventout += "\n          Number of TOB low TOF Hits collected:     ";
    eventout += j;
  }  
    
  // extract TOB high container
  edm::Handle<edm::PSimHitContainer> SiTOBHighContainer;
  iEvent.getByToken( edmPSimHitContainer_siTOBHigh_Token_, SiTOBHighContainer );
  if (!SiTOBHighContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsTOBHighTof in event!";
    return;
  }
  
  sysID = 40;   // TrackerHitsTOBHighTof
  j = 0;               
  for (itHit = SiTOBHighContainer->begin(); itHit != SiTOBHighContainer->end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through SiTOB Hits

  if (verbosity > 1) {
    eventout += "\n          Number of TOB high TOF Hits collected:     ";
    eventout += j;
  }  
  
  ///////////////////////////////////
  // get Silicon TID information
  ///////////////////////////////////
  // extract TID low container
  edm::Handle<edm::PSimHitContainer> SiTIDLowContainer;
  iEvent.getByToken( edmPSimHitContainer_siTIDLow_Token_, SiTIDLowContainer );
  if (!SiTIDLowContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsTIDLowTof in event!";
    return;
  }
  
  sysID = 50;   // TrackerHitsTIDLowTof
  j = 0;               
  for (itHit = SiTIDLowContainer->begin(); itHit != SiTIDLowContainer->end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through SiTID Hits

  if (verbosity > 1) {
    eventout += "\n          Number of TID low TOF Hits collected:     ";
    eventout += j;
  }  
      
  // extract TID high container
  edm::Handle<edm::PSimHitContainer> SiTIDHighContainer;
  iEvent.getByToken( edmPSimHitContainer_siTIDHigh_Token_, SiTIDHighContainer );
  if (!SiTIDHighContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsTIDHighTof in event!";
    return;
  }
  
  sysID = 60;   // TrackerHitsTIDHighTof
  j = 0;               
  for (itHit = SiTIDHighContainer->begin(); itHit != SiTIDHighContainer->end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through SiTID Hits

  if (verbosity > 1) {
    eventout += "\n          Number of TID high TOF Hits collected:     ";
    eventout += j;
  }  
  
  ///////////////////////////////////
  // get Silicon TEC information
  /////////////////////////////////// 
  // extract TEC low container
  edm::Handle<edm::PSimHitContainer> SiTECLowContainer;
  iEvent.getByToken( edmPSimHitContainer_siTECLow_Token_, SiTECLowContainer );
  if (!SiTECLowContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsTECLowTof in event!";
    return;
  }
  
  sysID = 70;   // TrackerHitsTECLowTof
  j = 0;               
  for (itHit = SiTECLowContainer->begin(); itHit != SiTECLowContainer->end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through SiTEC Hits

  if (verbosity > 1) {
    eventout += "\n          Number of TEC low TOF Hits collected:     ";
    eventout += j;
  }  
      
  
  // extract TEC high container
  edm::Handle<edm::PSimHitContainer> SiTECHighContainer;
  iEvent.getByToken( edmPSimHitContainer_siTECHigh_Token_, SiTECHighContainer );
  if (!SiTECHighContainer.isValid()) {
    edm::LogError("TrackerHitProducer::fillTrk")
      << "Unable to find TrackerHitsTECHighTof in event!";
    return;
  }
  sysID = 80;   // TrackerHitsTECHighTof
  j = 0;               
  for (itHit = SiTECHighContainer->begin(); itHit != SiTECHighContainer->end(); ++itHit) {

      // gather necessary information
      ++j;
      HitsSysID.push_back(sysID);
      HitsDuID.push_back(itHit->detUnitId());
      HitsTkID.push_back(itHit->trackId());
      HitsProT.push_back(itHit->processType());
      HitsParT.push_back(itHit->particleType());
      HitsP.push_back(itHit->pabs());
      
      HitsLpX.push_back(itHit->localPosition().x());
      HitsLpY.push_back(itHit->localPosition().y());
      HitsLpZ.push_back(itHit->localPosition().z());

      HitsLdX.push_back(itHit->localDirection().x());
      HitsLdY.push_back(itHit->localDirection().y());
      HitsLdZ.push_back(itHit->localDirection().z());
      HitsLdTheta.push_back(itHit->localDirection().theta());
      HitsLdPhi.push_back(itHit->localDirection().phi());

      HitsExPx.push_back(itHit->exitPoint().x());
      HitsExPy.push_back(itHit->exitPoint().y());
      HitsExPz.push_back(itHit->exitPoint().z());

      HitsEnPx.push_back(itHit->entryPoint().x());
      HitsEnPy.push_back(itHit->entryPoint().y());
      HitsEnPz.push_back(itHit->entryPoint().z());

      HitsEloss.push_back(itHit->energyLoss());
      HitsToF.push_back(itHit->tof());
      
  } // end loop through SiTEC Hits

  if (verbosity > 1) {
    eventout += "\n          Number of TEC high TOF Hits collected:     ";
    eventout += j;
  }  

  if (verbosity > 0)
    edm::LogInfo("TrackerHitProducer::fillTrk") << eventout;

  return;
}

void TrackerHitProducer::storeTrk(PTrackerSimHit& product)
{

/*
  if (verbosity > 2) {
    TString eventout("\nnPxlBrlHits        = ");
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
    edm::LogInfo("TrackerHitProducer::storeTrk") << eventout;
  } // end verbose output
*/
  product.putHits(HitsSysID, HitsDuID, HitsTkID, HitsProT, HitsParT, HitsP,
                  HitsLpX, HitsLpY, HitsLpZ, 
          HitsLdX, HitsLdY, HitsLdZ, HitsLdTheta, HitsLdPhi,
          HitsExPx, HitsExPy, HitsExPz,
          HitsEnPx, HitsEnPy, HitsEnPz,
          HitsEloss, HitsToF);
  
  return;
}

void TrackerHitProducer::clear()
{
  if (verbosity > 0)
    edm::LogInfo("GlobalValProducer::clear")
      << "Clearing event holders"; 

  // reset G4MC info
  nRawGenPart = 0;
  G4VtxX.clear();
  G4VtxY.clear();
  G4VtxZ.clear();
  G4TrkPt.clear();
  G4TrkE.clear();
  G4TrkEta.clear();
  G4TrkPhi.clear();
  // reset tracker info
  HitsSysID.clear();
  HitsDuID.clear();
  HitsTkID.clear(); 
  HitsProT.clear(); 
  HitsParT.clear(); 
  HitsP.clear();
  HitsLpX.clear(); 
  HitsLpY.clear(); 
  HitsLpZ.clear(); 
  HitsLdX.clear(); 
  HitsLdY.clear(); 
  HitsLdZ.clear(); 
  HitsLdTheta.clear(); 
  HitsLdPhi.clear();
  HitsExPx.clear(); 
  HitsExPy.clear(); 
  HitsExPz.clear();
  HitsEnPx.clear(); 
  HitsEnPy.clear(); 
  HitsEnPz.clear();
  HitsEloss.clear(); 
  HitsToF.clear();

  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerHitProducer);
