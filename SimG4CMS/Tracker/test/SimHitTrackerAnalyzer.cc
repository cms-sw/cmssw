// -*- C++ -*-
//
// Package:    SimHitTrackerAnalyzer
// Class:      SimHitTrackerAnalyzer
// 
/**\class SimHitTrackerAnalyzer SimHitTrackerAnalyzer.cc test/SimHitTrackerAnalyzer/src/SimHitTrackerAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tommaso Boccali
//         Created:  Tue Jul 26 08:47:57 CEST 2005
// $Id: SimHitTrackerAnalyzer.cc,v 1.3 2006/02/13 14:59:25 fambrogl Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "SimDataFormats/Track/interface/EmbdSimTrack.h"
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertex.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"

//
//
// class decleration
//

class SimHitTrackerAnalyzer : public edm::EDAnalyzer {
   public:
      explicit SimHitTrackerAnalyzer( const edm::ParameterSet& );
      ~SimHitTrackerAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
      // ----------member data ---------------------------
  std::string HepMCLabel;
  std::string SimTkLabel;
  std::string SimVtxLabel;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SimHitTrackerAnalyzer::SimHitTrackerAnalyzer( const edm::ParameterSet& iConfig ):
  HepMCLabel(iConfig.getUntrackedParameter("moduleLabelMC",std::string("PythiaSource"))),
  SimTkLabel(iConfig.getUntrackedParameter("moduleLabelTk",std::string("EmbdSimTrack"))),
  SimVtxLabel(iConfig.getUntrackedParameter("moduleLabelVtx",std::string("EmbdSimVertex")))
{
   //now do what ever initialization is needed

}


SimHitTrackerAnalyzer::~SimHitTrackerAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
SimHitTrackerAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   std::vector<PSimHit> theTrackerHits;
   std::vector<EmbdSimTrack> theSimTracks;
   std::vector<EmbdSimVertex> theSimVertexes;

   Handle<HepMCProduct> MCEvt;
   Handle<EmbdSimTrackContainer> SimTk;
   Handle<EmbdSimVertexContainer> SimVtx;
   Handle<PSimHitContainer> PixelBarrelHitsLowTof;
   Handle<PSimHitContainer> PixelBarrelHitsHighTof;
   Handle<PSimHitContainer> PixelEndcapHitsLowTof;
   Handle<PSimHitContainer> PixelEndcapHitsHighTof;
   Handle<PSimHitContainer> TIBHitsLowTof;
   Handle<PSimHitContainer> TIBHitsHighTof;
   Handle<PSimHitContainer> TIDHitsLowTof;
   Handle<PSimHitContainer> TIDHitsHighTof;
   Handle<PSimHitContainer> TOBHitsLowTof;
   Handle<PSimHitContainer> TOBHitsHighTof;
   Handle<PSimHitContainer> TECHitsLowTof;
   Handle<PSimHitContainer> TECHitsHighTof;


   iEvent.getByLabel(HepMCLabel, MCEvt);
   iEvent.getByLabel(SimTkLabel,SimTk);
   iEvent.getByLabel(SimVtxLabel,SimVtx);
   iEvent.getByLabel("SimG4Object","TrackerHitsPixelBarrelLowTof", PixelBarrelHitsLowTof);
   iEvent.getByLabel("SimG4Object","TrackerHitsPixelBarrelHighTof", PixelBarrelHitsHighTof);
   iEvent.getByLabel("SimG4Object","TrackerHitsPixelEndcapLowTof", PixelEndcapHitsLowTof);
   iEvent.getByLabel("SimG4Object","TrackerHitsPixelEndcapHighTof", PixelEndcapHitsHighTof);
   iEvent.getByLabel("SimG4Object","TrackerHitsTIBLowTof", TIBHitsLowTof);
   iEvent.getByLabel("SimG4Object","TrackerHitsTIBHighTof", TIBHitsHighTof);
   iEvent.getByLabel("SimG4Object","TrackerHitsTIDLowTof", TIDHitsLowTof);
   iEvent.getByLabel("SimG4Object","TrackerHitsTIDHighTof", TIDHitsHighTof);
   iEvent.getByLabel("SimG4Object","TrackerHitsTOBLowTof", TOBHitsLowTof);
   iEvent.getByLabel("SimG4Object","TrackerHitsTOBHighTof", TOBHitsHighTof);
   iEvent.getByLabel("SimG4Object","TrackerHitsTECLowTof", TECHitsLowTof);
   iEvent.getByLabel("SimG4Object","TrackerHitsTECHighTof", TECHitsHighTof);


   theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
   theSimVertexes.insert(theSimVertexes.end(),SimVtx->begin(),SimVtx->end());
   theTrackerHits.insert(theTrackerHits.end(), PixelBarrelHitsLowTof->begin(), PixelBarrelHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), PixelBarrelHitsHighTof->begin(), PixelBarrelHitsHighTof->end());
   theTrackerHits.insert(theTrackerHits.end(), PixelEndcapHitsLowTof->begin(), PixelEndcapHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), PixelEndcapHitsHighTof->begin(), PixelEndcapHitsHighTof->end());
   theTrackerHits.insert(theTrackerHits.end(), TIBHitsLowTof->begin(), TIBHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), TIBHitsHighTof->begin(), TIBHitsHighTof->end());
   theTrackerHits.insert(theTrackerHits.end(), TIDHitsLowTof->begin(), TIDHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), TIDHitsHighTof->begin(), TIDHitsHighTof->end());
   theTrackerHits.insert(theTrackerHits.end(), TOBHitsLowTof->begin(), TOBHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), TOBHitsHighTof->begin(), TOBHitsHighTof->end());
   theTrackerHits.insert(theTrackerHits.end(), TECHitsLowTof->begin(), TECHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), TECHitsHighTof->begin(), TECHitsHighTof->end());


   edm::ESHandle<TrackingGeometry> pDD;
   
   iSetup.get<TrackerDigiGeometryRecord> ().get(pDD);
   


   HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(MCEvt->GetEvent()));
   
   for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
	 p != myGenEvent->particles_end(); ++p ) {
     std::cout<< "Particle type form MC = "<< abs((*p)->pdg_id()) << std::endl; 
     std::cout<< "Particle momentum Pt form MC = "<< (*p)->momentum().perp() << std::endl;  
   }


   for (std::vector<EmbdSimTrack>::iterator isimtk = theSimTracks.begin();
	isimtk != theSimTracks.end(); ++isimtk){
     std::cout<<" Track momentum  x = "<<isimtk->momentum().x() <<" y = "<<isimtk->momentum().y() <<" z = "<< isimtk->momentum().z()<<std::endl;
     std::cout<<" Track momentum Ptx = "<<isimtk->momentum().perp() <<std::endl;
   }

   for (std::vector<EmbdSimVertex>::iterator isimvtx = theSimVertexes.begin();
	isimvtx != theSimVertexes.end(); ++isimvtx){
     std::cout<<" Vertex position  x = "<<isimvtx->position().x() <<" y = "<<isimvtx->position().y() <<" z = "<< isimvtx->position().z()<<std::endl;
   }


   std::map<unsigned int, std::vector<PSimHit>,std::less<unsigned int> > SimHitMap;

   for (std::vector<PSimHit>::iterator isim = theTrackerHits.begin();
	isim != theTrackerHits.end(); ++isim){
     SimHitMap[(*isim).detUnitId()].push_back((*isim));
     std::cout<<" SimHit position  x = "<<isim->localPosition().x() <<" y = "<<isim->localPosition().y() <<" z = "<< isim->localPosition().z()<<std::endl;
     std::cout<<" SimHit DetID = "<<isim->detUnitId()<<std::endl;	
     std::cout<<" Time of flight = "<<isim->timeOfFlight()<<std::endl;
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimHitTrackerAnalyzer)
