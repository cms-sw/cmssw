// -*- C++ -*-
//
// Package:    SimTrackerDumper
// Class:      SimTrackSimVertexDumper
// 
/*
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
//


// system include files
#include <memory>

#include "SimG4Core/Application/test/SimTrackSimVertexDumper.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
 
SimTrackSimVertexDumper::SimTrackSimVertexDumper( const edm::ParameterSet& iConfig ):
  HepMCLabel(iConfig.getParameter<edm::InputTag>("moduleLabelHepMC")),
  SimTkLabel(iConfig.getParameter<edm::InputTag>("moduleLabelTk")),
  SimVtxLabel(iConfig.getParameter<edm::InputTag>("moduleLabelVtx")),
  dumpHepMC(iConfig.getUntrackedParameter<bool>("dumpHepMC","false"))
{

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
SimTrackSimVertexDumper::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;
   using namespace HepMC;

   std::vector<SimTrack> theSimTracks;
   std::vector<SimVertex> theSimVertexes;

   Handle<HepMCProduct> MCEvt;
   Handle<SimTrackContainer> SimTk;
   Handle<SimVertexContainer> SimVtx;

   iEvent.getByLabel(HepMCLabel, MCEvt);
   const HepMC::GenEvent* evt = MCEvt->GetEvent();


   iEvent.getByLabel(SimTkLabel,SimTk);
   iEvent.getByLabel(SimVtxLabel,SimVtx);

   theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
   theSimVertexes.insert(theSimVertexes.end(),SimVtx->begin(),SimVtx->end());

   std::cout << "\n SimVertex / SimTrack structure dump \n" << std::endl;
   std::cout << " SimVertex in the event = " << theSimVertexes.size() << std::endl;
   std::cout << " SimTracks in the event = " << theSimTracks.size() << std::endl;
   std::cout << "\n" << std::endl;
   for (unsigned int isimvtx = 0; isimvtx < theSimVertexes.size(); isimvtx++){
     std::cout << "SimVertex " << isimvtx << " = " << theSimVertexes[isimvtx] << "\n" << std::endl;
     for (unsigned int isimtk = 0; isimtk < theSimTracks.size() ; isimtk++ ) {
       if ( theSimTracks[isimtk].vertIndex() >= 0 && std::abs(theSimTracks[isimtk].vertIndex()) == (int)isimvtx ) {
         std::cout<<"  SimTrack " << isimtk << " = "<< theSimTracks[isimtk] 
		  <<" Track Id = "<<theSimTracks[isimtk].trackId()<< std::endl;

         // for debugging purposes
         if (dumpHepMC ) {
           if ( theSimTracks[isimtk].genpartIndex() != -1 ) {
             HepMC::GenParticle* part = evt->barcode_to_particle( theSimTracks[isimtk].genpartIndex() ) ;
             if ( part ) { std::cout << "  ---> Corresponding to HepMC particle " << *part << std::endl; }
             else { std::cout << " ---> Corresponding HepMC particle to barcode " << theSimTracks[isimtk].genpartIndex() << " not in selected event " << std::endl; }
           }
         }
       }
     }
     std::cout << "\n" << std::endl;
   }
   
   for (std::vector<SimTrack>::iterator isimtk = theSimTracks.begin();
        isimtk != theSimTracks.end(); ++isimtk){
     if(isimtk->noVertex()){
       std::cout<<"SimTrack without an associated Vertex = "<< *isimtk <<std::endl;
     }
   }
   
   return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimTrackSimVertexDumper);
