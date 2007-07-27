#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/Common/interface/Handle.h"


#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"
#include "SimGeneral/TrackingAnalysis/interface/TrackingElectronProducer.h"
#include "SimGeneral/TrackingAnalysis/interface/TkNavigableSimElectronAssembler.h"

#include <map>

using namespace edm;
using namespace std; 
using CLHEP::HepLorentzVector;

typedef TkNavigableSimElectronAssembler::VertexPtr VertexPtr;

TrackingElectronProducer::TrackingElectronProducer(const edm::ParameterSet &conf) {
  produces<TrackingVertexCollection>("ElectronVertexTruth");
  produces<TrackingParticleCollection>("ElectronTrackTruth");

  std::cout << " TrackingElectronProducer CTOR " << std::endl;


  conf_ = conf;
}


void TrackingElectronProducer::produce(Event &event, const EventSetup &) {

//  TimerStack timers;  // Don't need the timers now, left for example
//  timers.push("TrackingTruth:Producer");
//  timers.push("TrackingTruth:Setup");
  // Get information out of event record

  std::cout << " TrackingElectronProducer produce 1 " << std::endl;

  edm::Handle<TrackingParticleCollection>  TruthTrackContainer ;
  //  event.getByLabel("trackingtruthprod","TrackingTruthProducer", 
  //		   trackingParticleHandle);
  event.getByType(TruthTrackContainer );
  std::cout << " TrackingElectronProducer produce 1.5 " << std::endl;
  
  const TrackingParticleCollection *etPC   = TruthTrackContainer.product();
  std::cout << " TrackingElectronProducer produce 2 " << std::endl;

    // now dumping electrons only
  listElectrons(*etPC);

  std::cout << " TrackingElectronProducer produce 3 " << std::endl;
  // now calling electron assembler and dumping assembled electrons 
  cout << "TrackingElectronProducer::now assembling electrons..." << endl;
  TkNavigableSimElectronAssembler assembler;
  std::vector<TrackingParticle*> particles;
  for (TrackingParticleCollection::const_iterator t = etPC -> begin(); t != etPC -> end(); ++t) {
    particles.push_back(const_cast<TrackingParticle*>(&(*t)));
  }

  TkNavigableSimElectronAssembler::ElectronList 
    electrons = assembler.assemble(particles);

  std::cout << "Electron segments found, now linking them " << std::endl;

  //
  // now add electron tracks and vertices to event
  //
  // prepare collections
  //
  std::auto_ptr<TrackingParticleCollection> trackingParticles(new TrackingParticleCollection);
  std::auto_ptr<TrackingVertexCollection> trackingVertices(new TrackingVertexCollection);

  std::cout << "now getting refprods " << std::endl;

  edm::RefProd<TrackingParticleCollection> refTPC =
    event.getRefBeforePut<TrackingParticleCollection>("ElectronTrackTruth");
  edm::RefProd<TrackingVertexCollection>   refTVC =
    event.getRefBeforePut<TrackingVertexCollection>("ElectronVertexTruth");

  std::cout << "now creating tk vertices " << std::endl;

  //
  // create TrackingVertices
  //
  typedef std::map<TrackingVertex const *,int> TVMap;
  TVMap trackingVertexMap;
  int ntv(0);
  // loop over electrons
  for ( TkNavigableSimElectronAssembler::ElectronList::const_iterator ie 
	  = electrons.begin(); ie != electrons.end(); ie++ ) {

    cout << "in loop on electron list" << endl;

    // store parent vertex of first track segment 
    // and decay vertex of last track segment
    TrackingVertexRef parentV = (*(*ie).front()).parentVertex();

    cout << "has got parent vertex" << endl;
    if (parentV.isNonnull()) {
    cout << "parent vertex is non null" << endl;
      (*trackingVertices).push_back(*parentV);
      trackingVertexMap[parentV.get()] = ntv++;
    }

    // for 131
    //    TrackingVertexRef decayV = (*(*ie).back()).decayVertex();    

    TrackingVertexRefVector decayVertices = (*(*ie).back()).decayVertices();
    if ( !decayVertices.empty() ) {

      // get first decay vertex
      TrackingVertexRef decayV = decayVertices.at(0);
      if (decayV.isNonnull()) {
	cout << "making map: decay vertex is non null" << endl;
	(*trackingVertices).push_back(*decayV);
	trackingVertexMap[decayV.get()] = ntv++;
      }
      else {
	cout << "making map: decay vertex is null" << endl;
      }
    }
  }
   
  //
  // create TrackingParticles
  //
  cout << "now creating tracking particles" << endl;
  typedef std::map<TrackingParticle const *,int> TPMap;
  TPMap trackingParticleMap;
  int ntp(0);
  // loop over electrons
  for ( TkNavigableSimElectronAssembler::ElectronList::const_iterator ie 
	  = electrons.begin(); ie != electrons.end(); ie++ ) {

    // create TrackingParticle from first track segment 
    TrackingParticle * tk = (*ie).front();
    CLHEP::HepLorentzVector hep4Pos = (*(*tk).parentVertex()).position();
    TrackingParticle::Point pos(hep4Pos.x(), hep4Pos.y(), hep4Pos.z());
    TrackingParticle tkp( (*tk).charge(), (*tk).p4(), pos, hep4Pos.t(), 
			  (*tk).pdgId(), (*tk).eventId() );

    // add G4 tracks and hits of all segments
    int ngenp = 0;
    for ( TkNavigableSimElectronAssembler::TrackList::const_iterator it 
	    = (*ie).begin(); it != (*ie).end(); it++ ) {

      for (TrackingParticle::genp_iterator uz=(*it)->genParticle_begin();
	   uz!=(*it)->genParticle_end();uz++) {
	tkp.addGenParticle(*uz);
	ngenp++;
      }
      addG4Track(tkp, *it);

    }
    if (ngenp > 1) cout << "ERROR::TrackingElectron has more than 1 GenParticle" << endl << "Nb of associated GenParticles = " << ngenp << endl;

    //    std::vector<PSimHit> hits = tkp.trackPSimHit();
    /*
    // count matched hits
    int totsimhit = 0;
    int oldlay = 0;
    int newlay = 0;
    int olddet = 0;
    int newdet = 0;
    for ( std::vector<PSimHit>::const_iterator ih = hits.begin(); 
	  ih != hits.end(); ih++ ) {
      unsigned int detid = (*ih).detUnitId();
      DetId detId = DetId(detid);
      oldlay = newlay;
      olddet = newdet;
      newlay = layerFromDetid(detid);
      newdet = detId.subdetId();
      
      // Count hits using layers for glued detectors
      if (oldlay != newlay || (oldlay==newlay && olddet!=newdet) ) {
	totsimhit++;
      }
    }
    */
    int totsimhit = 20; // FIXME temp. hack
    tkp.setMatchedHit(totsimhit);

    (*trackingParticles).push_back(tkp);
    trackingParticleMap[tk] = ntp++;
  }

  //
  // add references to vertices
  //

  cout << "Dumping assembled electrons..." << endl;

  cout << "Storing electron tracks" << endl;
  event.put(trackingParticles,"ElectronTrackTruth");

} 


void TrackingElectronProducer::listElectrons(
  const TrackingParticleCollection & tPC) const
{
  cout << "TrackingElectronProducer::printing electrons before assembly..." 
       << endl;
  for (TrackingParticleCollection::const_iterator it = tPC.begin();
       it != tPC.end(); it++) {
    if (abs((*it).pdgId()) == 11) {
      cout << "Electron: sim tk " << (*it).g4Tracks().front().trackId() 
	   << endl;
      
      TrackingVertexRef parentV = (*it).parentVertex();
      if (parentV.isNull()) {
	cout << " No parent vertex" << endl;
      } else {  
	cout << " Parent  vtx position " << parentV -> position() << endl;
      }  
      
      TrackingVertexRefVector decayVertices = (*it).decayVertices();
      if ( decayVertices.empty() ) {
	cout << " No decay vertex" << endl;
      } else {  
	cout << " Decay vtx position " 
	     << decayVertices.at(0) -> position() << endl;
      } 
    }
  }
}


void 
TrackingElectronProducer::addG4Track(TrackingParticle& e, 
				     const TrackingParticle * s) const
{

  // add G4 tracks
  std::vector<SimTrack> g4Tracks = (*s).g4Tracks();
  for ( std::vector<SimTrack>::const_iterator ig4 = g4Tracks.begin(); 
	ig4 != g4Tracks.end(); ig4++ ) {
    e.addG4Track(*ig4);
  }

  // add hits 
  // FIXME configurable for dropping delta-ray hits
  std::vector< PSimHit > hits = (*s).trackPSimHit();
  for ( std::vector<PSimHit>::const_iterator ih = hits.begin(); 
	ih != hits.end(); ih++ ) {
    e.addPSimHit(*ih);
  }
}


int TrackingElectronProducer::layerFromDetid(const unsigned int& detid ) {
  DetId detId = DetId(detid);
  int layerNumber=0;
  unsigned int subdetId = static_cast<unsigned int>(detId.subdetId());
  if ( subdetId == StripSubdetector::TIB)
    {
      TIBDetId tibid(detId.rawId());
      layerNumber = tibid.layer();
    }
  else if ( subdetId ==  StripSubdetector::TOB )
    {
      TOBDetId tobid(detId.rawId());
      layerNumber = tobid.layer();
    }
  else if ( subdetId ==  StripSubdetector::TID)
    {
      TIDDetId tidid(detId.rawId());
      layerNumber = tidid.wheel();
    }
  else if ( subdetId ==  StripSubdetector::TEC )
    {
      TECDetId tecid(detId.rawId());
      layerNumber = tecid.wheel();
    }
  else if ( subdetId ==  PixelSubdetector::PixelBarrel )
    {
      PXBDetId pxbid(detId.rawId());
      layerNumber = pxbid.layer();
    }
  else if ( subdetId ==  PixelSubdetector::PixelEndcap )
    {
      PXFDetId pxfid(detId.rawId());
      layerNumber = pxfid.disk();
    }
  else
    edm::LogVerbatim("TrackingTruthProducer") << "Unknown subdetid: " <<  subdetId;

  return layerNumber;
}


// DEFINE_FWK_MODULE(TrackingElectronProducer);
