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
typedef math::XYZTLorentzVectorD LorentzVector;

TrackingElectronProducer::TrackingElectronProducer(const edm::ParameterSet &conf) {
  produces<TrackingParticleCollection>("ElectronTrackTruth");

//  std::cout << " TrackingElectronProducer CTOR " << std::endl;


  conf_ = conf;
}


void TrackingElectronProducer::produce(Event &event, const EventSetup &) {

  // Get information out of event record

  edm::Handle<TrackingParticleCollection>  TruthTrackContainer ;
  event.getByType(TruthTrackContainer );

  TkNavigableSimElectronAssembler assembler;
  std::vector<TrackingParticle*> particles;

  for (TrackingParticleCollection::const_iterator t = TruthTrackContainer->begin();
       t != TruthTrackContainer->end(); ++t) {
    particles.push_back(const_cast<TrackingParticle*>(&(*t)));
  }

  TkNavigableSimElectronAssembler::ElectronList electrons = assembler.assemble(particles);

  //
  // now add electron tracks and vertices to event
  //
  // prepare collections
  //
  std::auto_ptr<TrackingParticleCollection> trackingParticles(new TrackingParticleCollection);

  edm::RefProd<TrackingParticleCollection> refTPC =
    event.getRefBeforePut<TrackingParticleCollection>("ElectronTrackTruth");

  //
  // create TrackingParticles
  //

  int totsimhit = 0; //fixing the electron hits
 // loop over electrons
  for ( TkNavigableSimElectronAssembler::ElectronList::const_iterator ie
          = electrons.begin(); ie != electrons.end(); ie++ ) {

    // create TrackingParticle from first track segment
    TrackingParticle * tk = (*ie).front();
    LorentzVector hep4Pos = (*(*tk).parentVertex()).position();
    TrackingParticle::Point pos(hep4Pos.x(), hep4Pos.y(), hep4Pos.z());
    TrackingParticle tkp( (*tk).charge(), (*tk).p4(), pos, hep4Pos.t(),
                          (*tk).pdgId(), (*tk).status(), (*tk).eventId() );

    // add G4 tracks and hits of all segments
    int ngenp = 0;
    totsimhit = 0;//initialize the number of matchedHits for each track
    for (TkNavigableSimElectronAssembler::TrackList::const_iterator it = (*ie).begin();
         it != (*ie).end(); it++ ) {

      for (TrackingParticle::genp_iterator uz=(*it)->genParticle_begin();
           uz!=(*it)->genParticle_end();uz++) {
        tkp.addGenParticle(*uz);
        ngenp++;
      }
      addG4Track(tkp, *it);
      totsimhit +=(*tk).matchedHit();
      
      /*
	std::cout << "Electron list of tracks Original Segment  = " << (*tk) 
	<< "\t Matched Hits = " << (*tk).matchedHit() 
	<< "\t SimHits = " << (*tk).trackPSimHit().size() 
	<< std::endl;
      */
      
    }

    //    int totsimhit = 20; // FIXME temp. hack
    tkp.setMatchedHit(totsimhit);

    //
    // add references to parent and decay vertices
    //
    TrackingVertexRef parentV = (*(*ie).front()).parentVertex();
    if (parentV.isNonnull()) {
      tkp.setParentVertex(parentV);
    }

    TrackingVertexRefVector decayVertices = (*(*ie).back()).decayVertices();
    if ( !decayVertices.empty() ) {
      // get first decay vertex
      TrackingVertexRef decayV = decayVertices.at(0);
      tkp.addDecayVertex(decayV);
    }

    //
    // put particle in transient store
    //
    (*trackingParticles).push_back(tkp);
  }

  event.put(trackingParticles,"ElectronTrackTruth");

}


void TrackingElectronProducer::listElectrons(
  const TrackingParticleCollection & tPC) const
{
//  cout << "TrackingElectronProducer::printing electrons before assembly..."
//       << endl;
  for (TrackingParticleCollection::const_iterator it = tPC.begin();
       it != tPC.end(); it++) {
    if (abs((*it).pdgId()) == 11) {
//      cout << "Electron: sim tk " << (*it).g4Tracks().front().trackId()
//         << endl;

      TrackingVertexRef parentV = (*it).parentVertex();
//      if (parentV.isNull()) {
//      cout << " No parent vertex" << endl;
//      } else {
//      cout << " Parent  vtx position " << parentV -> position() << endl;
//      }

      TrackingVertexRefVector decayVertices = (*it).decayVertices();
//      if ( decayVertices.empty() ) {
//      cout << " No decay vertex" << endl;
//      } else {
//      cout << " Decay vtx position "
//           << decayVertices.at(0) -> position() << endl;
//      }
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
