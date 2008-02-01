#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"

#include "SimGeneral/TrackingAnalysis/interface/MergedTruthProducer.h"

#include <map>
#include <set>

using namespace edm;
using namespace std;

MergedTruthProducer::MergedTruthProducer(const edm::ParameterSet &conf) {
  produces<TrackingVertexCollection>("MergedTrackTruth");
  produces<TrackingParticleCollection>("MergedTrackTruth");

  conf_ = conf;
  MessageCategory_ = "MergedTruthProducer";

}

void MergedTruthProducer::produce(Event &event, const EventSetup &) {

  edm::Handle<TrackingParticleCollection> elecPH;
  edm::Handle<TrackingParticleCollection> rawPH;
  edm::Handle<TrackingVertexCollection>   rawVH;

  event.getByLabel("trackingtruthprod",rawPH);
  event.getByLabel("trackingtruthprod",rawVH);
  event.getByLabel("electrontruth","ElectronTrackTruth",    elecPH);

//  std::auto_ptr<TrackingParticleCollection>  trackCollection(new TrackingParticleCollection(rawPH.product()));
//  std::auto_ptr<TrackingParticleCollection>   elecCollection(new TrackingParticleCollection(elecPH.product()));
//  std::auto_ptr<TrackingVertexCollection>   vertexCollection(new TrackingVertexCollection(rawVH.product()));

// Create collections of things we will put in event and size appropriately
  auto_ptr<TrackingParticleCollection> tPC(new TrackingParticleCollection);
  auto_ptr<TrackingVertexCollection>   tVC(new TrackingVertexCollection  );
  tPC->reserve(rawPH->size());
  tVC->reserve(rawVH->size());

// Get references before put so we can cross reference
  TrackingParticleRefProd refTPC = event.getRefBeforePut<TrackingParticleCollection>("MergedTrackTruth");
  TrackingVertexRefProd   refTVC = event.getRefBeforePut<TrackingVertexCollection>("MergedTrackTruth");

  std::set<EncodedTruthId> electronGID;  // Keeps track of GeantIDs of electron tracks we've added

// Copy vertices discarding parent & child tracks

  for (TrackingVertexCollection::const_iterator iVertex = rawVH->begin(); iVertex != rawVH->end(); ++iVertex) {
    TrackingVertex newVertex = (*iVertex);
    newVertex.clearDaughterTracks();
    newVertex.clearParentTracks();
    tVC->push_back(newVertex);
  }

  uint eIndex = 0;
  for (TrackingParticleCollection::const_iterator iTrack = elecPH->begin(); iTrack != elecPH->end(); ++iTrack, ++eIndex) {

// Copy references from old vertex, set on new vertex, see comments in next loop

    TrackingVertexRef       sourceV = iTrack->parentVertex();
    TrackingVertexRefVector decayVs = iTrack->decayVertices();
    TrackingParticle newTrack = *iTrack;
    newTrack.clearParentVertex();
    newTrack.clearDecayVertices();
    uint parentIndex = sourceV.key();
    newTrack.setParentVertex(TrackingVertexRef(refTVC,parentIndex));
    (tVC->at(parentIndex)).addDaughterTrack(TrackingParticleRef(refTPC,eIndex));
    for (TrackingVertexRefVector::const_iterator iDecayV = decayVs.begin(); iDecayV != decayVs.end(); ++iDecayV) {
      uint daughterIndex = iDecayV->key();
      newTrack.addDecayVertex(TrackingVertexRef(refTVC,daughterIndex));
      (tVC->at(daughterIndex)).addParentTrack(TrackingParticleRef(refTPC,eIndex));
    }
    tPC->push_back(newTrack);

// Keep track of which tracks we did so we can skip later

    for (TrackingParticle::g4t_iterator g4T = iTrack->g4Track_begin(); g4T !=  iTrack->g4Track_end(); ++g4T) {
      uint GID = g4T->trackId();
      if (GID) {
        electronGID.insert(EncodedTruthId(iTrack->eventId(),GID));
      }
    }
  }

  for (TrackingParticleCollection::const_iterator iTrack = rawPH->begin(); iTrack != rawPH->end(); ++iTrack) {
    bool addTrack = false;
    for (TrackingParticle::g4t_iterator g4T = iTrack->g4Track_begin(); g4T !=  iTrack->g4Track_end(); ++g4T) {
      uint GID = g4T->trackId();
      if (electronGID.count(EncodedTruthId(iTrack->eventId(),GID))) {
        // Do nothing
      } else {
        addTrack = true;
      }
    }

    if (addTrack) { // Skip tracks that were in electron list
      TrackingVertexRef       sourceV = iTrack->parentVertex();
      TrackingVertexRefVector decayVs = iTrack->decayVertices();
      TrackingParticle newTrack = *iTrack;
      newTrack.clearParentVertex();
      newTrack.clearDecayVertices();

      // Set vertex indices for new vertex product and track references in those vertices

      uint parentIndex = sourceV.key(); // Index of parent vertex in vertex container
      uint tIndex      = tPC->size();   // Index of this track in track container
      newTrack.setParentVertex(TrackingVertexRef(refTVC,parentIndex));             // Add vertex to track
      (tVC->at(parentIndex)).addDaughterTrack(TrackingParticleRef(refTPC,tIndex)); // Add track to vertex
      for (TrackingVertexRefVector::const_iterator iDecayV = decayVs.begin(); iDecayV != decayVs.end(); ++iDecayV) {
        uint daughterIndex = iDecayV->key();
        newTrack.addDecayVertex(TrackingVertexRef(refTVC,daughterIndex));            // Add vertex to track
        (tVC->at(daughterIndex)).addParentTrack(TrackingParticleRef(refTPC,tIndex)); // Add track to vertex
      }
      tPC->push_back(newTrack);
    }
  }

// Put TrackingParticles and TrackingVertices in event

  event.put(tPC,"MergedTrackTruth");
  event.put(tVC,"MergedTrackTruth");

//  timers.pop();
//  timers.pop();
}

