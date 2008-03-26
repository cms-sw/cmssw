#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/VertexAssociation/interface/VertexAssociatorByTracks.h"

using namespace reco;
using namespace std;

/* Constructor */
VertexAssociatorByTracks::VertexAssociatorByTracks (const edm::ParameterSet& conf) :
  conf_(conf) {}


/* Destructor */
VertexAssociatorByTracks::~VertexAssociatorByTracks() {
  //do cleanup here
}

//
//---member functions
//

VertexRecoToSimCollection VertexAssociatorByTracks::associateRecoToSim(
    edm::Handle<reco::VertexCollection>& vertexCollectionH,
    edm::Handle<TrackingVertexCollection>&  TVCollectionH,
    const edm::Event& event,
    reco::RecoToSimCollection& trackAssocResult) const {

//  const double minHitFraction = theMinHitFraction;
//  int nshared =0;
//  float fraction=0;
//  std::vector<unsigned int> SimTrackIds;
//  std::vector<unsigned int> matchedIds;

  using reco::VertexRef;

  VertexRecoToSimCollection  outputCollection;

  const TrackingVertexCollection tVC = *(TVCollectionH.product());
  const   reco::VertexCollection  vC = *(vertexCollectionH.product());

//  double minFraction = 0.01;
//  double fraction = 0.8;

  std::map<TrackingVertexRef,int> tVCount;

  int iv = 0;

  // Loop over reco::Vertex

  for (reco::VertexCollection::const_iterator vertex = vC.begin();
       vertex != vC.end(); ++vertex, ++iv) {
    tVCount.clear();
    VertexRef rVertexR = VertexRef(vertexCollectionH,iv);
    double nRecoTracks = vertex->tracksSize();

    // Loop over daughter tracks of reco::Vertex

    for (reco::Vertex::trackRef_iterator recoDaughter = vertex->tracks_begin();
         recoDaughter != vertex->tracks_end(); ++recoDaughter) {
      TrackRef tr = recoDaughter->castTo<TrackRef>();
      if (trackAssocResult[tr].size() > 0) {
        std::vector<std::pair<TrackingParticleRef, double> > tpV = trackAssocResult[tr];

        // Loop over TrackingParticles associated with reco::Track

        for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator match = tpV.begin();
             match != tpV.end(); ++match) {
          // ... and keep count of it's parent vertex
          TrackingParticleRef tp = match->first;
//          double trackFraction = match->second;
          TrackingVertexRef   tv = tp->parentVertex();
          ++tVCount[tv]; // Count matches to this reco:Vertex for this TrackingVertex
        }
      }
    }

    // Loop over map, set score, add to outputCollection

    for (std::map<TrackingVertexRef,int>::const_iterator match = tVCount.begin();
         match != tVCount.end(); ++match) {
      TrackingVertexRef tV = match->first;
      double nMatches      = match->second;
      outputCollection.insert(rVertexR,std::make_pair(tV,nMatches/nRecoTracks));
    }
  } // Loop on reco::Vertex

  return outputCollection;
}


VertexSimToRecoCollection VertexAssociatorByTracks::associateSimToReco(
    edm::Handle<reco::VertexCollection>&   vertexCollectionH,
    edm::Handle<TrackingVertexCollection>& TVCollectionH,
    const edm::Event& e,
    reco::SimToRecoCollection& trackAssocResult) const {

  const TrackingVertexCollection tVC = *(TVCollectionH.product());
  const   reco::VertexCollection  vC = *(vertexCollectionH.product());

  VertexSimToRecoCollection  outputCollection; // return value

  // Loop over TrackingVertexes

  std::map<VertexRef,int> vCount;
  int iTV = 0;
  for (TrackingVertexCollection::const_iterator tV = tVC.begin();
       tV != tVC.end(); ++tV, ++iTV) {
    vCount.clear();
    TrackingVertexRef tVertexR = TrackingVertexRef(TVCollectionH,iTV);
    double nSimTracks = (tV->daughterTracks()).size();

    // Loop over daughter tracks of TrackingVertex
    for (TrackingVertex::tp_iterator simDaughter = tV->daughterTracks_begin();
         simDaughter != tV->daughterTracks_end(); ++simDaughter) {
      TrackingParticleRef tp = *simDaughter;

      SimToRecoCollection::const_iterator daughterPosition = trackAssocResult.find(*simDaughter);
      if (daughterPosition != trackAssocResult.end()) {
        std::vector<std::pair<TrackRef, double> > recoTracks = trackAssocResult[*simDaughter];

       // Loop over reco::Tracks associated with TrackingParticle
        for (std::vector<std::pair<TrackRef, double> >::const_iterator match = recoTracks.begin();
             match != recoTracks.end(); ++match) {
          // ... and keep count of it's parent vertex

          TrackBaseRef track(match->first);
//          double   trackQuality = match->second;

          // Find vertex if any where this track comes from
          int iv = 0;
          for (reco::VertexCollection::const_iterator vertex = vC.begin();
              vertex != vC.end(); ++vertex,++iv) {
            VertexRef rVertexR = VertexRef(vertexCollectionH,iv);
            for (reco::Vertex::trackRef_iterator recoDaughter = vertex->tracks_begin();
                 recoDaughter != vertex->tracks_end(); ++recoDaughter) {
              if (*recoDaughter == track) {
                ++vCount[rVertexR]; // Count matches to this TrackingVertex for this reco:Vertex
              }
            }
          }
        }
      }
    }

    // Loop over map, set score, add to outputCollection
    for (std::map<VertexRef,int>::const_iterator match = vCount.begin(); match != vCount.end(); ++match) {
      VertexRef v = match->first;
      double nMatches      = match->second;
      outputCollection.insert(tVertexR,std::make_pair(v,nMatches/nSimTracks));
    }
  }

  return outputCollection;
}

