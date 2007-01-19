#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Handle.h"
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
  conf_(conf)//,
//  theMinHitFraction(conf_.getParameter<double>("MinHitFraction"))
{}


/* Destructor */
VertexAssociatorByTracks::~VertexAssociatorByTracks()
{
  //do cleanup here
}

//
//---member functions
//

VertexRecoToSimCollection VertexAssociatorByTracks::associateRecoToSim(
    edm::Handle<reco::VertexCollection>& vertexCollectionH,
    edm::Handle<TrackingVertexCollection>&  TVCollectionH,     
    const edm::Event& event, 
    reco::RecoToSimCollection& trackAssocResult)
{

//  const double minHitFraction = theMinHitFraction;
//  int nshared =0;
//  float fraction=0;
//  std::vector<unsigned int> SimTrackIds;
//  std::vector<unsigned int> matchedIds; 
  
  using reco::VertexRef;
  
  VertexRecoToSimCollection  outputCollection;
  
  const TrackingVertexCollection tVC = *(TVCollectionH.product());
  const   reco::VertexCollection  vC = *(vertexCollectionH.product()); 

  double minFraction = 0.01;  
  double fraction = 0.8;

  std::map<TrackingVertexRef,int> tVCount; 
  
  int iv = 0;
  for (reco::VertexCollection::const_iterator vertex = vC.begin(); 
       vertex != vC.end(); ++vertex, ++iv) {
    tVCount.clear();     
    VertexRef rVertexR = VertexRef(vertexCollectionH,iv);
    double nRecoTracks = vertex->tracksSize();
    
    // Loop over daughter tracks of reco::Vertex
    
    for (reco::track_iterator recoDaughter = vertex->tracks_begin();
         recoDaughter != vertex->tracks_end(); ++recoDaughter) {
      // If matched to any reco::Track in track associator output loop there ...    
      if (trackAssocResult[*recoDaughter].size() > 0) {
        std::vector<std::pair<TrackingParticleRef, double> > tpV = trackAssocResult[*recoDaughter];
        for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator match = tpV.begin();
             match != tpV.end(); ++match) {
          // ... and keep count of it's parent vertex
          TrackingParticleRef tp = match->first;
          double trackFraction = match->second;
          cout << "Track fraction: " << trackFraction << endl;
          TrackingVertexRef tv = tp->parentVertex();
          ++tVCount[tv];       
        }
      }  
    } 
    
    // Loop over map, set score, add to outputCollection
          
    for (std::map<TrackingVertexRef,int>::const_iterator match = tVCount.begin();
         match != tVCount.end(); ++match) {
      TrackingVertexRef tV = match -> first;
      double nMatches = match -> second;
      outputCollection.insert(rVertexR,std::make_pair(tV,nMatches/nRecoTracks));
    }  
  } // Loop on reco::Vertex

  return outputCollection;
}


VertexSimToRecoCollection VertexAssociatorByTracks::associateSimToReco(
    edm::Handle<reco::VertexCollection>&   vertexCollectionH,
    edm::Handle<TrackingVertexCollection>& TVCollectionH, 
    const edm::Event& e, 
    reco::RecoToSimCollection& trackAssocResult) 
{
  
  VertexSimToRecoCollection  outputCollection;
  return outputCollection;
}

