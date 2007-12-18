#ifndef RecoTracker_TkTrackingRegions_GlobalTrackingRegionWithVerticesProducer_H 
#define RecoTracker_TkTrackingRegions_GlobalTrackingRegionWithVerticesProducer_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class GlobalTrackingRegionWithVerticesProducer : public TrackingRegionProducer
{
public:

  GlobalTrackingRegionWithVerticesProducer(const edm::ParameterSet& cfg)
  { 
    edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");

    thePtMin            = regionPSet.getParameter<double>("ptMin");
    theOriginRadius     = regionPSet.getParameter<double>("originRadius");
    theOriginHalfLength = regionPSet.getParameter<double>("originHalfLength");
    theOriginZPos       = regionPSet.getParameter<double>("originZPos");
    thePrecise          = regionPSet.getParameter<bool>("precise"); 
    theSigmaZVertex     = regionPSet.getParameter<double>("sigmaZVertex");
    theFixedError       = regionPSet.getParameter<double>("fixedError");

    theUseFoundVertices = regionPSet.getParameter<bool>("useFoundVertices");
    theUseFixedError    = regionPSet.getParameter<bool>("useFixedError");
    vertexCollName      = regionPSet.getParameter<std::string>("VertexCollection");
  }   

  virtual ~GlobalTrackingRegionWithVerticesProducer(){}

  virtual std::vector<TrackingRegion* > regions
    (const edm::Event& ev, const edm::EventSetup&) const
  {
    std::vector<TrackingRegion* > result;
    if(theUseFoundVertices)
    {
      // Get originZPos from list of vertices (first or all)
      edm::Handle<reco::VertexCollection> vertexCollection;
      //    std::string vertexCollName = iConfig_.getParameter<std::string>("VertexCollection");
      ev.getByLabel(vertexCollName,vertexCollection);
      //  ev.getByLabel("pixelVertices",vertexCollection);

      if(vertexCollection->size() > 0) {
	for(reco::VertexCollection::const_iterator iV=vertexCollection->begin(); iV != vertexCollection->end() ; iV++) {
	  //	std::cerr << " [TrackProducer] using vertex at z=" << iV->z() << std::endl;
	  double theOriginZPos_       = iV->z();
	  double theOriginHalfLength_; 
	  if(!theUseFixedError) {
	    theOriginHalfLength_ = (iV->zError())*theSigmaZVertex; // correspond a theSigmaZVertex fois l'erreur sur le vertex cf. 15.9 pour le beamspot
	  }
	  if(theUseFixedError) {
	    theOriginHalfLength_ = theFixedError;
	  }

	  result.push_back( new GlobalTrackingRegion(thePtMin, theOriginRadius, theOriginHalfLength_, theOriginZPos_, thePrecise) );
	}
      }
      
      else {
        result.push_back( new GlobalTrackingRegion(thePtMin, theOriginRadius, theOriginHalfLength, theOriginZPos, thePrecise) );
      }
    }
    else
    {
      result.push_back(
        new GlobalTrackingRegion(thePtMin, theOriginRadius,
            theOriginHalfLength, theOriginZPos, thePrecise) );
    }

    return result;
  }

private:
  double thePtMin; 
  double theOriginRadius; 
  double theOriginHalfLength; 
  double theOriginZPos;
  double theSigmaZVertex;
  double theFixedError;
  bool thePrecise;
  
  bool theUseFoundVertices;
  bool theUseFixedError;
  std::string vertexCollName;
};

#endif 
