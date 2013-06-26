#ifndef RecoTracker_TkTrackingRegions_GlobalTrackingRegionWithVerticesProducer_H 
#define RecoTracker_TkTrackingRegions_GlobalTrackingRegionWithVerticesProducer_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Utilities/interface/InputTag.h"

class GlobalTrackingRegionWithVerticesProducer : public TrackingRegionProducer
{
public:

  GlobalTrackingRegionWithVerticesProducer(const edm::ParameterSet& cfg)
  { 
    edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");

    thePtMin            = regionPSet.getParameter<double>("ptMin");
    theOriginRadius     = regionPSet.getParameter<double>("originRadius");
    theNSigmaZ          = regionPSet.getParameter<double>("nSigmaZ");
    theBeamSpotTag      = regionPSet.getParameter<edm::InputTag>("beamSpot");
    thePrecise          = regionPSet.getParameter<bool>("precise"); 

    theSigmaZVertex     = regionPSet.getParameter<double>("sigmaZVertex");
    theFixedError       = regionPSet.getParameter<double>("fixedError");

    theUseFoundVertices = regionPSet.getParameter<bool>("useFoundVertices");
    theUseFixedError    = regionPSet.getParameter<bool>("useFixedError");
    vertexCollName      = regionPSet.getParameter<edm::InputTag>("VertexCollection");
  }   

  virtual ~GlobalTrackingRegionWithVerticesProducer(){}

  virtual std::vector<TrackingRegion* > regions
    (const edm::Event& ev, const edm::EventSetup&) const
  {
    std::vector<TrackingRegion* > result;

    GlobalPoint theOrigin;
    edm::Handle<reco::BeamSpot> bsHandle;
    ev.getByLabel( theBeamSpotTag, bsHandle);
    double bsSigmaZ;
    if(bsHandle.isValid()) {
      const reco::BeamSpot & bs = *bsHandle; 
      bsSigmaZ = theNSigmaZ*bs.sigmaZ();
      theOrigin = GlobalPoint(bs.x0(), bs.y0(), bs.z0());
    }else{
      throw cms::Exception("Seeding") << "ERROR: input beamSpot is not valid in GlobalTrackingRegionWithVertices";
    }

    if(theUseFoundVertices)
    {
      edm::Handle<reco::VertexCollection> vertexCollection;
      ev.getByLabel(vertexCollName,vertexCollection);

      for(reco::VertexCollection::const_iterator iV=vertexCollection->begin(); iV != vertexCollection->end() ; iV++) {
          if (iV->isFake() || !iV->isValid()) continue;
	  GlobalPoint theOrigin_       = GlobalPoint(iV->x(),iV->y(),iV->z());
	  double theOriginHalfLength_ = (theUseFixedError ? theFixedError : (iV->zError())*theSigmaZVertex); 
	  result.push_back( new GlobalTrackingRegion(thePtMin, theOrigin_, theOriginRadius, theOriginHalfLength_, thePrecise) );
      }
      
      if (result.empty()) {
        result.push_back( new GlobalTrackingRegion(thePtMin, theOrigin, theOriginRadius, bsSigmaZ, thePrecise) );
      }
    }
    else
    {
      result.push_back(
        new GlobalTrackingRegion(thePtMin, theOrigin, theOriginRadius, bsSigmaZ, thePrecise) );
    }

    return result;
  }

private:
  double thePtMin; 
  double theOriginRadius; 
  double theNSigmaZ;
  edm::InputTag theBeamSpotTag;

  double theSigmaZVertex;
  double theFixedError;
  bool thePrecise;
  
  bool theUseFoundVertices;
  bool theUseFixedError;
  edm::InputTag vertexCollName;


};

#endif 
