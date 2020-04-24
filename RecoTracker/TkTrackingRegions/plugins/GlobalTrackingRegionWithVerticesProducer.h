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
#include "FWCore/Framework/interface/ConsumesCollector.h"

class GlobalTrackingRegionWithVerticesProducer : public TrackingRegionProducer
{
public:

  GlobalTrackingRegionWithVerticesProducer(const edm::ParameterSet& cfg,
	   edm::ConsumesCollector && iC)
  { 
    edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");

    thePtMin            = regionPSet.getParameter<double>("ptMin");
    theOriginRadius     = regionPSet.getParameter<double>("originRadius");
    theNSigmaZ          = regionPSet.getParameter<double>("nSigmaZ");
    token_beamSpot      = iC.consumes<reco::BeamSpot>(regionPSet.getParameter<edm::InputTag>("beamSpot"));
    thePrecise          = regionPSet.getParameter<bool>("precise"); 
    theUseMS            = regionPSet.getParameter<bool>("useMultipleScattering");

    theSigmaZVertex     = regionPSet.getParameter<double>("sigmaZVertex");
    theFixedError       = regionPSet.getParameter<double>("fixedError");

    theMaxNVertices     = regionPSet.getParameter<int>("maxNVertices");

    theUseFoundVertices = regionPSet.getParameter<bool>("useFoundVertices");
    theUseFakeVertices  = regionPSet.getParameter<bool>("useFakeVertices");
    theUseFixedError    = regionPSet.getParameter<bool>("useFixedError");
    token_vertex      = iC.consumes<reco::VertexCollection>(regionPSet.getParameter<edm::InputTag>("VertexCollection"));
  }   

  ~GlobalTrackingRegionWithVerticesProducer() override{}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<bool>("precise", true);
    desc.add<bool>("useMultipleScattering", false);
    desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
    desc.add<bool>("useFixedError", true);
    desc.add<double>("originRadius", 0.2);
    desc.add<double>("sigmaZVertex", 3.0);
    desc.add<double>("fixedError", 0.2);
    desc.add<edm::InputTag>("VertexCollection", edm::InputTag("firstStepPrimaryVertices"));
    desc.add<double>("ptMin", 0.9);
    desc.add<bool>("useFoundVertices", true);
    desc.add<bool>("useFakeVertices", false);
    desc.add<int>("maxNVertices", -1)->setComment("-1 for all vertices");
    desc.add<double>("nSigmaZ", 4.0);

    // Only for backwards-compatibility
    edm::ParameterSetDescription descRegion;
    descRegion.add<edm::ParameterSetDescription>("RegionPSet", desc);

    descriptions.add("globalTrackingRegionWithVertices", descRegion);
  }

  std::vector<std::unique_ptr<TrackingRegion> > regions
    (const edm::Event& ev, const edm::EventSetup&) const override
  {
    std::vector<std::unique_ptr<TrackingRegion> > result;

    GlobalPoint theOrigin;
    edm::Handle<reco::BeamSpot> bsHandle;
    ev.getByToken( token_beamSpot, bsHandle);
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
      ev.getByToken(token_vertex,vertexCollection);

      for(reco::VertexCollection::const_iterator iV=vertexCollection->begin(); iV != vertexCollection->end() ; iV++) {
          if (!iV->isValid()) continue;
          if (iV->isFake() && !(theUseFakeVertices && theUseFixedError)) continue;
	  GlobalPoint theOrigin_       = GlobalPoint(iV->x(),iV->y(),iV->z());
	  double theOriginHalfLength_ = (theUseFixedError ? theFixedError : (iV->zError())*theSigmaZVertex); 
	  result.push_back( std::make_unique<GlobalTrackingRegion>(thePtMin, theOrigin_, theOriginRadius, theOriginHalfLength_, thePrecise, theUseMS) );
	  if(theMaxNVertices >= 0 && result.size() >= static_cast<unsigned>(theMaxNVertices))
	    break;
      }
      
      if (result.empty()) {
        result.push_back( std::make_unique<GlobalTrackingRegion>(thePtMin, theOrigin, theOriginRadius, bsSigmaZ, thePrecise, theUseMS) );
      }
    }
    else
    {
      result.push_back(
        std::make_unique<GlobalTrackingRegion>(thePtMin, theOrigin, theOriginRadius, bsSigmaZ, thePrecise, theUseMS) );
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
  int theMaxNVertices;
  bool thePrecise;
  bool theUseMS;
  
  bool theUseFoundVertices;
  bool theUseFakeVertices;
  bool theUseFixedError;
  edm::EDGetTokenT<reco::VertexCollection> 	 token_vertex; 
  edm::EDGetTokenT<reco::BeamSpot> 	 token_beamSpot; 


};

#endif 
