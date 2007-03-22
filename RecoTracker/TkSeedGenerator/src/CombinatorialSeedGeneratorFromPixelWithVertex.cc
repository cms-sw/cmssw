#include "RecoTracker/TkSeedGenerator/interface/CombinatorialSeedGeneratorFromPixelWithVertex.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/VertexReco/interface/Vertex.h"

#include <algorithm>
#include <vector>

void CombinatorialSeedGeneratorFromPixelWithVertex::init(const SiPixelRecHitCollection &coll ,
                                               const reco::VertexCollection &vtxcoll,
					       const edm::EventSetup& iSetup)
{
  pixelLayers->init(coll,iSetup);
  initPairGenerator(pixelLayers,iSetup);
  pixelVertices_ = &vtxcoll;
}

CombinatorialSeedGeneratorFromPixelWithVertex::CombinatorialSeedGeneratorFromPixelWithVertex(edm::ParameterSet const& conf)
  : SeedGeneratorFromLayerPairs(conf)
{  
  edm::ParameterSet conf_ = pSet();
  ptMin_            = conf_.getParameter<double>("ptMin");
  vertexRadius_     = conf_.getParameter<double>("vertexRadius");
  vertexDeltaZ_     = conf_.getParameter<double>("vertexDeltaZ");
  fallbackDeltaZ_   = conf_.getParameter<double>("fallbackDeltaZ");
  //vertexZSigmas_  = conf_.getParameter<double>("vertexZSigmas");
  numberOfVertices_ = conf_.getParameter<uint32_t>("numberOfVertices");
  mergeOverlaps_    = conf_.getParameter<bool>("mergeOverlaps");

  pixelLayers = new PixelSeedLayerPairs();
}

void CombinatorialSeedGeneratorFromPixelWithVertex::run(TrajectorySeedCollection &output,
        const edm::Event& ev, const edm::EventSetup& iSetup){
    if (pixelVertices_ == 0) throw cms::Exception("NO Pixel Vertices available");

    reco::VertexCollection::const_iterator it, end = pixelVertices_->end();
    GlobalTrackingRegion region;
    uint32_t nRemaining = numberOfVertices_;

    if ((pixelVertices_->size() == 0) && (fallbackDeltaZ_ > 0)) {
	region = GlobalTrackingRegion(ptMin_, vertexRadius_, fallbackDeltaZ_ , 0);
	seeds(output, ev, iSetup, region);
    } else {
	if (mergeOverlaps_) {
	    // collect [z1,z2] intervals to use
	    std::vector<std::pair<float,float> > in, out;
	    for (it = pixelVertices_->begin(); it < end; ++it) {
		if (--nRemaining < 0) break;
		float z = it->z();
		in.push_back(std::pair<float,float>(z - vertexDeltaZ_, z + vertexDeltaZ_));
	    }

	    // merge overlapping intervals
	    std::sort(in.begin(),in.end());
	    std::vector<std::pair<float,float> >::const_iterator it0,it1, itEnd;
	    for (it0 = in.begin(), itEnd = in.end(); it0 < itEnd;) {
		float start = it0->first;
		float end   = it0->second;
		for (it1 = it0+1; it1 < itEnd; ++it1) {
		    if (it1->first > end)  break; 
		    else if (it1->second > end) end = it1->second;
		}
		out.push_back(std::pair<float,float>(start,end));
		it0 = it1;
	    }

	    // seed from resulting intervals
	    for (it0 = out.begin(), itEnd = out.end(); it0 < itEnd; ++it0) {
		float z1 = it0->first, dz = 0.5*(it0->second - z1);
		region = GlobalTrackingRegion(ptMin_, vertexRadius_, dz, z1 + dz);
		seeds(output, ev, iSetup, region);
	    }
	} else {
	    for (it = pixelVertices_->begin(); it != end; ++it) {
		if (--nRemaining < 0) break;
		region = GlobalTrackingRegion(ptMin_, vertexRadius_, vertexDeltaZ_ , it->z());
		seeds(output, ev, iSetup, region);
	    }
	}
    }
}
