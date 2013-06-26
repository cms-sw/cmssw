#include <memory>
#include <set>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"

class VertexMerging {
    public:
	VertexMerging(const edm::ParameterSet &params);
	
	
        reco::VertexCollection mergeVertex(reco::VertexCollection & secondaryVertices);
	
	
    private:
	bool trackFilter(const reco::TrackRef &track) const;

	double					maxFraction;
	double					minSignificance;
};

