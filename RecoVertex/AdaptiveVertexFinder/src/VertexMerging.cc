#include "RecoVertex/AdaptiveVertexFinder/interface/VertexMerging.h"




VertexMerging::VertexMerging(const edm::ParameterSet &params) :
	maxFraction(params.getParameter<double>("maxFraction")),
	minSignificance(params.getParameter<double>("minSignificance"))
{
	
}

static double computeSharedTracks(const reco::Vertex &pv,
                                        const reco::Vertex &sv)
{
	std::set<reco::TrackRef> pvTracks;
	for(std::vector<reco::TrackBaseRef>::const_iterator iter = pv.tracks_begin();
	    iter != pv.tracks_end(); iter++) {
		if (pv.trackWeight(*iter) >= 0.5)
			pvTracks.insert(iter->castTo<reco::TrackRef>());
	}

	unsigned int count = 0, total = 0;
	for(std::vector<reco::TrackBaseRef>::const_iterator iter = sv.tracks_begin();
	    iter != sv.tracks_end(); iter++) {
		if (sv.trackWeight(*iter) >= 0.5) {
			total++;
			count += pvTracks.count(iter->castTo<reco::TrackRef>());
		}
	}

	return (double)count / (double)total;
}





reco::VertexCollection VertexMerging::mergeVertex(
	VertexCollection & secondaryVertices){



	
        VertexDistance3D dist;
	VertexCollection recoVertices;
	for(std::vector<reco::Vertex>::const_iterator sv = secondaryVertices.begin();
	    sv != secondaryVertices.end(); ++sv) {
          recoVertices.push_back(*sv);
        }
       for(std::vector<reco::Vertex>::iterator sv = recoVertices.begin();
	    sv != recoVertices.end(); ++sv) {

        bool shared=false;
       for(std::vector<reco::Vertex>::iterator sv2 = recoVertices.begin();
	    sv2 != recoVertices.end(); ++sv2) {
                  double fr=computeSharedTracks(*sv2, *sv);
        //        std::cout << sv2-recoVertices->begin() << " vs " << sv-recoVertices->begin() << " : " << fr << " "  <<  computeSharedTracks(*sv, *sv2) << " sig " << dist.distance(*sv,*sv2).significance() << std::endl;
          //      std::cout << (fr > maxFraction) << " && " << (dist.distance(*sv,*sv2).significance() < 2)  <<  " && " <<  (sv-sv2!=0)  << " && " <<  (fr >= computeSharedTracks(*sv2, *sv))  << std::endl;
		if (fr > maxFraction && dist.distance(*sv,*sv2).significance() < minSignificance && sv-sv2!=0 
                    && fr >= computeSharedTracks(*sv, *sv2) )
		  {
                      shared=true; 
                     // std::cout << "shared " << sv-recoVertices->begin() << " and "  << sv2-recoVertices->begin() << " fractions: " << fr << " , "  << computeSharedTracks(*sv2, *sv) << " sig: " <<  dist.distance(*sv,*sv2).significance() <<  std::endl;
         
                  }
                 

	}
        if(shared) { sv=recoVertices.erase(sv)-1; }
     //    std::cout << "it = " <<  sv-recoVertices->begin() << " new size is: " << recoVertices->size() <<   std::endl;
       }

	return recoVertices;


}

