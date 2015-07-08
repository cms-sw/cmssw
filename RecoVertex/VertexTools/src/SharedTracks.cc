#include "RecoVertex/VertexTools/interface/SharedTracks.h"
namespace vertexTools {
	using namespace reco;
	double	computeSharedTracks(const Vertex &pv, const std::vector<TrackRef> &svTracks,
				double minTrackWeight, float )
		{
			std::set<TrackRef> pvTracks;
			for(std::vector<TrackBaseRef>::const_iterator iter = pv.tracks_begin();
					iter != pv.tracks_end(); iter++)
				if (pv.trackWeight(*iter) >= minTrackWeight)
					pvTracks.insert(iter->castTo<TrackRef>());

			unsigned int count = 0;
			for(std::vector<TrackRef>::const_iterator iter = svTracks.begin();
					iter != svTracks.end(); iter++)
				count += pvTracks.count(*iter);

			return (double)count/(double)svTracks.size();
		}
	double	computeSharedTracks(const Vertex &pv, const std::vector<CandidatePtr> &svTracks,
				double minTrackWeight, float maxsigma)
		{
			unsigned int count = 0;
			for(std::vector<CandidatePtr>::const_iterator iter = svTracks.begin();
					iter != svTracks.end(); iter++)
			{
				if( std::abs((*iter)->bestTrack()->dz()-pv.z())/(*iter)->bestTrack()->dzError() < maxsigma &&
						std::abs((*iter)->bestTrack()->dxy(pv.position())/(*iter)->bestTrack()->dxyError()) < maxsigma 
				  )
					count++;
			}
			return (double)count/(double)svTracks.size();
		}
	double	computeSharedTracks(const VertexCompositePtrCandidate &sv2, const std::vector<CandidatePtr> &svTracks, double , float )
		{
			unsigned int count = 0;
			for(std::vector<CandidatePtr>::const_iterator iter = svTracks.begin();
					iter != svTracks.end(); iter++)
			{
				if(std::find(sv2.daughterPtrVector().begin(),sv2.daughterPtrVector().end(),*iter)!= sv2.daughterPtrVector().end())
					count++;
			}
			return (double)count/(double)svTracks.size();
		}


	double computeSharedTracks(const reco::Vertex &pv, const reco::VertexCompositePtrCandidate &sv, double minTrackWeight,float mindist)
	{
		return computeSharedTracks(pv,sv.daughterPtrVector(),minTrackWeight,mindist);
	}
	double computeSharedTracks(const reco::Vertex &pv, const reco::Vertex &sv, double minTrackWeight,float )
	{
		std::vector<TrackRef> svTracks;
		for(std::vector<TrackBaseRef>::const_iterator iter = sv.tracks_begin();
				iter != sv.tracks_end(); iter++)
			if (sv.trackWeight(*iter) >= minTrackWeight)
				svTracks.push_back(iter->castTo<TrackRef>());
		return computeSharedTracks(pv,svTracks,minTrackWeight);	

	}
	double computeSharedTracks(const reco::VertexCompositePtrCandidate &sv, const reco::VertexCompositePtrCandidate &sv2, double ,float )
	{
		return computeSharedTracks(sv,sv2.daughterPtrVector());
	}


}
