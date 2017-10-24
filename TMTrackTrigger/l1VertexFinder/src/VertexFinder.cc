
#include "TMTrackTrigger/VertexFinder/interface/VertexFinder.h"


#include "TMTrackTrigger/VertexFinder/interface/Settings.h"


using namespace std;


namespace vertexFinder {

void VertexFinder::GapClustering(){

    sort(fitTracks_.begin(), fitTracks_.end(), SortTracksByZ0());
	iterations_ = 0;
	RecoVertex Vertex;
	for (unsigned int i = 0; i < fitTracks_.size(); ++i)
	{
		Vertex.insert(fitTracks_[i]);
		iterations_++;
		if((i+1 < fitTracks_.size() and fitTracks_[i+1]->z0()-fitTracks_[i]->z0() > settings_->vx_distance()) or i==fitTracks_.size()-1){
			if(Vertex.numTracks() >= settings_->vx_minTracks()) {
				Vertex.computeParameters();
				vertices_.push_back(Vertex);
			}
			Vertex.clear();
		}
	}
}

void VertexFinder::SimpleMergeClustering(){
	iterations_ = 0;

	sort(fitTracks_.begin(), fitTracks_.end(), SortTracksByZ0());

	std::vector<RecoVertex> vClusters;
	vClusters.resize(fitTracks_.size());

	for(unsigned int i = 0; i < fitTracks_.size(); ++i){
		vClusters[i].insert(fitTracks_[i]);
		// iterations_++;
	}
	
	while(1){
		float MinimumScore = 9999;

		unsigned int clusterId0=0;
		unsigned int clusterId1=0;
		for(unsigned int iClust = 0 ; iClust < vClusters.size()-1 ; iClust++){
			iterations_++;

			float M = MaxDistance(vClusters[iClust], vClusters[iClust+1]);
			if(M < MinimumScore){
				MinimumScore = M;
				clusterId0 = iClust;
				clusterId1 = iClust+1;
			}
		}
		if(MinimumScore > settings_->vx_distance() or vClusters[clusterId1].tracks().empty() ) break;
		for(const L1fittedTrack* track : vClusters[clusterId0].tracks()){
			vClusters[clusterId1].insert(track);
		}
		vClusters.erase(vClusters.begin()+clusterId0);
	}

	for(RecoVertex clust : vClusters){
		if(clust.numTracks()>=settings_->vx_minTracks()){
			clust.computeParameters();
			vertices_.push_back(clust);
		}
	}
}

void VertexFinder::DBSCAN(){
	// std::vector<RecoVertex> vClusters;
	std::vector<unsigned int> visited;
	std::vector<unsigned int> saved;

	sort(fitTracks_.begin(), fitTracks_.end(), SortTracksByPt());
	iterations_ =0;

	for(unsigned int i = 0; i< fitTracks_.size(); ++i){
		if( find( visited.begin(), visited.end(), i) != visited.end() ) continue;

		
		visited.push_back(i);
		std::set<unsigned int> neighbourTrackIds;
		for(unsigned int k = 0; k < fitTracks_.size(); ++k){
			iterations_++;
			if(k!= i and fabs(fitTracks_[k]->z0()-fitTracks_[i]->z0()) < settings_->vx_distance()) neighbourTrackIds.insert(k); 
			
		}

		if(neighbourTrackIds.size() < settings_->vx_minTracks() ){
			// mark track as noise	
		} else{
			RecoVertex vertex;
			vertex.insert(fitTracks_[i]);
			saved.push_back(i);
			for(unsigned int id : neighbourTrackIds){
				if(find( visited.begin(), visited.end(), id) == visited.end()){
					visited.push_back(id);
					std::vector<unsigned int> neighbourTrackIds2;

					for(unsigned int k = 0; k < fitTracks_.size(); ++k){
						iterations_++;
						if(fabs(fitTracks_[k]->z0()-fitTracks_[id]->z0()) < settings_->vx_distance()) neighbourTrackIds2.push_back(k); 
					}

					if(neighbourTrackIds2.size() >= settings_->vx_minTracks()){
						for(unsigned int id2 : neighbourTrackIds2){
							neighbourTrackIds.insert(id2);
						}
					}
				}				
				if(find( saved.begin(), saved.end(), id) == saved.end()) vertex.insert(fitTracks_[id]);
			}
			vertex.computeParameters();
			vertices_.push_back(vertex);
		}
	}
}

void VertexFinder::PVR(){
	bool start = true;
	FitTrackCollection discardedTracks, acceptedTracks;
	iterations_ = 0;
	for(const L1fittedTrack* track : fitTracks_){
		acceptedTracks.push_back(track);
	}


	while(discardedTracks.size() >= settings_->vx_minTracks() or start == true){
		start = false;
		bool removing = true;
		discardedTracks.clear();
		while(removing){
			float oldDistance = 0.;

			if(settings_->debug() == 7) cout << "acceptedTracks "<< acceptedTracks.size() << endl;

			float z0start = 0;
			for(const L1fittedTrack* track : acceptedTracks){
				z0start += track->z0();
				iterations_++;
			}
	
			z0start /= acceptedTracks.size();
			if(settings_->debug() == 7) cout << "z0 vertex " << z0start << endl;
			FitTrackCollection::iterator badTrackIt = acceptedTracks.end();
			removing = false;
			
			for(FitTrackCollection::iterator it = acceptedTracks.begin(); it < acceptedTracks.end(); ++it ){
				const L1fittedTrack* track = *it;
				iterations_++;
				if(fabs(track->z0()-z0start) > settings_->vx_distance() and fabs(track->z0()-z0start) > oldDistance){
					badTrackIt = it;
					oldDistance = fabs(track->z0()-z0start);
					removing = true;
				}
			}

			if(removing){
				const L1fittedTrack* badTrack = *badTrackIt;
				if(settings_->debug() == 7) cout << "removing track "<< badTrack->z0() << " at distance "<< oldDistance << endl;
				discardedTracks.push_back(badTrack);
				acceptedTracks.erase(badTrackIt);
			}
		}

		if(acceptedTracks.size() >= settings_->vx_minTracks()){
			RecoVertex vertex;
			for(const L1fittedTrack* track :acceptedTracks ){
				vertex.insert(track);
			}		
			vertex.computeParameters();
			vertices_.push_back(vertex);
		}
		if(settings_->debug() == 7) cout << "discardedTracks size "<< discardedTracks.size() << endl;
		acceptedTracks.clear();
		acceptedTracks = discardedTracks;
	}
}

void VertexFinder::AdaptiveVertexReconstruction(){
	bool start = true;
	iterations_ =0 ;
	FitTrackCollection discardedTracks, acceptedTracks, discardedTracks2;

	for(const L1fittedTrack* track : fitTracks_){
		discardedTracks.push_back(track);
	}


	while(discardedTracks.size() >= settings_->vx_minTracks() or start == true){
		start = false;
		discardedTracks2.clear();
		FitTrackCollection::iterator it = discardedTracks.begin();
		const L1fittedTrack* track = *it;	
		acceptedTracks.push_back(track);
		float z0sum = track->z0();
				
		for(FitTrackCollection::iterator it2 = discardedTracks.begin(); it2 < discardedTracks.end(); ++it2){
			if(it2 != it){
				const L1fittedTrack* secondTrack = *it2;
				// Calculate new vertex z0 adding this track
				z0sum += secondTrack->z0();
				float z0vertex = z0sum/(acceptedTracks.size()+1);
				// Calculate chi2 of new vertex
				float chi2 = 0.;
				float dof = 0.;
				for(const L1fittedTrack* accTrack : acceptedTracks){
					iterations_++;
					float Residual = accTrack->z0() - z0vertex;
					if(fabs(accTrack->eta()) < 1.2) Residual /= 0.1812; // Assumed z0 resolution
					else if(fabs(accTrack->eta()) >=1.2 && fabs(accTrack->eta()) < 1.6) Residual /= 0.2912;
					else if(fabs(accTrack->eta()) >=1.6 && fabs(accTrack->eta()) < 2.) Residual /= 0.4628;
					else Residual /= 0.65;
					
					chi2 += Residual*Residual;
					dof = (acceptedTracks.size()+1)*2 - 1;
				}
				if(chi2/dof < settings_->vx_chi2cut()){
					acceptedTracks.push_back(secondTrack);
				} else{
					discardedTracks2.push_back(secondTrack);
					z0sum -= secondTrack->z0();
				}
			}
		}

		if(acceptedTracks.size() >= settings_->vx_minTracks()){
			RecoVertex vertex;
			for(const L1fittedTrack* track :acceptedTracks ){
				vertex.insert(track);
			}		
			vertex.computeParameters();
			vertices_.push_back(vertex);
		}

		acceptedTracks.clear();
		discardedTracks.clear();
		discardedTracks = discardedTracks2;
	}
}

void VertexFinder::HPV(){
	iterations_ = 0;
	sort(fitTracks_.begin(), fitTracks_.end(), SortTracksByPt());

	RecoVertex vertex;
	bool first = true;
	float z = 99.;
	for(const L1fittedTrack* track : fitTracks_){
		if(track->pt() < 50.) {
			if(first){
				first = false;
				z = track->z0();
				vertex.insert(track);
			} else{
				if(fabs(track->z0()-z) < settings_->vx_distance()) vertex.insert(track);
			}
		}
	}
	
	vertex.computeParameters();
	
	vertex.setZ(z);
	vertices_.push_back(vertex);

}

float VertexFinder::MaxDistance(RecoVertex cluster0, RecoVertex cluster1){
	float distance = 0;
	for(const L1fittedTrack* track0 : cluster0.tracks()){
		for(const L1fittedTrack* track1 : cluster1.tracks()){
			if(fabs(track0->z0()-track1->z0()) > distance){
				distance = fabs(track0->z0()-track1->z0());
			}
		}
	}

	return distance;
}

void VertexFinder::FindPrimaryVertex() {
	double vertexPt = 0;
	for(unsigned int i = 0; i < vertices_.size(); ++i){
		if(vertices_[i].pT() > vertexPt and vertices_[i].numHighPtTracks() >= settings_->vx_minHighPtTracks() ){
			vertexPt = vertices_[i].pT();
			pv_index_ = i;
		}
	}
}



void VertexFinder::TDRalgorithm(){
	float vxPt = 0.;
	
	for(float z=-14.95; z<15.; z += 0.1){
		RecoVertex vertex;
		FitTrackCollection tracks;
		
		for(const L1fittedTrack* track: fitTracks_){
			if(fabs(z-track->z0()) < settings_->tdr_vx_width() and track->pt() < 50.){
				vertex.insert(track);
			} else{
				tracks.push_back(track);
			}
		}
		vertex.computeParameters();
        vertex.setZ(z);
		if(vertex.pT() > vxPt){
			tdr_vertex_ = vertex;
			tdr_pileup_tracks_ = tracks;
			vxPt = vertex.pT();
		} 
	}

}

}