#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisRecoVertex.h"


void L1Analysis::L1AnalysisRecoVertex::SetVertices(const edm::Handle<reco::VertexCollection> vertices, unsigned maxVtx)
{  
   recoVertex_.nVtx = 0;
   for(reco::VertexCollection::const_iterator it=vertices->begin();
      it!=vertices->end() && recoVertex_.nVtx < maxVtx;
      ++it) {

      if (!it->isFake()) {
      
	recoVertex_.NDoF.push_back(it->ndof());
	recoVertex_.Z.push_back(it->z());
	recoVertex_.Rho.push_back(it->position().rho());
	
	recoVertex_.nVtx++;
      }
    }

}

