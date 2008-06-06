#ifndef _NuclearLikelihood_h_
#define _NuclearLikelihood_h_

#include "DataFormats/VertexReco/interface/Vertex.h"

class NuclearLikelihood {

        public :
            NuclearLikelihood():likelihood_(0.0) { }
            void calculate( const reco::Vertex& vtx );
            double result() const { return likelihood_; }
 
        private :
            int secondaryTrackMaxHits(const reco::Vertex& vtx , int& id);
            double likelihood_;
};

#endif
