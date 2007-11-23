#include "DataFormats/VertexReco/interface/Vertex.h"

class NuclearLikelihood {

        public :
            NuclearLikelihood():likelihood_(0.0) { }
            void calculate( const reco::Vertex& vtx );
            double result() const { return likelihood_; }
 
        private :
            int secondaryTrackMaxHits(const reco::Vertex& vtx );
            double likelihood_;
};
