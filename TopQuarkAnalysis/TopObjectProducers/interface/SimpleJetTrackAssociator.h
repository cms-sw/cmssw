#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Common/interface/Handle.h"

namespace helper {
class SimpleJetTrackAssociator {
        public:
                SimpleJetTrackAssociator() ;
                SimpleJetTrackAssociator(const edm::ParameterSet &iCfg) ;
                void associate(const math::XYZVector &dir, const edm::Handle<reco::TrackCollection> &in, reco::TrackRefVector &out);
        private:
                double deltaR_;
                int    nHits_;
                double chi2nMax_;
};
}

