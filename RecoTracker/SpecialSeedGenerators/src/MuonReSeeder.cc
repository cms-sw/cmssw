
//
//

/**
  \class    pat::MuonReSeeder MuonReSeeder.h "MuonAnalysis/MuonAssociators/interface/MuonReSeeder.h"
  \brief    Matcher of reconstructed objects to other reconstructed objects using the tracks inside them 
            
  \author   Giovanni Petrucciani
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"


class MuonReSeeder : public edm::EDProducer {
    public:
      explicit MuonReSeeder(const edm::ParameterSet & iConfig);
      virtual ~MuonReSeeder() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

    private:
      /// Labels for input collections
      edm::EDGetTokenT<edm::View<reco::Muon> > src_;

      /// Muon selection
      StringCutObjectSelector<reco::Muon> selector_;

      /// How many hits to keep from the muon trajectory
      int layersToKeep_;

      /// Do inside-out
      bool insideOut_;

      /// Dump deug information
      bool debug_;

      /// Track Transformer
      TrackTransformer refitter_;

};

MuonReSeeder::MuonReSeeder(const edm::ParameterSet & iConfig) :
    src_(consumes<edm::View<reco::Muon> >(iConfig.getParameter<edm::InputTag>("src"))),
    selector_(iConfig.existsAs<std::string>("cut") ? iConfig.getParameter<std::string>("cut") : "", true),
    layersToKeep_(iConfig.getParameter<int32_t>("layersToKeep")),
    insideOut_(iConfig.getParameter<bool>("insideOut")),
    debug_(iConfig.getUntrackedParameter<bool>("debug",false)),
    refitter_(iConfig)
{
    produces<std::vector<TrajectorySeed> >(); 
}

void 
MuonReSeeder::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    refitter_.setServices(iSetup);

    Handle<View<reco::Muon> > src;
    iEvent.getByToken(src_, src);

    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopo;
    iSetup.get<IdealGeometryRecord>().get(tTopo);

    auto_ptr<vector<TrajectorySeed> > out(new vector<TrajectorySeed>());
    unsigned int nsrc = src->size();
    out->reserve(nsrc);

    for (View<reco::Muon>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        const reco::Muon &mu = *it;
        if (mu.track().isNull() || !selector_(mu)) continue;
        std::vector<Trajectory> traj  = refitter_.transform(*mu.track());
        if (traj.size() != 1) continue;
        edm::OwnVector<TrackingRecHit> seedHits;
        const std::vector<TrajectoryMeasurement> & tms = traj.front().measurements();
        TrajectoryStateOnSurface tsos; const TrackingRecHit *hit  = 0;
        bool fromInside = (insideOut_ == (traj.front().direction() == alongMomentum));
        if (debug_) {
            std::cout << "Considering muon of pt " << mu.pt() << ", eta = " << mu.eta() << ", phi = " << mu.phi() << std::endl;
            std::cout << "Trajectory is " << (traj.front().direction() == alongMomentum ? "along" : "opposite") << " to momentum, so will start from " << (fromInside ? "inside" : "outside") << std::endl;
        }
        const TrajectoryMeasurement & tin = (fromInside ? tms.front() : tms.back());
        const TrajectoryMeasurement & tou = (fromInside ? tms.front() : tms.back());
        if (debug_) {
            std::cout << "IN state: subdetector   = " << tin.recHit()->geographicalId().subdetId() << std::endl;
            std::cout << "          global pos Rho  " << tin.updatedState().globalPosition().perp() << ", Z   " << tin.updatedState().globalPosition().z() << std::endl;
            std::cout << "OU state: subdetector   = " << tou.recHit()->geographicalId().subdetId() << std::endl;
            std::cout << "          global pos Rho  " << tou.updatedState().globalPosition().perp() << ", Z   " << tou.updatedState().globalPosition().z() << std::endl;
        }
        int lastSubdet = 0, lastLayer = -1;
        for (int i    = (fromInside ? 0 : tms.size()-1),
                 end  = (fromInside ? tms.size() : -1),
                 step = (fromInside ? +1 : -1),
                 taken = 0; (end-i)*step > 0; i += step) {
            const TrackingRecHit *lastHit = hit;
            hit = tms[i].recHit()->hit();
            if (debug_) std::cout << "  considering hit " << i << ": rechit on " << (hit ? hit->geographicalId().rawId() : -1) << std::endl;
            if (!hit) continue;
            int subdet = hit->geographicalId().subdetId();
	    int lay = tTopo->layer(hit->geographicalId());
            if (subdet != lastSubdet || lay != lastLayer) {
                // I'm on a new layer
                if (lastHit != 0 && taken == layersToKeep_) {
                    // I've had enough layers, I can stop here
                    hit = lastHit;
                    break;
                } 
                lastSubdet = subdet; lastLayer = lay;
                taken++;
            }
            seedHits.push_back(*hit); 
            tsos = tms[i].updatedState().isValid() ? tms[i].updatedState() :
                          (abs(i-end) < abs(i) ? tms[i].forwardPredictedState() : tms[i].backwardPredictedState());
            if (debug_) {
                std::cout << "     hit  : subdetector   = " << tms[i].recHit()->geographicalId().subdetId() << std::endl;
                if (hit->isValid()) {
                    std::cout << "            global pos Rho  " << tms[i].recHit()->globalPosition().perp() << ", Z   " << tms[i].recHit()->globalPosition().z() << std::endl;
                } else {
                    std::cout << "            invalid tracking rec hit, so no global position" << std::endl;
                }
                std::cout << "     state: global pos Rho  " << tsos.globalPosition().perp() << ", Z   " << tsos.globalPosition().z() << std::endl;
            }
        }
        if (!tsos.isValid()) continue;
        PTrajectoryStateOnDet const & PTraj = trajectoryStateTransform::persistentState(tsos, hit->geographicalId().rawId());
        TrajectorySeed seed(PTraj,std::move(seedHits),insideOut_ ? alongMomentum : oppositeToMomentum); 
        out->push_back(seed);
    }

    iEvent.put(out);
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonReSeeder);
