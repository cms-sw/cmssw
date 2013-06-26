
//
// $Id: OutsideInMuonSeeder.cc,v 1.2 2013/02/27 14:58:17 muzaffar Exp $
//

/**
  \class    pat::OutsideInMuonSeeder MuonReSeeder.h "MuonAnalysis/MuonAssociators/interface/MuonReSeeder.h"
  \brief    Matcher of reconstructed objects to other reconstructed objects using the tracks inside them 
            
  \author   Giovanni Petrucciani
  \version  $Id: OutsideInMuonSeeder.cc,v 1.2 2013/02/27 14:58:17 muzaffar Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

class OutsideInMuonSeeder : public edm::EDProducer {
    public:
      explicit OutsideInMuonSeeder(const edm::ParameterSet & iConfig);
      virtual ~OutsideInMuonSeeder() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

    private:
      /// Labels for input collections
      edm::InputTag src_;

      /// Muon selection
      StringCutObjectSelector<reco::Muon> selector_;

      /// How many layers to try
      int layersToTry_;

      /// How many hits to try on same layer
      int hitsToTry_;

      /// Do inside-out
      bool fromVertex_;

      /// How much to rescale errors from STA
      double errorRescaling_;

      std::string trackerPropagatorName_;
      std::string muonPropagatorName_; 
      std::string measurementTrackerName_; 
      std::string estimatorName_; 
      std::string updatorName_; 

      double minEtaForTEC_, maxEtaForTOB_;

      edm::ESHandle<MagneticField>          magfield_;
      edm::ESHandle<Propagator>             muonPropagator_;
      edm::ESHandle<Propagator>             trackerPropagator_;
      edm::ESHandle<GlobalTrackingGeometry> geometry_;
      edm::ESHandle<MeasurementTracker>     measurementTracker_;
      edm::ESHandle<Chi2MeasurementEstimatorBase>   estimator_;
      edm::ESHandle<TrajectoryStateUpdator>         updator_;

      /// Dump deug information
      bool debug_;
    
      /// Surface used to make a TSOS at the PCA to the beamline
      Plane::PlanePointer dummyPlane_;

      int doLayer(const GeometricSearchDet &layer, const TrajectoryStateOnSurface &state, std::vector<TrajectorySeed> &out) const ;
      void doDebug(const reco::Track &tk) const;

};

OutsideInMuonSeeder::OutsideInMuonSeeder(const edm::ParameterSet & iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    selector_(iConfig.existsAs<std::string>("cut") ? iConfig.getParameter<std::string>("cut") : "", true),
    layersToTry_(iConfig.getParameter<int32_t>("layersToTry")),
    hitsToTry_(iConfig.getParameter<int32_t>("hitsToTry")),
    fromVertex_(iConfig.getParameter<bool>("fromVertex")),
    errorRescaling_(iConfig.getParameter<double>("errorRescaleFactor")),
    trackerPropagatorName_(iConfig.getParameter<std::string>("trackerPropagator")),
    muonPropagatorName_(iConfig.getParameter<std::string>("muonPropagator")),
    estimatorName_(iConfig.getParameter<std::string>("hitCollector")),
    minEtaForTEC_(iConfig.getParameter<double>("minEtaForTEC")),
    maxEtaForTOB_(iConfig.getParameter<double>("maxEtaForTOB")),
    debug_(iConfig.getUntrackedParameter<bool>("debug",false)),
    dummyPlane_(Plane::build(Plane::PositionType(), Plane::RotationType()))
{
    produces<std::vector<TrajectorySeed> >(); 
    measurementTrackerName_ = "";
    updatorName_ = "KFUpdator";
}

void 
OutsideInMuonSeeder::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;
    iSetup.get<IdealMagneticFieldRecord>().get(magfield_);
    iSetup.get<TrackingComponentsRecord>().get(trackerPropagatorName_, trackerPropagator_);
    iSetup.get<TrackingComponentsRecord>().get(muonPropagatorName_, muonPropagator_);
    iSetup.get<GlobalTrackingGeometryRecord>().get(geometry_);
    iSetup.get<CkfComponentsRecord>().get(measurementTracker_);
    iSetup.get<TrackingComponentsRecord>().get(estimatorName_,estimator_);  
    iSetup.get<TrackingComponentsRecord>().get(updatorName_,updator_);  

    measurementTracker_->update(iEvent);

    Handle<View<reco::Muon> > src;
    iEvent.getByLabel(src_, src);

    
    auto_ptr<vector<TrajectorySeed> > out(new vector<TrajectorySeed>());

    for (View<reco::Muon>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        const reco::Muon &mu = *it;
        if (mu.outerTrack().isNull() || !selector_(mu)) continue;
        if (debug_ && mu.innerTrack().isNonnull()) doDebug(*mu.innerTrack());

        muonPropagator_->setPropagationDirection(fromVertex_ ? alongMomentum : oppositeToMomentum);
        trackerPropagator_->setPropagationDirection(alongMomentum);

        int sizeBefore = out->size();
        if (debug_) std::cout << "\n\n\nSeeding for muon of pt " << mu.pt() << ", eta " << mu.eta() << ", phi " << mu.phi() << std::endl;
        const reco::Track &tk = *mu.outerTrack();
        
        TrajectoryStateOnSurface state;
        if (fromVertex_) {
            FreeTrajectoryState fstate = trajectoryStateTransform::initialFreeState(tk, magfield_.product());
            dummyPlane_->move(fstate.position() - dummyPlane_->position());
            state = TrajectoryStateOnSurface(fstate, *dummyPlane_);
        } else {
            state = trajectoryStateTransform::innerStateOnSurface(tk, *geometry_, magfield_.product());
        }
        if (std::abs(tk.eta()) < maxEtaForTOB_) {
            std::vector< BarrelDetLayer * > const & tob = measurementTracker_->geometricSearchTracker()->tobLayers();
            int iLayer = 6, found = 0;
            for (std::vector<BarrelDetLayer *>::const_reverse_iterator it = tob.rbegin(), ed = tob.rend(); it != ed; ++it, --iLayer) {
                if (debug_) std::cout << "\n ==== Trying TOB " << iLayer << " ====" << std::endl;
                if (doLayer(**it, state, *out)) {
                    if (++found == layersToTry_) break;
                }
            }
        }
        if (tk.eta() > minEtaForTEC_) {
            int iLayer = 9, found = 0;
            std::vector< ForwardDetLayer * > const & tec = measurementTracker_->geometricSearchTracker()->posTecLayers();
            for (std::vector<ForwardDetLayer *>::const_reverse_iterator it = tec.rbegin(), ed = tec.rend(); it != ed; ++it, --iLayer) {
                if (debug_) std::cout << "\n ==== Trying TEC " << +iLayer << " ====" << std::endl;
                if (doLayer(**it, state, *out)) {
                    if (++found == layersToTry_) break;
                }
            }
        }
        if (tk.eta() < -minEtaForTEC_) {
            int iLayer = 9, found = 0;
            std::vector< ForwardDetLayer * > const & tec = measurementTracker_->geometricSearchTracker()->negTecLayers();
            for (std::vector<ForwardDetLayer *>::const_reverse_iterator it = tec.rbegin(), ed = tec.rend(); it != ed; ++it, --iLayer) {
                if (debug_) std::cout << "\n ==== Trying TEC " << -iLayer << " ====" << std::endl;
                if (doLayer(**it, state, *out)) {
                    if (++found == layersToTry_) break;
                }
            }
        }
        if (debug_) std::cout << "Outcome of seeding for muon of pt " << mu.pt() << ", eta " << mu.eta() << ", phi " << mu.phi() << ": found " << (out->size() - sizeBefore) << " seeds."<< std::endl;
        
    }

    iEvent.put(out);
}

int
OutsideInMuonSeeder::doLayer(const GeometricSearchDet &layer, const TrajectoryStateOnSurface &state, std::vector<TrajectorySeed> &out) const {
    TrajectoryStateOnSurface onLayer(state);
    onLayer.rescaleError(errorRescaling_);
    std::vector< GeometricSearchDet::DetWithState > dets;
    layer.compatibleDetsV(onLayer, *muonPropagator_, *estimator_, dets); 

    if (debug_) {
        std::cout << "Query on layer around x = " << onLayer.globalPosition() << 
            " with local pos error " << sqrt(onLayer.localError().positionError().xx()) << " ,  " << sqrt(onLayer.localError().positionError().yy()) << " ,  " << 
            " returned " << dets.size() << " compatible detectors" << std::endl;
    }

    std::vector<TrajectoryMeasurement> meas;
    for (std::vector<GeometricSearchDet::DetWithState>::const_iterator it = dets.begin(), ed = dets.end(); it != ed; ++it) {
        const MeasurementDet *det = measurementTracker_->idToDet(it->first->geographicalId());
        if (det == 0) { std::cerr << "BOGUS detid " << it->first->geographicalId().rawId() << std::endl; continue; }
        if (!it->second.isValid()) continue;
        std::vector < TrajectoryMeasurement > mymeas = det->fastMeasurements(it->second, state, *trackerPropagator_, *estimator_);
        if (debug_) std::cout << "Query on detector " << it->first->geographicalId().rawId() << " returned " << mymeas.size() << " measurements." << std::endl;
        for (std::vector<TrajectoryMeasurement>::const_iterator it2 = mymeas.begin(), ed2 = mymeas.end(); it2 != ed2; ++it2) {
            if (it2->recHit()->isValid()) meas.push_back(*it2);
        }
    }
    int found = 0;
    std::sort(meas.begin(), meas.end(), TrajMeasLessEstim());
    for (std::vector<TrajectoryMeasurement>::const_iterator it2 = meas.begin(), ed2 = meas.end(); it2 != ed2; ++it2) {
        if (debug_) {
            std::cout << "  inspecting Hit with chi2 = " << it2->estimate() << std::endl;
            std::cout << "        track state     " << it2->forwardPredictedState().globalPosition() << std::endl; 
            std::cout << "        rechit position " << it2->recHit()->globalPosition() << std::endl; 
        }
        TrajectoryStateOnSurface updated = updator_->update(it2->forwardPredictedState(), *it2->recHit());
        if (updated.isValid()) {
            if (debug_) std::cout << "          --> updated state: x = " << updated.globalPosition() << ", p = " << updated.globalMomentum() << std::endl; 
            edm::OwnVector<TrackingRecHit> seedHits;
            seedHits.push_back(*it2->recHit()->hit());
            PTrajectoryStateOnDet const & pstate = trajectoryStateTransform::persistentState(updated, it2->recHit()->geographicalId().rawId());
            TrajectorySeed seed(pstate, std::move(seedHits), oppositeToMomentum); 
            out.push_back(seed);
            found++; if (found == hitsToTry_) break;
        }
    }
    return found;
}

void
OutsideInMuonSeeder::doDebug(const reco::Track &tk) const {
    TrajectoryStateOnSurface tsos = trajectoryStateTransform::innerStateOnSurface(tk, *geometry_, &*magfield_);
    muonPropagator_->setPropagationDirection(alongMomentum);
    for (unsigned int i = 0; i < tk.recHitsSize(); ++i) {
        const TrackingRecHit *hit = &*tk.recHit(i);
        const GeomDet *det = geometry_->idToDet(hit->geographicalId()); 
        if (det == 0) continue;
        if (i != 0) tsos = muonPropagator_->propagate(tsos, det->surface());
        if (!tsos.isValid()) continue;
        std::cout << "  state " << i << " at x = " << tsos.globalPosition() << ", p = " << tsos.globalMomentum() << std::endl;
        if (hit->isValid()) {
            std::cout << "         valid   rechit on detid " << hit->geographicalId().rawId() << std::endl;
        } else {
            std::cout << "         invalid rechit on detid " << hit->geographicalId().rawId() << std::endl;
        }
    }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OutsideInMuonSeeder);
