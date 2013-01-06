#ifndef TrackingAnalysis_TrackingTruthProducer_h
#define TrackingAnalysis_TrackingTruthProducer_h

#include <map>

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/RecoAlgos/interface/TrackingParticleSelector.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"
#include "SimGeneral/TrackingAnalysis/interface/MuonPSimHitSelector.h"
#include "SimGeneral/TrackingAnalysis/interface/PixelPSimHitSelector.h"
#include "SimGeneral/TrackingAnalysis/interface/PSimHitSelector.h"
#include "SimGeneral/TrackingAnalysis/interface/TrackerPSimHitSelector.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "Utilities/Timing/interface/TimerStack.h"

class TrackerTopology;


class TrackingTruthProducer : public edm::EDProducer
{

public:

    explicit TrackingTruthProducer( const edm::ParameterSet & );

private:

    void produce( edm::Event &, const edm::EventSetup & );

    edm::ParameterSet conf_;

    std::vector<std::string> dataLabels_;
    bool                     useMultipleHepMCLabels_;
    double                   distanceCut_;
    std::vector<std::string> hitLabelsVector_;
    double                   volumeRadius_;
    double                   volumeZ_;
    bool                     mergedBremsstrahlung_;
    bool                     removeDeadModules_;
    std::string              simHitLabel_;

    std::string MessageCategory_;

    // Related to production

    std::vector<edm::Handle<edm::HepMCProduct> > hepMCProducts_;

    PSimHitSelector::PSimHitCollection        pSimHits_;

    PSimHitSelector                           pSimHitSelector_;
    PixelPSimHitSelector                      pixelPSimHitSelector_;
    TrackerPSimHitSelector                    trackerPSimHitSelector_;
    MuonPSimHitSelector                       muonPSimHitSelector_;

    std::auto_ptr<MixCollection<SimTrack> >   simTracks_;
    std::auto_ptr<MixCollection<SimVertex> >  simVertexes_;

    std::auto_ptr<TrackingParticleCollection> trackingParticles_;
    std::auto_ptr<TrackingVertexCollection>   trackingVertexes_;

    TrackingParticleRefProd refTrackingParticles_;
    TrackingVertexRefProd   refTrackingVertexes_;

    std::auto_ptr<TrackingParticleCollection> mergedTrackingParticles_;
    std::auto_ptr<TrackingVertexCollection>   mergedTrackingVertexes_;
    TrackingParticleRefProd refMergedTrackingParticles_;
    TrackingVertexRefProd   refMergedTrackingVertexes_;

    typedef std::map<EncodedEventId, unsigned int> EncodedEventIdToIndex;
    typedef std::map<EncodedTruthId, unsigned int> EncodedTruthIdToIndex;
    typedef std::multimap<EncodedTruthId, unsigned int> EncodedTruthIdToIndexes;

    EncodedEventIdToIndex   eventIdCounter_;
    EncodedTruthIdToIndexes trackIdToHits_;
    EncodedTruthIdToIndex   trackIdToIndex_;
    EncodedTruthIdToIndex   vertexIdToIndex_;

    bool selectorFlag_;
    TrackingParticleSelector selector_;

    void associator(
        std::vector<PSimHit> const &,
        EncodedTruthIdToIndexes &
    );

    void associator(
        std::auto_ptr<MixCollection<SimTrack> > const &,
        EncodedTruthIdToIndex &
    );

    void associator(
        std::auto_ptr<MixCollection<SimVertex> > const &,
        EncodedTruthIdToIndex &
    );

    void mergeBremsstrahlung();

    bool isBremsstrahlungVertex(
        TrackingVertex const & vertex,
        std::auto_ptr<TrackingParticleCollection> & tPC
    );

    void createTrackingTruth(const TrackerTopology *tTopo);

    bool setTrackingParticle(
        SimTrack const &,
        TrackingParticle &,
	const TrackerTopology *tTopo
    );

    int setTrackingVertex(
        SimVertex const &,
        TrackingVertex &
    );

    void addCloseGenVertexes(TrackingVertex &);

    unsigned long long m_vertexCounter ;
    unsigned long long m_noMatchVertexCounter ;

    std::vector<std::size_t> m_trackingVertexBins[ 10 ] ;
    double m_trackingVertexBinMins[ 10 ] ;
};


#endif
