#ifndef SimGeneral_PileupInformation_h
#define SimGeneral_PileupInformation_h

#include <map>

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "CommonTools/RecoAlgos/interface/TrackingParticleSelector.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupMixingContent.h"

#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "Utilities/Timing/interface/TimerStack.h"

class PileupInformation : public edm::EDProducer
{

public:

    explicit PileupInformation( const edm::ParameterSet & );

private:

    void produce( edm::Event &, const edm::EventSetup & );

    edm::ParameterSet conf_;

    typedef std::map<EncodedEventId, unsigned int> EncodedEventIdToIndex;
    typedef std::map< int, int > myindex;
    myindex event_index_;

    std::vector<float> zpositions;
    std::vector<float> sumpT_lowpT;
    std::vector<float> sumpT_highpT;
    std::vector<int> ntrks_lowpT;
    std::vector<int> ntrks_highpT;


    double                   distanceCut_;
    double                   volumeRadius_;
    double                   volumeZ_;
    double                   pTcut_1_;
    double                   pTcut_2_;

    edm::EDGetTokenT<TrackingParticleCollection>     trackingTruthT_;
    edm::EDGetTokenT<TrackingVertexCollection>     trackingTruthV_;
    edm::EDGetTokenT<PileupMixingContent>            PileupInfoLabel_;

    bool LookAtTrackingTruth_ ;

    std::string MessageCategory_;
    //std::string simHitLabel_;
    //std::auto_ptr<MixCollection<SimTrack> >   simTracks_;
    //std::auto_ptr<MixCollection<SimVertex> >  simVertexes_;



};


#endif
