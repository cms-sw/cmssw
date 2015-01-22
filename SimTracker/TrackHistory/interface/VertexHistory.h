#ifndef VertexHistory_h
#define VertexHistory_h

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimTracker/TrackHistory/interface/HistoryBase.h"
#include "SimTracker/TrackHistory/interface/Utils.h"
#include "SimTracker/VertexAssociation/interface/VertexAssociatorBase.h"

//! This class traces the simulated and generated history of a given track.
class VertexHistory : public HistoryBase
{

public:

    //! Constructor by pset.
    /* Creates a VertexHistory with association given by pset.

       /param[in] config with the configuration values
    */
    VertexHistory(const edm::ParameterSet &,
                  edm::ConsumesCollector&&);

    //! Pre-process event information (for accessing reconstruction information)
    void newEvent(const edm::Event &, const edm::EventSetup &);

    //! Evaluate track history using a TrackingParticleRef.
    /* Return false when the history cannot be determined upto a given depth.
       If not depth is pass to the function no restriction are apply to it.

       /param[in] trackingVertexRef of a simulated track
       /param[in] depth of the vertex history
       /param[out] boolean that is true when history can be determined
    */
    bool evaluate(TrackingVertexRef tvr)
    {
        if ( enableSimToReco_ )
        {

            std::pair<reco::VertexBaseRef, double> result =  match(tvr, simToReco_, bestMatchByMaxValue_);
            recovertex_ = result.first;
            quality_ =  result.second;
        }
        return HistoryBase::evaluate(tvr);
    }


    //! Evaluate reco::Vertex history using a given association.
    /* Return false when the track association is not possible (fake track).

       /param[in] VertexRef to a reco::track
       /param[out] boolean that is false when a fake track is detected
    */
    bool evaluate (reco::VertexBaseRef);

    //! Return a reference to the reconstructed track.
    const reco::VertexBaseRef & recoVertex() const
    {
        return recovertex_;
    }

    //! Return the quality of the match.
    double quality() const
    {
        return quality_;
    }


private:

    bool bestMatchByMaxValue_;

    bool enableRecoToSim_, enableSimToReco_;

    double quality_;

    edm::InputTag trackProducer_;

    edm::InputTag vertexProducer_;

    edm::InputTag trackingTruth_;

    edm::InputTag trackAssociator_;

    std::string vertexAssociator_;

    reco::VertexBaseRef recovertex_;

    reco::VertexRecoToSimCollection recoToSim_;

    reco::VertexSimToRecoCollection simToReco_;
};

#endif
