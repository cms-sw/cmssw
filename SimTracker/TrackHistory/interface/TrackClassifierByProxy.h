
#ifndef TrackClassifierByProxy_h
#define TrackClassifierByProxy_h

#include "DataFormats/Common/interface/AssociationMap.h"

#include "SimTracker/TrackHistory/interface/TrackClassifier.h"

//! Get track history and classification by proxy
template <typename Collection>
class TrackClassifierByProxy : public TrackClassifier
{

public:

    //! Association type.
    typedef edm::AssociationMap<edm::OneToMany<Collection, reco::TrackCollection> > Association;

    //! Constructor by ParameterSet.
    TrackClassifierByProxy(edm::ParameterSet const & config) : TrackClassifier(config),
            proxy_( config.getUntrackedParameter<edm::InputTag>("trackProducer") ) {}

    //! Pre-process event information (for accessing reconstraction information).
    virtual void newEvent(edm::Event const & event, edm::EventSetup const & config)
    {
        // Get the association part of the proxy to the collection
        event.getByLabel(proxy_, proxyHandler_);
        // Call the previous new event
        TrackClassifier::newEvent(event, config);
    }

    //! Classify the TrackingVertex in categories.
    TrackClassifierByProxy<Collection> const & evaluate (TrackingParticleRef const & track)
    {
        TrackClassifier::evaluate(track);
        return *this;
    }

    //! Classify any Tracks in categories.
    TrackClassifierByProxy<Collection> const & evaluate (edm::Ref<Collection> const & track, std::size_t index)
    {
        const reco::TrackRefVector * tracks = 0;

        try
        {
            // Get a reference to the vector of associated tracks
            tracks = proxyHandler_->find(track)->val;
        }
        catch (edm::Exception& e)
        {
            // If association fails define the track as unknown
            reset();
            unknownTrack();
            return *this;
        }

        // Evaluate the history for a given index
        TrackClassifier::evaluate( tracks->at(index) );

        return *this;
    }

    //! Classify any tracks in categories.
    TrackClassifierByProxy<Collection> const & evaluate (edm::Ref<Collection> const & track)
    {
        const reco::TrackRefVector * tracks = 0;

        try
        {
            // Get a reference to the vector of associated tracks
            tracks = proxyHandler_->find(track)->val;
        }
        catch (edm::Exception& e)
        {
            // If association fails define the track as unknown
            reset();
            unknownTrack();
            return *this;
        }

        // Loop over all the associated tracks
        for (std::size_t index = 0; index < tracks->size(); ++index)
        {
            // Copy the last status for all the flags
            Flags flags(flags_);

            // Evaluate the history for a given index
            TrackClassifier::evaluate( tracks->at(index) );

            // Combine OR the flag information
            for (std::size_t i = 0; i < flags_.size(); ++i)
                flags_[i] = flags_[i] | flags[i];
        }

        return *this;
    }

private:

    const edm::InputTag proxy_;

    edm::Handle<Association> proxyHandler_;

};

#endif
