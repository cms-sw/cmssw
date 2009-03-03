
#ifndef TrackClassifierByProxy_h
#define TrackClassifierByProxy_h

#include "DataFormats/Common/interface/AssociationMap.h"

#include "FWCore/Utilities/interface/Exception.h"

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
            proxy_( config.getUntrackedParameter<edm::InputTag>("vertexProducer") ) {}

    //! Pre-process event information (for accessing reconstraction information).
    virtual void newEvent(edm::Event const & event, edm::EventSetup const & config)
    {
        // Get the association part of the proxy to the collection
        event.getByLabel(proxy_, proxyHandler_);
    }

    //! Classify any Tracks in categories.
    TrackClassifierByProxy<Collection> const & evaluate (edm::Ref<Collection> const & track, std::size_t index)
    {
        // Find the set of tracks associated by the proxy
        typename Association::const_iterator itracks = proxyHandler_->find(track);

        // If not track is found there is something wrong with the collection or proxy
        if ( itracks == proxyHandler_->end() )
            cms::Exception("ProxyError") << "Track is not found in the given proxy.\n";

        // Get a reference to the vector of associated tracks
        const reco::TrackRefVector & tracks = itracks->val;

        // Evaluate the history for a given index
        TrackClassifier::evaluate( tracks.at(index) );

        return *this;
    }

    //! Classify any tracks in categories.
    TrackClassifierByProxy<Collection> const & evaluate (edm::Ref<Collection> const & track)
    {
        // Find the set of tracks associated by the proxy
        typename Association::const_iterator itracks = proxyHandler_->find(track);

        // If not track is found there is something wrong with the collection or proxy
        if ( itracks == proxyHandler_->end() )
            cms::Exception("ProxyError") << "Track is not found in the given proxy.\n";

        // Get a reference to the vector of associated tracks
        const reco::TrackRefVector & tracks = itracks->val;

        // Loop over all the associated tracks
        for (std::size_t index = 0; index < tracks.size(); ++index)
        {
            // Copy the last status for all the flags
            Flags flags(flags_);

            // Evaluate the history for a given index
            TrackClassifier::evaluate( tracks[index] );

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
