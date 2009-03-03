
#ifndef VertexClassifierByProxy_h
#define VertexClassifierByProxy_h

#include "DataFormats/Common/interface/AssociationMap.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "SimTracker/TrackHistory/interface/VertexClassifier.h"

//! Get track history and classification by proxy
template <typename Collection>
class VertexClassifierByProxy : public VertexClassifier
{

public:

    //! Association type.
    typedef edm::AssociationMap<edm::OneToMany<Collection, reco::VertexCollection> > Association;

    //! Constructor by ParameterSet.
    VertexClassifierByProxy(edm::ParameterSet const & config) : VertexClassifier(config),
            proxy_( config.getUntrackedParameter<edm::InputTag>("vertexProducer") ) {}

    //! Pre-process event information (for accessing reconstraction information).
    virtual void newEvent(edm::Event const & event, edm::EventSetup const & config)
    {
        // Get the association part of the proxy to the collection
        event.getByLabel(proxy_, proxyHandler_);
    }

    //! Classify any vertexes in categories.
    VertexClassifierByProxy<Collection> const & evaluate (edm::Ref<Collection> const & vertex, std::size_t index)
    {
        // Find the set of vertexes associated by the proxy
        typename Association::const_iterator ivertexes = proxyHandler_->find(vertex);

        // If not vertex is found there is something wrong with the collection or proxy
        if ( ivertexes == proxyHandler_->end() )
            cms::Exception("ProxyError") << "Vertex is not found in the given proxy.\n";

        // Get a reference to the vector of associated vertexes
        const reco::VertexRefVector & vertexes = ivertexes->val;

        // Evaluate the history for a given index
        VertexClassifier::evaluate( vertexes.at(index) );

        return *this;
    }

    //! Classify any vertexes in categories.
    VertexClassifierByProxy<Collection> const & evaluate (edm::Ref<Collection> const & vertex)
    {
        // Find the set of vertexes associated by the proxy
        typename Association::const_iterator ivertexes = proxyHandler_->find(vertex);

        // If not vertex is found there is something wrong with the collection or proxy
        if ( ivertexes == proxyHandler_->end() )
            cms::Exception("ProxyError") << "Vertex is not found in the given proxy.\n";

        // Get a reference to the vector of associated vertexes
        const reco::VertexRefVector & vertexes = ivertexes->val;

        // Loop over all the associated vertexes
        for (std::size_t index = 0; index < vertexes.size(); ++index)
        {
            // Copy the last status for all the flags
            Flags flags(flags_);

            // Evaluate the history for a given index
            VertexClassifier::evaluate( vertexes[index] );

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
