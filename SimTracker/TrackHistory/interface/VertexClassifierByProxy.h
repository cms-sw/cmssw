
#ifndef VertexClassifierByProxy_h
#define VertexClassifierByProxy_h

#include "DataFormats/Common/interface/AssociationMap.h"

#include "SimTracker/TrackHistory/interface/VertexClassifier.h"

//! Get track history and classification by proxy
template <typename Collection>
class VertexClassifierByProxy : public VertexClassifier {
public:
  //! Association type.
  typedef edm::AssociationMap<edm::OneToMany<Collection, reco::VertexCollection>> Association;

  //! Constructor by ParameterSet.
  VertexClassifierByProxy(edm::ParameterSet const &config, edm::ConsumesCollector &&collector)
      : VertexClassifier(config, std::move(collector)),
        proxy_(config.getUntrackedParameter<edm::InputTag>("vertexProducer")) {
    collector.consumes<Association>(proxy_);
  }

  //! Pre-process event information (for accessing reconstraction information).
  void newEvent(edm::Event const &event, edm::EventSetup const &config) override {
    // Get the association part of the proxy to the collection
    event.getByLabel(proxy_, proxyHandler_);
    // Call the previous new event
    VertexClassifier::newEvent(event, config);
  }

  //! Classify the TrackingVertex in categories.
  VertexClassifierByProxy<Collection> const &evaluate(TrackingVertexRef const &vertex) {
    VertexClassifier::evaluate(vertex);
    return *this;
  }

  //! Classify any vertexes in categories.
  VertexClassifierByProxy<Collection> const &evaluate(edm::Ref<Collection> const &vertex, std::size_t index) {
    const reco::VertexRefVector *vertexes = nullptr;

    try {
      // Get a reference to the vector of associated vertexes
      vertexes = &(proxyHandler_->find(vertex)->val);
    } catch (edm::Exception &e) {
      // If association fails define the vertex as unknown
      reset();
      unknownVertex();
      return *this;
    }

    // Evaluate the history for a given index
    VertexClassifier::evaluate(vertexes->at(index));

    return *this;
  }

  //! Classify any vertexes in categories.
  VertexClassifierByProxy<Collection> const &evaluate(edm::Ref<Collection> const &vertex) {
    const reco::VertexRefVector *vertexes = nullptr;

    try {
      // Get a reference to the vector of associated vertexes
      vertexes = &(proxyHandler_->find(vertex)->val);
    } catch (edm::Exception &e) {
      // If association fails define the vertex as unknown
      reset();
      unknownVertex();
      return *this;
    }

    // Loop over all the associated vertexes
    for (std::size_t index = 0; index < vertexes->size(); ++index) {
      // Copy the last status for all the flags
      Flags flags(flags_);

      // Evaluate the history for a given index
      VertexClassifier::evaluate(vertexes->at(index));

      // Combine OR the flag information
      for (std::size_t i = 0; i < flags_.size(); ++i)
        flags_[i] = flags_[i] || flags[i];
    }

    return *this;
  }

private:
  const edm::InputTag proxy_;

  edm::Handle<Association> proxyHandler_;
};

#endif
