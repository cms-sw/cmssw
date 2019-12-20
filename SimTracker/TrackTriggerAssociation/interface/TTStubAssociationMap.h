/*! \class   TTStubAssociationMap
 *  \brief   Class to store the MC truth of L1 Track Trigger stubs
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_ASSOCIATION_FORMAT_H
#define L1_TRACK_TRIGGER_STUB_ASSOCIATION_FORMAT_H

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
//#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"  /// NOTE: this is needed even if it seems not
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"

template <typename T>
class TTStubAssociationMap {
public:
  /// Constructors
  TTStubAssociationMap();

  /// Destructor
  ~TTStubAssociationMap();

  /// Data members:   getABC( ... )
  /// Helper methods: findABC( ... )

  /// Maps
  std::map<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> >, edm::Ptr<TrackingParticle> >
  getTTStubToTrackingParticleMap() const {
    return stubToTrackingParticleMap;
  }
  std::map<edm::Ptr<TrackingParticle>, std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > >
  getTrackingParticleToTTStubsMap() const {
    return trackingParticleToStubVectorMap;
  }

  void setTTStubToTrackingParticleMap(
      std::map<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> >, edm::Ptr<TrackingParticle> > aMap) {
    stubToTrackingParticleMap = aMap;
  }
  void setTrackingParticleToTTStubsMap(
      std::map<edm::Ptr<TrackingParticle>, std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > > aMap) {
    trackingParticleToStubVectorMap = aMap;
  }
  void setTTClusterAssociationMap(edm::RefProd<TTClusterAssociationMap<T> > aCluAssoMap) {
    theClusterAssociationMap = aCluAssoMap;
  }

  /// Operations
  edm::Ptr<TrackingParticle> findTrackingParticlePtr(edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > aStub) const;
  std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > findTTStubRefs(
      edm::Ptr<TrackingParticle> aTrackingParticle) const;

  /// MC Truth methods
  bool isGenuine(edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > aStub) const;
  bool isCombinatoric(edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > aStub) const;
  bool isUnknown(edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > aStub) const;

private:
  /// Data members
  std::map<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> >, edm::Ptr<TrackingParticle> >
      stubToTrackingParticleMap;
  std::map<edm::Ptr<TrackingParticle>, std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > >
      trackingParticleToStubVectorMap;
  edm::RefProd<TTClusterAssociationMap<T> > theClusterAssociationMap;

};  /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Default Constructor
/// NOTE: to be used with setSomething(...) methods
template <typename T>
TTStubAssociationMap<T>::TTStubAssociationMap() {
  /// Set default data members
  stubToTrackingParticleMap.clear();
  trackingParticleToStubVectorMap.clear();
  edm::RefProd<TTClusterAssociationMap<T> >* aRefProd = new edm::RefProd<TTClusterAssociationMap<T> >();
  theClusterAssociationMap = *aRefProd;
}

/// Destructor
template <typename T>
TTStubAssociationMap<T>::~TTStubAssociationMap() {}

/// Operations
template <typename T>
edm::Ptr<TrackingParticle> TTStubAssociationMap<T>::findTrackingParticlePtr(
    edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > aStub) const {
  if (stubToTrackingParticleMap.find(aStub) != stubToTrackingParticleMap.end()) {
    return stubToTrackingParticleMap.find(aStub)->second;
  }

  /// Default: return NULL
  //edm::Ptr< TrackingParticle >* temp = new edm::Ptr< TrackingParticle >();
  return edm::Ptr<TrackingParticle>();
}

template <typename T>
std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > TTStubAssociationMap<T>::findTTStubRefs(
    edm::Ptr<TrackingParticle> aTrackingParticle) const {
  if (trackingParticleToStubVectorMap.find(aTrackingParticle) != trackingParticleToStubVectorMap.end()) {
    return trackingParticleToStubVectorMap.find(aTrackingParticle)->second;
  }

  std::vector<edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > > tempVector;
  tempVector.clear();
  return tempVector;
}

/// MC truth
template <typename T>
bool TTStubAssociationMap<T>::isGenuine(edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > aStub) const {
  /// Check if there is a SimTrack
  if ((this->findTrackingParticlePtr(aStub)).isNull())
    return false;

  return true;
}

template <typename T>
bool TTStubAssociationMap<T>::isCombinatoric(edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > aStub) const {
  /// Defined by exclusion
  if (this->isGenuine(aStub))
    return false;

  if (this->isUnknown(aStub))
    return false;

  return true;
}

template <typename T>
bool TTStubAssociationMap<T>::isUnknown(edm::Ref<edmNew::DetSetVector<TTStub<T> >, TTStub<T> > aStub) const {
  /// UNKNOWN means that both clusters are unknown
  //std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > > theseClusters = aStub->getClusterRefs();

  /// Sanity check
  if (theClusterAssociationMap.isNull()) {
    return true;
  }

  if (theClusterAssociationMap->isUnknown(aStub->clusterRef(0)) &&
      theClusterAssociationMap->isUnknown(aStub->clusterRef(1)))
    return true;

  return false;
}

#endif
