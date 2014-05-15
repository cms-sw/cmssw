/*! \class   TTClusterAssociator
 *  \brief   Plugin to create the MC truth for TTClusters.
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_ASSOCIATOR_H
#define L1_TRACK_TRIGGER_CLUSTER_ASSOCIATOR_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/TrackTrigger/interface/classNameFinder.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <memory>
#include <map>
#include <vector>

template< typename T >
class TTClusterAssociator : public edm::EDProducer
{
  /// NOTE since pattern hit correlation must be performed within a stacked module, one must store
  /// Clusters in a proper way, providing easy access to them in a detector/member-wise way
  public:
    /// Constructors
    explicit TTClusterAssociator( const edm::ParameterSet& iConfig );

    /// Destructor
    ~TTClusterAssociator();

  private:
    /// Data members
    edm::Handle< edm::DetSetVector< PixelDigiSimLink > >   thePixelDigiSimLinkHandle;
    edm::Handle< edm::SimTrackContainer >                  theSimTrackHandle;
    edm::InputTag                                          simTrackInputTag;
    std::vector< edm::InputTag >                           TTClustersInputTags;
    const StackedTrackerGeometry                           *theStackedTrackers;
    //unsigned int                                           ADCThreshold;

    /// Mandatory methods
    virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Constructors
template< typename T >
TTClusterAssociator< T >::TTClusterAssociator( const edm::ParameterSet& iConfig )
{
  simTrackInputTag = iConfig.getParameter< edm::InputTag >( "simTrackHits" );
  TTClustersInputTags = iConfig.getParameter< std::vector< edm::InputTag > >( "TTClusters" );
  for ( unsigned int iTag = 0; iTag < TTClustersInputTags.size(); iTag++ )
  {
    produces< TTClusterAssociationMap< T > >( TTClustersInputTags.at(iTag).instance() );
  }
}

/// Destructor
template< typename T >
TTClusterAssociator< T >::~TTClusterAssociator(){}

/// Begin run
template< typename T >
void TTClusterAssociator< T >::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Get the geometry
  edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
  iSetup.get< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  theStackedTrackers = StackedTrackerGeomHandle.product();

  /// Print some information when loaded
  std::cout << std::endl;
  std::cout << "TTClusterAssociator< " << templateNameFinder< T >() << " > loaded."
            << std::endl;
  std::cout << std::endl;
}

/// End run
template< typename T >
void TTClusterAssociator< T >::endRun( const edm::Run& run, const edm::EventSetup& iSetup ){}

/// Implement the producer
template< >
void TTClusterAssociator< Ref_PixelDigi_ >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

#endif

