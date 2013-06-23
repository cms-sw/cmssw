/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Andrew W. Rose, IC                   ///
///                                      ///
/// 2008                                 ///
/// ////////////////////////////////////////

#ifndef CLUSTERING_ALGORITHM_a_H
#define CLUSTERING_ALGORITHM_a_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithmRecord.h"

#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class ClusteringAlgorithm_a : public ClusteringAlgorithm< T >
  {
    private:
      /// Data members

    public:
      /// Constructor
      ClusteringAlgorithm_a( const StackedTrackerGeometry *aStackedTracker )
        : ClusteringAlgorithm< T >( aStackedTracker,__func__ ) {}

      /// Destructor
      ~ClusteringAlgorithm_a(){}

      /// Clustering operations  
      void Cluster( std::vector< std::vector< T > > &output, const std::vector< T > &input ) const;

  }; /// Close class

  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// Clustering operations
  /// NOTE: in this case, clustering is dummy and each hit is
  /// treated as a different already-ok cluster
  template< typename T >
  void ClusteringAlgorithm_a< T >::Cluster( std::vector< std::vector< T > > &output,
                                            const std::vector< T > &input ) const
  {
    /// Prepare output
    output.clear();
    /// Loop over all hits
    typename std::vector< T >::const_iterator inputIterator;
    for( inputIterator = input.begin();
         inputIterator != input.end();
         ++inputIterator ) 
    {
      std::vector< T > temp;
      temp.push_back(*inputIterator);
      output.push_back(temp);
    } /// End of loop over all hits
  } /// End of ClusteringAlgorithm_a< ... >::Cluster( ... )



/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template< typename T >
class  ES_ClusteringAlgorithm_a : public edm::ESProducer
{

  private:
    /// Data members
    boost::shared_ptr< ClusteringAlgorithm< T > > _theAlgo;

  public:
    /// Constructor
    ES_ClusteringAlgorithm_a( const edm::ParameterSet & p )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_ClusteringAlgorithm_a(){}

    /// Implement the producer
    boost::shared_ptr< ClusteringAlgorithm< T > > produce( const ClusteringAlgorithmRecord & record )
    { 
      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
 
      ClusteringAlgorithm< T >* ClusteringAlgo =
        new ClusteringAlgorithm_a< T >( &(*StackedTrackerGeomHandle) );

      _theAlgo = boost::shared_ptr< ClusteringAlgorithm< T > >( ClusteringAlgo );
      return _theAlgo;
    }

}; /// Close class

#endif

