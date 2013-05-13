/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Kristofer Henriksson                 ///
/// ////////////////////////////////////////

/// ////////////////////////////////////////
/// This is a greedy clustering to be    ///
/// used for diagnostic purposes, which  ///
/// will make clusters as large as       ///
/// possible by including all contiguous ///
/// hits in a single cluster.            ///
/// ////////////////////////////////////////

#ifndef CLUSTERING_ALGORITHM_neighbor_H
#define CLUSTERING_ALGORITHM_neighbor_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithmRecord.h"

#include <boost/shared_ptr.hpp>
#include <string>
#include <cstdlib>
#include <map>

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class ClusteringAlgorithm_neighbor : public ClusteringAlgorithm< T >
  {
    private:
      /// Data members
      /// Other stuff
    
    public:
      /// Constructor
      ClusteringAlgorithm_neighbor( const StackedTrackerGeometry *aStackedTracker )
        : ClusteringAlgorithm< T >( aStackedTracker,__func__ ) {}

      /// Destructor
      ~ClusteringAlgorithm_neighbor(){}

      /// Clustering operations  
      void Cluster( std::vector< std::vector< T > > &output,
                    const std::vector< T > &input) const;

      /// Needed for neighbours
      bool isANeighbor( const T& center, const T& mayNeigh) const;
      void addNeighbors( std::vector< T >& cluster, const std::vector< T >& input, unsigned int start, std::vector<bool> &masked ) const;

  }; /// Close class

  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// Clustering operations
  template< typename T >
  void ClusteringAlgorithm_neighbor< T >::Cluster( std::vector<std::vector< T > > &output,
                                                   const std::vector< T > &input ) const
  {
    /// Prepare output
    output.clear();
    /// Loop over all input hits and delete
    /// them once clustered
    std::vector<bool> used(input.size(),false);

    for ( unsigned int i=0; i<input.size(); i++) {
      if (used[i]) continue;
      std::vector<T> cluster;
      cluster.push_back(input[i]);
      used[i]=true;
      if (i<input.size()-1)
	addNeighbors( cluster, input, i+1, used );
      output.push_back(cluster);
    } /// End of iteration
  }

  /// Check if the hit is a neighbour
  template< typename T >
  bool ClusteringAlgorithm_neighbor< T >::isANeighbor( const T& center,
                                                       const T& mayNeigh ) const
  {
    unsigned int rowdist = abs(center->row() - mayNeigh->row());
    unsigned int coldist = abs(center->column() - mayNeigh->column());
    return rowdist <= 1 && coldist <= 1;
  }

  /// Add neighbours to the cluster
  template< typename T >
  void ClusteringAlgorithm_neighbor< T >::addNeighbors( std::vector< T >& cluster,
                                                        const std::vector< T >& input,
							unsigned int startVal,
							std::vector<bool>& used) const
  {
    /// This following line is necessary to ensure the
    /// iterators afterward remain valid.
    cluster.reserve(input.size());
    typename std::vector< T >::iterator clusIter;
    typename std::vector< T >::iterator inIter;

    /// Loop over hits
    for ( clusIter = cluster.begin();
          clusIter < cluster.end();
          clusIter++ )
    {
      /// Loop over candidate neighbours
      for ( unsigned int i=startVal; i<input.size(); i++) 
      {
        /// Is it really a neighbour?
        if ( isANeighbor(*clusIter, input[i]) )
	  {
	    cluster.push_back(input[i]);
	    used[i]=true;
	  }
      } /// End of loop over candidate neighbours
    } /// End of loop over hits
  }



/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template< typename T >
class ES_ClusteringAlgorithm_neighbor : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< ClusteringAlgorithm< T > > _theAlgo;    

  public:
    /// Constructor
    ES_ClusteringAlgorithm_neighbor( const edm::ParameterSet & p )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_ClusteringAlgorithm_neighbor(){}

    /// Implement the producer
    boost::shared_ptr< ClusteringAlgorithm< T > > produce( const ClusteringAlgorithmRecord & record )
    { 
      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );

      ClusteringAlgorithm< T >* ClusteringAlgo =
        new ClusteringAlgorithm_neighbor< T >( &*StackedTrackerGeomHandle );

      _theAlgo = boost::shared_ptr< ClusteringAlgorithm< T > >( ClusteringAlgo );
      return _theAlgo;
    }

}; /// Close class

#endif

