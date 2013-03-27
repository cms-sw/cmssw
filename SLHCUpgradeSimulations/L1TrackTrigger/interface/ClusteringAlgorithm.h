/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Andrew W. Rose, IC                   ///
///                                      ///
/// 2008                                 ///
/// ////////////////////////////////////////

#ifndef CLUSTERING_ALGO_BASE_H
#define CLUSTERING_ALGO_BASE_H

#include <sstream>
#include <map>

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class ClusteringAlgorithm
  {
    protected:
      /// Data members
      const StackedTrackerGeometry *theStackedTracker;

    public:
      /// Constructors
      ClusteringAlgorithm( const StackedTrackerGeometry *aStackedTracker )
        : theStackedTracker( aStackedTracker ){}

      /// Destructor
      virtual ~ClusteringAlgorithm(){}

      /// Clustering operations
      virtual void Cluster( std::vector< std::vector< T > > &output, const std::vector< T > &input ) const
      {
        output.clear();
      }

      /// Algorithm name
      virtual std::string AlgorithmName() const { return ""; }

  }; /// Close class



#endif

