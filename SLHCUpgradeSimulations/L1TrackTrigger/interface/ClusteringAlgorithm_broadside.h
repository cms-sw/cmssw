/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Andrew W. Rose, IC                   ///
///                                      ///
/// 2008                                 ///
/// ////////////////////////////////////////

#ifndef CLUSTERING_ALGORITHM_broadside_H
#define CLUSTERING_ALGORITHM_broadside_H

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
  class ClusteringAlgorithm_broadside : public ClusteringAlgorithm< T >
  {
    private:
      /// Data members
      int                          mWidthCut; /// Cluster max width

    public:
      /// Constructor
      ClusteringAlgorithm_broadside( const StackedTrackerGeometry *aStackedTracker, int aWidthCut )
        : ClusteringAlgorithm< T >( aStackedTracker, __func__ )
      {
        mWidthCut = aWidthCut;
      }

      /// Destructor
      ~ClusteringAlgorithm_broadside(){}

      /// Clustering operations  
      void Cluster( std::vector< std::vector< T > > &output, const std::vector< T > &input ) const;

  }; /// Close class

  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// Clustering operations
  template<typename T>
  void ClusteringAlgorithm_broadside< T >::Cluster( std::vector< std::vector< T > > &output,
                                                    const std::vector< T > &input ) const
  {
    /// Prepare the output
    output.clear();

    /// Prepare a proper hit container
    std::map< unsigned int, std::vector< T > >                            local;

    /// Map all the hits by column index
    typename std::vector< T >::const_iterator inputIterator;
    inputIterator = input.begin();
    while ( inputIterator != input.end() )
     {
      local[(**inputIterator).column()].push_back(*inputIterator);
      ++inputIterator;
    }

    /// Loop over the mapped hits
    typename std::map< unsigned int, std::vector< T > >::iterator mapIterator;
    mapIterator = local.begin();
    while ( mapIterator != local.end() )
    {
      /// Collect hits sharing column index and
      /// differing by 1 in row index
      typename std::vector< T >::iterator inputIterator;
      inputIterator = mapIterator->second.begin();

      /// Loop over single column
      while( inputIterator != mapIterator->second.end() )
      {
        std::vector< T > temp;
        temp.push_back(*inputIterator);
        inputIterator = mapIterator->second.erase(inputIterator);
        typename std::vector< T >::iterator inputIterator2;
        inputIterator2 = inputIterator;

        /// Nested loop
        while( inputIterator2 != mapIterator->second.end() )
        {
          /// Check col/row and add to the cluster
          if( (temp.back()->column() == (**inputIterator2).column()) &&
              ((**inputIterator2).row() - temp.back()->row() == 1) )
          {
            temp.push_back(*inputIterator2);
            inputIterator2 = mapIterator->second.erase(inputIterator2);
          }
          else
            break;

        } /// End of nested loop

        /// Reject all clusters large than the allowed size
        if ( (mWidthCut < 1) || (int(temp.size()) <= mWidthCut) ) output.push_back(temp);
        inputIterator = inputIterator2;

      } /// End of loop over single column
      ++mapIterator;

    } /// End of loop over mapped hits
  } /// End of ClusteringAlgorithm_broadside< ... >::Cluster( ... )



/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template< typename T >
class  ES_ClusteringAlgorithm_broadside: public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< ClusteringAlgorithm<T> > _theAlgo;
    int mWidthCut;

  public:
    /// Constructor
    ES_ClusteringAlgorithm_broadside( const edm::ParameterSet & p )
      : mWidthCut( p.getParameter< int >("WidthCut") )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_ClusteringAlgorithm_broadside(){}

    /// Implement the producer
    boost::shared_ptr< ClusteringAlgorithm< T > > produce( const ClusteringAlgorithmRecord & record )
    { 
      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  
      ClusteringAlgorithm< T >* ClusteringAlgo =
        new ClusteringAlgorithm_broadside< T >( &(*StackedTrackerGeomHandle), mWidthCut );

      _theAlgo = boost::shared_ptr< ClusteringAlgorithm< T > >( ClusteringAlgo );
      return _theAlgo;
    } 

}; /// Close class

#endif

