/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Andrew W. Rose, IC                   ///
///                                      ///
/// 2008                                 ///
/// ////////////////////////////////////////

#ifndef CLUSTERING_ALGORITHM_2d_H
#define CLUSTERING_ALGORITHM_2d_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithmRecord.h"

#include <boost/shared_ptr.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <bitset>
#include <map>

  /// Container of pixel information
  template< typename T >
  struct pixelContainer
  {
    /// Pixel and neighbours
    const T*         centrePixel;
    std::bitset< 8 > neighbours;

    /// Kill bits (2 of many)
    bool kill0;
    bool kill1;
  };

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class ClusteringAlgorithm_2d : public ClusteringAlgorithm< T >
  {
    private:
      /// Data members
      bool                         mDoubleCountingTest; /// This is to manage double counting

    public:
      /// Constructor
      ClusteringAlgorithm_2d( const StackedTrackerGeometry *aStackedTracker, bool aDoubleCountingTest )
        : ClusteringAlgorithm< T >( aStackedTracker,__func__ )
      { 
        mDoubleCountingTest = aDoubleCountingTest;
      }

      /// Destructor
      ~ClusteringAlgorithm_2d(){}

      /// Clustering operations
      void Cluster( std::vector< std::vector< T > > &output, const std::vector< T > &input ) const;

  }; /// Close class

  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// Clustering operations
  template< typename T >
  void ClusteringAlgorithm_2d< T >::Cluster( std::vector< std::vector< T > > &output,
                                             const std::vector< T > &input ) const
  {
    /// Prepare the output
    output.clear();

    /// Prepare a proper hit container
    std::map< std::pair< unsigned int, unsigned int>, pixelContainer< T > >                     hitContainer;
    typename std::map< std::pair< unsigned int, unsigned int >, pixelContainer< T > >::iterator centralPixel;

    /// First fill all, put the hits into a grid
    /// Loop over all hits
    typename std::vector< T >::const_iterator inputIterator;
    for( inputIterator = input.begin();
         inputIterator != input.end();
         ++inputIterator )
    {
      /// Assign central Pixel
      /// Assign kill bits
      /// Assign neighbours
      hitContainer[ std::make_pair( (**inputIterator).row(), (**inputIterator).column() ) ].centrePixel = &(*inputIterator);
      hitContainer[ std::make_pair( (**inputIterator).row(), (**inputIterator).column() ) ].kill0 = false;
      hitContainer[ std::make_pair( (**inputIterator).row(), (**inputIterator).column() ) ].kill1 = false;
      hitContainer[ std::make_pair( (**inputIterator).row(), (**inputIterator).column() ) ].neighbours = 0x00;
    } /// End of loop over all hits

    /// Then search to see if neighbour hits exist
    /// Loop over all central pixels
    for( centralPixel = hitContainer.begin();
         centralPixel != hitContainer.end();
         ++centralPixel )
    {
      /// Get the coordinates
      unsigned int row = centralPixel->first.first;
      unsigned int col = centralPixel->first.second;

      /// Layout of the grid to understand what follows
      ///    a  b  c     0  1  2          -->r/phi = increasing row
      ///    d  x  e  =  3  x  4          |
      ///    f  g  h     5  6  7          V  z = decreasing column

      /// Just check if there are neighbours and, if so,
      /// assign the corresponding bit to be true/false

      /// Column +1, rows from -1 to +1
      centralPixel->second.neighbours[0] = ( hitContainer.find( std::make_pair( row-1, col+1 ) ) != hitContainer.end() );
      centralPixel->second.neighbours[1] = ( hitContainer.find( std::make_pair( row  , col+1 ) ) != hitContainer.end() );
      centralPixel->second.neighbours[2] = ( hitContainer.find( std::make_pair( row+1, col+1 ) ) != hitContainer.end() );

      /// Column 0, rows -1 and +1
      centralPixel->second.neighbours[3] = ( hitContainer.find( std::make_pair( row-1, col   ) ) != hitContainer.end() );
      centralPixel->second.neighbours[4] = ( hitContainer.find( std::make_pair( row+1, col   ) ) != hitContainer.end() );

      /// Column -1, rows from -1 to +1
      centralPixel->second.neighbours[5] = ( hitContainer.find( std::make_pair( row-1, col-1 ) ) != hitContainer.end() );
      centralPixel->second.neighbours[6] = ( hitContainer.find( std::make_pair( row  , col-1 ) ) != hitContainer.end() );
      centralPixel->second.neighbours[7] = ( hitContainer.find( std::make_pair( row+1, col-1 ) ) != hitContainer.end() );

    } /// End of loop over all central pixels

    /// Then fill the kill bits
    /// Loop over all central pixels
    for( centralPixel = hitContainer.begin();
         centralPixel != hitContainer.end();
         ++centralPixel )
    {
      /// KB 1) The first kill bit, kill0, prevents a cluster to be larger than 2 pixels in r-phi: if both columns
      /// adf and ceh contain at least one pixel over threshold each, this bit is set to 1, otherwise  it is set to 0
      /// KB 2) The second kill bit, kill1, makes the cluster to be built only if pix is in the leftmostbottom position
      /// within the cluster: if there is a pixel over threshold either in column adf or in position g,
      /// this bit is set to 1, otherwise it is set to 0

      /// Check row -1
      bool adf = centralPixel->second.neighbours[0] | centralPixel->second.neighbours[3] | centralPixel->second.neighbours[5]  ;
      /// Check row +1
      bool ceh = centralPixel->second.neighbours[2] | centralPixel->second.neighbours[4] | centralPixel->second.neighbours[7]  ;

      /// Kill bits are set here
      centralPixel->second.kill0 = ( adf & ceh );
      centralPixel->second.kill1 = ( adf | centralPixel->second.neighbours[6] );

    } /// End of loop over all central pixels

    /// Then cross check for the final kill bit
    /// Loop over all central pixels
    for( centralPixel = hitContainer.begin();
         centralPixel != hitContainer.end();
         ++centralPixel )
    {
      /// Get the coordinates
      unsigned int row = centralPixel->first.first;
      unsigned int col = centralPixel->first.second;

      /// KB 3) if at least one of the pixels, in ceh column, fired and features its kill0 = 1, let a      /// third kill bit kill2 be 1, otherwise set it to 0
      /// NOTE that kill2 prevents the pixel to report a cluster when looking at its size out of the 3x3
      /// pixel window under examination
      bool kill2 = false;
      typename std::map< std::pair< unsigned int, unsigned int >, pixelContainer< T > >::iterator rhs;

      if ( ( rhs = hitContainer.find( std::make_pair( row+1, col-1 ) ) ) != hitContainer.end() ) kill2 |= rhs->second.kill0;
      if ( ( rhs = hitContainer.find( std::make_pair( row+1, col   ) ) ) != hitContainer.end() ) kill2 |= rhs->second.kill0;
      if ( ( rhs = hitContainer.find( std::make_pair( row+1, col+1 ) ) ) != hitContainer.end() ) kill2 |= rhs->second.kill0;

      /// If all the kill bits are fine,
      /// then the Cluster can be prepared for output
      if ( !centralPixel->second.kill0 && !centralPixel->second.kill1 && !kill2 )
      {
        /// Store the central pixel
        std::vector< T > temp;
        temp.push_back( *hitContainer[ std::make_pair( row , col ) ].centrePixel );
        /// Store all the neighbours
        if( centralPixel->second.neighbours[0] ) temp.push_back ( *hitContainer[ std::make_pair( row-1, col+1 ) ].centrePixel );
        if( centralPixel->second.neighbours[1] ) temp.push_back ( *hitContainer[ std::make_pair( row  , col+1 ) ].centrePixel );
        if( centralPixel->second.neighbours[2] ) temp.push_back ( *hitContainer[ std::make_pair( row+1, col+1 ) ].centrePixel );
        if( centralPixel->second.neighbours[3] ) temp.push_back ( *hitContainer[ std::make_pair( row-1, col   ) ].centrePixel );
        if( centralPixel->second.neighbours[4] ) temp.push_back ( *hitContainer[ std::make_pair( row+1, col   ) ].centrePixel );
        if( centralPixel->second.neighbours[5] ) temp.push_back ( *hitContainer[ std::make_pair( row-1, col-1 ) ].centrePixel );
        if( centralPixel->second.neighbours[6] ) temp.push_back ( *hitContainer[ std::make_pair( row  , col-1 ) ].centrePixel );
        if( centralPixel->second.neighbours[7] ) temp.push_back ( *hitContainer[ std::make_pair( row+1, col-1 ) ].centrePixel );
        output.push_back(temp);

      } /// End of "all the kill bits are fine"
    } /// End of loop over all central pixels

    /// Eventually, if needed, do the
    /// test for double counting!
    if( mDoubleCountingTest )
    {
      std::set< std::pair< unsigned int, unsigned int > > test;
      std::set< std::pair< unsigned int, unsigned int > > doubles;
      typename std::vector< std::vector< T > >::iterator outputIterator1;
      typename std::vector< T >::iterator                outputIterator2;
      /// Loop over Clusters
      for ( outputIterator1 = output.begin();
            outputIterator1 != output.end();
            ++outputIterator1 )
      {
        /// Loop over Hits inside each Cluster
        for ( outputIterator2 = outputIterator1->begin();
              outputIterator2 != outputIterator1->end();
              ++outputIterator2 )
        {
          /// Are there Hits with same coordinates?
          /// If yes, put in doubles vector, else in test one
          if ( test.find( std::make_pair( (**outputIterator2).row(), (**outputIterator2).column() ) ) != test.end() )
            doubles.insert( std::make_pair( (**outputIterator2).row(), (**outputIterator2).column() ) );
          else
            test.insert( std::make_pair( (**outputIterator2).row(), (**outputIterator2).column() ) );

        } /// End of loop over Hits inside each Cluster
      } /// End of loop over Clusters

      /// If we found duplicates
      /// WARNING is it really doing something
      /// more than printout???????
      if ( doubles.size() )
      {
        std::set< std::pair< unsigned int, unsigned int> >::iterator it;
        std::stringstream errmsg;
        /// Printout double Pixel
        for ( it = doubles.begin(); it != doubles.end(); ++it )
        {
          errmsg << "Double counted pixel: (" << it->first << "," << it->second << ")\n";
        }

        /// Loop over Clusters
        for ( outputIterator1 = output.begin();
              outputIterator1 != output.end();
              ++outputIterator1 )
        {
          errmsg <<  "cluster: ";
          /// Loop over Hits inside each Cluster
          for ( outputIterator2 = outputIterator1->begin();
                outputIterator2 != outputIterator1->end();
                ++outputIterator2 )
          {
            errmsg << "| (" <<  (**outputIterator2).row() <<","<< (**outputIterator2).column()<< ") ";
          }
          errmsg << "|\n";
        } /// End of loop over Clusters

        edm::LogError("ClusteringAlgorithm_2d") << errmsg.str();

      } /// End of "if we found duplicates"
    } /// End of test for double counting
  } /// End of ClusteringAlgorithm_2d< ... >::Cluster( ... )



/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template< typename T >
class  ES_ClusteringAlgorithm_2d : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< ClusteringAlgorithm< T > > _theAlgo;
    bool mDoubleCountingTest;

  public:
    /// Constructor
    ES_ClusteringAlgorithm_2d( const edm::ParameterSet & p )
      : mDoubleCountingTest( p.getParameter< bool >("DoubleCountingTest") )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_ClusteringAlgorithm_2d(){}

    /// Implement the producer
    boost::shared_ptr< ClusteringAlgorithm< T > > produce( const ClusteringAlgorithmRecord & record )
    {
      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );

      ClusteringAlgorithm< T >* ClusteringAlgo =
        new ClusteringAlgorithm_2d< T >( &(*StackedTrackerGeomHandle), mDoubleCountingTest );

      _theAlgo = boost::shared_ptr< ClusteringAlgorithm< T > >( ClusteringAlgo );
      return _theAlgo;
    } 

}; /// Close class

#endif

