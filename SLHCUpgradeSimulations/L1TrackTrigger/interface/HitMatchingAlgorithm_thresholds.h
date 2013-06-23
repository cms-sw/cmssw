/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon,  UNIPD              ///
///                                      ///
/// 2010, May                            ///
/// 2011, June                           ///
/// ////////////////////////////////////////

#ifndef L1TRIGGEROFFLINE_TRIGGERSIMULATION_HITMATCHINGALGORITHM_THRESHOLDS_H
#define L1TRIGGEROFFLINE_TRIGGERSIMULATION_HITMATCHINGALGORITHM_THRESHOLDS_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithmRecord.h"

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template < class T >
  class HitMatchingAlgorithm_thresholds : public HitMatchingAlgorithm< T >
  {  
    private:
      /// Data members
      std::vector< std::vector< unsigned int > >  rowcuts_,rowoffsets_,rowwindows_;
      std::vector<unsigned int> layers_;
      std::vector<unsigned int> columnmin_;
      std::vector<unsigned int> columnmax_;


    public:
      using HitMatchingAlgorithm< T >::theStackedTracker;

      /// Constructor
      HitMatchingAlgorithm_thresholds( const StackedTrackerGeometry *aStackedTracker,
                                       const edm::ParameterSet& pset )
        : HitMatchingAlgorithm< T >( aStackedTracker,__func__ ) {

	std::vector< edm::ParameterSet > vpset=(pset.getParameter< std::vector< edm::ParameterSet > >("Thresholds"));
	std::vector<edm::ParameterSet>::const_iterator ipset;
	for (ipset = vpset.begin(); ipset!=vpset.end(); ipset++)
	  {   
	    /// Extract row threshold options
	    layers_.push_back(ipset->getParameter<unsigned int>("Layer"));
	    rowcuts_.push_back(ipset->getParameter< std::vector< unsigned int > >("RowCuts"));
	    rowoffsets_.push_back(ipset->getParameter< std::vector< unsigned int > >("RowOffsets"));
	    rowwindows_.push_back(ipset->getParameter< std::vector< unsigned int > >("RowWindows"));
	    columnmin_.push_back(ipset->getParameter<unsigned int>("ColumnCutMin"));
	    columnmax_.push_back(ipset->getParameter<unsigned int>("ColumnCutMax"));
	  }
      }

      /// Destructor
      ~HitMatchingAlgorithm_thresholds(){}

      /// Matching operations
      void CheckTwoMemberHitsForCompatibility( bool &aConfirmation, int &aDisplacement, int &anOffset, const L1TkStub< T > &aL1TkStub ) const;


  }; /// Close class

  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// Matching operations
  template< typename T >
  void HitMatchingAlgorithm_thresholds< T >::CheckTwoMemberHitsForCompatibility( bool &aConfirmation, int &aDisplacement, int &anOffset, const L1TkStub< T > &aL1TkStub ) const
  {
    /// Convert DetId
    StackedTrackerDetId stDetId( aL1TkStub.getDetId() );

    /// Force this to be a BARREL-only algorithm
    if ( stDetId.isEndcap() )
    {
      aConfirmation = false;
      return;
    }

    
    /// Find hit positions
    MeasurementPoint inner = aL1TkStub.getClusterPtr(0)->findAverageLocalCoordinates();
    MeasurementPoint outer = aL1TkStub.getClusterPtr(1)->findAverageLocalCoordinates();
    GlobalPoint pos = HitMatchingAlgorithm< T >::theStackedTracker->findAverageGlobalPosition( aL1TkStub.getClusterPtr(1).get() ); 

    /// Layer number
    unsigned int layer = stDetId.layer();
    
    //std::vector<edm::ParameterSet>::const_iterator ipset;
    //for (ipset = vpset_.begin(); ipset!=vpset_.end(); ipset++)
    for ( unsigned int j=0; j<layers_.size(); j++)  
    {  
      /// Check layer
      if (layers_[j]!=layer) continue;

      /// Find row cut
      unsigned int i = 0;
      for (i=0; i<rowcuts_[j].size(); i++)
      {
        if (outer.y()<rowcuts_[j][i]) break;
      }

      /// Set row thresholds
      unsigned int rowoffset = rowoffsets_[j][i]; 
      unsigned int rowwindow = rowwindows_[j][i];
      
      /// Set column thresholds
      unsigned int columncut = (pos.eta()>0)?outer.x()-inner.x():inner.x()-outer.x();

      /// Decision
      bool row = ((int)inner.y()-(int)outer.y()-(int)rowoffset>=0)&&(inner.y()-outer.y()-rowoffset<rowwindow);
      bool col = (columncut>=columnmin_[j])&&(columncut<=columnmax_[j]);

      /// Return comparison  
      if ( row && col )
      {
        aConfirmation = true;

        /// Calculate output
        /// NOTE this assumes equal pitch in both sensors!
        MeasurementPoint mp0 = aL1TkStub.getClusterPtr(0)->findAverageLocalCoordinates();
        MeasurementPoint mp1 = aL1TkStub.getClusterPtr(1)->findAverageLocalCoordinates();
        aDisplacement = 2*(mp1.x() - mp0.x()); /// In HALF-STRIP units!

        /// By default, assigned as ZERO
        anOffset = 0;

      }
    }
  }
 


/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template < class T >
class  ES_HitMatchingAlgorithm_thresholds : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< HitMatchingAlgorithm< T > > algo_;
    edm::ParameterSet pset_;

  public:
    /// Constructor
    ES_HitMatchingAlgorithm_thresholds( const edm::ParameterSet & p )
      : pset_(p)
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_HitMatchingAlgorithm_thresholds(){}
  
    /// Implement the producer
    boost::shared_ptr< HitMatchingAlgorithm< T > > produce( const HitMatchingAlgorithmRecord & record )
    { 
      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );

      algo_ = boost::shared_ptr< HitMatchingAlgorithm< T > >( new HitMatchingAlgorithm_thresholds< T >( &(*StackedTrackerGeomHandle),
                                                                           pset_ ) );
      return algo_;
    } 

}; /// Close class

#endif

