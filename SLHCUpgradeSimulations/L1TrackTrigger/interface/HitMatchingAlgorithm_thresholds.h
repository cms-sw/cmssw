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
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/classInfo.h"

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
      std::vector< edm::ParameterSet > vpset_;  
      const classInfo     *info_;

    public:
      using HitMatchingAlgorithm< T >::theStackedTracker;

      /// Constructor
      HitMatchingAlgorithm_thresholds( const StackedTrackerGeometry *aStackedTracker,
                                       const edm::ParameterSet& pset )
        : HitMatchingAlgorithm< T >( aStackedTracker ),
          vpset_(pset.getParameter< std::vector< edm::ParameterSet > >("Thresholds")),
          info_( new classInfo(__PRETTY_FUNCTION__)){}

      /// Destructor
      ~HitMatchingAlgorithm_thresholds(){}

      /// Matching operations
      void CheckTwoMemberHitsForCompatibility( bool &aConfirmation, int &aDisplacement, int &anOffset, const L1TkStub< T > &aL1TkStub ) const;

      /// Algorithm name
      std::string AlgorithmName() const 
      { 
        return ( (info_->FunctionName()) + "<" + (info_->TemplateTypes().begin()->second) + ">" );
      }

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
    GlobalPoint pos = aL1TkStub.getClusterPtr(1)->findAverageGlobalPosition( HitMatchingAlgorithm< T >::theStackedTracker ); 

    /// Layer number
    unsigned int layer = stDetId.layer();
    
    std::vector<edm::ParameterSet>::const_iterator ipset;
    for (ipset = vpset_.begin(); ipset!=vpset_.end(); ipset++)
    {  
      /// Check layer
      if (ipset->getParameter<unsigned int>("Layer")!=layer) continue;

      /// Extract row threshold options
      std::vector< unsigned int > rowcuts = ipset->getParameter< std::vector< unsigned int > >("RowCuts");
      std::vector< unsigned int > rowoffsets = ipset->getParameter< std::vector< unsigned int > >("RowOffsets");
      std::vector< unsigned int > rowwindows = ipset->getParameter< std::vector< unsigned int > >("RowWindows");

      /// Find row cut
      unsigned int i = 0;
      for (i=0; i<rowcuts.size(); i++)
      {
        if (outer.y()<rowcuts[i]) break;
      }

      /// Set row thresholds
      unsigned int rowoffset = rowoffsets[i]; 
      unsigned int rowwindow = rowwindows[i];
      
      /// Set column thresholds
      unsigned int columncut = (pos.eta()>0)?outer.x()-inner.x():inner.x()-outer.x();
      unsigned int columnmin = ipset->getParameter<unsigned int>("ColumnCutMin");
      unsigned int columnmax = ipset->getParameter<unsigned int>("ColumnCutMax");

      /// Decision
      bool row = ((int)inner.y()-(int)outer.y()-(int)rowoffset>=0)&&(inner.y()-outer.y()-rowoffset<rowwindow);
      bool col = (columncut>=columnmin)&&(columncut<=columnmax);

      /// Return comparison  
      if ( row && col )
      {
        aConfirmation = true;

        /// Calculate output
        /// NOTE this assumes equal pitch in both sensors!
        MeasurementPoint mp0 = aL1TkStub.getClusterPtr(0)->findAverageLocalCoordinates();
        MeasurementPoint mp1 = aL1TkStub.getClusterPtr(1)->findAverageLocalCoordinates();
        aDisplacement = mp1.x() - mp0.x();

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

