/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, Sept                           ///
///                                      ///
/// ////////////////////////////////////////

#ifndef TRACKING_ALGORITHM_a_H
#define TRACKING_ALGORITHM_a_H

#include <memory>
#include <string>
#include <map>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithmRecord.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

#include <boost/shared_ptr.hpp>

namespace cmsUpgrades{

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template<  typename T  >
  class TrackingAlgorithm_a : public TrackingAlgorithm< T > {

    private:
      /// Data members
      double mMagneticFieldStrength;
      /// Other stuff
      const cmsUpgrades::classInfo *mClassInfo;

    public:
      /// Constructor
      TrackingAlgorithm_a( const cmsUpgrades::StackedTrackerGeometry *i,
                           double aMagneticFieldStrength ) :
                           TrackingAlgorithm< T >( i ),
                           mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) ),
                           mMagneticFieldStrength( aMagneticFieldStrength ) {}

      /// Destructor
      ~TrackingAlgorithm_a() {}

      /// ////////////// ///
      /// HELPER METHODS ///
      /// Seed propagation
      std::vector< cmsUpgrades::L1TkTrack< T > > PropagateSeed( edm::Ptr< cmsUpgrades::L1TkTracklet< T > > aSeed,
                                                                std::vector< edm::Ptr< cmsUpgrades::L1TkStub< T > > > aBricks  ) const {
        // As a test, the seed tracklet is not propagated at all
        std::vector< edm::Ptr< cmsUpgrades::L1TkStub< T > > > tempChain;
        tempChain.clear();
        for ( unsigned int iStub = 0; iStub < 2; iStub++ ) tempChain.push_back( aSeed->getStubRef( iStub ) );
        /// Here the Track constructor
        cmsUpgrades::L1TkTrack< T > tempTrack( tempChain, aSeed );
        /// Output
        std::vector< cmsUpgrades::L1TkTrack< T > > tempTrackColl;
        tempTrackColl.push_back( tempTrack );
        return tempTrackColl;

      }

      /// Algorithm name
      std::string AlgorithmName() const { 
        return ( (mClassInfo->FunctionName())+"<"+(mClassInfo->TemplateTypes().begin()->second)+">" );
      }

  }; /// Close class

} /// Close namespace



/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template<  typename T  >
class  ES_TrackingAlgorithm_a: public edm::ESProducer{
  public:
    ES_TrackingAlgorithm_a(const edm::ParameterSet & p){setWhatProduced( this );}
    virtual ~ES_TrackingAlgorithm_a() {}

    boost::shared_ptr< cmsUpgrades::TrackingAlgorithm<T> > produce(const cmsUpgrades::TrackingAlgorithmRecord & record)
    {
      /// This is for compatibility with bpphel algorithm
      /// and to guarantee any Track could be fitted

      /// Get magnetic field
      edm::ESHandle<MagneticField> magnet;
      record.getRecord<IdealMagneticFieldRecord>().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();

      /// Calculate B rounded to 4.0 or 3.8
      mMagneticFieldStrength = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0;

      edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
      record.getRecord<cmsUpgrades::StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );
  
      cmsUpgrades::TrackingAlgorithm<T>* TrackingAlgo = new cmsUpgrades::TrackingAlgorithm_a< T >( &(*StackedTrackerGeomHandle),
                                                                                                   mMagneticFieldStrength );

      _theAlgo  = boost::shared_ptr< cmsUpgrades::TrackingAlgorithm< T > >( TrackingAlgo );
      return _theAlgo;

    } 

  private:
    boost::shared_ptr< cmsUpgrades::TrackingAlgorithm<T> > _theAlgo;
};

#endif
