/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, Oct                            ///
///                                      ///
/// ////////////////////////////////////////

#ifndef STUB_PAIRING_ALGORITHM_globalgeometry_H
#define STUB_PAIRING_ALGORITHM_globalgeometry_H

#include <memory>
#include <string>
#include <map>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/StubPairingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/StubPairingAlgorithmRecord.h"

#include "SLHCUpgradeSimulations/Utilities/interface/constants.h"
#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

#include <boost/shared_ptr.hpp>

namespace cmsUpgrades {

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template<  typename T  >
  class StubPairingAlgorithm_globalgeometry : public StubPairingAlgorithm< T > {

    private:
      /// Data members
      /// Matching operations
      double mPtThreshold;
      double mCompatibilityScalingFactor;
      double mFastPhiCut;
      double mIPWidth;
      double mPtFactor;
      /// Other stuff
      const cmsUpgrades::classInfo *mClassInfo;

    public:
      /// Constructor
      StubPairingAlgorithm_globalgeometry( const cmsUpgrades::StackedTrackerGeometry *i ,
                                           double aCompatibilityScalingFactor ,
                                           double aIPWidth,
                                           double aPtFactor,
                                           double aFastPhiCut ) : 
                                           cmsUpgrades::StubPairingAlgorithm< T >( i ),
                                           mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) ),
                                           mCompatibilityScalingFactor(aCompatibilityScalingFactor),
                                           mIPWidth(aIPWidth),
                                           mPtFactor(aPtFactor),
                                           mFastPhiCut(aFastPhiCut){}
      /// Destructor
      ~StubPairingAlgorithm_globalgeometry(){}

      /// ////////////// ///
      /// HELPER METHODS ///
      /// Matching operations
      bool CheckTwoStackStubsForCompatibility( edm::Ptr< L1TkStub< T > > innerStub, edm::Ptr< L1TkStub< T > > outerStub ) const;

      /// Algorithm name
      std::string AlgorithmName() const { 
        return ( (mClassInfo->FunctionName())+"<"+(mClassInfo->TemplateTypes().begin()->second)+">" );
      }

  }; /// Close class



  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// ////////////// ///
  /// HELPER METHODS ///
  /// ////////////// ///

  /// Matching operations
  template<typename T>
  bool StubPairingAlgorithm_globalgeometry< T >::CheckTwoStackStubsForCompatibility( edm::Ptr< L1TkStub< T > > innerStub, edm::Ptr< L1TkStub< T > > outerStub ) const
  {
    /// NOTE
    /// This is just as HitMatchingAlgorithm_globalgeometry

    /// Get average position of Stubs composing the Tracklet
    GlobalPoint innerStubPosition = (*innerStub).getPosition();
    GlobalPoint outerStubPosition = (*outerStub).getPosition();

    /// Get useful quantities
    double outerPointRadius = outerStubPosition.perp();
    double innerPointRadius = innerStubPosition.perp();
    double outerPointPhi = outerStubPosition.phi();
    double innerPointPhi = innerStubPosition.phi();

    /// Check for seed compatibility given a pt cut
    /// Threshold computed from radial location of hits    
    double deltaRadius = outerPointRadius - innerPointRadius;
    double deltaPhiThreshold = deltaRadius * mCompatibilityScalingFactor;  

    /// Calculate angular displacement from hit phi locations
    /// and renormalize it, if needed
    double deltaPhi = outerPointPhi - innerPointPhi;
    //if (deltaPhi < 0) deltaPhi = -deltaPhi;
    //if (deltaPhi > cmsUpgrades::KGMS_PI) deltaPhi = 2*cmsUpgrades::KGMS_PI - deltaPhi;
    if ( fabs(deltaPhi) >= cmsUpgrades::KGMS_PI) {
      if ( deltaPhi>0 ) deltaPhi = deltaPhi - 2*cmsUpgrades::KGMS_PI;
      else deltaPhi = 2*cmsUpgrades::KGMS_PI - fabs(deltaPhi);
    }
    double deltaPhiC = deltaPhi; /// This is for charge
    deltaPhi = fabs(deltaPhi);

    /// Rough search in Phi
    if ( deltaPhi < mFastPhiCut ) {
      /// Rough search in Z
      if ( (innerStubPosition.z()>0 && outerStubPosition.z()>-mIPWidth) ||
           (innerStubPosition.z()<0 && outerStubPosition.z()<+mIPWidth) ) {
        /// Detailed search in Phi
        if ( deltaPhi < deltaPhiThreshold ) {
          /// Calculate projections to the beampline
          double positiveZBoundary = (mIPWidth - outerStubPosition.z()) * deltaRadius;
          double negativeZBoundary = -(mIPWidth + outerStubPosition.z()) * deltaRadius;
          double multipliedLocation = (innerStubPosition.z() - outerStubPosition.z()) * outerPointRadius;
          /// Detailed search in Z
          if ( ( multipliedLocation < positiveZBoundary ) &&
               ( multipliedLocation > negativeZBoundary ) ) {
                  
            /// We have the tracklet
            return true;

          } /// End of detailed search in Z
        } /// End of detailed search in Phi
      } /// End of rough search by Z sector
    } /// End of rough search in Phi

    /// Default
    return false;
  }
 
} /// Close namespace



/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template< typename T >
class  ES_StubPairingAlgorithm_globalgeometry: public edm::ESProducer{

  private:
    /// Data members
    boost::shared_ptr< cmsUpgrades::StubPairingAlgorithm<T> > _theAlgo;
    double mPtThreshold;
    double mIPWidth;
    double mFastPhiCut;

  public:
    /// Constructor
    ES_StubPairingAlgorithm_globalgeometry( const edm::ParameterSet & p ) :
                                            mPtThreshold( p.getParameter<double>("minPtThreshold") ),
                                            mIPWidth( p.getParameter<double>("ipWidth") ),
                                            mFastPhiCut( p.getParameter<double>("fastPhiCut") )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_StubPairingAlgorithm_globalgeometry(){}

    /// ///////////////// ///
    /// MANDATORY METHODS ///
    /// Implement the producer
    boost::shared_ptr< cmsUpgrades::StubPairingAlgorithm<T> > produce( const cmsUpgrades::StubPairingAlgorithmRecord & record )
    {
      /// Get magnetic field
      edm::ESHandle<MagneticField> magnet;
      record.getRecord<IdealMagneticFieldRecord>().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();

      /// Calculate scaling factor based on B and Pt threshold
      double mCompatibilityScalingFactor = (cmsUpgrades::KGMS_C * mMagneticFieldStrength) / (100.0 * 2.0e+9 * mPtThreshold);

      /// Calculate factor for rough Pt estimate
      /// B rounded to 4.0 or 3.8
      /// This is B * C / 2 * appropriate power of 10
      /// So it's B * 0.0015
      double mPtFactor = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0*0.0015;

      edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
      record.getRecord<cmsUpgrades::StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );
  
      cmsUpgrades::StubPairingAlgorithm< T >* StubPairingAlgo =
        new cmsUpgrades::StubPairingAlgorithm_globalgeometry< T >( &(*StackedTrackerGeomHandle),
                                                                   mCompatibilityScalingFactor, mIPWidth, mPtFactor, mFastPhiCut );

      _theAlgo  = boost::shared_ptr< cmsUpgrades::StubPairingAlgorithm< T > >( StubPairingAlgo );
      return _theAlgo;
    }

}; /// Close class

#endif

