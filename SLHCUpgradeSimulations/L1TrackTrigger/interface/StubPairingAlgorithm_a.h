/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, Oct                            ///
///                                      ///
/// ////////////////////////////////////////

#ifndef STUB_PAIRING_ALGORITHM_a_H
#define STUB_PAIRING_ALGORITHM_a_H

#include <memory>
#include <string>
#include <map>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/StubPairingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/StubPairingAlgorithmRecord.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

#include <boost/shared_ptr.hpp>

namespace cmsUpgrades{

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template<  typename T  >
  class StubPairingAlgorithm_a : public StubPairingAlgorithm< T > {

    private:
      /// Data members
      /// Other stuff
      const cmsUpgrades::classInfo *mClassInfo;

    public:
      /// Constructor
      StubPairingAlgorithm_a( const cmsUpgrades::StackedTrackerGeometry *i ) :
                              StubPairingAlgorithm< T >( i ),
                              mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) ) {}
      /// Destructor
      ~StubPairingAlgorithm_a() {}

      /// ////////////// ///
      /// HELPER METHODS ///
      /// Matching operations
      bool CheckTwoStackStubsForCompatibility( edm::Ptr< L1TkStub< T > > innerStub, edm::Ptr< L1TkStub< T > > outerStub ) const
      { 
        /// As a test, we will accept all pairs of hits in a stack
        return true;
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
class  ES_StubPairingAlgorithm_a: public edm::ESProducer{
  public:
    ES_StubPairingAlgorithm_a(const edm::ParameterSet & p){setWhatProduced( this );}
    virtual ~ES_StubPairingAlgorithm_a() {}

    boost::shared_ptr< cmsUpgrades::StubPairingAlgorithm<T> > produce(const cmsUpgrades::StubPairingAlgorithmRecord & record)
    { 
      edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
      record.getRecord<cmsUpgrades::StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );
  
      cmsUpgrades::StubPairingAlgorithm<T>* HitMatchingAlgo = new cmsUpgrades::StubPairingAlgorithm_a<T>( &(*StackedTrackerGeomHandle) );

      _theAlgo  = boost::shared_ptr< cmsUpgrades::StubPairingAlgorithm<T> >( HitMatchingAlgo );

      return _theAlgo;
    } 

  private:
    boost::shared_ptr< cmsUpgrades::StubPairingAlgorithm<T> > _theAlgo;
};

#endif

