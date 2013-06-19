/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Sebastien Viret                      ///
/// Nicola Pozzobon                      ///
///                                      ///
/// 2013                                 ///
/// ////////////////////////////////////////

#ifndef TRACKING_ALGO_ASSOBE_H
#define TRACKING_ALGO_ASSOBE_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithmRecord.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "CLHEP/Units/PhysicalConstants.h"

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
class TrackingAlgorithm_associativeBarrelEndcap : public TrackingAlgorithm< T >
{
  private :
    /// Data members
    double                       mMagneticField;
    unsigned int                 nSectors;
    unsigned int                 nWedges;

/*
    std::vector< std::vector< double > > tableRPhiBB;
    std::vector< std::vector< double > > tableZBB;
    std::vector< std::vector< double > > tableRPhiBE;
    std::vector< std::vector< double > > tableZBE;
    std::vector< std::vector< double > > tableRPhiEB;
    std::vector< std::vector< double > > tableZEB;
    std::vector< std::vector< double > > tableRPhiEE;
    std::vector< std::vector< double > > tableZEE;
*/
  public:
    /// Constructors
    TrackingAlgorithm_associativeBarrelEndcap( const StackedTrackerGeometry *aStackedGeom,
                                                double aMagneticField, unsigned int aSectors, unsigned int aWedges)
/*                                         std::vector< std::vector< double > > aTableRPhiBB,
                                         std::vector< std::vector< double > > aTableZBB,
                                         std::vector< std::vector< double > > aTableRPhiBE,
                                         std::vector< std::vector< double > > aTableZBE,
                                         std::vector< std::vector< double > > aTableRPhiEB,
                                         std::vector< std::vector< double > > aTableZEB,
                                         std::vector< std::vector< double > > aTableRPhiEE,
                                         std::vector< std::vector< double > > aTableZEE )
*/

      : TrackingAlgorithm< T > ( aStackedGeom, __func__ )
    {
      mMagneticField = aMagneticField;
      nSectors = aSectors;
      nWedges = aWedges;

/*
      tableRPhiBB = aTableRPhiBB;
      tableZBB = aTableZBB;
      tableRPhiBE = aTableRPhiBE;
      tableZBE = aTableZBE;
      tableRPhiEB = aTableRPhiEB;
      tableZEB = aTableZEB;
      tableRPhiEE = aTableRPhiEE;
      tableZEE = aTableZEE;
*/
    }

    /// Destructor
    ~TrackingAlgorithm_associativeBarrelEndcap(){}

    /// Pattern Finding
    void PatternFinding() const;

    /// Pattern Recognition
    void PatternRecognition() const;

    /// Return the number of Sectors
    unsigned int ReturnNumberOfSectors() const { return nSectors; } /// Phi
    unsigned int ReturnNumberOfWedges() const { return nWedges; } /// Eta

    /// Return the value of the magnetic field
    double ReturnMagneticField() const { return mMagneticField; }

    /// Fit the Track
    void FitTrack( L1TkTrack< T > &track ) const;

}; /// Close class

/** ***************************** **/
/**                               **/
/**   IMPLEMENTATION OF METHODS   **/
/**                               **/
/** ***************************** **/

template< typename T >
void TrackingAlgorithm_associativeBarrelEndcap< T >::PatternFinding() const
{
  std::cerr << "Pattern Finding" << std::endl;
}

template< typename T >
void TrackingAlgorithm_associativeBarrelEndcap< T >::PatternRecognition() const
{
  std::cerr << "Pattern Recognition" << std::endl;
}

/// Fit the track
template< typename T >
void TrackingAlgorithm_associativeBarrelEndcap< T >::FitTrack( L1TkTrack< T > &track ) const
{
  std::cerr << "HOUGH!!!" << std::endl;
}

/** ********************** **/
/**                        **/
/**   DECLARATION OF THE   **/
/**    ALGORITHM TO THE    **/
/**       FRAMEWORK        **/
/**                        **/
/** ********************** **/

template< typename T >
class ES_TrackingAlgorithm_associativeBarrelEndcap : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TrackingAlgorithm< T > > _theAlgo;

    /// Number of Sectors
    unsigned int  mSectors;
    unsigned int  mWedges;

  public:
    /// Constructor
    ES_TrackingAlgorithm_associativeBarrelEndcap( const edm::ParameterSet & p )
      : mSectors( p.getParameter< int >("NumSectors") ), mWedges( p.getParameter< int >("NumWedges") )
    {

      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TrackingAlgorithm_associativeBarrelEndcap() {}

    /// Implement the producer
    boost::shared_ptr< TrackingAlgorithm< T > > produce( const TrackingAlgorithmRecord & record )
    {
      /// Get magnetic field
      edm::ESHandle< MagneticField > magnet;
      record.getRecord< IdealMagneticFieldRecord >().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();
      double mMagneticFieldRounded = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0;

      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );

      TrackingAlgorithm< T >* TrackingAlgo =
        new TrackingAlgorithm_associativeBarrelEndcap< T >( &(*StackedTrackerGeomHandle),
                                                    mMagneticFieldRounded, mSectors, mWedges
);

      _theAlgo = boost::shared_ptr< TrackingAlgorithm< T > >( TrackingAlgo );
      return _theAlgo;
    }

};

#endif

