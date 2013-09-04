/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon,  UNIPD              ///
///                                      ///
/// 2011, September                      ///
/// ////////////////////////////////////////

#ifndef TRACKING_ALGO_BASE_H
#define TRACKING_ALGO_BASE_H

#include <sstream>
#include <map>
#include <string>
#include "classNameFinder.h"

#include "SimDataFormats/SLHC/interface/L1TkTrack.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

/** ************************ **/
/**                          **/
/**   DECLARATION OF CLASS   **/
/**                          **/
/** ************************ **/

template< typename T >
class TrackingAlgorithm
{
  protected:
    /// Data members
    const StackedTrackerGeometry *theStackedTracker;
    std::string                  className_;

  public:
    /// Constructors
    TrackingAlgorithm( const StackedTrackerGeometry *aStackedGeom, std::string fName )
      : theStackedTracker( aStackedGeom )
    {
      className_=classNameFinder<T>(fName);
    }

    /// Destructor
    virtual ~TrackingAlgorithm() {}

    /// Seed creation
    virtual void CreateSeeds( std::vector< L1TkTrack< T > > &output,
                              std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< L1TkStub< T > > > > *outputSectorMap,
                              edm::Handle< std::vector< L1TkStub< T > > > &input ) const
    {
      output.clear();
    }

    /// Match a Stub to a Seed/Track
    virtual void AttachStubToSeed( L1TkTrack< T > &seed, edm::Ptr< L1TkStub< T > > &candidate ) const
    {
      seed.addStubPtr( candidate );
    }

    /// AM Pattern Finding
    virtual void PatternFinding() const
    {}

    /// AM Pattern Recognition
    virtual void PatternRecognition() const
    {}

    virtual unsigned int ReturnNumberOfSectors() const { return 1; } 
    virtual unsigned int ReturnNumberOfWedges() const { return 1; }
    virtual double ReturnMagneticField() const { return 1.0; }

    /// Fit the Track
    virtual void FitTrack( L1TkTrack< T > &seed ) const;

    /// Algorithm name
    virtual std::string AlgorithmName() const { return className_; }

}; /// Close class

/// Fit the track
template< typename T >
void TrackingAlgorithm< T >::FitTrack( L1TkTrack< T > &seed ) const
{
  /// First step is to calculate derivatives
  /// First index is parameter, second index is stub
  /// Even stub index is for transverse plane, odd for r-z
  /// Fit parameters are phi0, rInv, cotTheta0, z0
  double D_[4][40] = {{0.}}; /// NOTE: in the note, Dij is derivative wrt. var j at point i
                             /// here in the code D_ is the transpose of ij
  double M_[4][8] = {{0.}};
  double MinvDt_[4][40] = {{0.}};
  unsigned int js = 0; /// Use js as looping index over stubs in matrices

  /// Get the Stubs and other information from the seed
  std::vector< edm::Ptr< L1TkStub< T > > > curStubs = seed.getStubPtrs();
  GlobalVector seedMomentum = seed.getMomentum();
  GlobalPoint seedVertex = seed.getVertex();
  double seedRInv = seed.getRInv();
  double seedPhi0 = seedMomentum.phi();
  double seedZ0 = seedVertex.z();
  double seedCotTheta0 = tan( M_PI_2 - seedMomentum.theta() );

  /// Loop over the Stubs and calculate derivatives at each point
  /// Limit the procedure to 20 stubs
  if ( curStubs.size() > 20 )
    return;
  for ( unsigned int is = 0; is < curStubs.size(); is++ )
  {
    /// Get the stub global position
    GlobalPoint curPos = TrackingAlgorithm< T >::theStackedTracker->findGlobalPosition( curStubs.at(is).get() );
    double stubRho = curPos.perp();

    /// Resolution from Pixel Pitch
    StackedTrackerDetId stDetId( curStubs.at(is)->getClusterPtr(0)->getDetId() );
    const GeomDetUnit* det0 = TrackingAlgorithm< T >::theStackedTracker->idToDetUnit( stDetId, 0 );
    const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
    const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
    std::pair< float, float > pitch0 = top0->pitch();
    double sigmaRhoPhi = pitch0.first / sqrt(12) * curStubs.at(is)->getClusterPtr(0)->findWidth(); /// Take into account cluster width!
    double sigmaZ = pitch0.second / sqrt(12); /// This is valid for both Barrel and Endcap Stubs,
                                              /// it is the calculation of derivatives that changes

    if ( stDetId.isBarrel() )
    {
      /// Transverse plane: [0] rInv, [1] phi0
      /// d(rho*phi)/d(rInv)
      D_[0][js] = -0.5 * stubRho * stubRho / sqrt( 1 - 0.25*stubRho*stubRho*seedRInv*seedRInv ) / sigmaRhoPhi;
      /// d(rho*phi)/d(phi0)
      D_[1][js] = stubRho / sigmaRhoPhi;
      /// d(rho*phi)/d(cotTheta0) = 0
      /// d(rho*phi)/d(z0) = 0
      js++;

      /// Longitudinal plane: [2] cotTheta0, [3] z0
      /// d(z)/d(rInv) = 0
      /// d(z)/d(phi0) = 0
      /// d(z)/d(cotTheta0)
      D_[2][js] = 2. / seedRInv * asin( 0.5*seedRInv*stubRho ) / sigmaZ;
      /// d(z)/d(z0)
      D_[3][js] = 1.0 / sigmaZ;
      js++;
    }
    else if ( stDetId.isEndcap() )
    {

      double stubZ = curPos.z();
      double stubPhi = curPos.phi();

      double dRho_dRInv      = - 2.0 * sin( 0.5 * seedRInv * ( stubZ - seedZ0 ) / seedCotTheta0 ) / ( seedRInv * seedRInv )
                               +       cos( 0.5 * seedRInv * ( stubZ - seedZ0 ) / seedCotTheta0 ) * ( stubZ - seedZ0 ) / ( seedRInv * seedCotTheta0 );
      double dRho_dPhi0      = 0;
      double dRho_dCotTheta0 = - cos( 0.5 * seedRInv * ( stubZ - seedZ0 ) / seedCotTheta0 ) * ( stubZ - seedZ0 ) / ( seedCotTheta0 * seedCotTheta0 );
      double dRho_dZ0        = - cos( 0.5 * seedRInv * ( stubZ - seedZ0 ) / seedCotTheta0 ) / ( seedCotTheta0 );

      /// Radial plane
      /// d(rho)/d(rInv)
      D_[0][js] = dRho_dRInv / sigmaZ;
      /// d(rho)/d(phi0)
      D_[1][js] = dRho_dPhi0 / sigmaZ;
      /// d(rho)/d(cotTheta0)
      D_[2][js] = dRho_dCotTheta0 / sigmaZ;
      /// d(rho)/d(z0)
      D_[3][js] = dRho_dZ0 / sigmaZ;
      js++;

      double dPhi_dRInv      = - 0.5 * ( stubZ - seedZ0 ) / ( seedCotTheta0 );
      double dPhi_dPhi0      = 1;
      double dPhi_dCotTheta0 = 0.5 * seedRInv * ( stubZ - seedZ0 ) / ( seedCotTheta0 * seedCotTheta0 ); 
      double dPhi_dZ0        = 0.5 * seedRInv / seedCotTheta0;

      /// Transverse plane
      /*
      /// OLD VERSION
      /// d(rho*phi)/d(rInv)
      D_[0][js] = stubRho * dPhi_dRInv / sigmaRhoPhi;
      /// d(rho*phi)/d(phi0)
      D_[1][js] = stubRho * dPhi_dPhi0 / sigmaRhoPhi;
      /// d(rho*phi)/d(cotTheta0)
      D_[2][js] = stubRho * dPhi_dCotTheta0 / sigmaRhoPhi;
      /// d(rho*phi)/d(z0)
      D_[3][js] = stubRho * dPhi_dZ0 / sigmaRhoPhi;
      */

      /// NEW VERSION - non-pointing strips correction
      double rhoTrack = 2.0 * sin( 0.5 * seedRInv * ( stubZ - seedZ0 ) / seedCotTheta0 ) / seedRInv;
      double phiTrack = seedPhi0 - 0.5 * seedRInv * ( stubZ - seedZ0 ) / seedCotTheta0;

      /// Calculate a correction for non-pointing-strips in square modules
      /// Relevant angle is the one between hit and module center, with
      /// vertex at (0, 0). Take snippet from HitMatchingAlgorithm_window201*
      /// POSITION IN TERMS OF PITCH MULTIPLES:
      ///       0 1 2 3 4 5 5 6 8 9 ...
      /// COORD: 0 1 2 3 4 5 6 7 8 9 ...
      /// OUT   | | | | | |x| | | | | | | | | |
      ///
      /// IN    | | | |x|x| | | | | | | | | | |
      ///             THIS is 3.5 (COORD) and 4.0 (POS)
      /// The center of the module is at NROWS/2 (position) and NROWS-0.5 (coordinates)
      MeasurementPoint stubCoord = curStubs.at(is)->getClusterPtr(0)->findAverageLocalCoordinates();
      double stubTransvDispl = pitch0.first * ( stubCoord.x() - (top0->nrows()/2 - 0.5) ); /// Difference in coordinates is the same as difference in position
      if ( stubZ > 0 )
      {
        stubTransvDispl = - stubTransvDispl;
      }
      double stubPhiCorr = asin( stubTransvDispl / stubRho );
      double rhoMultiplier = - sin( stubPhiCorr - phiTrack + stubPhi );
      double phiMultiplier = rhoTrack * cos( stubPhiCorr - phiTrack + stubPhi ); 

      /// d(rho*phi)/d(rInv)
      D_[0][js] = ( phiMultiplier * dPhi_dRInv + rhoMultiplier * dRho_dRInv ) / sigmaRhoPhi;
      /// d(rho*phi)/d(phi0)
      D_[1][js] = ( phiMultiplier * dPhi_dPhi0 + rhoMultiplier * dRho_dPhi0 ) / sigmaRhoPhi;
      /// d(rho*phi)/d(cotTheta0)
      D_[2][js] = ( phiMultiplier * dPhi_dCotTheta0 + rhoMultiplier * dRho_dCotTheta0 ) / sigmaRhoPhi;
      /// d(rho*phi)/d(z0)
      D_[3][js] = ( phiMultiplier * dPhi_dZ0 + rhoMultiplier * dRho_dZ0 ) / sigmaRhoPhi;
      js++;
    }
  }

  /// Calculate M-matrix
  /// now, by construction, js = 2*curStubs.size()
  for ( unsigned int i1 = 0; i1 < 4; i1++ )
  {
    for ( unsigned int i2 = 0; i2 < 4; i2++ )
    {
      for ( unsigned int i3 = 0; i3 < js; i3++ )
      {
        M_[i1][i2] += D_[i1][i3]*D_[i2][i3];
      }
    }
  }

  /// Invert the M-matrix
  double ratio, a;

  /// Step 1
  for ( unsigned int i = 0; i < 4; i++ )
  {
    for ( unsigned int j = 4; j < 8; j++ )
    {
      if ( i == (j - 4) )
        M_[i][j] = 1.0;
      else
        M_[i][j] = 0.0;
    }
  }

  /// Step 2
  for ( unsigned int i = 0; i < 4; i++ )
  {
    for ( unsigned int j = 0; j < 4; j++ )
    {
      if ( i != j )
      {
        ratio = M_[j][i]/M_[i][i];

        for ( unsigned int k = 0; k < 8; k++ )
        {
           M_[j][k] -= ratio * M_[i][k];
        }
      }
    }
  }

  /// Step 3
  for ( unsigned int i = 0; i < 4; i++ )
  {
    a = M_[i][i];
    for ( unsigned int j = 0; j < 8; j++)
    {
        M_[i][j] /= a;
    }
  }

  /// Calculate MinvDt-matrix
  /// now, by construction, js = 2*curStubs.size()
  for ( unsigned int j = 0; j < js; j++ )
  {
    for ( unsigned int i1 = 0; i1 < 4; i1++ )
    {
      for ( unsigned int i2 = 0; i2 < 4; i2++ )
      {
        MinvDt_[i1][j] += M_[i1][i2+4] * D_[i2][j];
      }
    }
  }

  /// Calculate the residuals
  double delta[40] = {0};
  double chiSquare = 0;
  js = 0; /// re-initialize js

  for ( unsigned int is = 0; is < curStubs.size(); is++ )
  {
    /// Get the stub global position
    GlobalPoint curPos = TrackingAlgorithm< T >::theStackedTracker->findGlobalPosition( curStubs.at(is).get() );
    double stubRho = curPos.perp();
    double stubPhi = curPos.phi();
    double stubZ = curPos.z();

    /// Resolution from Pixel Pitch
    StackedTrackerDetId stDetId( curStubs.at(is)->getClusterPtr(0)->getDetId() );
    const GeomDetUnit* det0 = TrackingAlgorithm< T >::theStackedTracker->idToDetUnit( stDetId, 0 );
    const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
    const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
    std::pair< float, float > pitch0 = top0->pitch();
    double sigmaRhoPhi = pitch0.first / sqrt(12);
    double sigmaZ = pitch0.second / sqrt(12);

    /// Residuals!
    if ( stDetId.isBarrel() )
    {
      double deltaPhi = seedPhi0 - asin( 0.5*stubRho*seedRInv ) - stubPhi;
      if ( deltaPhi > M_PI )
        deltaPhi -= 2*M_PI;
      if ( deltaPhi < -1*M_PI )
        deltaPhi += 2*M_PI;

      /// Transverse plane  
      delta[js++] = stubRho * deltaPhi / sigmaRhoPhi;
      /// Longitudinal plane
      delta[js++] = ( seedZ0 + (2./seedRInv)*seedCotTheta0*asin( 0.5*stubRho*seedRInv ) - stubZ ) / sigmaZ;
    }
    else if ( stDetId.isEndcap() )
    {
      /*
      double deltaPhi = seedPhi0 - 0.5 * seedRInv * ( stubZ - seedZ0 ) / seedCotTheta0 - stubPhi;
      if ( deltaPhi > M_PI )
        deltaPhi -= 2*M_PI;
      if ( deltaPhi < -1*M_PI )
        deltaPhi += 2*M_PI;
      */
      /*
      /// OLD VERSION
      /// Radial plane
      delta[js++] = ( 2.0 * sin ( 0.5 * seedRInv * ( stubZ - seedZ0 ) / seedCotTheta0 ) / seedRInv - stubRho ) / sigmaZ;
      /// Transverse plane
      delta[js++] = deltaPhi / ( sigmaRhoPhi / stubRho );
      */

      double rhoTrack = 2.0 * sin( 0.5 * seedRInv * ( stubZ - seedZ0 ) / seedCotTheta0 ) / seedRInv;
      double phiTrack = seedPhi0 - 0.5 * seedRInv * ( stubZ - seedZ0 ) / seedCotTheta0;
      double stubPhi = curPos.phi();

      /// Calculate a correction for non-pointing-strips in square modules
      /// Relevant angle is the one between hit and module center, with
      /// vertex at (0, 0). Take snippet from HitMatchingAlgorithm_window201*
      /// POSITION IN TERMS OF PITCH MULTIPLES:
      ///       0 1 2 3 4 5 5 6 8 9 ...
      /// COORD: 0 1 2 3 4 5 6 7 8 9 ...
      /// OUT   | | | | | |x| | | | | | | | | |
      ///
      /// IN    | | | |x|x| | | | | | | | | | |
      ///             THIS is 3.5 (COORD) and 4.0 (POS)
      /// The center of the module is at NROWS/2 (position) and NROWS-0.5 (coordinates)
      MeasurementPoint stubCoord = curStubs.at(is)->getClusterPtr(0)->findAverageLocalCoordinates();
      double stubTransvDispl = pitch0.first * ( stubCoord.x() - (top0->nrows()/2 - 0.5) ); /// Difference in coordinates is the same as difference in position
      if ( stubZ > 0 )
      {
        stubTransvDispl = - stubTransvDispl;
      }
      double stubPhiCorr = asin( stubTransvDispl / stubRho );
      //double rhoMultiplier = - sin( stubPhiCorr - phiTrack + stubPhi );
      //double phiMultiplier = rhoTrack * cos( stubPhiCorr - phiTrack + stubPhi );
      double deltaRhoPhiCorr = stubTransvDispl - rhoTrack*sin( stubPhiCorr - phiTrack + stubPhi );

      /// NEW VERSION
      /// Radial plane
      delta[js++] = ( rhoTrack - stubRho ) / sigmaZ;
      /// Transverse plane
      delta[js++] = deltaRhoPhiCorr / sigmaRhoPhi;
    }

    /// Update the chiSquare
    chiSquare += ( delta[js-2]*delta[js-2] + delta[js-1]*delta[js-1] );
  }

  /// Calculate the steps in the linearized fit
  double dRInv = 0.0;
  double dPhi0 = 0.0;
  double dCotTheta0 = 0.0;
  double dZ0 = 0.0;
  double dRInv_c = 0.0;
  double dPhi0_c = 0.0;
  double dCotTheta0_c = 0.0;
  double dZ0_c = 0.0;

  /// Matrix glory in action
  for ( unsigned int j = 0; j < js; j++ )
  {
    dRInv      -= MinvDt_[0][j]*delta[j];
    dPhi0      -= MinvDt_[1][j]*delta[j];
    dCotTheta0 -= MinvDt_[2][j]*delta[j];
    dZ0        -= MinvDt_[3][j]*delta[j];

    dRInv_c      += D_[0][j]*delta[j];
    dPhi0_c      += D_[1][j]*delta[j];
    dCotTheta0_c += D_[2][j]*delta[j];
    dZ0_c        += D_[3][j]*delta[j];
  }

  double deltaChiSquare = dRInv*dRInv_c + dPhi0*dPhi0_c + dCotTheta0*dCotTheta0_c + dZ0*dZ0_c;
  chiSquare += deltaChiSquare;

  /// Update the Track
  double newRInv = seedRInv + dRInv;
  double newPhi0 = seedPhi0 + dPhi0;
  double newCotTheta0 = seedCotTheta0 + dCotTheta0;

  double mMagneticField = this->ReturnMagneticField();

  double newPt = fabs( mMagneticField*0.003 / newRInv );
  seed.setMomentum( GlobalVector( newPt*cos(newPhi0),
                                  newPt*sin(newPhi0),
                                  newPt*newCotTheta0 ) );
  seed.setVertex( GlobalPoint( 0, 0, seedZ0 + dZ0 ) );
  seed.setRInv( newRInv );
  seed.setChi2( chiSquare );
}

#endif

