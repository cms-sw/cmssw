/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon, UNIPD               ///
/// Emmanuele Salvati, Cornell           ///
///                                      ///
/// 2011, June                           ///
/// 2011, October                        ///
/// 2013, January                        ///
/// ////////////////////////////////////////

#include "SimDataFormats/SLHC/interface/L1TkCluster.h"

  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// Get cluster width
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  unsigned int L1TkCluster< edm::Ref< edm::PSimHitContainer > >::findWidth() const
  {
    return theHits.size();
  }

  /// Get hit local coordinates
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  MeasurementPoint L1TkCluster< edm::Ref< edm::PSimHitContainer > >::findHitLocalCoordinates( unsigned int hitIdx ) const
  {
    MeasurementPoint mp( 0, 0 ); /// Dummy values
    return mp;
  }
  
  /// Unweighted average local cluster coordinates
  /// Default template for PixelDigis in *.h
  /// Specialize the template for PSimHits
  template<>
  MeasurementPoint L1TkCluster< edm::Ref< edm::PSimHitContainer > >::findAverageLocalCoordinates() const
  {
    MeasurementPoint mp( 0, 0 ); /// Dummy values
    return mp;
  }

