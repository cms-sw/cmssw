/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, Nov                            ///
///                                      ///
/// ////////////////////////////////////////

#ifndef STACKED_TRACKER_L1TK_BEAM_FORMAT_H
#define STACKED_TRACKER_L1TK_BEAM_FORMAT_H

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

namespace cmsUpgrades{

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  class L1TkBeam {

    public:

    private:
      /// Data members
      unsigned int theBeamType; /// 0 = Beta, 1 = Gauss, 2 = Flat
      GlobalPoint  theBeamPosition;
      double       theSizeX;
      double       theSizeY;
      double       theSizeZ;

    public:
      /// Constructors
      L1TkBeam()
      {
        /// Set default data members
        theBeamType = 99;
        theBeamPosition = GlobalPoint(0.0, 0.0, 0.0);
        theSizeX = 0.0;
        theSizeY = 0.0;
        theSizeZ = 0.0;
      }

      /// Destructor
      ~L1TkBeam()
      {
        /// Nothing to be done
      }

      /// //////////////////////// ///
      /// METHODS FOR DATA MEMBERS ///
      /// Beam position
      GlobalPoint getBeamPosition() const
      {
        return theBeamPosition;    
      }

      void setBeamPosition( double aPosX, double aPosY, double aPosZ )
      {
        theBeamPosition = GlobalPoint( aPosX, aPosY, aPosZ );
      }

      /// Beam size
      double getSizeX() const
      {
        return theSizeX;
      }

      double getSizeY() const
      {
        return theSizeY;
      }

      double getSizeZ() const
      {
        return theSizeZ;
      }

      void setBeamSize( double aSizeX, double aSizeY, double aSizeZ )
      {
        theSizeX = aSizeX;
        theSizeY = aSizeY;
        theSizeZ = aSizeZ;
      }



      /// Beam type
      unsigned int getBeamType() const
      {
        return theBeamType;
      }

      void setBeamType( unsigned int aBeamType )
      {
        theBeamType = aBeamType;
      }

      /// ////////////// ///
      /// HELPER METHODS ///

      /// /////////////////// ///
      /// INFORMATIVE METHODS ///

  }; /// Close class


  /** ***************************** **/
  /**                               **/
  /**   IMPLEMENTATION OF METHODS   **/
  /**                               **/
  /** ***************************** **/

  /// THIS IS NOT A TEMPLATE CLASS,
  /// so the definition is typed within the first instance
  /// found by the compiler, otherwise, to implement it
  /// outside of the class implementation, you need the inline
  /// keyword to be used

} /// Close namespace

#endif




