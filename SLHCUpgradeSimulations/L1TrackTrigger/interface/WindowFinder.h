/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// ////////////////////////////////////////

#ifndef WINDOW_AUX_H
#define WINDOW_AUX_H

#include "Geometry/CommonTopologies/interface/Topology.h" 
#include "Geometry/TrackerGeometryBuilder/src/PixelGeomDetUnit.cc"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include <memory>


  class StackedTrackerWindow
  {
    public:
      StackedTrackerWindow() {}
      StackedTrackerWindow( double aMinrow, double aMaxrow, double aMincol, double aMaxcol )
                          : mMinrow(aMinrow),
                            mMaxrow(aMaxrow),
                            mMincol(aMincol),
                            mMaxcol(aMaxcol){}
      virtual ~StackedTrackerWindow(){}
      double mMinrow;
      double mMaxrow;
      double mMincol;
      double mMaxcol;
  };

  class WindowFinder
  {
    public:
      explicit WindowFinder( const StackedTrackerGeometry *aGeometry,
                             double aPtScalingFactor,
                             double aIPwidth,
                             double aRowResolution,
                             double aColResolution );
      virtual ~WindowFinder();
      void dumphit( const StackedTrackerDetId & anId,
                    unsigned int hitIdentifier,
                    const double & aInnerRow,
                    const double & aInnerColumn );

      StackedTrackerWindow getWindow( const StackedTrackerDetId & anId,
                                      const double & aInnerRow,
                                      const double & aInnerColumn );
  
    private:
      const StackedTrackerGeometry *mGeometry;
      double mPtScalingFactor;
      double mIPwidth;
      double mRowResolution;
      double mColResolution;
      /// These are the variables which need to be filled!
      double mMinrow , mMaxrow , mMincol , mMaxcol;
      /// As all hits in the same stack are tested sequentially, cache the sensor parameters for speed!
      StackedTrackerDetId mLastId;
      PixelGeomDetUnit *mInnerDet , *mOuterDet;
      double mSeparation;
      double mHalfPixelLength;
      double mInnerDetRadius, mInnerDetPhi;
      double mlastInnerRow, mlastInnerCol;
  };



#endif

