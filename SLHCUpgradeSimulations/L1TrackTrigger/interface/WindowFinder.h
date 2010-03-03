// system include files
#include <memory>

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"

#include "Geometry/TrackerGeometryBuilder/src/PixelGeomDetUnit.cc"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h" 

namespace cmsUpgrades{

/*=======================================================================================================================================================*/

class StackedTrackerWindow{
	public:
		StackedTrackerWindow(){}

		StackedTrackerWindow( double aMinrow , double aMaxrow , double aMincol , double aMaxcol):	mMinrow(aMinrow),
														mMaxrow(aMaxrow),
														mMincol(aMincol),
														mMaxcol(aMaxcol)
													{}

		virtual ~StackedTrackerWindow(){}

		double mMinrow;
		double mMaxrow;
		double mMincol;
		double mMaxcol;
};

/*=======================================================================================================================================================*/

class WindowFinder{
	public:
		explicit WindowFinder( const cmsUpgrades::StackedTrackerGeometry *aGeometry , double aPtScalingFactor , double aIPwidth , double aRowResolution , double aColResolution );
		virtual ~WindowFinder();

		void dumphit( const StackedTrackerDetId & anId , unsigned int hitIdentifier , const double & aInnerRow , const double & aInnerColumn );

	  	StackedTrackerWindow getWindow( const StackedTrackerDetId & anId , const double & aInnerRow , const double & aInnerColumn );

	private:
		const cmsUpgrades::StackedTrackerGeometry *mGeometry;
		double mPtScalingFactor;
		double mIPwidth;

		double mRowResolution;
		double mColResolution;

		//these are the variables which need to be filled!
		double mMinrow , mMaxrow , mMincol , mMaxcol;

		//As all hits in the same stack are tested sequentially, cache the sensor parameters for speed!
		StackedTrackerDetId mLastId;
		PixelGeomDetUnit *mInnerDet , *mOuterDet;
		double mSeparation;
		double mHalfPixelLength;

		double mInnerDetRadius , mInnerDetPhi;

		double mlastInnerRow , mlastInnerCol;



};

/*=======================================================================================================================================================*/

}

