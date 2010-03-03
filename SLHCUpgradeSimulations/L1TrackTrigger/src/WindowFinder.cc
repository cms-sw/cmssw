#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/WindowFinder.h"
#include <iostream>

cmsUpgrades::WindowFinder::WindowFinder( const cmsUpgrades::StackedTrackerGeometry *aGeometry , double aPtScalingFactor , double aIPwidth , double aRowResolution , double aColResolution )
:	mGeometry(aGeometry), 
	mPtScalingFactor( aPtScalingFactor ), 
	mIPwidth( aIPwidth ),
	mRowResolution( aRowResolution ),
	mColResolution( aColResolution ),
	mMinrow(0) , mMaxrow(0) , mMincol(0) , mMaxcol(0) ,
	mLastId(0),
	mlastInnerRow(-1),
	mlastInnerCol(-1)
{
//		std::cout<<"mPtScalingFactor:" << mPtScalingFactor << " & mIPwidth:" << mIPwidth << std::endl;
}


cmsUpgrades::WindowFinder::~WindowFinder()
{
}	

void cmsUpgrades::WindowFinder::dumphit( const StackedTrackerDetId & anId , unsigned int hitIdentifier , const double & aInnerRow , const double & aInnerColumn )
{
	const PixelGeomDetUnit* detunit = reinterpret_cast< const PixelGeomDetUnit* > (mGeometry -> idToDetUnit( anId , hitIdentifier ));

	MeasurementPoint mp	( aInnerRow + (0.5*mRowResolution), aInnerColumn + (0.5*mColResolution) ); // Centre of the pixel.
	LocalPoint LP  = detunit->topology().localPosition( mp )  ;
	GlobalPoint GP = detunit->surface().toGlobal( LP );
	std::cout << (hitIdentifier?"INNER":"OUTER") << " -> eta = " << GP.eta() << std::endl;
}


cmsUpgrades::StackedTrackerWindow cmsUpgrades::WindowFinder::getWindow( const StackedTrackerDetId & anId , const double & aInnerRow , const double & aInnerColumn )
{

	if( (anId == mLastId) && (mlastInnerRow == aInnerRow) && ( mlastInnerCol == aInnerColumn) )
		return cmsUpgrades::StackedTrackerWindow::StackedTrackerWindow (  mMinrow , mMaxrow , mMincol , mMaxcol  );

	mlastInnerRow = aInnerRow;
	mlastInnerCol = aInnerColumn;

	if(anId!=mLastId){
		mLastId = anId;
		mInnerDet = const_cast< PixelGeomDetUnit* >(reinterpret_cast< const PixelGeomDetUnit* > (mGeometry	-> idToDetUnit( anId , 0 )));
		mOuterDet = const_cast< PixelGeomDetUnit* >(reinterpret_cast< const PixelGeomDetUnit* > (mGeometry	-> idToDetUnit( anId , 1 )));

		mHalfPixelLength = mInnerDet->specificTopology().pitch().second * mColResolution * 0.5;
		mSeparation = mInnerDet->surface().localZ( mOuterDet->position() );
		if (mSeparation<0)mSeparation=-mSeparation;

		mInnerDetRadius	= mInnerDet->position().perp();
		mInnerDetPhi			=	mInnerDet->position().phi();
	}

//find the bounds of the inner "pixel" in pixel units
	MeasurementPoint		MP_INNER			( aInnerRow + (0.5*mRowResolution), aInnerColumn + (0.5*mColResolution) ); 	// centre of the "pixel".
//find the bounds of the inner "pixel" in cm
	LocalPoint 						LP_INNER			=		mInnerDet ->topology().localPosition( MP_INNER );
//find the positions of the inner "pixels" corners in global coordinates
	GlobalPoint 					GP_INNER			= 		mInnerDet ->surface().toGlobal( LP_INNER );
//calculate the maximum allowed track angle to the tangent at the "pixels" bounds
	double 							PHI						=		asin( mPtScalingFactor * GP_INNER.perp() );
//calculate the angle of the sensor to the tangent at the bounds
	double 							PixelAngle			=		acos(  sin(	mInnerDetPhi - GP_INNER.phi() ) * mInnerDetRadius / LP_INNER.x() );
//calculate the deviation in the r-phi direction
	double							deltaXminus		=		( mSeparation * tan( PixelAngle - PHI ));	
	double							deltaXplus		=		( mSeparation * tan( PixelAngle + PHI ));	

//inner pixel z-bounds
	double							PIXEL_Z_PLUS		=		GP_INNER.z()+mHalfPixelLength;
	double							PIXEL_Z_MINUS	=		GP_INNER.z()-mHalfPixelLength;
//	std::cout << PIXEL_Z_PLUS << " , " << PIXEL_Z_MINUS << std::endl;
//IP z-bounds
	double							IP_Z_PLUS			=		mIPwidth;
	double							IP_Z_MINUS			=		-mIPwidth;
//	std::cout << IP_Z_PLUS << " , " << IP_Z_MINUS << std::endl;
//stack radial separation through inner hit
	double							R_SEPARATION	=		mSeparation / cos( PixelAngle );
	if (R_SEPARATION<0)	R_SEPARATION	=		-R_SEPARATION;
//calculate the deviation in the z direction
	double							deltaZminus		=		(PIXEL_Z_MINUS-IP_Z_PLUS) * R_SEPARATION / GP_INNER.perp();	
	double							deltaZplus		=		(PIXEL_Z_PLUS-IP_Z_MINUS) * R_SEPARATION / GP_INNER.perp();		
//	std::cout << deltaZminus << " , " << deltaZplus << std::endl;

// --- make boundary points in the inner reference frame --- //
	LocalPoint LP_OUTER_PLUS( LP_INNER.x()-deltaXplus , LP_INNER.y()-mHalfPixelLength-deltaZplus , -mSeparation );
	LocalPoint LP_OUTER_MINUS( LP_INNER.x()-deltaXminus , LP_INNER.y()+mHalfPixelLength-deltaZminus , -mSeparation );
// --- migrate into the global frame --- //
	GlobalPoint GP_OUTER_PLUS = mInnerDet ->surface().toGlobal(LP_OUTER_PLUS);
	GlobalPoint GP_OUTER_MINUS = mInnerDet ->surface().toGlobal(LP_OUTER_MINUS);
// --- migrate into the local frame of the outer det--- //
	LocalPoint LP_OUTER_PLUS_2 = mOuterDet ->surface().toLocal(GP_OUTER_PLUS);
	LocalPoint LP_OUTER_MINUS_2 = mOuterDet ->surface().toLocal(GP_OUTER_MINUS);
// --- convert into pixel units --- //
	std::pair<float,float> PLUS = mOuterDet -> specificTopology().pixel(LP_OUTER_PLUS_2);
	std::pair<float,float> MINUS = mOuterDet -> specificTopology().pixel(LP_OUTER_MINUS_2);

	mMinrow	= mRowResolution * floor( PLUS.first / mRowResolution ); 
	mMincol	= mColResolution * floor( PLUS.second / mColResolution );
	mMaxrow	= mRowResolution * floor( MINUS.first / mRowResolution );
	mMaxcol = mColResolution * floor( MINUS.second / mColResolution ); 

	if(mMinrow>mMaxrow)std::swap(mMinrow,mMaxrow);
	if(mMincol>mMaxcol)std::swap(mMincol,mMaxcol);

// -------------------------------- //


/*
// test 
	double phidiffa = GP_OUTER_PLUS.phi() - GP_INNER.phi();
	double r1a = GP_INNER.perp()/100;
	double r2a = GP_OUTER_PLUS.perp()/100;
	double x2a = r1a*r1a + r2a*r2a - 2*r1a*r2a*cos(phidiffa);
//	double pta = 0.6*sqrt(x2a)/sin(fabs(phidiffa));
	double pta = 0.6*sqrt(x2a)/sin(phidiffa);
	double za = PIXEL_Z_PLUS - ((GP_OUTER_PLUS.z()-PIXEL_Z_PLUS)*GP_INNER.perp()/R_SEPARATION);

	double phidiffb = GP_OUTER_MINUS.phi() - GP_INNER.phi();
	double r1b = GP_INNER.perp()/100;
	double r2b = GP_OUTER_MINUS.perp()/100;
	double x2b = r1b*r1b + r2b*r2b - 2*r1b*r2b*cos(phidiffb);
//	double ptb = 0.6*sqrt(x2b)/sin(fabs(phidiffb));
	double ptb = 0.6*sqrt(x2b)/sin(phidiffb);
	double zb = PIXEL_Z_MINUS - ((GP_OUTER_MINUS.z()-PIXEL_Z_MINUS)*GP_INNER.perp()/R_SEPARATION);

	std::cout<<"PT -> " << pta << "\t" << ptb << std::endl;
	std::cout<<"Z -> " << za << "\t" << zb << std::endl;

	std::cout<< "( " << aInnerRow << " , " << aInnerColumn << " ) -> ( "
	<< MINUS.first <<" , "<< MINUS.second <<" ) & ( "<< PLUS.first <<" , "<<PLUS.second <<" ) -> ( "
	<< mMinrow <<" , "<< mMincol <<" ) & ( "<< mMaxrow <<" , "<< mMaxcol <<" ) " << std::endl;


	MeasurementPoint mpa	( mMinrow, mMincol );
	GlobalPoint gpa =  mOuterDet ->surface().toGlobal( mOuterDet ->topology().localPosition( mpa )  ) ;
	phidiffa = gpa.phi() - GP_INNER.phi();
	r1a = GP_INNER.perp()/100;
	r2a = gpa.perp()/100;
	x2a = r1a*r1a + r2a*r2a - 2*r1a*r2a*cos(phidiffa);
	pta = 0.6*sqrt(x2a)/sin(phidiffa);
	za = GP_INNER.z()- (  (gpa.z()-GP_INNER.z())*GP_INNER.perp()/R_SEPARATION);

	MeasurementPoint mpb	( mMaxrow+0.5, mMaxcol+0.5 );
	GlobalPoint gpb =  mOuterDet ->surface().toGlobal( mOuterDet ->topology().localPosition( mpb )  ) ;
	phidiffb = gpb.phi() - GP_INNER.phi();
	r1b = GP_INNER.perp()/100;
	r2b = gpb.perp()/100;
	x2b = r1b*r1b + r2b*r2b - 2*r1b*r2b*cos(phidiffb);
	ptb = 0.6*sqrt(x2b)/sin(phidiffb);
	zb = GP_INNER.z()- (  (gpb.z()-GP_INNER.z())*GP_INNER.perp()/R_SEPARATION);

	std::cout<<"PT -> " << pta << "\t" << ptb << std::endl;
	std::cout<<"Z -> " << za << "\t" << zb << std::endl;
*/


	return cmsUpgrades::StackedTrackerWindow::StackedTrackerWindow (  mMinrow , mMaxrow , mMincol , mMaxcol  );
}

