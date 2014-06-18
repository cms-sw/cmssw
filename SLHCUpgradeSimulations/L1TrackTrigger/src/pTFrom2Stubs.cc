
#include <iostream> 
#include <memory>
#include <cmath>
#include <assert.h>

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/pTFrom2Stubs.h"

namespace pTFrom2Stubs{

	//====================
	float rInvFrom2(std::vector< TTTrack< Ref_PixelDigi_> >::const_iterator trk, const StackedTrackerGeometry* theStackedGeometry){

		//vector of R, r and phi for each stub
		std::vector< std::vector<float> > riPhiStubs(0);
		//get stub reference
		std::vector< edm::Ref<edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > vecStubRefs = trk->getStubRefs();

		//loop over L1Track's stubs 
		int rsize =vecStubRefs.size();
		for(int j =0; j< rsize; ++j){

			edm::Ref<edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > stubRef =vecStubRefs.at(j) ;
			const TTStub<Ref_PixelDigi_>* stub=&(*stubRef) ;

			GlobalPoint stubPosition = theStackedGeometry->findGlobalPosition(stub);

			std::vector<float> tmp(0);
			float Rad = sqrt(stubPosition.x()*stubPosition.x() + stubPosition.y()*stubPosition.y() + stubPosition.z()+stubPosition.z());
			float r_i = sqrt(stubPosition.x()*stubPosition.x() + stubPosition.y()*stubPosition.y());
			float  phi_i=stubPosition.phi();

			tmp.push_back(Rad);
			tmp.push_back(r_i);
			tmp.push_back(phi_i);

			riPhiStubs.push_back(tmp);
		}

		std::sort(riPhiStubs.begin(), riPhiStubs.end());
		//now calculate the curvature from first 2 stubs
		float nr1   = (riPhiStubs[0])[1];
		float nphi1 = (riPhiStubs[0])[2];

		float nr2   = (riPhiStubs[1])[1];
		float nphi2 = (riPhiStubs[1])[2];

		float ndeltaphi=nphi1 -nphi2;
		static float two_pi=8*atan(1.0);
		if (ndeltaphi>0.5*two_pi) ndeltaphi-=two_pi;
		if (ndeltaphi<-0.5*two_pi) ndeltaphi+=two_pi;
		float ndist=sqrt(nr2*nr2+nr1*nr1-2*nr1*nr2*cos(ndeltaphi));

		float curvature = 2*sin(ndeltaphi)/ndist;
		return curvature;
	}
	//====================
	float pTFrom2(std::vector< TTTrack< Ref_PixelDigi_> >::const_iterator trk, const StackedTrackerGeometry* theStackedGeometry){

		float rinv= rInvFrom2(trk, theStackedGeometry);
		return fabs( 0.00299792*3.8/rinv); 

	}
	//====================
}
