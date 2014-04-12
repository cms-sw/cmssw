#include "RecoVertex/KinematicFit/interface/MultiTrackPointingKinematicConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

AlgebraicVector MultiTrackPointingKinematicConstraint::value(const std::vector<KinematicState> &states, const GlobalPoint& point) const{
	int num = states.size();
	if(num<2) throw VertexException("MultiTrackPointingKinematicConstraint::value <2 states passed");

	//2 equations (for all tracks)
	AlgebraicVector  vl(2,0);
	double dx = point.x() - refPoint.x();
	double dy = point.y() - refPoint.y();
	double dz = point.z() - refPoint.z();
	double dT = sqrt(pow(dx,2) + pow(dy,2));
	double ds = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));

	double pxSum=0, pySum=0, pzSum=0;
	for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++)
	{
		double a = - i->particleCharge() * i->magneticField()->inInverseGeV(i->globalPosition()).z();

		pxSum += i->kinematicParameters()(3) - a*(point.y() - i->kinematicParameters()(1));
		pySum += i->kinematicParameters()(4) + a*(point.x() - i->kinematicParameters()(0));
		pzSum += i->kinematicParameters()(5);
	}
	
	double pT = sqrt(pow(pxSum,2) + pow(pySum,2));
	double pSum = sqrt(pow(pxSum,2) + pow(pySum,2) + pow(pzSum,2));
	
	vl(1) = (dT - dx)/dy + (pxSum - pT)/pySum;
	vl(2) = (ds - dT)/dz + (pT - pSum)/pzSum;
	
	return vl;
}

AlgebraicMatrix MultiTrackPointingKinematicConstraint::parametersDerivative(const std::vector<KinematicState> &states, const GlobalPoint& point) const{
	int num = states.size();
	if(num<2) throw VertexException("MultiTrackPointingKinematicConstraint::parametersDerivative <2 states passed");
	
	//2 equations (for all tracks)
	AlgebraicMatrix  matrix(2,num*7,0);//AlgebraicMatrix starts from 1
	
	double pxSum=0, pySum=0, pzSum=0;
	for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++)
	{
		double a = - i->particleCharge() * i->magneticField()->inInverseGeV(i->globalPosition()).z();
		
		pxSum += i->kinematicParameters()(3) - a*(point.y() - i->kinematicParameters()(1));
		pySum += i->kinematicParameters()(4) + a*(point.x() - i->kinematicParameters()(0));
		pzSum += i->kinematicParameters()(5);
	}

	double pT = sqrt(pow(pxSum,2) + pow(pySum,2));
	double pSum = sqrt(pow(pxSum,2) + pow(pySum,2) + pow(pzSum,2));

	int col=0;
	for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++){
		double a = - i->particleCharge() * i->magneticField()->inInverseGeV(i->globalPosition()).z();

		matrix(1,1+col*7) =	a*(1/pT + (-pT + pxSum)/pow(pySum,2));//dH/dx
		matrix(1,2+col*7) =	(a - (a*pxSum)/pT)/pySum;//dH/dy
		//dH/dz=0
		matrix(1,4+col*7) =	(pT - pxSum)/(pT*pySum);//dH/dpx
		matrix(1,5+col*7) =	-(1/pT) + (pT - pxSum)/pow(pySum,2);//dH/dpy		
		//dH/dpz=0
		//dH/dm=0
		matrix(2,1+col*7) =	(a*(-pSum + pT)*pySum)/(pSum*pT*pzSum);//dH/dx
		matrix(2,2+col*7) =	(a*( pSum - pT)*pxSum)/(pSum*pT*pzSum);//dH/dy
		//dH/dz
		matrix(2,4+col*7) =	((-(1/pSum) + 1/pT)*pxSum)/pzSum;//dH/dpx
		matrix(2,5+col*7) =	((-(1/pSum) + 1/pT)*pySum)/pzSum;//dH/dpy
		matrix(2,6+col*7) =	-(1/pSum) + (pSum - pT)/pow(pzSum,2);//dH/dpz
		//dH/dm=0		

		col++;
	}
	
	return matrix;
}

AlgebraicMatrix MultiTrackPointingKinematicConstraint::positionDerivative(const std::vector<KinematicState> &states, const GlobalPoint& point) const{
	int num = states.size();
	if(num<2) throw VertexException("MultiTrackPointingKinematicConstraint::positionDerivative <2 states passed");
	
	//2 equations (for all tracks)
	AlgebraicMatrix  matrix(2,3,0);
	double dx = point.x() - refPoint.x();
	double dy = point.y() - refPoint.y();
	double dz = point.z() - refPoint.z();
	double dT = sqrt(pow(dx,2) + pow(dy,2));
	double ds = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));
	
	double pxSum=0, pySum=0, pzSum=0, aSum = 0;
	for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++){
		double a = - i->particleCharge() * i->magneticField()->inInverseGeV(i->globalPosition()).z();
		aSum += a;
		
		pxSum += i->kinematicParameters()(3) - a*(point.y() - i->kinematicParameters()(1));
		pySum += i->kinematicParameters()(4) + a*(point.x() - i->kinematicParameters()(0));
		pzSum += i->kinematicParameters()(5);
	}
	double pT = sqrt(pow(pxSum,2) + pow(pySum,2));
	double pSum = sqrt(pow(pxSum,2) + pow(pySum,2) + pow(pzSum,2));

	matrix(1,1) = (-1 + dx/dT)/dy - aSum/pT + (aSum*(pT - pxSum))/pow(pySum,2);//dH/dxv
	matrix(1,2) = 1/dT + (-dT + dx)/pow(dy,2) - (aSum*(pT - pxSum))/(pT*pySum);//dH/dyv
	//dH/dzv=0
	matrix(2,1) = ((1/ds - 1/dT)*dx)/dz + (aSum*(pSum - pT)*pySum)/(pSum*pT*pzSum);//dH/dxv
	matrix(2,2) = ((1/ds - 1/dT)*dy)/dz - (aSum*(pSum - pT)*pxSum)/(pSum*pT*pzSum);//dH/dyv
	matrix(2,3) = 1/ds + (-ds + dT)/pow(dz,2);//dH/dzv
	
	return matrix;
}

int MultiTrackPointingKinematicConstraint::numberOfEquations() const{
	return 2;
}
