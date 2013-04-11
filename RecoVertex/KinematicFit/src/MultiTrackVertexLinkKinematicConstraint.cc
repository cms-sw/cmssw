#include "RecoVertex/KinematicFit/interface/MultiTrackVertexLinkKinematicConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

AlgebraicVector MultiTrackVertexLinkKinematicConstraint::value(const std::vector<KinematicState> &states, const GlobalPoint& point) const{
	int num = states.size();
	if(num<2) throw VertexException("MultiTrackVertexLinkKinematicConstraint::value <2 states passed");

	//2 equations (for all tracks)
	AlgebraicVector  vl(2,0);
	double dx = point.x() - refPoint.x();
	double dy = point.y() - refPoint.y();
	double dz = point.z() - refPoint.z();
	double dT = sqrt(pow(dx,2) + pow(dy,2));
	double ds = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));

	double pxSum=0, pySum=0, pzSum=0;
	double aSum = 0;
	for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++)
	{
		double a = - i->particleCharge() * i->magneticField()->inInverseGeV(i->globalPosition()).z();
		aSum += a;

		pxSum += i->kinematicParameters()(3) - a*(point.y() - i->kinematicParameters()(1));
		pySum += i->kinematicParameters()(4) + a*(point.x() - i->kinematicParameters()(0));
		pzSum += i->kinematicParameters()(5);
	}
	
	double pT = sqrt(pow(pxSum,2) + pow(pySum,2));
	double pSum = sqrt(pow(pxSum,2) + pow(pySum,2) + pow(pzSum,2));
	
	vl(1) = (dT - dx)/dy + (-2*pT + sqrt(-(pow(aSum,2)*pow(dT,2)) + 4*pow(pT,2)))/(aSum*dT) + (-pT + pxSum)/pySum;
	vl(2) = (ds - dT)/dz + (pT - pSum)/pzSum;
	
	return vl;
}

AlgebraicMatrix MultiTrackVertexLinkKinematicConstraint::parametersDerivative(const std::vector<KinematicState> &states, const GlobalPoint& point) const{
	int num = states.size();
	if(num<2) throw VertexException("MultiTrackVertexLinkKinematicConstraint::parametersDerivative <2 states passed");
	
	//2 equations (for all tracks)
	AlgebraicMatrix  matrix(2,num*7,0);//AlgebraicMatrix starts from 1
	double dx = point.x() - refPoint.x();
	double dy = point.y() - refPoint.y();
	double dT = sqrt(pow(dx,2) + pow(dy,2));
	
	double pxSum=0, pySum=0, pzSum=0;
	double aSum = 0;
	for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++)
	{
		double a = - i->particleCharge() * i->magneticField()->inInverseGeV(i->globalPosition()).z();
		aSum += a;

		pxSum += i->kinematicParameters()(3) - a*(point.y() - i->kinematicParameters()(1));
		pySum += i->kinematicParameters()(4) + a*(point.x() - i->kinematicParameters()(0));
		pzSum += i->kinematicParameters()(5);
	}

	double pT = sqrt(pow(pxSum,2) + pow(pySum,2));
	double pSum = sqrt(pow(pxSum,2) + pow(pySum,2) + pow(pzSum,2));

	int col=0;
	for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++){
		double a = - i->particleCharge() * i->magneticField()->inInverseGeV(i->globalPosition()).z();

		matrix(1,1+col*7) =	a*(-(pT/pow(pySum,2)) + pxSum/pow(pySum,2) - (4*pySum)/(aSum*dT*sqrt(-(pow(aSum,2)*pow(dT,2)) + 4*pow(pT,2))) + (1 + (2*pySum)/(aSum*dT))/pT);//dH/dx
		matrix(1,2+col*7) =	(a*(aSum*dT*(pT - pxSum) + 2*(-1 + (2*pT)/sqrt(-(pow(aSum,2)*pow(dT,2)) + 4*pow(pT,2)))*pxSum*pySum))/(aSum*dT*pT*pySum);//dH/dy
		//dH/dz=0
		matrix(1,4+col*7) =	(aSum*dT*(pT - pxSum) + 2*(-1 + (2*pT)/sqrt(-(pow(aSum,2)*pow(dT,2)) + 4*pow(pT,2)))*pxSum*pySum)/(aSum*dT*pT*pySum);//dH/dpx
		matrix(1,5+col*7) =	pT/pow(pySum,2) - pxSum/pow(pySum,2) + (4*pySum)/(aSum*dT*sqrt(-(pow(aSum,2)*pow(dT,2)) + 4*pow(pT,2))) + (-1 - (2*pySum)/(aSum*dT))/pT;//dH/dpy
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

AlgebraicMatrix MultiTrackVertexLinkKinematicConstraint::positionDerivative(const std::vector<KinematicState> &states, const GlobalPoint& point) const{
	int num = states.size();
	if(num<2) throw VertexException("MultiTrackVertexLinkKinematicConstraint::positionDerivative <2 states passed");
	
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

	matrix(1,1) = (-1 + dx/dT)/dy + (2*dx*pT*(1 - (2*pT)/sqrt(-(pow(aSum,2)*pow(dT,2)) + 4*pow(pT,2))))/(aSum*pow(dT,3)) + aSum*(-(1/pT) + pT/pow(pySum,2) - pxSum/pow(pySum,2)) + (2*(-(1/pT) + 2/sqrt(-(pow(aSum,2)*pow(dT,2)) + 4*pow(pT,2)))*pySum)/dT;//dH/dxv
	matrix(1,2) = 1/dT + (-dT + dx)/pow(dy,2) - (dy*(-2*pT + sqrt(-(pow(aSum,2)*pow(dT,2)) + 4*pow(pT,2))))/(aSum*pow(dT,3)) - ((-2 + sqrt(4 - (pow(aSum,2)*pow(dT,2))/pow(pT,2)))*pxSum)/(dT*pT) - (aSum*(dy*pow(pT,2) + aSum*pow(dT,2)*pxSum))/(dT*pow(pT,2)*sqrt(-(pow(aSum,2)*pow(dT,2)) + 4*pow(pT,2))) + (aSum*(-pT + pxSum))/(pT*pySum);//dH/dyv
	//dH/dzv=0
	matrix(2,1) = ((1/ds - 1/dT)*dx)/dz + (aSum*(pSum - pT)*pySum)/(pSum*pT*pzSum);//dH/dxv
	matrix(2,2) = ((1/ds - 1/dT)*dy)/dz - (aSum*(pSum - pT)*pxSum)/(pSum*pT*pzSum);//dH/dyv
	matrix(2,3) = 1/ds + (-ds + dT)/pow(dz,2);//dH/dzv
	
	return matrix;
}

int MultiTrackVertexLinkKinematicConstraint::numberOfEquations() const{
	return 2;
}
