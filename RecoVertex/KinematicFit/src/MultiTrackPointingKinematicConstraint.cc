#include "RecoVertex/KinematicFit/interface/MultiTrackPointingKinematicConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

AlgebraicVector MultiTrackPointingKinematicConstraint::value(const std::vector<KinematicState> states, const GlobalPoint& point) const{
	int num = states.size();
	if(num<2) throw VertexException("MultiTrackPointingKinematicConstraint::value <2 states passed");

	//2 equations (for all tracks)
	AlgebraicVector  vl(2,0);
	double dx = point.x() - refPoint.x();
	double dy = point.y() - refPoint.y();
	double dz = point.z() - refPoint.z();
	double ds = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));

	double pxSum=0, pySum=0, pzSum=0;	
	for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++){
		GlobalVector mom = i->globalMomentum();
		pxSum += mom.x();
		pySum += mom.y();
		pzSum += mom.z();
	}
	
	double pSum = sqrt(pow(pxSum,2) + pow(pySum,2) + pow(pzSum,2));
	
	vl(1) = -(dx/dy) + sqrt(pow(dx,2) + pow(dy,2))/dy + pxSum/pySum - sqrt(pow(pxSum,2) + pow(pySum,2))/pySum;
	vl(2) = ds/dz - sqrt(pow(dx,2) + pow(dy,2))/dz - sqrt(pow(pSum,2))/pzSum + sqrt(pow(pxSum,2) + pow(pySum,2))/pzSum;
	
	return vl;
}

AlgebraicMatrix MultiTrackPointingKinematicConstraint::parametersDerivative(const std::vector<KinematicState> states, const GlobalPoint& point) const{
	int num = states.size();
	if(num<2) throw VertexException("MultiTrackPointingKinematicConstraint::parametersDerivative <2 states passed");
	
	//2 equations (for all tracks)
	AlgebraicMatrix  matrix(2,num*7,0);

	double pxSum=0, pySum=0, pzSum=0;	
	for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++)
	{
		GlobalVector mom = i->globalMomentum();
		pxSum += mom.x();
		pySum += mom.y();
		pzSum += mom.z();
	}
	double pSum = sqrt(pow(pxSum,2) + pow(pySum,2) + pow(pzSum,2));

	int col=0;
	for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++){
		matrix(1,4+col*7) =	1/pySum - pxSum/(pySum*sqrt(pow(pxSum,2) + pow(pySum,2)));
		matrix(1,5+col*7) =	-(pxSum/pow(pySum,2)) + pow(pxSum,2)/(pow(pySum,2)*sqrt(pow(pxSum,2) + pow(pySum,2)));

		matrix(2,4+col*7) =	-(pxSum/(pSum*pzSum)) + pxSum/(sqrt(pow(pxSum,2) + pow(pySum,2))*pzSum);
		matrix(2,5+col*7) =	-(pySum/(pSum*pzSum)) + pySum/(sqrt(pow(pxSum,2) + pow(pySum,2))*pzSum);
		matrix(2,6+col*7) =	-(1/pSum) + pSum/pow(pzSum,2) - sqrt(pow(pxSum,2) + pow(pySum,2))/pow(pzSum,2);
		col++;
	}
	
	return matrix;
}

AlgebraicMatrix MultiTrackPointingKinematicConstraint::positionDerivative(const std::vector<KinematicState> states, const GlobalPoint& point) const{
	int num = states.size();
	if(num<2) throw VertexException("MultiTrackPointingKinematicConstraint::positionDerivative <2 states passed");
	
	//2 equations (for all tracks)
	AlgebraicMatrix  matrix(2,3,0);
	double dx = point.x() - refPoint.x();
	double dy = point.y() - refPoint.y();
	double dz = point.z() - refPoint.z();
	double ds = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));
	
	double pxSum=0, pySum=0, pzSum=0;	
	for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++){
		GlobalVector mom = i->globalMomentum();
		pxSum += mom.x();
		pySum += mom.y();
		pzSum += mom.z();
	}

	matrix(1,1) =	-(1/dy) + dx/(dy*sqrt(pow(dx,2) + pow(dy,2)));
	matrix(1,2) =	dx/pow(dy,2) - pow(dx,2)/(pow(dy,2)*sqrt(pow(dx,2) + pow(dy,2)));
	matrix(2,1) =	-(dx/(sqrt(pow(dx,2) + pow(dy,2))*dz));
	matrix(2,2) =	-(dy/(sqrt(pow(dx,2) + pow(dy,2))*dz));
	matrix(2,3) =	-(ds/pow(dz,2)) + sqrt(pow(dx,2) + pow(dy,2))/pow(dz,2);
	
	return matrix;
}

int MultiTrackPointingKinematicConstraint::numberOfEquations() const{
	return 2;
}
