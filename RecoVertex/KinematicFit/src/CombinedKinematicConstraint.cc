#include "RecoVertex/KinematicFit/interface/CombinedKinematicConstraint.h"

AlgebraicVector  CombinedKinematicConstraint::value(const std::vector<KinematicState> &states, const GlobalPoint& point) const
{
	AlgebraicVector tmpValue;
	int size = 0;
	for (std::vector<MultiTrackKinematicConstraint* >::const_iterator it=constraints.begin(); it!=constraints.end(); ++it){
		tmpValue = (*it)->value(states, point);
		size += tmpValue.num_row();
	}
	AlgebraicVector values(size);
	int position = 1;
	for (std::vector<MultiTrackKinematicConstraint* >::const_iterator it=constraints.begin(); it!=constraints.end(); ++it){
		tmpValue =  (*it)->value(states, point);
		values.sub(position, tmpValue);
		position += tmpValue.num_row();
	}
	
	return values;
}

AlgebraicMatrix CombinedKinematicConstraint::parametersDerivative(const std::vector<KinematicState> &states, const GlobalPoint& point) const
{
	AlgebraicMatrix tmpMatrix;
	int row = 0;
	for (std::vector<MultiTrackKinematicConstraint* >::const_iterator it=constraints.begin(); it!=constraints.end(); ++it){
		tmpMatrix = (*it)->parametersDerivative(states, point);
		row += tmpMatrix.num_row();
	}
	AlgebraicMatrix matrix(row,7*states.size(),0);
	int posRow = 1;
	for (std::vector<MultiTrackKinematicConstraint* >::const_iterator it=constraints.begin(); it!=constraints.end(); ++it){
		tmpMatrix =  (*it)->parametersDerivative(states, point);
		matrix.sub(posRow, 1, tmpMatrix);
		posRow += tmpMatrix.num_row();
	}
	
	return matrix;
}

AlgebraicMatrix CombinedKinematicConstraint::positionDerivative(const std::vector<KinematicState> &states, const GlobalPoint& point) const
{
	AlgebraicMatrix tmpMatrix;
	int row = 0;
	for (std::vector<MultiTrackKinematicConstraint* >::const_iterator it=constraints.begin(); it!=constraints.end(); ++it){
		tmpMatrix = (*it)->positionDerivative(states, point);
		row += tmpMatrix.num_row();
	}
	AlgebraicMatrix matrix(row,3,0);
	int posRow = 1;
	for (std::vector<MultiTrackKinematicConstraint* >::const_iterator it=constraints.begin(); it!=constraints.end(); ++it){
		tmpMatrix =  (*it)->positionDerivative(states, point);
		matrix.sub(posRow, 1, tmpMatrix);
		posRow += tmpMatrix.num_row();
	}
	
	return matrix;
}

int CombinedKinematicConstraint::numberOfEquations() const
{
	int noEq = 0;
	for (std::vector<MultiTrackKinematicConstraint* >::const_iterator it=constraints.begin(); it!=constraints.end(); ++it) noEq += (*it)->numberOfEquations();
	
	return noEq;
}
