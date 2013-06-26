#include "RecoVertex/KinematicFit/interface/MultiTrackMassKinematicConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"


AlgebraicVector  MultiTrackMassKinematicConstraint::value(const std::vector<KinematicState> &states,
                        const GlobalPoint& point) const
{
 if(states.size()<nPart) throw VertexException("MultiTrackMassKinematicConstraint::not enough states given");

 double sumEnergy = 0, sumPx=0, sumPy=0., sumPz=0.;

 double a;
 for (unsigned int i=0;i<nPart;++i) {
    a = -states[i].particleCharge() * states[i].magneticField()->inInverseGeV(states[i].globalPosition()).z();

    sumEnergy += states[i].kinematicParameters().energy();
    sumPx += states[i].kinematicParameters()(3) - a*(point.y() - states[i].kinematicParameters()(1));
    sumPy += states[i].kinematicParameters()(4) + a*(point.x() - states[i].kinematicParameters()(0));
    sumPz += states[i].kinematicParameters()(5);
 }

 double j_m = sumPx*sumPx + sumPy*sumPy + sumPz*sumPz;

 AlgebraicVector res(1,0);
 res(1)  = sumEnergy*sumEnergy - j_m - mass*mass;
 return res;
}

AlgebraicMatrix MultiTrackMassKinematicConstraint::parametersDerivative(const std::vector<KinematicState> &states,
                                      const GlobalPoint& point) const
{
  if(states.size()<nPart) throw VertexException("MultiTrackMassKinematicConstraint::not enough states given");

  AlgebraicMatrix res(1,states.size()*7,0);

  double sumEnergy = 0, sumPx=0, sumPy=0., sumPz=0.;

 double a;
  for (unsigned int i=0;i<nPart;++i) {
    a = -states[i].particleCharge() * states[i].magneticField()->inInverseGeV(states[i].globalPosition()).z();

    sumEnergy += states[i].kinematicParameters().energy();
    sumPx += states[i].kinematicParameters()(3) - a*(point.y() - states[i].kinematicParameters()(1));
    sumPy += states[i].kinematicParameters()(4) + a*(point.x() - states[i].kinematicParameters()(0));
    sumPz += states[i].kinematicParameters()(5);
  }

  for (unsigned int i=0;i<nPart;++i) {
    a = -states[i].particleCharge() * states[i].magneticField()->inInverseGeV(states[i].globalPosition()).z();

 //x derivatives:
    res(1,1+i*7) = 2*a*sumPy;

 //y derivatives:
    res(1,2+i*7) = -2*a*sumPx;

 //z components:
    res(1,3+i*7)  = 0.;

 //px components:
    res(1,4+i*7)  = 2*sumEnergy/states[i].kinematicParameters().energy()*states[i].kinematicParameters()(3) - 2*sumPx;

 //py components:
    res(1,5+i*7)  = 2*sumEnergy/states[i].kinematicParameters().energy()*states[i].kinematicParameters()(4) - 2*sumPy;

 //pz1 components:
    res(1,6+i*7)  = 2*sumEnergy/states[i].kinematicParameters().energy()*states[i].kinematicParameters()(5) - 2*sumPz;

 //mass components:
    res(1,7+i*7)  = 2*states[i].kinematicParameters().mass()*sumEnergy/states[i].kinematicParameters().energy();
  }
  return res;
}

AlgebraicMatrix MultiTrackMassKinematicConstraint::positionDerivative(const std::vector<KinematicState> &states,
                                    const GlobalPoint& point) const
{
  AlgebraicMatrix res(1,3,0);
  if(states.size()<nPart) throw VertexException("MultiTrackMassKinematicConstraint::not enough states given");

  double sumA = 0, sumPx=0, sumPy=0.;

  double a;
  for (unsigned int i=0;i<nPart;++i) {
    a = -states[i].particleCharge() * states[i].magneticField()->inInverseGeV(states[i].globalPosition()).z();
    sumA += a;

    sumPx += states[i].kinematicParameters()(3) - a*(point.y() - states[i].kinematicParameters()(1));
    sumPy += states[i].kinematicParameters()(4) + a*(point.x() - states[i].kinematicParameters()(0));
  }

 //xv component
  res(1,1) = - 2 * sumPy * sumA;

 //yv component
  res(1,2) =   2 * sumPx * sumA;

 //zv component
  res(1,3) = 0.;

  return res;
}
