#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticle.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"

KinematicParticle::~KinematicParticle() { delete lConstraint;} 

bool KinematicParticle::operator<(const KinematicParticle& other)const
{
 bool res = false;
 if(this < &other) res=true;
 return res;
}

KinematicState KinematicParticle::initialState()const
{return initState;}

KinematicState KinematicParticle::currentState()const
{return cState;}

KinematicConstraint * KinematicParticle::lastConstraint()const
{return lConstraint;}

ReferenceCountingPointer<KinematicParticle> KinematicParticle::previousParticle()const
{return pState;} 

KinematicTree * KinematicParticle::correspondingTree()const
{return tree;}

float KinematicParticle::chiSquared()const
{return chi2;}

float KinematicParticle::degreesOfFreedom()const
{return ndf;}

void KinematicParticle::setTreePointer(KinematicTree * tr)const
{tree = tr;}

reco::TransientTrack KinematicParticle::refittedTransientTrack() const
{
  TransientTrackFromFTSFactory factory;
  return factory.build(currentState().freeTrajectoryState());
}
