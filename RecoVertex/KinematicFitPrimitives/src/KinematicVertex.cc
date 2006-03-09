#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertex.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticle.h"

KinematicVertex::KinematicVertex()
{vl = false;}

KinematicVertex::KinematicVertex(const VertexState state, float totalChiSq, 
                                                             float degreesOfFr):
		  theState(state),theChiSquared(totalChiSq),theNDF(degreesOfFr)                
					                           
{
 vl = true;
 tree = 0;
 pVertex = 0;
}

KinematicVertex::KinematicVertex(const CachingVertex& vertex)                                              
{
// theVertexPosition = vertex.position();
// theVPositionError = vertex.error();
 vl = true;
 theState = VertexState(vertex.position(), vertex.error());
 theChiSquared = vertex.totalChiSquared();
 theNDF = vertex.degreesOfFreedom();
 tree = 0;
 pVertex = 0;
}		 

KinematicVertex::KinematicVertex(const VertexState state, 
                         const ReferenceCountingPointer<KinematicVertex> prVertex,
                                    float totalChiSq, float degreesOfFr):
				    theState(state) ,
				    theChiSquared(totalChiSq),theNDF(degreesOfFr) , pVertex(prVertex)
{
 vl = true;
 tree = 0;
}



bool KinematicVertex::operator==(const KinematicVertex& other)const
{
 bool res = false;
 if(vertexIsValid()&& other.vertexIsValid())
 {
  GlobalPoint cPos = this->position();
  GlobalPoint oPos = other.position();
  AlgebraicMatrix cCov = this->error().matrix();
  AlgebraicMatrix oCov = other.error().matrix();
  if((cPos.x()==oPos.x())&&(cPos.y()==oPos.y())&&(cPos.z()==oPos.z())
                                                      &&(cCov==oCov))
  res = true;
 }else if(!(vertexIsValid()) && !(other.vertexIsValid())){
  if(this == &other) res = true;
 } 
 return res;
}

bool KinematicVertex::operator==(ReferenceCountingPointer<KinematicVertex> other)const
{
 bool res = false;
 if(*this == *other) res = true;
 return res;
}


bool KinematicVertex::operator<(const KinematicVertex& other)const
{ 
 bool res = false;
 if(this < &other) res=true;
 return res;
}	

bool KinematicVertex::vertexIsValid() const
{return vl;}

KinematicVertex::~KinematicVertex()
{}

GlobalPoint KinematicVertex::position() const
{
 return theState.position(); 
}
 
GlobalError KinematicVertex::error() const
{
 return theState.error();
}
 
float  KinematicVertex::chiSquared() const
{return theChiSquared;}
 
float  KinematicVertex::degreesOfFreedom() const
{return theNDF;}

KinematicTree *  KinematicVertex::correspondingTree() const
{return tree;}

void KinematicVertex::setTreePointer(KinematicTree * tr) const
{tree = tr;}

ReferenceCountingPointer<KinematicVertex>  KinematicVertex::vertexBeforeConstraint() const
{return pVertex;}

VertexState KinematicVertex:: vertexState() const
{return theState;}

