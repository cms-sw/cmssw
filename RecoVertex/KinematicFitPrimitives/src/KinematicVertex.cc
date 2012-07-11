#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertex.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicParticle.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/GsfTransientTrack.h"

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

KinematicVertex::KinematicVertex(const CachingVertex<6>& vertex)                                              
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

bool KinematicVertex::operator==(const ReferenceCountingPointer<KinematicVertex> other)const
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
{ tree = tr;}

ReferenceCountingPointer<KinematicVertex>  KinematicVertex::vertexBeforeConstraint() const
{return pVertex;}

VertexState KinematicVertex::vertexState() const
{return theState;}

KinematicVertex::operator reco::Vertex() 
{
   //If the vertex is invalid, return an invalid TV !
  if (!vertexIsValid() || tree==0) return reco::Vertex();

//accessing the tree components, move pointer to top
  if (!tree->findDecayVertex(this)) return reco::Vertex();
  std::vector<RefCountedKinematicParticle> daughters = tree->daughterParticles();

  reco::Vertex vertex(reco::Vertex::Point(theState.position()),
// 	RecoVertex::convertError(theVertexState.error()), 
	theState.error().matrix_new(), 
	chiSquared(), degreesOfFreedom(), daughters.size() );

  for (std::vector<RefCountedKinematicParticle>::const_iterator i = daughters.begin();
       i != daughters.end(); ++i) {

    const TransientTrackKinematicParticle * ttkp = dynamic_cast<const TransientTrackKinematicParticle * >(&(**i));
    if(ttkp != 0) {
      const reco::TrackTransientTrack * ttt = dynamic_cast<const reco::TrackTransientTrack*>(ttkp->initialTransientTrack()->basicTransientTrack());
      if ((ttt!=0) && (ttt->persistentTrackRef().isNonnull())) {
	reco::TrackRef tr = ttt->persistentTrackRef();
	vertex.add(reco::TrackBaseRef(tr), ttkp->refittedTransientTrack().track(), 1.);
      } else {
	const reco::GsfTransientTrack * ttt = dynamic_cast<const reco::GsfTransientTrack*>(ttkp->initialTransientTrack()->basicTransientTrack());
	if ((ttt!=0) && (ttt->persistentTrackRef().isNonnull())) {
	  reco::GsfTrackRef tr = ttt->persistentTrackRef();
	  vertex.add(reco::TrackBaseRef(tr), ttkp->refittedTransientTrack().track(), 1.);
	}
      }
    }
  }
  return vertex;
}

