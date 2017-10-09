#ifndef KinematicVertex_H
#define KinematicVertex_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
class KinematicTree;

/**
 * Class representing a Decay Vertex
 * Caches a vertex position, covariance
 * matrix, chi squared and number of
 * degrees of freedom. Class is usually
 * created by KinematicParticleVertexFitter
 *
 * Kirill Prokofiev, December 2002
 */


class KinematicVertex : public ReferenceCounted
{
public:

 friend class KinematicTree;

/**
 * Empty default constructor
 * for invalid vertices
 */
 KinematicVertex();


/**
 * Constructor with vertex state, chi2 and ndf.
 * Previous state of the vertex pointer is set to 0.
 */
 KinematicVertex(const VertexState state, float totalChiSq, float degreesOfFr);

/**
 * Constructor with previous (before constraint)
 * state of the vertex
 */
 KinematicVertex(const VertexState state,
             const ReferenceCountingPointer<KinematicVertex> prVertex,
                                    float totalChiSq, float degreesOfFr);

/**
 * Direct transformation from caching vertex
 */
 KinematicVertex(const CachingVertex<6>& vertex);


 virtual ~KinematicVertex();

/**
 * Comparison by contents operator
 * is _true_ if position AND
 * covariance match
 */
 bool operator==(const KinematicVertex& other) const;

 bool operator==(const ReferenceCountingPointer<KinematicVertex> other) const;

/**
 * comparison by adress operator
 * Has NO physical meaning
 * To be used inside the graph only
 */

 bool operator<(const KinematicVertex& other)const;
/**
 * Access methods
 */

/**
 * Checking the validity of the vertex
 * Example: production vertex for the
 * first decayed particle or decay vertices
 * of final state particles can be invalid
 * since we don't know them.
 */
 bool vertexIsValid() const;

/**
 * Returns the pointer to the kinematic
 * tree (if any) current vertex belongs to
 * returned in case of not any tree build yet
 */
 KinematicTree * correspondingTree() const;

/**
 * Previous (before constraint) state of the vertex
 */
 ReferenceCountingPointer<KinematicVertex> vertexBeforeConstraint() const;


 VertexState vertexState() const;

 GlobalPoint position() const;

 GlobalError error() const;

 float chiSquared() const;

 float degreesOfFreedom() const;

 operator reco::Vertex();

private:

 void setTreePointer(KinematicTree * tr) const;

//kinematic tree this
//vertex belongs to (can be 0)
 mutable KinematicTree * tree;
 mutable bool vl;

 VertexState theState;
// GlobalPoint theVertexPosition;
// GlobalError theVPositionError;
 float theChiSquared;
 float theNDF;
 mutable ReferenceCountingPointer<KinematicVertex> pVertex;
};

#endif
