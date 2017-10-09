#ifndef KinematicTree_H
#define KinematicTree_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/Math/interface/Graph.h"
#include "DataFormats/Math/interface/GraphWalker.h"

/**
 * Class describing the decay
 * chain inside the kinematic fit.
 * Uses the boost graph based DDD
 * library. KinematicVertices are
 * vertices, kinemtaic particles become nodes.
 *
 * Kirill Prokofiev, April 2003
 */
/** 
 * WARNING:  before using any of these
 * methods please make sure you _understand correctly_
 * what part of tree are you looking at now and
 * where the pointer will be after the desiring 
 * operation.
 *
 * "Left" and "Right" notation reflects the order
 * of creating a three: from bottom to top,
 * from left to right
 * EXAMPLE: Bs->J/Psi Phi -> mumu KK
 * First creating a mumu branch, fit it to J/Psi,
 * Then adding Phi->KK and reconstructing a Bs will
 * look like: left bottom particle: muon, top of
 * the tree is Bs, right bottom is K.
 */ 

class KinematicTree : public ReferenceCounted
{
 public:

/**
 * Constructor initializing
 * everything and setting all values to 0 
 */ 
 KinematicTree();

 
 virtual ~KinematicTree();
 
/**
 * Access methods
 */ 
 bool isEmpty() const;
 bool isValid() const  {return !empt;}

/**
 * This method checks if the tree
 * is consistent, i.e. the top vertex is
 * only one.
 */
   
 bool isConsistent() const;

/**
 * Methods adding nodes and 
 * edges to the graph representation 
 * of the Kinematic Tree
 */   
 void addParticle(RefCountedKinematicVertex prodVtx, 
                  RefCountedKinematicVertex  decVtx, 
	          RefCountedKinematicParticle part);

/**
 * Kinematic Tree  navigation methods
 */ 
 
/** 
 * Returns the complete vector of final state
 * particles for the whole decay tree.
 * Pointer is NOT moved after this operation
 */
 std::vector<RefCountedKinematicParticle> finalStateParticles() const;
 
/**
 * Returns the top particle of
 * decay. Pointer is moved to the
 * TOP of the decay tree.
 */  
 RefCountedKinematicParticle topParticle() const;
 
/**
 * Returns a current decay vertex
 * pointer is NOT moved
 */ 
 RefCountedKinematicVertex currentDecayVertex() const;
 
/**
 * Returns a current production vertex
 * pointer is NOT moved
 */ 
 RefCountedKinematicVertex currentProductionVertex() const;
 
/**
 * Returns a current particle
 * pointer is NOT moved
 */ 
 RefCountedKinematicParticle currentParticle() const;
 
/**
 * Returns _true_ and state of mother particle
 * if successfull, _false_ and state of current particle
 * in case of failure
 * Pointer is NOT moved.
 */
 std::pair<bool,RefCountedKinematicParticle>  motherParticle() const;
 
 
/**
 * Returns a non-zero vector in case of success and
 * 0 vector in case of failure  
 * Pointer is NOT moved
 */ 
 std::vector<RefCountedKinematicParticle> daughterParticles() const;
   
/**
 *  Puts the pointer to the top (root)
 *  of the tree. The  Graph walker
 * object gets recreated inside the tree.
 */   
 void movePointerToTheTop() const; 
 
/**
 * Pointer goes to the mother
 * particle (if any) with respest to
 * the current one
 */ 
 bool movePointerToTheMother() const;

/**
 * Pointer goes to the first 
 * child(if any) in the list
 */ 
 bool movePointerToTheFirstChild() const;
 
/**
 * Pointer goes to the next 
 * child in the list (if any)
 */ 
 bool movePointerToTheNextChild() const;

/**
 * Pointer goes to the needed
 * particle inside the tree if
 * found (true). Pointer goes
 * to the top of the tree if not 
 * found (false)
 */
 bool findParticle(const RefCountedKinematicParticle part) const;
 
/**
 * Pointer goes to the particle
 * for which the neede vertex will be 
 * the decay one (true case)
 * Or pointer stays at the top of teh tree
 * if search is unsuccessfull (false case). 
 */
 bool findDecayVertex(const RefCountedKinematicVertex vert) const;
 bool findDecayVertex(KinematicVertex * vert) const;

/**
 * Methods replacing Particles and Vertices
 * inside the tree during the refit. Methods
 * should used by corresponding KinematicFitter
 * classe only. Replace the current track or
 * current (decay) vertex 
 * WARNING: replace methods are not available
 * at this version of KinematicFitPrimitives.
 */ 
 void replaceCurrentParticle(RefCountedKinematicParticle newPart) const;
   
 
/**
 * Replaces _decay_ vertex of the 
 * current particle with the given value
 */   
 void replaceCurrentVertex(RefCountedKinematicVertex newVert)     const; 
 
/**
 * Method adding a tree _tr_ to the vertex vtx
 * of current tree in such a way, that vtx become a 
 * production vertex for the top particle of the _tr_.
 * The whole contetnts of _tr_ is the rewritten to
 * current tree. This method is purely technical
 * and contains no mathematics. To be used by
 * KinematicParticleVertexFitter after the corresponding fit.
 */ 
 void addTree(RefCountedKinematicVertex vtx, KinematicTree * tr);
private:
 
/**
 * Private methods to  walk around the tree:
 * Needed to collect final state or 
 * particle search.
 */ 
 
 bool leftBranchSearch(RefCountedKinematicParticle part) const;
 
 bool leftBranchVertexSearch(RefCountedKinematicVertex vtx) const; 
 
 bool leftFinalParticle() const;
 
 void leftBranchAdd(KinematicTree * otherTree, RefCountedKinematicVertex vtx);

 mutable bool empt;
 
 mutable math::Graph<RefCountedKinematicVertex,RefCountedKinematicParticle> treeGraph;
 mutable math::GraphWalker<RefCountedKinematicVertex, RefCountedKinematicParticle> * treeWalker; 

};
#endif 
