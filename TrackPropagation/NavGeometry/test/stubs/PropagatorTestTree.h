#ifndef PROPAGATORTESTTREE_H_
#define PROPAGATORTESTTREE_H_

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class TFile;
class TTree;

/** creates / fills tree with comparative values of two propagators
 *  for a series of steps along a helix.
 */
class PropagatorTestTree {
public:
  // Constructor: initialise histogramming
  PropagatorTestTree (const char* file = "PropagatorTest.root");
  // Destructor
  ~PropagatorTestTree();

  /** Filling of helix parameters: starting point and momentum,
   *    helix center, curvature and charge.
   */
  void fillHelix (const GlobalPoint&,
		  const GlobalVector&,
		  const GlobalPoint&,
		  const float,
		  const int);
  /** Filling of values for one point along the helix:
   *    path length, generated point and momentum,
   *    plane position and normal vector,
   *    trajectory state from the two propagators.
   */
  void addStep (const float,
		const GlobalPoint&,
		const GlobalVector&,
		const GlobalPoint&,
		const GlobalVector&,
		const TrajectoryStateOnSurface&,
		const TrajectoryStateOnSurface&);
  /// Filling of the tree for one helix
  void fill();
  /// Creation of arrays
  void createArrays();
  /// Deletion of arrays
  void removeArrays();
  /// Booking of tree
  void bookTree();

private:
  /// allocation of arrays for the 3 co-ordinates
  void createFloatArrays (float* arrayPtr[3]) const;
  /// deletion of arrays for the 3 co-ordinates
  void removeFloatArrays (float* arrayPtr[3]) const;
  /// storing of cartesian co-ordinates (point)
  inline void storeCartesian (const GlobalPoint& aPoint, float** arrayPtr, const int index) const
  {
    arrayPtr[0][index] = aPoint.x();
    arrayPtr[1][index] = aPoint.y();
    arrayPtr[2][index] = aPoint.z();
  }
  /// storing of cartesian co-ordinates (point, relative)
  inline void storeCartesian (const GlobalPoint& aPoint, const GlobalPoint& refPoint,
			      float** arrayPtr, const int index) const
  {
    GlobalVector d(aPoint-refPoint);
    arrayPtr[0][index] = d.x();
    arrayPtr[1][index] = d.y();
    arrayPtr[2][index] = d.z();
  }
  /// storing of cartesian co-ordinates (vector)
  inline void storeCartesian (const GlobalVector& aVector, 
			      float** arrayPtr, const int index) const
  {
    arrayPtr[0][index] = aVector.x();
    arrayPtr[1][index] = aVector.y();
    arrayPtr[2][index] = aVector.z();
  }
  /// storing of cartesian co-ordinates (vector, relative)
  inline void storeCartesian (const GlobalVector& aVector, const GlobalVector& refVector,
			      float** arrayPtr, const int index) const
  {
    GlobalVector d(aVector-refVector);
    arrayPtr[0][index] = d.x();
    arrayPtr[1][index] = d.y();
    arrayPtr[2][index] = d.z();
  }
  /// storing of angles + transverse component (vector)
  inline void storeAngles (const GlobalVector& aVector, 
			   float** arrayPtr, const int index) const
  {
    arrayPtr[0][index] = aVector.theta();
    arrayPtr[1][index] = aVector.phi();
    arrayPtr[2][index] = aVector.perp();
  }
  /// storing of angles + transverse component (vector, relative)
  inline void storeAngles (const GlobalVector& aVector, const GlobalVector& refVector,
			   float** arrayPtr, const int index) const
  {
    arrayPtr[0][index] = aVector.theta() - refVector.theta();
    arrayPtr[1][index] = aVector.phi() - refVector.phi();
    arrayPtr[2][index] = aVector.perp() - refVector.perp();
  }

private:
  const unsigned int theMaxSteps;
  TFile* theFile;
  TTree* theTree;

  float thePreviousPathLength;

  float theStart[6];
  float theCenter[3];
  float theRho;
  char theCharge;
  unsigned int theNrOfSteps;
  unsigned int theNrOfFwdSteps;
  float* thePathLengths;
  float* theGeneratedPoints[3];
  float* theGeneratedMomenta[3];
  float* thePlanePoints[3];
  float* thePlaneNormals[3];
  unsigned char* theOldStatus;
  float* theOldDPoints[3];
  float* theOldDMomenta[3];
  unsigned char* theNewStatus;
  float* theNewDPoints[3];
  float* theNewDMomenta[3];

};

#endif
