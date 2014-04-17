#ifndef CommonDet_Trajectory_H
#define CommonDet_Trajectory_H

#include "DataFormats/Common/interface/RefToBase.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <algorithm>
#include <boost/shared_ptr.hpp>

/** A class for detailed particle trajectory representation.
 *  It is used during trajectory building to "grow" a trajectory.
 *  The trajectory is represented as an ordered sequence of 
 *  TrajectoryMeasurement objects with a stack-like interface.
 *  The measurements are added to the Trajectory in the order of
 *  increasing precision: each new TrajectoryMeasurement is assumed to improve
 *  the precision of the last one, normally by adding a constraint from 
 *  a new RecHit.
 *     However the Trajectory class does not have the means to verify
 *  that measurements are added in the correct order, and thus cannot
 *  guarantee the order, which is the responsibility of the 
 *  TrajectoryBuilder. The Trajectory provides some security by
 *  allowing to add or remove measurements only on one of it's ends,
 *  with push(TM) and pop() methods. The last measurement in a Trajectory
 *  can thus be either the innermost (closest to the interaction point)
 *  or the outermost, depending on the way the Trajectory was built.
 *  The direction of building is represented as a PropagationDirection,
 *  which has two possible values: alongMomentum (outwards) and
 *  oppositeToMomentum (inwards), and is accessed with the direction()
 *  method.
 */


class Trajectory
{
public:

  typedef std::vector<TrajectoryMeasurement>                   DataContainer;
  typedef TransientTrackingRecHit::ConstRecHitContainer        ConstRecHitContainer;
  typedef ConstRecHitContainer                                 RecHitContainer;


  /** Default constructor of an empty trajectory with undefined seed and 
   * undefined direction. This constructor is necessary in order to transiently
   * copy vector<Trajectory> in the edm::Event
   */
  
  Trajectory() : 
    theSeed(), seedRef_(),
    theChiSquared(0), theChiSquaredBad(0),
    theNumberOfFoundHits(0), theNumberOfLostHits(0),
    theDirection(anyDirection), theDirectionValidity(false), theValid(false),theDPhiCache(0),theNLoops(0)
    {}


  /** Constructor of an empty trajectory with undefined direction.
   *  The direction will be defined at the moment of the push of a second
   *  measurement, from the relative radii of the first and second 
   *  measurements.
   */
    
  explicit Trajectory( const TrajectorySeed& seed) : 
    theSeed( new TrajectorySeed(seed) ), seedRef_(),
    theChiSquared(0), theChiSquaredBad(0),
    theNumberOfFoundHits(0), theNumberOfLostHits(0),
    theDirection(anyDirection), theDirectionValidity(false), theValid(true),theDPhiCache(0),theNLoops(0)
  {}

  /** Constructor of an empty trajectory with defined direction.
   *  No check is made in the push method that measurements are
   *  added in the correct direction.
   */
  Trajectory( const TrajectorySeed& seed, PropagationDirection dir) : 
    theSeed( new TrajectorySeed(seed) ), seedRef_(),
    theChiSquared(0), theChiSquaredBad(0),
    theNumberOfFoundHits(0), theNumberOfLostHits(0),
    theDirection(dir), theDirectionValidity(true), theValid(true),theDPhiCache(0),theNLoops(0)
   
  {}

  /** Constructor of an empty trajectory with defined direction.
   *  No check is made in the push method that measurements are
   *  added in the correct direction.
   */
  Trajectory( const boost::shared_ptr<const TrajectorySeed> & seed, PropagationDirection dir) : 
    theSeed( seed ), seedRef_(),
    theChiSquared(0), theChiSquaredBad(0),
    theNumberOfFoundHits(0), theNumberOfLostHits(0),
    theDirection(dir), theDirectionValidity(true), theValid(true),theDPhiCache(0),theNLoops(0)
  {}

  /** Constructor of an empty trajectory with defined direction.
   *  No check is made in the push method that measurements are
   *  added in the correct direction.
   */
  explicit Trajectory(PropagationDirection dir) : 
    theSeed(), seedRef_(),
    theChiSquared(0), theChiSquaredBad(0),
    theNumberOfFoundHits(0), theNumberOfLostHits(0),
    theDirection(dir), theDirectionValidity(true), theValid(true),theDPhiCache(0),theNLoops(0)
   
  {}


#if defined( __GXX_EXPERIMENTAL_CXX0X__)
 
  Trajectory(Trajectory const & rh) = default;
  Trajectory & operator=(Trajectory const & rh) = default;

  Trajectory(Trajectory && rh) : 
    theSeed(std::move(rh.theSeed)), seedRef_(std::move(rh.seedRef_)),
    theData(std::move(rh.theData)),
    theChiSquared(rh.theChiSquared), theChiSquaredBad(rh.theChiSquaredBad),
    theNumberOfFoundHits(rh.theNumberOfFoundHits), theNumberOfLostHits(rh.theNumberOfLostHits),
    theDirection(rh.theDirection), theDirectionValidity(rh.theDirectionValidity), theValid(rh.theValid),
    theDPhiCache(rh.theDPhiCache),theNLoops(rh.theNLoops)  
  {}

  Trajectory & operator=(Trajectory && rh) {
    using std::swap;
    swap(theData,rh.theData);
    theChiSquared=rh.theChiSquared;
    theChiSquaredBad=rh.theChiSquaredBad;
    theValid=rh.theValid;
    theDPhiCache=rh.theDPhiCache;
    theNLoops=rh.theNLoops;  
    theNumberOfFoundHits=rh.theNumberOfFoundHits;
    theNumberOfLostHits=rh.theNumberOfLostHits;
    theDirection=rh.theDirection; 
    theDirectionValidity=rh.theDirectionValidity;
    swap(theSeed,rh.theSeed);
    swap(seedRef_,rh.seedRef_);

    return *this;

  }

#else
//  private:
//  Trajectory(Trajectory const & rh){}	
//  Trajectory & operator=(Trajectory const & rh){ return *this;}
//  public:
#endif

  /** Reserves space in the vector to avoid lots of allocations when 
      push_back-ing measurements */
  void reserve (unsigned int n) { theData.reserve(n); }
  
  /** Add a new measurement to a Trajectory.
   *  The Chi2 of the trajectory is incremented by the value
   *  of tm.estimate() . 
   */
  void push(const TrajectoryMeasurement& tm);
  /** same as the one-argument push, but the trajectory Chi2 is incremented
   *  by chi2Increment. Useful e.g. in trajectory smoothing.
   */
  void push(const TrajectoryMeasurement & tm, double chi2Increment);

#if defined( __GXX_EXPERIMENTAL_CXX0X__)
  void push(TrajectoryMeasurement&& tm);
  void push(TrajectoryMeasurement&& tm, double chi2Increment);
#endif

  /** Remove the last measurement from the trajectory.
   */
  void pop();

  /** Access to the last measurement.
   *  It's the most precise one in a trajectory before smoothing.
   *  It's the outermost measurement if direction() == alongMomentum,
   *  the innermost one if direction() == oppositeToMomentum.
   */
  TrajectoryMeasurement const & lastMeasurement() const {
    check(); 
    if (theData.back().recHitR().hit()!=0) return theData.back();
    else if (theData.size()>2) return *(theData.end()-2);
    else throw cms::Exception("TrajectoryMeasurement::lastMeasurement - Too few measurements in trajectory");
  }

  /** Access to the first measurement.
   *  It is the least precise one in a trajectory before smoothing.
   *  It is precise in a smoothed trajectory. 
   *  It's the innermost measurement if direction() == alongMomentum,
   *  the outermost one if direction() == oppositeToMomentum.
   */
  TrajectoryMeasurement const & firstMeasurement() const {
    check(); 
    if (theData.front().recHitR().hit()!=0) return theData.front();
    else if (theData.size()>2) return *(theData.begin()+1);
    else throw cms::Exception("TrajectoryMeasurement::firstMeasurement - Too few measurements in trajectory");
  }
  
  /** Return all measurements in a container.
   */
  DataContainer const & measurements() const { return theData;}
  DataContainer & measurements() { return theData;}

  /// obsolete name, use measurements() instead.
  DataContainer const & data() const { return measurements();}

  /** Return all RecHits in a container.
   */
  ConstRecHitContainer recHits() const {
    ConstRecHitContainer hits;
    hits.reserve(theData.size());
    for (Trajectory::DataContainer::const_iterator itm
           = theData.begin(); itm != theData.end(); itm++){
      hits.push_back((*itm).recHit());
    }
    return hits;
  }

  /** Just valid hits..
   *
   */
  void validRecHits(ConstRecHitContainer & cont) const;

  /** Number of valid RecHits used to determine the trajectory.
   *  Can be less than the number of measurements in data() since
   *  detector layers crossed without using RecHits from them are also 
   *  stored as measurements.
   */

  int foundHits() const { return theNumberOfFoundHits;}

  /** Number of detector layers crossed without valid RecHits.
   *  Used mainly as a criteria for abandoning a trajectory candidate
   *  during trajectory building.
   */

  int lostHits() const { return theNumberOfLostHits;}
  
  /// True if trajectory has no measurements.
  bool empty() const { return theData.empty();}

  /// - Trajectories with at least 1 valid hit:
  ///     value of the raw Chi2 of the trajectory, not normalised to the N.D.F.
  ///     (evaluated using only valid hits)
  /// - Trajectories with only invalid hits:
  ///     raw Chi2 (not norm.) of invalid hits w.r.t. the "default" trajectory
  ///     (traj. containing only the seed information)
  float chiSquared() const { return (theNumberOfFoundHits ? theChiSquared : theChiSquaredBad);}

  /// Number of dof of the trajectory. The method accepts a bool in order to properly 
  /// take into account the presence of magnetic field in the dof computation.
  /// By default the MF is considered ON.
  int ndof(bool bon = true) const;


  /** Direction of "growing" of the trajectory. 
   *  Possible values are alongMomentum (outwards) and 
   *  oppositeToMomentum (inwards).
   */
  PropagationDirection const & direction() const;

  /** Returns true if the Trajectory is valid.
   *  Trajectories are invalidated e.g. during ambiguity resolution.
   */
  bool isValid() const { return theValid;}

  /// Method to invalidate a trajectory. Useful during ambiguity resolution.
  void invalidate() { theValid = false;}

  /// Access to the seed used to reconstruct the Trajectory
  TrajectorySeed const & seed() const { return *theSeed;}


  /** Definition of inactive Det from the Trajectory point of view.
   */
  static bool inactive(//const Det& det
		       ){return false;}//FIXME

  /** Definition of what it means for a hit to be "lost".
   *  This definition is also used by the TrajectoryBuilder.
   */
  static bool lost( const TransientTrackingRecHit& hit);

  /** Returns true if the hit type is TrackingRecHit::bad
   *  Used in stand-alone trajectory construction
   */
  static bool isBad( const TransientTrackingRecHit& hit);

  /// Redundant method, returns the layer of lastMeasurement() .
  const DetLayer* lastLayer() const {
    check();
    if (theData.back().recHit()->hit()!=0) return theData.back().layer();
    else if (theData.size()>2) return (theData.end()-2)->layer();
    else throw cms::Exception("TrajectoryMeasurement::lastMeasurement - Too few measurements in trajectory");
  }

  /**  return the Reference to the trajectory seed in the original
   *   seeds collection. If the collection has been dropped from the
   *   Event, the reference may be invalid. Its validity should be tested,
   *   before the reference is actually used. 
   */
  edm::RefToBase<TrajectorySeed> seedRef(void) const { return seedRef_; }
  
  void setSeedRef(const edm::RefToBase<TrajectorySeed> & seedRef) { seedRef_ = seedRef ; } 

  TrajectoryStateOnSurface geometricalInnermostState() const;

  TrajectoryMeasurement const & closestMeasurement(GlobalPoint) const; 

  /// Reverse the propagation direction and the order of the trajectory measurements.
  /// It doesn't reverse the forward and backward predicted states within each trajectory measurement
  void reverse() ;

  const boost::shared_ptr<const TrajectorySeed> & sharedSeed() const { return theSeed; }
  void setSharedSeed(const boost::shared_ptr<const TrajectorySeed> & seed) { theSeed=seed;}

  /// accessor to the delta phi angle betweem the directions of the two measurements on the last 
  /// two layers crossed by the trajectory
   float dPhiCacheForLoopersReconstruction() const { return theDPhiCache;}

  /// method to set the delta phi angle betweem the directions of the two measurements on the last 
  /// two layers crossed by the trajectory
   void setDPhiCacheForLoopersReconstruction(float dphi) {  theDPhiCache = dphi;}

   bool isLooper() const { return (theNLoops>0);}
   signed char nLoops() const {return theNLoops;}

   void setNLoops(signed char value) { theNLoops=value;}
   void incrementLoops() {theNLoops++;}

private:

  void pushAux(double chi2Increment);


  boost::shared_ptr<const TrajectorySeed>    theSeed;
  edm::RefToBase<TrajectorySeed> seedRef_;

  DataContainer theData;
  float theChiSquared;
  float theChiSquaredBad;

  signed short theNumberOfFoundHits;
  signed short theNumberOfLostHits;

  PropagationDirection theDirection;
  bool                 theDirectionValidity;
  bool theValid;

  float theDPhiCache;
  signed char theNLoops;

  void check() const;
};

#endif
