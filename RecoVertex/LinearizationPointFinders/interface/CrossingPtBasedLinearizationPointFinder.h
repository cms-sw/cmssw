#ifndef CrossingPtBasedLinearizationPointFinder_H
#define CrossingPtBasedLinearizationPointFinder_H

#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexTools/interface/ModeFinder3d.h"
#include "RecoVertex/VertexTools/interface/RecTracksDistanceMatrix.h"

  /** A linearization point finder. It works the following way:
   *  1. Calculate in an optimal way 'n_pairs' different crossing points.
   *     Optimal in this context means the following:
   *     a. Try to use as many different tracks as possible;
   *        avoid using the same track all the time.
   *     b. Use the most energetic tracks.
   *     c. Try not to group the most energetic tracks together.
   *        Try to group more energetic tracks with less energetic tracks.
   *        We assume collimated bundles here, so this is why.
   *     d. Perform optimally. Do not sort more tracks (by total energy, see b)
   *        than necessary.
   *     e. If n_pairs >= (number of all possible combinations),
   *        do not leave any combinations out.
   *     ( a. and e. are almost but not entirely fulfilled in the current impl )
   *  2. Apply theAlgo on the n points.
   */

class CrossingPtBasedLinearizationPointFinder : public LinearizationPointFinder
{
public:
  /** \param n_pairs: how many track pairs will be considered (maximum)
   *                  a value of -1 means full combinatorics.
   */
  CrossingPtBasedLinearizationPointFinder( const ModeFinder3d & algo,
     const signed int n_pairs = 5 );

  /** This constructor exploits the information stored in a 
   *  RecTracksDistanceMatrix object.
   *  \param n_pairs: how many track pairs will be considered (maximum)
   *                  a value of -1 means full combinatorics.
   */

  CrossingPtBasedLinearizationPointFinder( 
      const RecTracksDistanceMatrix * m, const ModeFinder3d & algo,
      const signed int n_pairs = -1 );

  CrossingPtBasedLinearizationPointFinder(
      const CrossingPtBasedLinearizationPointFinder & );

  ~CrossingPtBasedLinearizationPointFinder() override;

/** Method giving back the Initial Linearization Point.
 */
  GlobalPoint getLinearizationPoint(const std::vector<reco::TransientTrack> & ) const override;
  GlobalPoint getLinearizationPoint(const std::vector<FreeTrajectoryState> & ) const override;

  CrossingPtBasedLinearizationPointFinder * clone() const override {
    return new CrossingPtBasedLinearizationPointFinder ( * this );
  };
protected:
  const bool useMatrix;
  signed int theNPairs;
  const RecTracksDistanceMatrix *theMatrix;

private:
  /// calls (*theAglo) (input)
  /// can optionally save input / output in .root file
  GlobalPoint find ( const std::vector<std::pair <GlobalPoint , float> > & ) const;
private:
  ModeFinder3d * theAlgo;

  /** Private struct to order tracks by momentum
   */
  struct CompareTwoTracks {
    int operator() ( const reco::TransientTrack & a, const reco::TransientTrack & b ) {
            return a.initialFreeState().momentum().mag() >
        	   b.initialFreeState().momentum().mag();
//       return a.p() > b.p();
    };
  };
  std::vector <reco::TransientTrack> getBestTracks ( const std::vector<reco::TransientTrack> & ) const;
  GlobalPoint useFullMatrix ( const std::vector<reco::TransientTrack> & ) const;
  GlobalPoint useAllTracks  ( const std::vector<reco::TransientTrack> & ) const;
};

#endif
