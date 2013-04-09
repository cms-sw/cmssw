#ifndef DataFormats_GEMRecHit_H
#define DataFormats_GEMRecHit_H

/** \class GEMRecHit
 *
 *  RecHit for GEM 
 *
 *  $Date: 2007/08/02 05:20:41 $
 *  $Revision: 1.8 $
 *  \author M. Maggi -- INFN Bari 
 */

#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"


class GEMRecHit : public RecHit2DLocalPos {
 public:

  GEMRecHit(const GEMDetId& rpcId,
	    int bx);

  /// Default constructor
  GEMRecHit();

  /// Constructor from a local position, rpcId and digi time.
  /// The 3-dimensional local error is defined as
  /// resolution (the cell resolution) for the coordinate being measured
  /// and 0 for the two other coordinates
  GEMRecHit(const GEMDetId& rpcId,
	    int bx,
	    const LocalPoint& pos);
  

  /// Constructor from a local position and error, rpcId and bx.
  GEMRecHit(const GEMDetId& rpcId,
	    int bx,
	    const LocalPoint& pos,
	    const LocalError& err);
  

  /// Constructor from a local position and error, rpcId, bx, frist strip of cluster and cluster size.
  GEMRecHit(const GEMDetId& rpcId,
	    int bx,
	    int firstStrip,
	    int clustSize,
	    const LocalPoint& pos,
	    const LocalError& err);
  
  /// Destructor
  virtual ~GEMRecHit();


  /// Return the 3-dimensional local position
  virtual LocalPoint localPosition() const {
    return theLocalPosition;
  }


  /// Return the 3-dimensional error on the local position
  virtual LocalError localPositionError() const {
    return theLocalError;
  }


  virtual GEMRecHit* clone() const;

  
  /// Access to component RecHits.
  /// No components rechits: it returns a null vector
  virtual std::vector<const TrackingRecHit*> recHits() const;


  /// Non-const access to component RecHits.
  /// No components rechits: it returns a null vector
  virtual std::vector<TrackingRecHit*> recHits();


  /// Set local position 
  void setPosition(LocalPoint pos) {
    theLocalPosition = pos;
  }

  
  /// Set local position error
  void setError(LocalError err) {
    theLocalError = err;
  }


  /// Set the local position and its error
  void setPositionAndError(LocalPoint pos, LocalError err) {
    theLocalPosition = pos;
    theLocalError = err;
  }
  

  /// Return the rpcId
  GEMDetId rpcId() const {
    return theGEMId;
  }
 
  int BunchX() const {
    return theBx;
  }

  int firstClusterStrip() const {
    return theFirstStrip;
  }

  int clusterSize() const {
    return theClusterSize;
  }

  /// Comparison operator, based on the rpcId and the digi time
  bool operator==(const GEMRecHit& hit) const;

 private:
  GEMDetId theGEMId;
  int theBx;
  int theFirstStrip;
  int theClusterSize;
  // Position and error in the Local Ref. Frame of the GEMLayer
  LocalPoint theLocalPosition;
  LocalError theLocalError;

};
#endif

/// The ostream operator
std::ostream& operator<<(std::ostream& os, const GEMRecHit& hit);
