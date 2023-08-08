#ifndef PSimHit_H
#define PSimHit_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

class TrackingSlaveSD;  // for friend declaration only

/** 
 * Persistent-capable SimHit.
 * Suitable for tracking detectors.
 */

class PSimHit {
public:
  static constexpr unsigned int k_tidOffset = 200000000;

  PSimHit() : theDetUnitId(0) {}

  PSimHit(const Local3DPoint& entry,
          const Local3DPoint& exit,
          float pabs,
          float tof,
          float eloss,
          int particleType,
          unsigned int detId,
          unsigned int trackId,
          float theta,
          float phi,
          unsigned short processType = 0)
      : theEntryPoint(entry),
        theSegment(exit - entry),
        thePabs(pabs),
        theEnergyLoss(eloss),
        theThetaAtEntry(theta),
        thePhiAtEntry(phi),
        theTof(tof),
        theParticleType(particleType),
        theProcessType(processType),
        theDetUnitId(detId),
        theTrackId(trackId) {}

  /// Entry point in the local Det frame
  Local3DPoint entryPoint() const { return theEntryPoint; }

  /// Exit point in the local Det frame
  Local3DPoint exitPoint() const { return theEntryPoint + theSegment; }

  /** Local position in the Det frame.
   *  Normally it is on the detection surface, but this is not
   *  checked. It is computed as the middle point between entry and exit.
   */
  Local3DPoint localPosition() const { return theEntryPoint + 0.5 * theSegment; }

  /// The momentum of the track that produced the hit, at entry point.
  LocalVector momentumAtEntry() const { return LocalVector(thetaAtEntry(), phiAtEntry(), pabs()); }

  /// Obsolete. Same as momentumAtEntry().unit(), for backward compatibility.
  LocalVector localDirection() const { return LocalVector(thetaAtEntry(), phiAtEntry(), 1.f); }

  /// fast and more accurate access to momentumAtEntry().theta()
  Geom::Theta<float> thetaAtEntry() const { return Geom::Theta<float>(theThetaAtEntry); }

  /// fast and more accurate access to momentumAtEntry().phi()
  Geom::Phi<float> phiAtEntry() const { return Geom::Phi<float>(thePhiAtEntry); }

  /// fast and more accurate access to momentumAtEntry().mag()
  float pabs() const { return thePabs; }

  /** Time of flight in nanoseconds from the primary interaction
   *  to the entry point. Always positive in a PSimHit,
   *  but may become negative in a SimHit due to bunch assignment.
   */
  float timeOfFlight() const { return tof(); }

  /// deprecated name for timeOfFlight()
  float tof() const { return theTof; }

  /// The energy deposit in the PSimHit, in ???.
  float energyLoss() const { return theEnergyLoss; }

  /** The particle type of the track that produced this hit,
   *  in standard PDG code. 
   *  NB: This differs from ORCA5 and earlier, where the code was Geant3.
   *  The particle type of the hit may differ from the particle type of
   *  the SimTrack with id trackId(). This happends if the hit was created
   *  by a secondary track (e.g. a delta ray) originating from the 
   *  trackId() and not existing as a separate SimTrack.
   */
  int particleType() const { return theParticleType; }

  /** The DetUnit identifier, to be interpreted in the context of the
   *  detector system that produced the hit. E.g. in the Tracker
   *  this is index used with DetUnitNumbering<TrackerSimHitTag>.
   *  Currently the context is not deducible from the PSimHit and
   *  must be known when the PSimHit is created/accessed.
   */
  unsigned int detUnitId() const { return theDetUnitId; }

  /** The SimTrack ID of the "mother" track. This may be the actual
   *  charged track that produced the hit, or a "mother" of this
   *  track, in case the track that produced the hit was not
   *  saved as a SimTrack.
   *  This ID must be interpreted in the context of the SimEvent
   *  to which the PSimHit belongs.
   */
  unsigned int trackId() const { return theTrackId; }

  /** In case te SimTrack ID is incremented by the k_tidOffset for hit category definition, this
   * methods returns the original theTrackId value directly.
   */
  unsigned int originalTrackId() const { return (theTrackId > k_tidOffset) ? theTrackId % k_tidOffset : theTrackId; }

  unsigned int offsetTrackId() const { return (theTrackId > k_tidOffset) ? theTrackId / k_tidOffset : theTrackId; }

  static unsigned int addTrackIdOffset(unsigned int tId, unsigned int offset) { return offset * k_tidOffset + tId; }

  EncodedEventId eventId() const { return theEventId; }

  void setEventId(EncodedEventId e) { theEventId = e; }

  /** The ID of the physics process that created the track that produced 
   *  the hit. This is useful for identifying hits from secondary interactions,
   *  especially in the case when the track that produced the hit was not saved 
   *  as a SimTrack.
   *  The meaning of the ID is defined outside of the PSimHit; The only 
   *  value with special significance is zero (for "undefined"), so zero should
   *  not be the ID of any process.
   */
  unsigned short processType() const { return theProcessType; }

  void setTof(float tof) { theTof = tof; }

protected:
  // properties
  Local3DPoint theEntryPoint;  // position at entry
  Local3DVector theSegment;    // exitPos - entryPos
  float thePabs;               // momentum
  float theEnergyLoss;         // Energy loss
  float theThetaAtEntry;
  float thePhiAtEntry;

  float theTof;  // Time Of Flight
  int theParticleType;
  unsigned short theProcessType;  // ID of the process which created the track
                                  // which created the PSimHit

  // association
  unsigned int theDetUnitId;
  unsigned int theTrackId;
  EncodedEventId theEventId;

  friend class TrackingSlaveSD;
};

std::ostream& operator<<(std::ostream& o, const PSimHit& hit);

#endif  // PSimHit_H
