#ifndef TkNavigableSimElectronAssembler_h_
#define TkNavigableSimElectronAssembler_h_

#include <vector>
#include <list>

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

/** For each generator or Geant electron, finds the list of track segments 
 *  before first brem, in between brems, and after last brem. 
 *  Adapted from a code by Wolfgang Adam. 
 */
class TkNavigableSimElectronAssembler {

public:
  typedef TrackingParticle* TrackPtr;
  typedef std::vector<TrackPtr> TrackPtrContainer;
  typedef TrackingVertex* VertexPtr;
  typedef std::list<TrackPtr> TrackList;
  typedef std::list<TrackList> ElectronList;

private:

public:
  /// constructor
  TkNavigableSimElectronAssembler () {}
  /// destructor
  virtual ~TkNavigableSimElectronAssembler () {}
  /// generation of composite electrons
  ElectronList assemble (TrackPtrContainer& simTracks) const;

private:
  /// selection of electron segments
  /// on output, allTracks contains non-electron tracks only
  TrackList electronFilter (TrackList& allTracks) const;
  /// building track opposite to track direction
  void searchInwards (TrackList& electronTracks,
		      const TrackPtr startSegment,
		      TrackList& trackSegments) const;
  /// incoming electron
  const TrackPtr findParent (const TrackingParticle& track) const;
  /// building track along track direction
  void searchOutwards (TrackList& electronTracks,
		       const TrackPtr startSegment,
		       TrackList& trackSegments) const;
  /// outgoing electron
  const TrackPtr findChild (const TrackingParticle& track) const;
  /// verification of Bremsstrahlung
  std::pair<TrackPtr, TrackPtr> checkVertex (const TrackingVertex*) const;

  /// creation of combined electron track
  //  TrackingParticleRef createTrack (const TrackList& segments) const;
  /// find edm::Ref of initial track segment
  //  TrackingParticleRef findInitialSegmentRef (const TrackPtr& firstSegment) 
  //    const;


};
#endif // TkNavigableSimElectronAssembler_h_
