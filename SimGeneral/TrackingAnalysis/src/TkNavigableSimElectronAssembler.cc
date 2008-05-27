#include "SimGeneral/TrackingAnalysis/interface/TkNavigableSimElectronAssembler.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

using namespace std;

/** Assembles electron track segments to combined tracks, adds those
 *  tracks to the container and hides the segments.
 */

TkNavigableSimElectronAssembler::ElectronList
TkNavigableSimElectronAssembler::assemble (TrackPtrContainer& allTracks) const
{
  //
  // Create lists of electron tracks and non-electron tracks
  // (use lists for constant time removal, see below)
  //
  TrackList tracks(allTracks.begin(), allTracks.end());
  TrackList electronTracks(electronFilter(tracks));
//  std::cout << "TkNavigableSimElectronAssembler: found "
//          << electronTracks.size()
//          << " electron segments" << std::endl;
  if ( electronTracks.empty() ) return ElectronList();
  //
  // build assembled tracks
  //
  ElectronList electrons;
  TrackList trackSegments;
  while ( electronTracks.size() ) {
    //
    // initialise new combined electron
    //
    const TrackPtr startSegment = electronTracks.front();
    electronTracks.pop_front();
    trackSegments.clear();
    trackSegments.push_back(startSegment);
//    std::cout << "Starting assembly with segment at " << *startSegment
//              << std::endl;
//       << " (p=" << startSegment->momentum().mag()
//       << ",r_vtx=";
//     if ( startSegment->vertex() )
//       std::cout << startSegment->vertex()->position().perp();
//     std::cout << ",&vtx=" << startSegment->vertex() << ")" << std::endl;
    //
    // add segments before current segment
    //
    searchInwards(electronTracks, startSegment, trackSegments);
//    std::cout << "nb segments after inward search " << trackSegments.size()
//            << std::endl;
    //
    // add segments after current segment
    //
    searchOutwards(electronTracks, startSegment, trackSegments);
//    std::cout << "nb segments after outward search " << trackSegments.size()
//            << std::endl;
    //
    // store list of segments
    //
    electrons.push_back(trackSegments);
  }

  return electrons;
}


void
TkNavigableSimElectronAssembler::searchInwards (TrackList& electronTracks,
                                                const TrackPtr startSegment,
                                                TrackList& trackSegments) const
{

  TrackPtr currentSegment(startSegment);
//  TrackPtr debug = findParent(*currentSegment);
//  std::cout << "searchInwards: parent " << debug << std::endl;
  while ( TrackPtr nextSegment = findParent(*currentSegment) ) {
    trackSegments.push_front(nextSegment);
    electronTracks.remove(nextSegment);
    currentSegment = nextSegment;
//     EncodedSimTrackId packedId(nextSegment->id());
//     if ( packedId.bunch()!=startId.bunch() ||
//       packedId.eventInBunch()!=startId.eventInBunch() )
//       std::cout << "*** inconsistency in bunch / event number! ***" << std::endl;
//     std::cout << "Adding parent at " << nextSegment
//       << " (p=" << nextSegment->momentum().mag()
//       << ",r_vtx=";
//     if ( nextSegment->vertex() )
//       std::cout << nextSegment->vertex()->position().perp();
//     std::cout << ",&vtx=" << nextSegment->vertex() << ")" << std::endl;
  }
}


const TkNavigableSimElectronAssembler::TrackPtr
TkNavigableSimElectronAssembler::findParent (const TrackingParticle& track)
  const
{
  //
  // verify Bremsstrahlung
  //
  std::pair<const TrackPtr, const TrackPtr> segmentPair
    = checkVertex(track.parentVertex().get());
  //
  return segmentPair.first;
}


void
TkNavigableSimElectronAssembler::searchOutwards (TrackList& electronTracks,
                                                 const TrackPtr startSegment,
                                                 TrackList& trackSegments)
  const
{
//   EncodedSimTrackId startId(startSegment->id());
  TrackPtr currentSegment(startSegment);
  while ( TrackPtr nextSegment = findChild(*currentSegment) ) {
    trackSegments.push_back(nextSegment);
    electronTracks.remove(nextSegment);
    currentSegment = nextSegment;
//     EncodedSimTrackId packedId(nextSegment->id());
//     if ( packedId.bunch()!=startId.bunch() ||
//       packedId.eventInBunch()!=startId.eventInBunch() )
//       std::cout << "*** inconsistency in bunch / event number! ***" << std::endl;
//     std::cout << "Adding child at " << nextSegment
//       << " (p=" << nextSegment->momentum().mag()
//       << ",r_vtx=";
//     if ( nextSegment->vertex() )
//       std::cout << nextSegment->vertex()->position().perp();
//     std::cout << ",&vtx=" << nextSegment->vertex() << ")" << std::endl;
  }
}


const TkNavigableSimElectronAssembler::TrackPtr
TkNavigableSimElectronAssembler::findChild (const TrackingParticle& track)
  const
{
  //
  // verify bremsstrahlung
  //

  // for 131
  //  std::pair<TrackPtr, TrackPtr> segmentPair
  //    = checkVertex(track.decayVertex().get());
  //  std::pair<TrackPtr, TrackPtr> result(0,0);

  TrackingVertexRefVector decayVertices = track.decayVertices();
  if ( decayVertices.empty() ) {
//    std::cout << "Decay vertex is null " << std::endl;
    return 0;
  }

  std::pair<TrackPtr, TrackPtr> segmentPair
    //    = checkVertex(&(*track.decayVertices().at(0)));
    = checkVertex( &(*decayVertices.at(0)) );
  //
  return segmentPair.second;
}


/** Verify bremsstrahlung vertex: ask for one incoming electron and
 *  one outgoing electron + 0 or 1 outgoing photons.
 *  \return if bremsstrahlung: pointers to electrons, otherwise 0/0
 */
std::pair<TkNavigableSimElectronAssembler::TrackPtr,
          TkNavigableSimElectronAssembler::TrackPtr>
TkNavigableSimElectronAssembler::checkVertex (const TrackingVertex* vertex) const
{
  std::pair<TrackPtr, TrackPtr> result(0,0);
  //
  // check vertex & parent
  //
  if ( vertex==0 )  return result;
  const TrackingParticleRefVector parents(vertex->sourceTracks());
  if ( parents.empty() ) {
//     std::cout << "No parent track at vertex" << std::endl;
    return result;
  }
  if ( parents.size() != 1 ) {
//     std::cout << "More than 1 parent track at vertex" << std::endl;
    return result;
  }
  if ( abs( (**parents.begin()).pdgId()) != 11 ) {
//    std::cout << "Found parent track of type "
//            << (**parents.begin()).pdgId() << " at vertex" << std::endl;
    return result;
  }
  //
  // check secondaries
  //
  const TrackingParticleRefVector secondaries(vertex->daughterTracks());
  TrackPtr child(0);
  int nPhoton(0);
//   std::cout << "Types at vertex =";
  for ( TrackingVertex::tp_iterator it=vertex->daughterTracks_begin();
        it!=vertex->daughterTracks_end(); it++ ) {
//     std::cout << " " << (*it).pdgId();
    if ( abs((**it).pdgId()) == 11 ) {
      // accept only one electron in list of secondaries
      if ( child )  std::cout << std::endl << "Found several electrons at vertex" << std::endl;
      if ( child )  return result;
      child = const_cast<TrackPtr>(&(**it));
    }
    // accept <= 1 photon
    else if ( (**it).pdgId() == 22 ) {
      nPhoton++;
      if ( nPhoton>1 ) {
//      std::cout << "Found several photons at vertex" << std::endl;
        return result;
      }
    }
    else {
//      std::cout << std::endl << "Found track of type "
//              << (**parents.begin()).pdgId() << " at vertex" << std::endl;
      return result;
    }
  }
//  std::cout << std::endl;
//  if ( child==0 ) {
//    std::cout << "No electron found at vertex" << std::endl;
//  }
//  else {
//    std::cout << "ElectronAssembler::electron child" << std::endl;
//    std::cout << *child << std::endl;
//  }

  //
  result.first = const_cast<TrackPtr>(&(**parents.begin()));
  result.second = child;
  return result;
}


/** selection of electron segments
 *  on output, allTracks contains non-electron tracks only
 */
TkNavigableSimElectronAssembler::TrackList
TkNavigableSimElectronAssembler::electronFilter (TrackList& allTracks) const
{
  TrackList electrons;
  TrackList others;

  for ( TrackList::iterator it = allTracks.begin();
        it != allTracks.end(); it++ ) {
    if ( abs((**it).pdgId())==11 ) {
      electrons.push_back(*it);
      //      allTracks.erase(it);
    }
    else {
      others.push_back(*it);
    }
  }

  // on output, argument contains only non-electrons
  allTracks = others;
  return electrons;
}


/** Find edm::Ref of first segment of electron track
 */
/*
TrackingParticleRef TkNavigableSimElectronAssembler::findInitialSegmentRef(
  const TrackPtr& firstSegment) const
{
  TrackingVertexRef startV = (*firstSegment).parentVertex();

  TrackingParticleRefVector daughters = (*startV).daughterTracks();
  for (TrackingParticleRefVector::iterator it = daughters.begin();
       it != daughters.end(); it++) {
    TrackingParticleRef d = (*it);

    // TrackingParticles pointed to by firstSegment and one of vertex
    // daughters match ?
    // comparison temporarily done through Geant track ID and type
    // should be done through edm::Ref
    TrackingParticle::g4t_iterator gTkf = (*firstSegment).g4Track_begin();
    TrackingParticle::g4t_iterator gTkd = (*d.get()).g4Track_begin();

    if ( ( (*gTkf).trackId() == (*gTkd).trackId() )
         && ( (*gTkf).type() == (*gTkd).type() ) ) {
      return d;
    }
  }

//  std::cout << "PASCAL:: Now make producer of electron tracks" << std::endl;
//  std::cout << "Question: do-able without crossing frame info ?? " << std::endl;

  return TrackingParticleRef();
}
*/
