// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Regression guard for the SimTrack/SimVertex history reconnection logic
// (SimTrackManager::computeDroppedAncestorRedirect). At a high PersistencyEmin the
// low-energy intermediate ancestors are dropped, so a stored secondary's immediate
// parent may not be persisted; the redirect walks the full parentID map up to the
// nearest stored ancestor so the production SimVertex is never left an orphan
// (parentIndex = -1) and the truth graph stays connected to the generator.
//
// This guards the *algorithm*. The invariant it enforces - every stored, non-primary
// track resolves to a stored ancestor - is exactly what breaks if a simulation
// change (e.g. a GPU port) stops recording the per-track parent history.

#include "SimG4Core/Notification/interface/SimTrackManager.h"

#include <cppunit/extensions/HelperMacros.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class testSimTrackManagerReconnect : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSimTrackManagerReconnect);
  CPPUNIT_TEST(droppedImmediateParentIsReattached);
  CPPUNIT_TEST(deepChainOfDroppedAncestors);
  CPPUNIT_TEST(fullyStoredChainNeedsNoRedirect);
  CPPUNIT_TEST(malformedParentLoopTerminates);
  CPPUNIT_TEST(noStoredTrackIsLeftOrphan);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void droppedImmediateParentIsReattached();
  void deepChainOfDroppedAncestors();
  void fullyStoredChainNeedsNoRedirect();
  void malformedParentLoopTerminates();
  void noStoredTrackIsLeftOrphan();

private:
  // The invariant the truth graph relies on: with the redirect applied, every
  // stored non-primary track resolves to a stored ancestor (its immediate parent
  // is stored, or the redirect points at a stored ancestor), so no production
  // SimVertex is an orphan. Returns the trackIDs that violate it.
  static std::vector<int> orphanTracks(const std::vector<std::pair<int, int> >& stored,
                                       const std::unordered_map<int, int>& redirect) {
    std::unordered_set<int> savedIds;
    for (auto const& [id, parent] : stored)
      savedIds.insert(id);
    std::vector<int> orphans;
    for (auto const& [id, parent] : stored) {
      if (parent <= 0)
        continue;  // primary: attaches directly to the generator
      if (savedIds.count(parent) != 0)
        continue;  // immediate parent is stored
      auto it = redirect.find(id);
      if (it == redirect.end() || savedIds.count(it->second) == 0)
        orphans.push_back(id);
    }
    return orphans;
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSimTrackManagerReconnect);

void testSimTrackManagerReconnect::droppedImmediateParentIsReattached() {
  // 1(primary) -> 2(dropped) -> 3(stored); 1 -> 4(stored).
  const std::vector<std::pair<int, int> > stored = {{1, 0}, {3, 2}, {4, 1}};
  const std::unordered_map<int, int> parentOfAll = {{1, 0}, {2, 1}, {3, 2}, {4, 1}};

  const auto redirect = SimTrackManager::computeDroppedAncestorRedirect(stored, parentOfAll);

  CPPUNIT_ASSERT_EQUAL(std::size_t(1), redirect.size());
  CPPUNIT_ASSERT(redirect.count(3) == 1);
  CPPUNIT_ASSERT_EQUAL(1, redirect.at(3));  // 3 reattached to nearest stored ancestor (1)
  CPPUNIT_ASSERT(redirect.count(4) == 0);   // 4's immediate parent (1) is already stored
  CPPUNIT_ASSERT(orphanTracks(stored, redirect).empty());
}

void testSimTrackManagerReconnect::deepChainOfDroppedAncestors() {
  // 1(primary) -> 2 -> 3 -> 4 -> 5(stored); only 1 and 5 are stored.
  const std::vector<std::pair<int, int> > stored = {{1, 0}, {5, 4}};
  const std::unordered_map<int, int> parentOfAll = {{1, 0}, {2, 1}, {3, 2}, {4, 3}, {5, 4}};

  const auto redirect = SimTrackManager::computeDroppedAncestorRedirect(stored, parentOfAll);

  CPPUNIT_ASSERT(redirect.count(5) == 1);
  CPPUNIT_ASSERT_EQUAL(1, redirect.at(5));  // walked 4 -> 3 -> 2 -> 1
  CPPUNIT_ASSERT(orphanTracks(stored, redirect).empty());
}

void testSimTrackManagerReconnect::fullyStoredChainNeedsNoRedirect() {
  // Every ancestor stored: getOrCreateVertex resolves everything, no redirect.
  const std::vector<std::pair<int, int> > stored = {{1, 0}, {2, 1}, {3, 2}};
  const std::unordered_map<int, int> parentOfAll = {{1, 0}, {2, 1}, {3, 2}};

  const auto redirect = SimTrackManager::computeDroppedAncestorRedirect(stored, parentOfAll);

  CPPUNIT_ASSERT(redirect.empty());
  CPPUNIT_ASSERT(orphanTracks(stored, redirect).empty());
}

void testSimTrackManagerReconnect::malformedParentLoopTerminates() {
  // Defensive: a cyclic parent map (2 <-> 3) must terminate and must not invent a
  // bogus ancestor for a track whose chain never reaches a stored id.
  const std::vector<std::pair<int, int> > stored = {{1, 0}, {4, 2}};
  const std::unordered_map<int, int> parentOfAll = {{1, 0}, {2, 3}, {3, 2}, {4, 2}};

  const auto redirect = SimTrackManager::computeDroppedAncestorRedirect(stored, parentOfAll);

  CPPUNIT_ASSERT(redirect.count(4) == 0);  // no stored ancestor reachable; terminates, no bogus entry
}

void testSimTrackManagerReconnect::noStoredTrackIsLeftOrphan() {
  // Realistic shower: a primary with many generations, every other generation
  // dropped (sub-PersistencyEmin). Whatever is stored must stay connected to the
  // generator - this is the property that fails if the sim drops the parentage.
  std::unordered_map<int, int> parentOfAll;
  std::vector<std::pair<int, int> > stored;
  parentOfAll[1] = 0;
  stored.emplace_back(1, 0);  // primary
  int nextId = 2;
  int previousStored = 1;
  for (int generation = 0; generation < 8; ++generation) {
    const int dropped = nextId++;  // intermediate, NOT stored
    const int kept = nextId++;     // secondary, stored
    parentOfAll[dropped] = previousStored;
    parentOfAll[kept] = dropped;
    stored.emplace_back(kept, dropped);
    previousStored = kept;
  }

  const auto redirect = SimTrackManager::computeDroppedAncestorRedirect(stored, parentOfAll);

  const auto orphans = orphanTracks(stored, redirect);
  CPPUNIT_ASSERT_MESSAGE("history reconnection left orphan SimTracks (truth graph fragmented)", orphans.empty());
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
