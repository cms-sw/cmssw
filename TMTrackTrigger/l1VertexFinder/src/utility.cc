
#include "TMTrackTrigger/l1VertexFinder/interface/utility.h"


#include "TMTrackTrigger/l1VertexFinder/interface/TP.h"
#include "TMTrackTrigger/l1VertexFinder/interface/Settings.h"
#include "TMTrackTrigger/l1VertexFinder/interface/Stub.h"

#include "FWCore/Utilities/interface/Exception.h"


namespace vertexFinder {
namespace utility {

//=== Count number of tracker layers a given list of stubs are in.
//=== By default, consider both PS+2S modules, but optionally consider only the PS ones.

unsigned int countLayers(const Settings* settings, const vector<const Stub*>& vstubs, bool disableReducedLayerID, bool onlyPS) {

  //=== Unpack configuration parameters

  // Note if using reduced layer ID, so tracker layer can be encoded in 3 bits.
  static bool  reduceLayerID           = settings->reduceLayerID();
  // Define layers using layer ID (true) or by bins in radius of 5 cm width (false).
  static bool  useLayerID              = settings->useLayerID();
  // When counting stubs in layers, actually histogram stubs in distance from beam-line with this bin size.
  static float layerIDfromRadiusBin    = settings->layerIDfromRadiusBin();
  // Inner radius of tracker.
  static float trackerInnerRadius      = settings->trackerInnerRadius();

  // Disable use of reduced layer ID if requested, otherwise take from cfg.
  bool reduce  =  (disableReducedLayerID)  ?  false  :  reduceLayerID;

  const int maxLayerID(30);
  vector<bool> foundLayers(maxLayerID, false);

  if (useLayerID) {
    // Count layers using CMSSW layer ID.
    for (const Stub* stub: vstubs) {
      if ( (! onlyPS) || stub->psModule()) { // Consider only stubs in PS modules if that option specified.
	// Use either normal or reduced layer ID depending on request.
	int layerID = reduce  ?  stub->layerIdReduced()  :  stub->layerId();
	if (layerID >= 0 && layerID < maxLayerID) {
	  foundLayers[layerID] = true;
	} else {
	  throw cms::Exception("Utility::invalid layer ID");
	} 
      }
    }

  } else {
    // Count layers by binning stub distance from beam line.
    for (const Stub* stub: vstubs) {
      if ( (! onlyPS) || stub->psModule()) { // Consider only stubs in PS modules if that option specified.
	// N.B. In this case, no concept of "reduced" layer ID has been defined yet, so don't depend on "reduce";
	int layerID = (int) ( (stub->r() - trackerInnerRadius) / layerIDfromRadiusBin );
	if (layerID >= 0 && layerID < maxLayerID) {
	  foundLayers[layerID] = true;
	} else {
	  throw cms::Exception("Utility::invalid layer ID");
	} 
      }
    }
  }

  unsigned int ncount = 0;
  for (const bool& found: foundLayers) {
    if (found) ncount++;
  }
  
  return ncount;
}


//=== Given a set of stubs (presumably on a reconstructed track candidate)
//=== return the best matching Tracking Particle (if any),
//=== the number of tracker layers in which one of the stubs matched one from this tracking particle,
//=== and the list of the subset of the stubs which match those on the tracking particle.

const TP* matchingTP(const Settings* settings, const vector<Stub*>& vstubs,
                        unsigned int& nMatchedLayersBest, vector<const Stub*>& matchedStubsBest)
{
  // Get matching criteria
  const double        minFracMatchStubsOnReco = settings->minFracMatchStubsOnReco();
  const double        minFracMatchStubsOnTP   = settings->minFracMatchStubsOnTP();
  const unsigned int  minNumMatchLayers       = settings->minNumMatchLayers();
  const unsigned int  minNumMatchPSLayers     = settings->minNumMatchPSLayers();

  // Loop over the given stubs, looking at the TP that produced each one.

  map<const TP*, vector<const Stub*> > tpsToStubs;
  map<const TP*, vector<const Stub*> > tpsToStubsStrict;

  for (const Stub* s : vstubs) {
    // If this stub was produced by one or more TPs, store a link from the TPs to the stub.
    // (The assocated TPs here are influenced by config param "StubMatchStrict"). 
    for (const TP* tp_i : s->assocTPs()) {
      tpsToStubs[ tp_i ].push_back( s );
    }
    // To resolve tie-break situations, do the same, but now only considering strictly associated TP, where the TP contributed
    // to both clusters making up stub.
    if (s->assocTP() != nullptr) {
      tpsToStubsStrict[ s->assocTP() ].push_back( s );
    }
  }

  // Loop over all the TP that matched the given stubs, looking for the best matching TP.

  nMatchedLayersBest = 0;        // initialize
  unsigned int nMatchedLayersStrictBest = 0; // initialize
  matchedStubsBest.clear();     // initialize
  const TP* tpBest = nullptr;   // initialize

  for (const auto& iter: tpsToStubs) {
    const TP*                 tp                 = iter.first;
    const vector<const Stub*> matchedStubsFromTP = iter.second;

    const vector<const Stub*> matchedStubsStrictFromTP = tpsToStubsStrict[tp]; // Empty vector, if this TP didnt produce both clusters in stub.

    // Count number of the given stubs that came from this TP.
    unsigned int nMatchedStubs  = matchedStubsFromTP.size();
    // Count number of tracker layers in which the given stubs came from this TP.
    unsigned int nMatchedLayers = countLayers( settings, matchedStubsFromTP, true );
    unsigned int nMatchedPSLayers = countLayers( settings, matchedStubsFromTP, true, true );

    // For tie-breaks, count number of tracker layers in which both clusters of the given stubs came from this TP.
    unsigned int nMatchedLayersStrict = countLayers( settings, matchedStubsStrictFromTP, true );

    // If enough layers matched, then accept this tracking particle.
    // Of the three criteria used here, usually only one is used, with the cuts on the other two set ultra loose.
        
    if (nMatchedStubs >= minFracMatchStubsOnReco * vstubs.size() && // Fraction of matched stubs relative to number of given stubs  
  nMatchedStubs >= minFracMatchStubsOnTP   * tp->numAssocStubs() && // Fraction of matched stubs relative to number of stubs on TP.
  nMatchedLayers >= minNumMatchLayers && nMatchedPSLayers >= minNumMatchPSLayers) { // Number of matched layers
      // In case more than one matching TP found in this cell, note best match based on number of matching layers.
      // In tie-break situation, count layers in which both clusters in stub came from same TP.
      if (nMatchedLayersBest < nMatchedLayers || (nMatchedLayersBest == nMatchedLayers && nMatchedLayersStrictBest < nMatchedLayersStrict)) {
  // Store data for this TP match.
  nMatchedLayersBest = nMatchedLayers;
  matchedStubsBest   = matchedStubsFromTP;
  tpBest             = tp;
      }
    }
  }

  return tpBest;
}


} // end namespace utility
} // end namespace vertexFinder
