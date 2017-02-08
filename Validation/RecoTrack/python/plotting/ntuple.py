import ROOT

class SubDet:
    BPix = 1
    FPix = 2
    TIB = 3
    TID = 4
    TOB = 5
    TEC = 6

    _toString = {
        BPix: "BPix",
        FPix: "FPix",
        TIB:  "TIB",
        TID:  "TID",
        TOB:  "TOB",
        TEC:  "TEC",
    }

    @staticmethod
    def toString(det):
        return SubDet._toString[det]

# to be kept is synch with enum HitSimType in TrackingNtuple.py
class HitSimType:
    Signal = 0
    ITPileup = 1
    OOTPileup = 2
    Noise = 3
    Unknown = 99

class _Collection(object):
    """Adaptor class representing a collection of objects.

    Concrete collection classes should inherit from this class.

    """
    def __init__(self, tree, sizeBranch, objclass):
        """Constructor.

        Arguments:
        tree        -- TTree object
        sizeBranch  -- Name of the branch to be used in size()
        objclass    -- Class to be used for the objects in __getitem__()
        """
        super(_Collection, self).__init__()
        self._tree = tree
        self._sizeBranch = sizeBranch
        self._objclass = objclass

    def size(self):
        """Number of objects in the collection."""
        return int(getattr(self._tree, self._sizeBranch).size())

    def __len__(self):
        """Number of objects in the collection."""
        return self.size()

    def __getitem__(self, index):
        """Get object 'index' in the collection."""
        return self._objclass(self._tree, index)

    def __iter__(self):
        """Returns generator for the objects."""
        for index in xrange(self.size()):
            yield self._objclass(self._tree, index)

class _Object(object):
    """Adaptor class representing a single object in a collection.

    The member variables of the object are obtained from the branches
    with common prefix and a given index.

    Concrete object classes should inherit from this class.
    """
    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree   -- TTree object
        index  -- Index for this object
        prefix -- Prefix of the branchs
        """
        super(_Object, self).__init__()
        self._tree = tree
        self._index = index
        self._prefix = prefix

    def __getattr__(self, attr):
        """Return object member variable.

        'attr' is translated as a branch in the TTree (<prefix>_<attr>).
        """
        self._checkIsValid()
        return lambda: getattr(self._tree, self._prefix+"_"+attr)[self._index]

    def _checkIsValid(self):
        """Raise an exception if the object index is not valid."""
        if not self.isValid():
            raise Exception("%s is not valid" % self.__class__.__name__)

    def isValid(self):
        """Check if object index is valid."""
        return self._index != -1

    def index(self):
        """Return object index."""
        return self._index

class _HitObject(_Object):
    """Adaptor class for pixel/strip hit objects."""
    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree   -- TTree object
        index  -- Index for this object
        prefix -- Prefix of the branchs
        """
        """Constructor
        """
        super(_HitObject, self).__init__(tree, index, prefix)

    def ntracks(self):
        """Returns number of tracks containing this hit."""
        self._checkIsValid()
        return getattr(self._tree, self._prefix+"_trkIdx")[self._index].size()

    def tracks(self):
        """Returns generator for tracks containing this hit.

        The generator returns Track objects
        """
        self._checkIsValid()
        for itrack in getattr(self._tree, self._prefix+"_trkIdx")[self._index]:
            yield Track(self._tree, itrack)

    def nseeds(self):
        """Returns number of seeds containing this hit."""
        self._checkIsValid()
        return getattr(self._tree, self._prefix+"_seeIdx")[self._index].size()

    def seeds(self):
        """Returns generator for tracks containing this hit.

        The generator returns Seed objects
        """
        self._checkIsValid()
        for iseed in getattr(self._tree, self._prefix+"_seeIdx")[self._index]:
            yield Seed(self._tree, iseed)


class _RecoHitAdaptor(object):
    """Adaptor class for objects containing hits (e.g. tracks)"""
    def __init__(self):
        super(_RecoHitAdaptor, self).__init__()

    def _hits(self):
        """Internal method to generate pairs of hit index and type."""
        for ihit, hitType in zip(self.hitIdx(), self.hitType()):
            yield (ihit, hitType)

    def hits(self):
        """Returns generator for hits.

        Generator returns PixelHit/StripHit/GluedHit depending on the
        hit type.

        """
        for ihit, hitType in self._hits():
            if hitType == 0:
                yield PixelHit(self._tree, ihit)
            elif hitType == 1:
                yield StripHit(self._tree, ihit)
            elif hitType == 2:
                yield GluedHit(self._tree, ihit)
            elif hitType == 3:
                yield InvalidHit(self._tree, ihit)
            else:
                raise Exception("Unknown hit type %d" % hitType)

    def pixelHits(self):
        """Returns generator for pixel hits."""
        self._checkIsValid()
        for ihit, hitType in self._hits():
            if hitType != 0:
                continue
            yield PixelHit(self._tree, ihit)

    def stripHits(self):
        """Returns generator for strip hits."""
        self._checkIsValid()
        for ihit, hitType in self._hits():
            if hitType != 1:
                continue
            yield StripHit(self._tree, ihit)

    def gluedHits(self):
        """Returns generator for matched strip hits."""
        self._checkIsValid()
        for ihit, hitType in self._hits():
            if hitType != 2:
                continue
            yield GluedHit(self._tree, ihit)

    def invalidHits(self):
        """Returns generator for invalid hits."""
        self._checkIsValid()
        for ihit, hitType in self._hits():
            if hitType != 3:
                continue
            yield InvalidHit(self._tree, ihit)

class _SimHitAdaptor(object):
    """Adaptor class for objects containing or matched to SimHits (e.g. TrackingParticles, reco hits)."""
    def __init__(self):
        super(_SimHitAdaptor, self).__init__()

    def nSimHits(self):
        self._checkIsValid()
        return self.simHitIdx().size()

    def simHits(self):
        """Returns generator for SimHits."""
        self._checkIsValid()
        for ihit in self.simHitIdx():
            yield SimHit(self._tree, ihit)

class _LayerStrAdaptor(object):
    """Adaptor class for layerStr() method."""
    def __init__(self):
        super(_LayerStrAdaptor, self).__init__()

    def layerStr(self):
        """Returns a string describing the layer of the hit."""
        self._checkIsValid()
        return "%s%d" % (SubDet.toString(getattr(self._tree, self._prefix+"_det")[self._index]),
                         getattr(self._tree, self._prefix+"_lay")[self._index])

class _TrackingParticleMatchAdaptor(object):
    """Adaptor class for objects matched to TrackingParticles."""
    def __init__(self):
        super(_TrackingParticleMatchAdaptor, self).__init__()

    def _nMatchedTrackingParticles(self):
        """Internal method to get the number of matched TrackingParticles."""
        return getattr(self._tree, self._prefix+"_simTrkIdx")[self._index].size()

    def nMatchedTrackingParticles(self):
        """Returns the number of matched TrackingParticles."""
        self._checkIsValid()
        return self._nMatchedTrackingParticles()

    def matchedTrackingParticleInfos(self):
        """Returns a generator for matched TrackingParticles.

        The generator returns TrackingParticleMatchInfo objects.

        """
        self._checkIsValid()
        for imatch in xrange(self._nMatchedTrackingParticles()):
            yield TrackingParticleMatchInfo(self._tree, self._index, imatch, self._prefix)

##########
class TrackingNtuple(object):
    """Class abstracting the whole ntuple/TTree.

    Main benefit is to provide nice interface for
    - iterating over events
    - querying whether hit/seed information exists
    """
    def __init__(self, fileName, tree="trackingNtuple/tree"):
        """Constructor.

        Arguments:
        fileName -- String for path to the ROOT file
        tree     -- Name of the TTree object inside the ROOT file (default: 'trackingNtuple/tree')
        """
        super(TrackingNtuple, self).__init__()
        self._file = ROOT.TFile.Open(fileName)
        self._tree = self._file.Get(tree)
        self._entries = self._tree.GetEntriesFast()

    def tree(self):
        return self._tree

    def nevents(self):
        return self._entries

    def hasHits(self):
        """Returns true if the ntuple has hit information."""
        return hasattr(self._tree, "pix_isBarrel")

    def hasSeeds(self):
        """Returns true if the ntuple has seed information."""
        return hasattr(self._tree, "see_fitok")

    def __iter__(self):
        """Returns generator for iterating over TTree entries (events)

        Generator returns Event objects.

        """
        for jentry in xrange(self._entries):
            # get the next tree in the chain and verify
            ientry = self._tree.LoadTree( jentry )
            if ientry < 0: break
            # copy next entry into memory and verify
            nb = self._tree.GetEntry( jentry )
            if nb <= 0: continue

            yield Event(self._tree, jentry)

##########
class Event(object):
    """Class abstracting a single event.

    Main benefit is to provide nice interface to get various objects
    or collections of objects.
    """
    def __init__(self, tree, entry):
        """Constructor.

        Arguments:
        tree  -- TTree object
        entry -- Entry number in the tree
        """
        super(Event, self).__init__()
        self._tree = tree
        self._entry = entry

    def entry(self):
        return self._entry

    def event(self):
        return self._tree.event

    def lumi(self):
        """Returns lumisection number."""
        return self._tree.lumi

    def run(self):
        """Returns run number."""
        return self._tree.run

    def beamspot(self):
        """Returns BeamSpot object."""
        return BeamSpot(self._tree)

    def tracks(self):
        """Returns Tracks object."""
        return Tracks(self._tree)

    def pixelHits(self):
        """Returns PixelHits object."""
        return PixelHits(self._tree)

    def stripHits(self):
        """Returns StripHits object."""
        return StripHits(self._tree)

    def gluedHits(self):
        """Returns GluedHits object."""
        return GluedHits(self._tree)

    def seeds(self):
        """Returns Seeds object."""
        return Seeds(self._tree)

    def trackingParticles(self):
        """Returns TrackingParticles object."""
        return TrackingParticles(self._tree)

    def vertices(self):
        """Returns Vertices object."""
        return Vertices(self._tree)

    def trackingVertices(self):
        """Returns TrackingVertices object."""
        return TrackingVertices(self._tree)

##########
class BeamSpot(object):
    """Class representing the beam spot."""
    def __init__(self, tree):
        """Constructor.

        Arguments:
        tree -- TTree object
        """
        super(BeamSpot, self).__init__()
        self._tree = tree
        self._prefix = "bsp"

    def __getattr__(self, attr):
        """Return object member variable.

        'attr' is translated as a branch in the TTree (bsp_<attr>).
        """
        return lambda: getattr(self._tree, self._prefix+"_"+attr)

##########
class TrackingParticleMatchInfo(_Object):
    """Class representing a match to a TrackingParticle.

    The point of this class is to provide, in addition to the matched
    TrackingParticle, also other information about the match (e.g.
    shared hit fraction for tracks/seeds or SimHit information for hits.
    """
    def __init__(self, tree, index, tpindex, prefix):
        """Constructor.

        Arguments:
        tree    -- TTree object
        index   -- Index of the object (track/seed/hit) matched to TrackingParticle
        tpindex -- Index of the TrackingParticle match (second index in _simTrkIdx branch)
        prefix  -- String for prefix of the object (track/seed/hit) matched to TrackingParticle
        """
        super(TrackingParticleMatchInfo, self).__init__(tree, index, prefix)
        self._tpindex = tpindex

    def __getattr__(self, attr):
        """Custom __getattr__ because of the second index needed to access the branch."""
        return lambda: super(TrackingParticleMatchInfo, self).__getattr__(attr)()[self._tpindex]

    def trackingParticle(self):
        """Returns matched TrackingParticle."""
        self._checkIsValid()
        return TrackingParticle(self._tree, getattr(self._tree, self._prefix+"_simTrkIdx")[self._index][self._tpindex])

class TrackMatchInfo(_Object):
    """Class representing a match to a Track.

    The point of this class is to provide, in addition to the matched
    Track, also other information about the match (e.g. shared hit fraction.
    """
    def __init__(self, tree, index, trkindex, prefix):
        """Constructor.

        Arguments:
        tree     -- TTree object
        index    -- Index of the object (TrackingParticle) matched to track
        trkindex -- Index of the track match (second index in _trkIdx branch)
        prefix   -- String for prefix of the object (TrackingParticle) matched to track
        """
        super(TrackMatchInfo, self).__init__(tree, index, prefix)
        self._trkindex = trkindex

    def track(self):
        """Returns matched Track."""
        self._checkIsValid()
        return Track(self._tree, getattr(self._tree, self._prefix+"_trkIdx")[self._index][self._trkindex])

##########
class Track(_Object, _RecoHitAdaptor, _TrackingParticleMatchAdaptor):
    """Class presenting a track."""
    def __init__(self, tree, index):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the track
        """
        super(Track, self).__init__(tree, index, "trk")

    def seed(self):
        """Returns Seed of the track."""
        self._checkIsValid()
        return Seed(self._tree, self._tree.trk_seedIdx[self._index])

    def vertex(self):
        """Returns Vertex that used this track in its fit."""
        self._checkIsValid()
        return Vertex(self._tree, self._tree.trk_vtxIdx[self._index])

class Tracks(_Collection):
    """Class presenting a collection of tracks."""
    def __init__(self, tree):
        """Constructor.

        Arguments:
        tree -- TTree object
        """
        super(Tracks, self).__init__(tree, "trk_pt", Track)

##########
class PixelHit(_HitObject, _LayerStrAdaptor, _SimHitAdaptor):
    """Class representing a pixel hit."""
    def __init__(self, tree, index):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the hit
        """
        super(PixelHit, self).__init__(tree, index, "pix")

    def isValidHit(self):
        return True

class PixelHits(_Collection):
    """Class presenting a collection of pixel hits."""
    def __init__(self, tree):
        """Constructor.

        Arguments:
        tree -- TTree object
        """
        super(PixelHits, self).__init__(tree, "pix_isBarrel", PixelHit)

##########
class StripHit(_HitObject, _LayerStrAdaptor, _SimHitAdaptor):
    """Class representing a strip hit."""
    def __init__(self, tree, index):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the hit
        """
        super(StripHit, self).__init__(tree, index, "str")

    def isValidHit(self):
        return True

class StripHits(_Collection):
    """Class presenting a collection of strip hits."""
    def __init__(self, tree):
        """Constructor.

        Arguments:
        tree -- TTree object
        """
        super(StripHits, self).__init__(tree, "str_isBarrel", StripHit)

##########
class GluedHit(_Object):
    """Class representing a matched strip hit."""
    def __init__(self, tree, index):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the hit
        """
        super(GluedHit, self).__init__(tree, index, "glu")

    def isValidHit(self):
        return True

    def monoHit(self):
        """Returns a StripHit for the mono hit."""
        self._checkIsValid()
        return StripHit(self._tree, self._tree.glu_monoIdx[self._index])

    def stereoHit(self):
        """Returns a StripHit for the stereo hit."""
        self._checkIsValid()
        return StripHit(self._tree, self._tree.glu_stereoIdx[self._index])

    def nseeds(self):
        """Returns the number of seeds containing this hit."""
        self._checkIsValid()
        return self._tree.glu_seeIdx[self._index].size()

    def seeds(self):
        """Returns generator for seeds containing this hit.

        The generator returns Seed objects
        """
        self._checkIsValid()
        for iseed in self._tree.glu_seeIdx[self._index]:
            yield Seed(self._tree, iseed)

class GluedHits(_Collection):
    """Class presenting a collection of matched strip hits."""
    def __init__(self, tree):
        """Constructor.

        Arguments:
        tree -- TTree object
        """
        super(GluedHits, self).__init__(tree, "glu_isBarrel", GluedHit)

##########
class InvalidHit(_Object):
    # repeating TrackingRecHit::Type
    class Type:
        missing = 1
        inactive = 2
        bad = 3
        missing_inner = 4
        missing_outer = 5

        _toString = {
            missing: "missing",
            inactive: "inactive",
            bad: "bad",
            missing_inner: "missing_inner",
            missing_outer: "missing_outer",
        }

    """Class representing an invalid hit."""
    def __init__(self, tree, index):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the hit
        """
        super(InvalidHit, self).__init__(tree, index, "inv")

    def isValidHit(self):
        return False

    def layerStr(self):
        """Returns a string describing the layer of the hit."""
        det = self._tree.inv_det[self._index]
        invalid_type = self._tree.inv_type[self._index]
        return "%s%d (%s)" % (SubDet.toString(det), self._tree.inv_lay[self._index], InvalidHit.Type._toString[invalid_type])

##########
def _seedOffsetForAlgo(tree, algo):
    """Internal function for returning a pair of indices for the beginning of seeds of a given 'algo', and the one-beyond-last index of the seeds."""
    for ioffset, offset in enumerate(tree.see_offset):
        if tree.see_algo[offset] == algo:
            next_offset = tree.see_offset[ioffset+1] if ioffset < tree.see_offset.size()-1 else tree.see_algo.size()
            return (offset, next_offset)
    return (-1, -1)

class Seed(_Object, _RecoHitAdaptor, _TrackingParticleMatchAdaptor):
    """Class presenting a seed."""
    def __init__(self, tree, index):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the seed
        """
        super(Seed, self).__init__(tree, index, "see")

    def indexWithinAlgo(self):
        """Returns the seed index within the seeds of the same algo.

        In case of errors, -1 is returned.
        """
        self._checkIsValid()
        algo = self._tree.see_algo[self._index]
        (offset, next_offset) = _seedOffsetForAlgo(self._tree, algo)
        if offset == -1: # algo not found
            return -1
        return self._index - offset

    def track(self):
        """Returns Track that was made from this seed."""
        self._checkIsValid()
        return Track(self._tree, self._tree.see_trkIdx[self._index])

class Seeds(_Collection):
    """Class presenting a collection of seeds."""
    def __init__(self, tree):
        """Constructor.

        Arguments:
        tree -- TTree object
        """
        super(Seeds, self).__init__(tree, "see_pt", Seed)

    def nSeedsForAlgo(self, algo):
        """Returns the number of seeds for a given 'algo'."""
        (offset, next_offset) = _seedOffsetForAlgo(self._tree, algo)
        return next_offset - offset

    def seedsForAlgo(self, algo):
        """Returns gnerator iterating over the seeds of a given 'algo'.

        Generator returns Seed object.
        """
        (offset, next_offset) = _seedOffsetForAlgo(self._tree, algo)
        for isee in xrange(offset, next_offset):
            yield Seed(self._tree, isee)

##########
class SimHit(_Object, _LayerStrAdaptor, _RecoHitAdaptor):
    """Class representing a SimHit which has not induced a RecHit."""
    def __init__(self, tree, index):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the SimHit
        """
        super(SimHit, self).__init__(tree, index, "simhit")

    def nRecHits(self):
        self._checkIsValid()
        return self._tree.simhit_hitIdx[self._index].size()

    def trackingParticle(self):
        self._checkIsValid()
        return TrackingParticle(self._tree, getattr(self._tree, self._prefix+"_simTrkIdx")[self._index])

##########
class TrackingParticle(_Object, _SimHitAdaptor):
    """Class representing a TrackingParticle."""
    def __init__(self, tree, index):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the TrackingParticle
        """
        super(TrackingParticle, self).__init__(tree, index, "sim")

    def _nMatchedTracks(self):
        """Internal function to get the number of matched tracks."""
        return self._tree.sim_trkIdx[self._index].size()

    def nMatchedTracks(self):
        """Returns the number of matched tracks."""
        self._checkIsValid()
        return self._nMatchedTracks()

    def matchedTrackInfos(self):
        """Returns a generator for matched tracks.

        The generator returns TrackMatchInfo objects.
        """
        self._checkIsValid()
        for imatch in xrange(self._nMatchedTracks()):
            yield TrackMatchInfo(self._tree, self._index, imatch, self._prefix)

    def parentVertex(self):
        """Returns the parent TrackingVertex."""
        self._checkIsValid()
        return TrackingVertex(self._tree, self._tree.sim_parentVtxIdx[self._index])

    def decayVertices(self):
        """Returns a generator for decay vertices.

        The generator returns TrackingVertex objects.
        """
        self._checkIsValid()
        for ivtx in self._tree.sim_decayVtxIdx[self._index]:
            yield TrackingVertex(self._tree, ivtx)

class TrackingParticles(_Collection):
    """Class presenting a collection of TrackingParticles."""
    def __init__(self, tree):
        """Constructor.

        Arguments:
        tree -- TTree object
        """
        super(TrackingParticles, self).__init__(tree, "sim_pt", TrackingParticle)

##########
class Vertex(_Object):
    """Class presenting a primary vertex."""
    def __init__(self, tree, index):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the vertex
        """
        super(Vertex, self).__init__(tree, index, "vtx")

    def nTracks(self):
        """Returns the number of tracks used in the vertex fit."""
        self._checkIsValid()
        return self._tree.vtx_trkIdx[self._index].size()

    def tracks(self):
        """Returns a generator for the tracks used in the vertex fit.

        The generator returns Track object.
        """
        self._checkIsValid()
        for itrk in self._tree.vtx_trkIdx[self._index]:
            yield Track(self._tree, itrk)

class Vertices(_Collection):
    """Class presenting a collection of vertices."""
    def __init__(self, tree):
        """Constructor.

        Arguments:
        tree -- TTree object
        """
        super(Vertices, self).__init__(tree, "vtx_valid", Vertex)

##########
class TrackingVertex(_Object):
    """Class representing a TrackingVertex."""
    def __init__(self, tree, index):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the TrackingVertex
        """
        super(TrackingVertex, self).__init__(tree, index, "simvtx")

    def nSourceTrackingParticles(self):
        """Returns the number of source TrackingParticles."""
        self._checkIsValid()
        return self._tree.simvtx_sourceSimIdx[self._index].size()

    def nDaughterTrackingParticles(self):
        """Returns the number of daughter TrackingParticles."""
        self._checkIsValid()
        return self._tree.simvtx_daughterSimIdx[self._index].size()

    def sourceTrackingParticles(self):
        """Returns a generator for the source TrackingParticles."""
        self._checkIsValid()
        for isim in self._tree.simvtx_sourceSimIdx[self._index]:
            yield TrackingParticle(self._tree, isim)

    def daughterTrackingParticles(self):
        """Returns a generator for the daughter TrackingParticles."""
        self._checkIsValid()
        for isim in self._tree.simvtx_daughterSimIdx[self._index]:
            yield TrackingParticle(self._tree, isim)

class TrackingVertices(_Collection, TrackingVertex):
    """Class presenting a collection of TrackingVertices."""
    def __init__(self, tree):
        """Constructor.

        Arguments:
        tree -- TTree object
        """
        super(TrackingVertex, self).__init__(tree, "simvtx_x")
