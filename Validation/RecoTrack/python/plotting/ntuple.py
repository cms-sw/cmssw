import ROOT

import sys
import difflib

class Detector:
#    class Phase0: pass # not supported yet
    class Phase1: pass
#    class Phase2: pass # not supported yet

    def __init__(self):
        self._detector = self.Phase1

    def set(self, det):
        self._detector = det

    def get(self):
        return self._detector

detector = Detector()

# Poor-man enum class with string conversion
class _Enum:
    def __init__(self, **values):
        self._reverse = {}
        for key, value in values.iteritems():
            setattr(self, key, value)
            if value in self._reverse:
                raise Exception("Value %s is already used for a key %s, tried to re-add it for key %s" % (value, self._reverse[value], key))
            self._reverse[value] = key

    def toString(self, val):
        return self._reverse[val]

SubDet = _Enum(
    BPix = 1,
    FPix = 2,
    TIB = 3,
    TID = 4,
    TOB = 5,
    TEC = 6
)

# Needs to be kept consistent with
# DataFormats/TrackReco/interface/TrackBase.h
Algo = _Enum(
    undefAlgorithm = 0, ctf = 1,
    duplicateMerge = 2, cosmics = 3,
    initialStep = 4,
    lowPtTripletStep = 5,
    pixelPairStep = 6,
    detachedTripletStep = 7,
    mixedTripletStep = 8,
    pixelLessStep = 9,
    tobTecStep = 10,
    jetCoreRegionalStep = 11,
    conversionStep = 12,
    muonSeededStepInOut = 13,
    muonSeededStepOutIn = 14,
    outInEcalSeededConv = 15, inOutEcalSeededConv = 16,
    nuclInter = 17,
    standAloneMuon = 18, globalMuon = 19, cosmicStandAloneMuon = 20, cosmicGlobalMuon = 21,
    # Phase1
    highPtTripletStep = 22, lowPtQuadStep = 23, detachedQuadStep = 24,
    reservedForUpgrades1 = 25, reservedForUpgrades2 = 26,
    bTagGhostTracks = 27,
    beamhalo = 28,
    gsf = 29,
    # HLT algo name
    hltPixel = 30,
    # steps used by PF
    hltIter0 = 31,
    hltIter1 = 32,
    hltIter2 = 33,
    hltIter3 = 34,
    hltIter4 = 35,
    # steps used by all other objects @HLT
    hltIterX = 36,
    # steps used by HI muon regional iterative tracking
    hiRegitMuInitialStep = 37,
    hiRegitMuLowPtTripletStep = 38,
    hiRegitMuPixelPairStep = 39,
    hiRegitMuDetachedTripletStep = 40,
    hiRegitMuMixedTripletStep = 41,
    hiRegitMuPixelLessStep = 42,
    hiRegitMuTobTecStep = 43,
    hiRegitMuMuonSeededStepInOut = 44,
    hiRegitMuMuonSeededStepOutIn = 45,
    algoSize = 46
)

# Needs to kept consistent with
# DataFormats/TrackReco/interface/TrajectoryStopReasons.h
StopReason = _Enum(
  UNINITIALIZED = 0,
  MAX_HITS = 1,
  MAX_LOST_HITS = 2,
  MAX_CONSECUTIVE_LOST_HITS = 3,
  LOST_HIT_FRACTION = 4,
  MIN_PT = 5,
  CHARGE_SIGNIFICANCE = 6,
  LOOPER = 7,
  MAX_CCC_LOST_HITS = 8,
  NO_SEGMENTS_FOR_VALID_LAYERS = 9,
  SEED_EXTENSION = 10,
  SIZE = 12,
  NOT_STOPPED = 255
)

# From DataFormats/DetId/interface/DetId.h
class DetId(object):
    def __init__(self, *args, **kwargs):
        super(DetId, self).__init__()
        if len(args) == 1 and len(kwargs) == 0:
            self.detid = args[0]
        else:
            self.detid = kwargs["detid"]
        self.det = (self.detid >> 28) & 0xF
        self.subdet = (self.detid >> 25) & 0x7
# From Geometry/TrackerNumberingBuilder/README.md
class _Side(DetId):
    SideMinus = 1
    SidePlus = 2
    def __init__(self, *args, **kwargs):
        super(_Side, self).__init__(*args, **kwargs)
class _ModuleType(DetId):
    TypePair = 0
    TypeStereo = 1
    TypeRPhi = 2
    def __init__(self, *args, **kwargs):
        super(_ModuleType, self).__init__(*args, **kwargs)
        self.moduleType = self.detid & 0x3
class BPixDetIdPhase1(DetId):
    def __init__(self, detid):
        super(BPixDetIdPhase1, self).__init__(detid=detid)
        self.layer = (detid >> 20) & 0xF
        self.ladder = (detid >> 12) & 0xFF
        self.module = (detid >> 2) & 0x3FF
    def __str__(self):
        return "layer %d ladder %d module %d" % (self.layer, self.ladder, self.module)
class FPixDetIdPhase1(_Side, DetId):
    PanelForward = 1
    PanelBackward = 2
    def __init__(self, detid):
        super(FPixDetIdPhase1, self).__init__(detid=detid)
        self.side = (detid >> 23) & 0x3
        self.disk = (detid >> 18) & 0xF
        self.blade = (detid >> 12) & 0x3F
        self.panel = (detid >> 10) & 0x3
        self.module = (detid >> 2) & 0xFF
    def __str__(self):
        return "side %d disk %d blade %d panel %d" % (self.side, self.disk, self.blade, self.panel)
class TIBDetId(_Side, _ModuleType, DetId):
    OrderInternal = 1
    OrderExternal = 2
    def __init__(self, detid):
        super(TIBDetId, self).__init__(detid=detid)
        self.layer = (detid >> 14) & 0x7
        self.side = (detid >> 12) & 0x3
        self.order = (detid >> 10) & 0x3
        self.string = (detid >> 4) & 0x3F
        self.module = (detid >> 2) & 0x3
    def __str__(self):
        return "layer %d order %d string %d module %d" % (self.layer, self.order, self.string, self.module)
class TIDDetId(_Side, _ModuleType, DetId):
    OrderBack = 1
    OrderFront = 1
    def __init__(self, detid):
        super(TIDDetId, self).__init__(detid=detid)
        self.side = (detid >> 13) & 0x3
        self.disk = (detid >> 11) & 0x3
        self.ring = (detid >> 9) & 0x3
        self.order = (detid >> 7) & 0x3
        self.module = (detid >> 2) & 0x1F
    def __str__(self):
        return "side %d disk %d ring %d order %d module %d" % (self.side, self.disk, self.ring, self.order, self.module)
class TOBDetId(_Side, _ModuleType, DetId):
    def __init__(self, detid):
        super(TOBDetId, self).__init__(detid=detid)
        self.layer = (detid >> 14) & 0x7
        self.side = (detid >> 12) & 0x3
        self.rod = (detid >> 5) & 0x7F
        self.module = (detid >> 2) & 0x7
    def __str__(self):
        return "layer %d rod %d module %d" % (self.layer, self.rod, self.module)
class TECDetId(_Side, _ModuleType, DetId):
    OrderBack = 1
    OrderFront = 1
    def __init__(self, detid):
        super(TECDetId, self).__init__(detid=detid)
        self.side = (detid >> 18) & 0x3
        self.wheel = (detid >> 14) & 0xF
        self.order = (detid >> 12) & 0x3
        self.petal = (detid >> 8) & 0xF
        self.ring = (detid >> 5) & 0x7
        self.module = (detid >> 2) & 0x7
    def __str__(self):
        return "side %d wheel %d order %d petal %d ring %d module %d" % (self.side, self.wheel, self.order, self.petal, self.ring, self.module)

def parseDetId(detid):
    subdet = DetId(detid).subdet
    if detector.get() == Detector.Phase1:
        if subdet == SubDet.BPix: return BPixDetIdPhase1(detid)
        if subdet == SubDet.FPix: return FPixDetIdPhase1(detid)
        if subdet == SubDet.TIB: return TIBDetId(detid)
        if subdet == SubDet.TID: return TIDDetId(detid)
        if subdet == SubDet.TOB: return TOBDetId(detid)
        if subdet == SubDet.TEC: return TECDetId(detid)
        raise Exception("Got unknown subdet %d" % subdet)
    raise Exception("Supporting only phase1 DetIds at the moment")

# Common diff helpers, used in the printout helpers
def _difflist(list1, list2):
    diff = difflib.unified_diff(list1, list2, lineterm="", n=len(list1))
    for item in diff:
        if item[:2] == "@@":
            break
    return list(diff)

def _makediff(funcs, obj1, obj2):
    different = False
    ret = []
    for f in funcs:
        out1 = f(obj1)
        out2 = f(obj2)
        dff = _difflist(out1, out2)
        if len(dff) == 0:
            ret.extend([" "+s for s in out1])
        else:
            ret.extend(dff)
            different = True
    if different:
        return ret
    else:
        return []


# Common detailed printout helpers
class _RecHitPrinter(object):
    def __init__(self, indent=0):
        self._prefix = " "*indent
        self._backup = []

    def _indent(self, num):
        if num > 0:
            self._prefix += " "*num
        elif num < 0:
            self._prefix = self._prefix[:num]

    def indent(self, num):
        self._backup.append(self._prefix)
        self._indent(num)

    def setIndentFrom(self, printer, adjust=0):
        self._backup.append(self._prefix)
        self._prefix = printer._prefix
        self._indent(adjust)

    def restoreIndent(self):
        self._prefix = self._backup.pop()

    def _printHits(self, hits):
        lst = []
        for hit in hits:
            matched = ""
            coord = ""
            if hit.isValidHit():
                if isinstance(hit, _SimHitAdaptor):
                    matched = "matched to TP:SimHit " + ",".join(["%d:%d"%(sh.trackingParticle().index(), sh.index()) for sh in hit.simHits()])
                coord = "x,y,z %f,%f,%f" % (hit.x(), hit.y(), hit.z())
            detId = parseDetId(hit.detId())
            lst.append(self._prefix+"%s %d detid %d %s %s %s" % (hit.layerStr(), hit.index(), detId.detid, str(detId), coord, matched))
        return lst

class SeedPrinter(_RecHitPrinter):
    def __init__(self, *args, **kwargs):
        super(SeedPrinter, self).__init__(*args, **kwargs)

    def printSeed(self, seed):
        lst = []
        track = seed.track()
        madeTrack = "did not make a track"
        if track.isValid():
            madeTrack = "made track %d" % track.index()

        lst.append(self._prefix+"Seed %d algo %s %s" % (seed.indexWithinAlgo(), Algo.toString(seed.algo()), madeTrack))
        lst.append(self._prefix+" starting state: pT %f local pos x,y %f,%f mom x,y,z %f,%f,%f" % (seed.statePt(), seed.stateTrajX(), seed.stateTrajY(), seed.stateTrajPx(), seed.stateTrajPy(), seed.stateTrajPz()))
        lst.append(self._prefix+" hits")
        self.indent(2)
        lst.extend(self._printHits(seed.hits()))
        self.restoreIndent()
        return lst

    def __call__(self, seed, out=sys.stdout):
        if isinstance(out, list):
            lst = out
        else:
            lst = []

        lst.extend(self.printSeed(seed))

        if not isinstance(out, list):
            for line in lst:
                out.write(line),
                out.write("\n")

class TrackPrinter(_RecHitPrinter):
    def __init__(self, indent=0, hits=True, seedPrinter=SeedPrinter(), trackingParticles=True, trackingParticlePrinter=None):
        super(TrackPrinter, self).__init__(indent)
        self._hits = hits
        self._seedPrinter = seedPrinter
        self._trackingParticles = trackingParticles
        self._trackingParticlePrinter = trackingParticlePrinter

    def printHeader(self, track):
        lst = []
        lst.append(self._prefix+"Track %d pT %f eta %f phi %f dxy %f err %f dz %f err %f" % (track.index(), track.pt(), track.eta(), track.phi(), track.dxy(), track.dxyErr(), track.dz(), track.dzErr()))
        algo = track.algo()
        oriAlgo = track.originalAlgo()
        lst.append(self._prefix+" pixel hits %d strip hits %d" % (track.nPixel(), track.nStrip()))
        lst.append(self._prefix+" HP %s algo %s originalAlgo %s stopReason %s" % (str(track.isHP()), Algo.toString(track.algo()), Algo.toString(track.originalAlgo()), StopReason.toString(track.stopReason())))
        return lst

    def printHits(self, track):
        lst = []
        lst.append(self._prefix+" hits")
        self.indent(2)
        lst.extend(self._printHits(track.hits()))
        self.restoreIndent()
        return lst

    def printSeed(self, track):
        self._seedPrinter.setIndentFrom(self, adjust=1)
        lst = self._seedPrinter.printSeed(track.seed())
        self._seedPrinter.restoreIndent()
        return lst

    def printTrackingParticles(self, track):
        lst = []
        if track.nMatchedTrackingParticles == 0:
            lst.append(self._prefix+" not matched to any TP")
        elif self._trackingParticlePrinter is None:
            lst.append(self._prefix+" matched to TPs "+",".join([str(tpInfo.trackingParticle().index()) for tpInfo in track.matchedTrackingParticleInfos()]))
        else:
            lst.append(self._prefix+" matched to TPs")
            self._trackingParticlePrinter.indent(2)
            for tpInfo in track.matchedTrackingParticleInfos():
                lst.extend(self._trackingParticlePrinter.printTrackingParticle(tpInfo.trackingParticle()))
            self._trackingParticlePrinter.indent(-2)
        return lst

    def printTrack(self, track):
        lst = []
        lst.extend(self.printHeader(track))

        if self._hits:
            lst.extend(self.printHits(track))

        if self._seedPrinter:
            lst.extend(self.printSeed(track))

        return lst

    def printMatchedTrackingParticles(self, track):
        lst = []
        if self._trackingParticles:
            lst.extend(self.printTrackingParticles(track))
        return lst

    def __call__(self, track, out=sys.stdout):
        if isinstance(out, list):
            lst = out
        else:
            lst = []

        lst.extend(self.printTrack(track))
        lst.extend(self.printTrackingParticles(track))

        if not isinstance(out, list):
            for line in lst:
                out.write(line)
                out.write("\n")

    def diff(self, track1, track2):
        generators = [
            self.printHeader
        ]
        if self._hits:
            generators.append(self.printHits)
        if self._seedPrinter:
            generators.append(self.printSeed)

        return _makediff(generators, track1, track2)

class TrackingParticlePrinter:
    def __init__(self, indent=0, parentage=True, hits=True, tracks=True, trackPrinter=None, seedPrinter=SeedPrinter()):
        self._prefix = " "*indent
        self._parentage = parentage
        self._hits = hits
        self._tracks = tracks
        self._trackPrinter = trackPrinter
        self._seedPrinter = seedPrinter

    def indent(self, num):
        if num > 0:
            self._prefix += " "*num
        elif num < 0:
            self._prefix = self._prefix[:num]

    def _printTP(self, tp):
        genIds = ""
        if len(tp.genPdgIds()) > 0:
            genIds = " genPdgIds "+",".join([str(pdgId) for pdgId in tp.genPdgIds()])
        fromB = ""
        if tp.isFromBHadron():
            fromB = " from B hadron"
        return [self._prefix+"TP %d pdgId %d%s%s pT %f eta %f phi %f" % (tp.index(), tp.pdgId(), genIds, fromB, tp.pt(), tp.eta(), tp.phi())]
        

    def _parentageChain(self, tp):
        lst = []
        prodVtx = tp.parentVertex()
        if prodVtx.nSourceTrackingParticles() == 1:
            lst.extend(self._printTP(next(prodVtx.sourceTrackingParticles())))
        elif prodVtx.nSourceTrackingParticles() >= 2:
            self.indent(1)
            for tp in prodVtx.sourceTrackingParticles():
                self._printTP(tp, out)
                self.indent(1)
                lst.extend(self._parentageChain(tp))
                self.indent(-1)
            self.indent(-1)
        return lst

    def printTrackingParticle(self, tp):
        lst = []
        lst.extend(self._printTP(tp))
        if self._parentage:
            if tp.parentVertex().nSourceTrackingParticles() > 0:
                lst.append(self._prefix+" parentage chain")
                self.indent(2)
                lst.extend(self._parentageChain(tp))
                self.indent(-2)
        return lst

    def printHits(self, tp):
        lst = []
        if self._hits:
            lst.append(self._prefix+" sim hits")
            for simhit in tp.simHits():
                detId = parseDetId(simhit.detId())
                lst.append(self._prefix+"  %d: pdgId %d process %d %s detId %d %s x,y,z %f,%f,%f" % (simhit.index(), simhit.particle(), simhit.process(), simhit.layerStr(), detId.detid, str(detId), simhit.x(), simhit.y(), simhit.z()))
        return lst

    def printMatchedTracks(self, tp):
        lst = []
        if tp.nMatchedTracks() == 0:
            lst.append(self._prefix+" not matched to any track")
        elif self._trackPrinter is None:
            lst.append(self._prefix+" matched to tracks"+",".join([str(trkInfo.track().index()) for trkInfo in tp.matchedTrackInfos()]))
        else:
            lst.append(self._prefix+" matched to tracks")
            self._trackPrinter.indent(2)
            for trkInfo in tp.matchedTrackInfos():
                lst.extend(self._trackPrinter.printTrack(trkInfo.track()))
            self._trackPrinter.restoreIndent()
        return lst

    def printMatchedSeeds(self, tp):
        lst = []
        if self._seedPrinter:
            if tp.nMatchedSeeds() == 0:
                lst.append(self._prefix+ " not matched to any seed")
            else:
                lst.append(self._prefix+" matched to seeds")
                self._seedPrinter.setIndentFrom(self, adjust=2)
                for seedInfo in tp.matchedSeedInfos():
                    self._seedPrinter(seedInfo.seed(), lst)
                self._seedPrinter.restoreIndent()
        return lst

    def __call__(self, tp, out=sys.stdout):
        if isinstance(out, list):
            lst = out
        else:
            lst = []

        lst.extend(self.printTrackingParticle(tp))
        lst.extend(self.printHits(tp))
        lst.extend(self.printMatchedTracks(tp))
        lst.extend(self.printMatchedSeeds(tp))

        for line in lst:
            out.write(line)
            out.write("\n")

    def diff(self, tp1, tp2):
        generators = [
            self.printTrackingParticle,
            self.printHits,
            self.printMatchedTracks,
            self.printMatchedSeeds
        ]
        return _makediff(generators, tp1, tp2)

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

        Generator returns PixelHit/StripHit/GluedHit/Phase2OT depending on the
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
            elif hitType == 4:
                yield Phase2OTHit(self._tree, ihit)
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

    def phase2OTHits(self):
        """Returns generator for phase2 outer tracker hits."""
        self._checkIsValid()
        for ihit, hitType in self._hits():
            if hitType != 4:
                continue
            yield Phase2OTHit(self._tree, ihit)

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
        subdet = getattr(self._tree, self._prefix+"_det")[self._index]
        side = ""
        if subdet in [SubDet.FPix, SubDet.TID, SubDet.TEC]:
            detid = parseDetId(getattr(self._tree, self._prefix+"_detId")[self._index])
            if detid.side == 1:
                side = "-"
            elif detid.side == 2:
                side = "+"
            else:
                side = "?"
        return "%s%d%s" % (SubDet.toString(subdet),
                           getattr(self._tree, self._prefix+"_lay")[self._index],
                           side)

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

    Note that to iteratate over the evets with zip(), you should use
    itertools.izip() instead.
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
        """Returns event number."""
        return self._tree.event

    def lumi(self):
        """Returns lumisection number."""
        return self._tree.lumi

    def run(self):
        """Returns run number."""
        return self._tree.run

    def eventId(self):
        """Returns (run, lumi, event) tuple."""
        return (self._tree.run, self._tree.lumi, self._tree.event)

    def eventIdStr(self):
        """Returns 'run:lumi:event' string."""
        return "%d:%d:%d" % self.eventId()

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

    def phase2OTHits(self):
        """Returns Phase2OTHits object."""
        return Phase2OTHits(self._tree)

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

class SeedMatchInfo(_Object):
    """Class representing a match to a Seed.

    The point of this class is to provide an interface compatible with
    all other "MatchInfo" classes

    """
    def __init__(self, tree, index, seedindex, prefix):
        """Constructor.

        Arguments:
        tree     -- TTree object
        index    -- Index of the object (TrackingParticle) matched to seed
        seedindex -- Index of the seed match (second index in _trkIdx branch)
        prefix   -- String for prefix of the object (TrackingParticle) matched to seed
        """
        super(SeedMatchInfo, self).__init__(tree, index, prefix)
        self._seedindex = seedindex

    def seed(self):
        """Returns matched Seed."""
        self._checkIsValid()
        return Seed(self._tree, getattr(self._tree, self._prefix+"_seedIdx")[self._index][self._seedindex])

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
class GluedHit(_Object, _LayerStrAdaptor):
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
class Phase2OTHit(_HitObject, _LayerStrAdaptor, _SimHitAdaptor):
    """Class representing a phase2 OT hit."""
    def __init__(self, tree, index):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the hit
        """
        super(Phase2OTHit, self).__init__(tree, index, "ph2")

    def isValidHit(self):
        return True

class Phase2OTHits(_Collection):
    """Class presenting a collection of phase2 OT hits."""
    def __init__(self, tree):
        """Constructor.

        Arguments:
        tree -- TTree object
        """
        super(Phase2OTHits, self).__init__(tree, "ph2_isBarrel", Phase2OTHit)

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

    def _nMatchedSeeds(self):
        """Internal function to get the number of matched seeds."""
        return self._tree.sim_seedIdx[self._index].size()

    def nMatchedSeeds(self):
        """Returns the number of matched seeds."""
        self._checkIsValid()
        return self._nMatchedSeeds()

    def matchedSeedInfos(self):
        """Returns a generator for matched tracks.

        The generator returns SeedMatchInfo objects.
        """
        self._checkIsValid()
        for imatch in xrange(self._nMatchedSeeds()):
            yield SeedMatchInfo(self._tree, self._index, imatch, self._prefix)

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

    def isLooper(self):
        """Returns True if this TrackingParticle is a looper.

        Note that the check involves looping over the SimHits, so it is not too cheap."""
        self._checkIsValid()
        prevr = 0
        for ihit in self.simHitIdx():
            hit = SimHit(self._tree, ihit)
            r = hit.x()**2 + hit.y()**2
            if r < prevr:
                return True
            prevr = r
        return False


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
