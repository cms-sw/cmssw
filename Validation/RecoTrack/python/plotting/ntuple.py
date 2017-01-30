import ROOT

import re
import sys
import math
import difflib
import collections

class Detector:
#    class Phase0: pass # not supported yet
    class Phase1: pass
    class Phase2: pass

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

# to be kept is synch with enum HitSimType in TrackingNtuple.py
HitSimType = _Enum(
    Signal = 0,
    ITPileup = 1,
    OOTPileup = 2,
    Noise = 3,
    Unknown = 99
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
class TIDDetIdPhase2(_Side, DetId):
    PanelForward = 1
    PanelBackward = 1
    def __init__(self, detid):
        super(TIDDetIdPhase2, self).__init__(detid=detid)
        self.side = (detid >> 23) & 0x3
        self.disk = (detid >> 18) & 0xF
        self.ring = (detid >> 12) & 0x3F
        self.panel = (detid >> 10) & 0x3
        self.module = (detid >> 2) & 0xFF
    def __str__(self):
        return "side %d disk %d ring %d panel %d" % (self.side, self.disk, self.ring, self.panel)
class TOBDetIdPhase2(DetId):
    def __init__(self, detid):
        super(TOBDetIdPhase2, self).__init__(detid=detid)
        self.layer = (detid >> 20) & 0xF
        self.side = (detid >> 18) & 0x3
        self.ladder = (detid >> 10) & 0xFF
        self.module = (detid >> 2) & 0x3FF
    def __str__(self):
        return "layer %d side %d ladder %d module %d" % (self.layer, self.side, self.ladder, self.module)

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
    elif detector.get() == Detector.Phase2:
        if subdet == SubDet.BPix: return BPixDetIdPhase1(detid)
        if subdet == SubDet.FPix: return FPixDetIdPhase1(detid)
        if subdet == SubDet.TIB: raise Exception("TIB not included in subDets for Phase2")
        if subdet == SubDet.TID: return TIDDetIdPhase2(detid)
        if subdet == SubDet.TOB: return TOBDetIdPhase2(detid)
        if subdet == SubDet.TEC: raise Exception("TEC not included in subDets for Phase2")
        raise Exception("Got unknown subdet %d" % subdet)
    raise Exception("Supporting only phase1 and phase2 DetIds at the moment")

# Common track-track matching by hits (=clusters)
def _commonHits(trk1, trk2):
    """Returns the number of common hits in trk1 and trk2. Matching is
    done via the hit type and index, so effectively the matching is
    done by clusters. Invalid hits are ignored.

    """
    hits1 = set()
    for hit in trk1.hits():
        if not hit.isValidHit(): continue
        hits1.add( (type(hit), hit.index()) )

    ncommon = 0
    for hit in trk2.hits():
        if not hit.isValidHit(): continue
        if (type(hit), hit.index()) in hits1:
            ncommon += 1

    return ncommon

def _matchTracksByHits(reftrk, trklist):
    if len(trklist) == 0:
        return None

    hits1 = set()
    for hit in reftrk.hits():
        if not hit.isValidHit(): continue
        hits1.add( (type(hit), hit.index()) )

    best = (None, 0)
    for trk in trklist:
        ncommon = 0
        for hit in trk.hits():
            if not hit.isValidHit(): continue
            if (type(hit), hit.index()) in hits1:
                ncommon += 1
        if ncommon > best[1]:
            best = (trk, ncommon)

    return best

def _matchTracksByTrackingParticle(reftrk, trklist, othertrklist, othertplist):
    """
    reftrk and trklist must come from same ntuple
    othertrklist and othertplist must come from the same ntuple, can be different from above
    """
    ref_viatp = []
    other_viatp = []
    if reftrk.nMatchedTrackingParticles() == 0:
        return (ref_viatp, other_viatep)

    for tpInfo1 in reftrk.matchedTrackingParticleInfos():
        tp1 = tpInfo1.trackingParticle()

        for trkInfo1 in tp1.matchedTrackInfos():
            # Is there any way to avoid the linear search?
            # I tried to create a map from track.index()
            # to index in trks1/2, but that fails because
            # I remove items in both
            # Hmm, maybe if I add boolean lists to mark if a trk1/2 is already used?
            t1Index = trkInfo1.track().index()
            if t1Index != reftrk.index():
                t1 = next(t for t in trklist if t.index() == t1Index)
                ref_viatp.append(t1)

        tp2 = othertplist[tp1.index()]
        for trkInfo2 in tp2.matchedTrackInfos():
            t2Index = trkInfo2.track().index()
            try:
                t2 = next(t for t in othertrklist if t.index() == t2Index)
                other_viatp.append(t2)
            except StopIteration:
                # I thought first that in case the TP is matched to
                # track2 NOT in othertrklist, I want to print that
                # anyway because it is interesting to note that that
                # track1 is reconstructed/something in ntuple2, but
                # does not enter in othertrklist. But now I think the
                # opposite.
                pass
                #other_viatp_notintrks2.append(trkInfo2.track())

    return (ref_viatp, other_viatp)



# Common diff helpers, used in the printout helpers
class _DiffResult(object):
    def __init__(self, diff=[], hasDifference=False):
        self._diff = []
        self._hasDifference = hasDifference
        self.extend(diff)

    def setDifference(self, diff=True):
        self._hasDifference = diff

    def hasDifference(self):
        return self._hasDifference

    def extend(self, diff):
        if isinstance(diff, _DiffResult):
            self._diff.append(diff)
            if diff.hasDifference():
                self.setDifference()
        else:
            self._diff.extend(diff)

    def lines(self):
        for line in self._diff:
            if isinstance(line, _DiffResult):
                for l in line.lines():
                    yield l
            else:
                yield line

    def __str__(self):
        return "\n".join(filter(lambda s: s != "", (str(item) for item in self._diff)))

    def __len__(self):
        return len(self._diff)

def _difflist(list1, list2):
    diff = difflib.unified_diff(list1, list2, lineterm="", n=len(list1))
    for item in diff:
        if item[:2] == "@@":
            break
    return list(diff)

def _makediff(list1, list2, equalPrefix=" "):
    diff = _difflist(list1, list2)
    if len(diff) == 0:
        return _DiffResult([equalPrefix+s for s in list1], hasDifference=False)
    else:
        return _DiffResult(diff, hasDifference=True)

def _mapdiff(func, obj1, obj2):
    lst1 = func(obj1) if obj1 is not None else []
    lst2 = func(obj2) if obj2 is not None else []
    return _DiffResult(_makediff(lst1, lst2))

def diffTrackListsFromSameTrackingParticle(trackPrinter, lst1, lst2, diffByHitsOnly=False):
    diff = _DiffResult()
    trks1 = list(lst1)
    trks2 = list(lst2) # make copy because it is modified

    trks1Empty = (len(trks1) == 0)
    trks2Empty = (len(trks2) == 0)

    if trks1Empty and trks2Empty:
        return diff

    # make sure all tracks really come from a single TP
    # just to simplify the work loop, generalization can be considered later
    commonTP = None
    def _findCommonTP(_lst, _commonTP, _name):
        for trk in _lst:
            if trk.nMatchedTrackingParticles() != 1:
                raise Exception("Track %d from %s is matched to %d TPs. This is not supported by this function yet." % (trk.index(), _name, trk.nMatchedTrackingParticles()))
            if _commonTP is None:
                _commonTP = next(trk.matchedTrackingParticleInfos()).trackingParticle()
            else:
                tp = next(trk.matchedTrackingParticleInfos()).trackingParticle()
                if tp.index() != _commonTP.index():
                    raise Exception("Track %d from %s is matched to TP %d, which differs from the TP %d of already processed tracks." % (trk.index(), _name, _commonTP.index(), tp.index()))
        return _commonTP
    commonTP = _findCommonTP(trks1, commonTP, "lst1")
    commonTP = _findCommonTP(trks2, commonTP, "lst2")

    # Need some tracks from trks1 and trks2 to print the TrackingParticle information
    someTrk1 = trks1[0] if not trks1Empty else None
    someTrk2 = trks2[0] if not trks2Empty else None

    for trk1 in trks1:
        (matchedTrk2, ncommon) = _matchTracksByHits(trk1, trks2)

        # no more tracks in tp2
        if matchedTrk2 is None:
            diff.extend(_makediff(trackPrinter.printTrack(trk1), []))
        else: # diff trk1 to best-matching track from trks2
            someTrk2 = matchedTrk2
            trks2.remove(matchedTrk2)
            tmp = trackPrinter.diff(trk1, matchedTrk2, diffTrackingParticles=False)
            if diffByHitsOnly and ncommon == trk1.nValid() and ncommon == matchedTrk2.nValid():
                tmp.setDifference(False)
            diff.extend(tmp)

    for trk2 in trks2: # remaining tracks in trks2
        diff.extend(_makediff([], trackPrinter.printTrack(trk2)))

    # finally add information of the trackingParticle
    # easiest is to pass a track matched to the TP
    tmp = _mapdiff(trackPrinter.printTrackingParticles, someTrk1, someTrk2)
    tmp.setDifference(False) # we know the TP is the same, even if the "track match" information will differ
    diff.extend(tmp)

    return diff

def diffTrackListsGeneric(trackPrinter, lst1, lst2):
    diff = _DiffResult()
    trks1 = list(lst1)
    trks2 = list(lst2) # make copy because it is modified

    # sort in eta
    trks1.sort(key=lambda t: t.eta())
    trks2.sort(key=lambda t: t.eta())

    # Bit of a hack...
    tps1 = None
    tps2 = None
    if len(trks1) > 0:
        tps1 = TrackingParticles(trks1[0]._tree)
    if len(trks2) > 0:
        tps2 = TrackingParticles(trks2[0]._tree)

    while len(trks1) > 0:
        trk1 = trks1.pop(0)

        #print trk1.index(), [t.index() for t in trks2]

        # first try via TrackingParticles
        if trk1.nMatchedTrackingParticles() > 0 and tps2:
            (trks1_viatp, trks2_viatp) = _matchTracksByTrackingParticle(trk1, trks1, trks2, tps2)
            if len(trks2_viatp) > 0:
                for t1 in trks1_viatp:
                    trks1.remove(t1)
                for t2 in trks2_viatp:
                    trks2.remove(t2)

                #if debug:
                #    print trk1.index()
                #    print [t.index() for t in trks1_viatp]
                #    print [t.index() for t in trks2_viatp]
                #    print [t.index() for t in trks2_viatp_notintrks2]

                tmp = diffTrackListsFromSameTrackingParticle(trackPrinter, [trk1]+trks1_viatp, trks2_viatp, diffByHitsOnly=True)

                #if debug:
                #    print str(tmp)

                #print "##########"
                #s = str(tmp)
                #if len(s) > 0:
                #    print s
                #print "%%%%%"
                #print tmp._diff
                #print tmp.hasDifference()
                #print "----------"

                if tmp.hasDifference():
                    diff.extend(tmp)
                continue

        # If there is a trk2 having smaller eta than trk1, check first
        # if there is a matching track in trks1. If yes, do
        # nothing (as the pair will be printed later, in the order
        # of trk1s). If no, print the trk2 here so that tracks are
        # printed in eta order.
        if len(trks2) > 0:
            i = 0
            while i < len(trks2) and trks2[i].eta() < trk1.eta():
                # is there any way to speed these up?
                if trks2[i].nMatchedTrackingParticles() > 0 and tps1:
                    (trks2_viatp, trks1_viatp) = _matchTracksByTrackingParticle(trks2[i], trks2, trks1, tps1)
                    if len(trks1_viatp) > 0:
                        i += 1
                        continue

                (matchedTrk1, nc1) = _matchTracksByHits(trks2[i], [trk1]+trks1)
                if matchedTrk1 is not None:
                    i += 1
                    continue
                else:
                    diff.extend(_makediff([], trackPrinter.printTrackAndMatchedTrackingParticles(trks2[i])))
                    del trks2[i]

        # if no matching tracks in trk2 via TrackingParticles, then
        # proceed finding the best match via hits
        (matchedTrk2, ncommon) = _matchTracksByHits(trk1, trks2)

        # no match, more tracks in tp2, or too few common hits
        if matchedTrk2 is None or ncommon < 3:
            diff.extend(_makediff(trackPrinter.printTrackAndMatchedTrackingParticles(trk1), []))
        else: # diff trk1 to best-matching track from trks2
            trks2.remove(matchedTrk2)
            if ncommon == trk1.nValid() and ncommon == matchedTrk2.nValid():
                continue

            diff.extend(trackPrinter.diff(trk1, matchedTrk2))

    for trk2 in trks2: # remaining tracks in trks2
        diff.extend(_makediff([], trackPrinter.printTrackAndMatchedTrackingParticles(trk2)))

    return diff


def _formatHitDiffForTwiki(diffHits, prefix):
    #line_re = re.compile("(?P<sign>[\+- ])\s+(?P<det>[a-zA-Z]+)(?P<lay>\d+)")
    #line_re = re.compile("(?P<sign>[ \-+])\s+(?P<det>[a-zA-Z]+)(?P<lay>\d+)\D*?(?P<missing>inactive)?")
    line_re = re.compile("(?P<sign>[ \-+])\s+(?P<det>[a-zA-Z]+)(?P<lay>\d+)\D*?(\((?P<missing>missing|inactive)\))?\s+\d+")

    summary = []
    prevdet = ""
    prevsign = " "
    diffLines = diffHits.lines()

    # skip header
    for line in diffLines:
        if "hits" in line:
            break

    for line in diffLines:
        m = line_re.search(line)
        if not m:
            break
            raise Exception("regex not found from line %s" % line.rstrip())
        sign = m.group("sign")
        det = m.group("det")
        lay = m.group("lay")

        if det != prevdet:
            if prevsign != " ":
                summary.append("%ENDCOLOR%")
                prevsign = " "
            summary.extend([" ", det])
            prevdet = det

        if sign != prevsign:
            if prevsign != " ":
                summary.append("%ENDCOLOR%")
            if sign == "-":
                summary.append("%RED%")
            elif sign == "+":
                summary.append("%GREEN%")
            prevsign = sign

        #print sign, det, lay
        #if len(summary) > 0:
        #    print " ", summary[-1]

        #if det != prevdet:
        #    if prevsign != " ":
        #        #if len(summary) > 0:
        #        #    if 
        #        summary.append("%ENDCOLOR")
        #    summary.extend([" ", det])
        #    if prevsign == "-":
        #        summary.append("%RED%")
        #    elif prevsign == "+":
        #        summary.append("%GREEN%")
        #    prevdet = det
        summary.append(lay)
        if m.group("missing"):
            if m.group("missing") == "missing":
                summary.append("(m)")
            elif m.group("missing") == "inactive":
                summary.append("(i)")

    if prevsign != " ":
        summary.append("%ENDCOLOR%")
    # prune "changed" missing/inactive hits
    i = 2
    while i < len(summary)-5:
        if summary[i] == "(i)" or summary[i] == "(m)":
            if summary[i-2] == "%RED%" and summary[i+1] == "%ENDCOLOR%" and summary[i+2] == "%GREEN%" and summary[i+3] == summary[i-1] and summary[i+4] == summary[i] and summary[i+5] == "%ENDCOLOR%":
                summary[i-2:i+6] = [summary[i-1], summary[i]]
        i += 1

    line = " "+"".join(summary)
    return ["?"+prefix+line]

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
                    #matched = "matched to TP:SimHit " + ",".join(["%d:%d"%(sh.trackingParticle().index(), sh.index()) for sh in hit.simHits()])
                    matched = "from %s " % HitSimType.toString(hit.simType())
                    matches = ["%d:%d"%(sh.trackingParticle().index(), sh.index()) for sh in hit.simHits()]
                    if len(matches) == 0:
                        matched += "not matched to any TP/SimHit"
                    else:
                        matched += "matched to TP:SimHit "+",".join(matches)

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
    def __init__(self, indent=0, hits=True, seedPrinter=SeedPrinter(), trackingParticles=True, trackingParticlePrinter=None, bestMatchingTrackingParticle=True, diffForTwiki=False):
        super(TrackPrinter, self).__init__(indent)
        self._hits = hits
        self._seedPrinter = seedPrinter
        self._trackingParticles = trackingParticles
        self._trackingParticlePrinter = trackingParticlePrinter
        self._bestMatchingTrackingParticle = bestMatchingTrackingParticle
        self._diffForTwiki = diffForTwiki

    def printHeader(self, track):
        lst = []
        lst.append(self._prefix+"Track %d pT %f eta %f phi %f dxy %f err %f dz %f err %f" % (track.index(), track.pt(), track.eta(), track.phi(), track.dxy(), track.dxyErr(), track.dz(), track.dzErr()))

        hp = "loose"
        if track.isHP():
            hp = "HP"

        algo = track.algo()
        oriAlgo = track.originalAlgo()
        algos = []
        algoMask = track.algoMask()
        for i in xrange(Algo.algoSize):
            if algoMask & 1:
                algos.append(Algo.toString(i))
            algoMask = algoMask >> 1
        algoMaskStr = ""
        if len(algos) >= 2:
            algoMaskStr = " algoMask "+",".join(algos)


        lst.append(self._prefix+" pixel hits %d strip hits %d chi2/ndof %f" % (track.nPixel(), track.nStrip(), track.nChi2()))
        lst.append(self._prefix+" is %s algo %s originalAlgo %s%s stopReason %s" % (hp, Algo.toString(track.algo()), Algo.toString(track.originalAlgo()), algoMaskStr, StopReason.toString(track.stopReason())))
        lst.append(self._prefix+" px %f py %f pz %f p %f" % (track.px(), track.py(), track.pz(), math.sqrt(track.px()**2+track.py()**2+track.pz()**2)))
        return lst

    def printHits(self, track):
        lst = []
        if self._hits:
            lst.append(self._prefix+" hits")
            self.indent(2)
            lst.extend(self._printHits(track.hits()))
            self.restoreIndent()
        return lst

    def printSeed(self, track):
        lst = []
        if self._seedPrinter:
            self._seedPrinter.setIndentFrom(self, adjust=1)
            lst.extend(self._seedPrinter.printSeed(track.seed()))
            self._seedPrinter.restoreIndent()
        return lst

    def diffSeeds(self, track1, track2):
        ret = _DiffResult()
        if self._seedPrinter:
            self._seedPrinter.setIndentFrom(self, adjust=1)
            diffSeed = _makediff(self._seedPrinter.printSeed(track1.seed()), self._seedPrinter.printSeed(track2.seed()))
            ret.extend(diffSeed)
            if self._diffForTwiki and diffSeed.hasDifference():
                ret.extend(_formatHitDiffForTwiki(diffSeed, self._prefix+" "))
            self._seedPrinter.restoreIndent()
        return ret

    def _printTrackingParticles(self, tps, header):
        lst = []
        if self._trackingParticlePrinter is None:
            lst.append(self._prefix+" "+header+" "+",".join([str(tp.index()) for tp in tps]))
        else:
            lst.append(self._prefix+" "+header)
            self._trackingParticlePrinter.indent(2)
            for tp in tps:
                lst.extend(self._trackingParticlePrinter.printTrackingParticle(tp))
                lst.extend(self._trackingParticlePrinter.printHits(tp))
                lst.extend(self._trackingParticlePrinter.printMatchedTracks(tp, useTrackPrinter=False))
            self._trackingParticlePrinter.indent(-2)
        return lst

    def printTrackingParticles(self, track):
        lst = []
        if track.nMatchedTrackingParticles() == 0:
            if self._bestMatchingTrackingParticle:
                bestTP = track.bestMatchingTrackingParticle()
                if bestTP is not None:
                    lst.extend(self._printTrackingParticles([bestTP], "not matched to any TP, but a following TP with >= 3 matched hits is found"))
                else:
                    lst.append(self._prefix+" not matched to any TP")
            else:
                lst.append(self._prefix+" not matched to any TP")
        else:
            lst.extend(self._printTrackingParticles([tpInfo.trackingParticle() for tpInfo in track.matchedTrackingParticleInfos()], "matched to TPs"))
        return lst

    def printTrack(self, track):
        lst = self.printHeader(track)
        lst.extend(self.printHits(track))
        lst.extend(self.printSeed(track))
        return lst


    def printMatchedTrackingParticles(self, track):
        lst = []
        if self._trackingParticles:
            lst.extend(self.printTrackingParticles(track))
        return lst

    def printTrackAndMatchedTrackingParticles(self, track):
        lst = []
        lst.extend(self.printTrack(track))
        lst.extend(self.printMatchedTrackingParticles(track))
        return lst

    def __call__(self, track, out=sys.stdout):
        if isinstance(out, list):
            lst = out
        else:
            lst = []

        lst.extend(self.printTrackAndMatchedTrackingParticles(track))

        if not isinstance(out, list):
            for line in lst:
                out.write(line)
                out.write("\n")

    def diff(self, track1, track2, diffTrackingParticles=True):
        if track1 is None:
            lst = self.printTrack(track2) + self.printTrackingParticles(track2)
            return _makediff([], lst)
        if track2 is None:
            lst = self.printTrack(track1) + self.printTrackingParticles(track1)
            return _makediff(lst, [])

        ret = _DiffResult()
        ret.extend(_mapdiff(self.printHeader, track1, track2))
        if self._diffForTwiki:
            lst = [
                self._prefix+" parameters",
                self._prefix+"  pt %RED%{pt1:.3g}%ENDCOLOR% %GREEN%{pt2:.3g}%ENDCOLOR%".format(pt1=track1.pt(), pt2=track2.pt()),
                self._prefix+"  eta %RED%{eta1:.3g}%ENDCOLOR% %GREEN%{eta2:.3g}%ENDCOLOR%".format(eta1=track1.eta(), eta2=track2.eta()),
                self._prefix+"  phi %RED%{phi1:.3g}%ENDCOLOR% %GREEN%{phi2:.3g}%ENDCOLOR%".format(phi1=track1.phi(), phi2=track2.phi()),
                self._prefix+"  dxy %RED%{dxy1:.3g}%ENDCOLOR% %GREEN%{dxy2:.3g}%ENDCOLOR% ({dxy1rel:.2f}*err1, {dxy2rel:.2f}*err2)".format(dxy1=track1.dxy(), dxy2=track2.dxy(), dxy1rel=(track2.dxy()-track1.dxy())/track1.dxyErr(), dxy2rel=(track2.dxy()-track1.dxy())/track2.dxyErr()),
                self._prefix+"  dz %RED%{dz1:.3g}%ENDCOLOR% %GREEN%{dz2:.3g}%ENDCOLOR% ({dz1rel:.2f}*err1, {dz2rel:.2f}*err2)".format(dz1=track1.dz(), dz2=track2.dz(), dz1rel=(track2.dz()-track1.dz())/track1.dzErr(), dz2rel=(track2.dz()-track1.dz())/track2.dzErr()),
                self._prefix+"  chi2/ndof %RED%{chi1:.3g}%ENDCOLOR% %GREEN%{chi2:.3g}%ENDCOLOR%".format(chi1=track1.nChi2(), chi2=track2.nChi2()),
            ]
            ret.extend(_makediff(lst, lst, equalPrefix="?"))

        diffHits = _mapdiff(self.printHits, track1, track2)
        ret.extend(diffHits)
        if self._hits and self._diffForTwiki:
            ret.extend(_formatHitDiffForTwiki(diffHits, self._prefix))

        ret.extend(self.diffSeeds(track1, track2))
        if diffTrackingParticles:
            ret.extend(_mapdiff(self.printTrackingParticles, track1, track2))
        return ret

class TrackingParticlePrinter:
    def __init__(self, indent=0, parentage=True, hits=True, tracks=True, trackPrinter=None, bestMatchingTrack=True, seedPrinter=SeedPrinter()):
        self._prefix = " "*indent
        self._parentage = parentage
        self._hits = hits
        self._tracks = tracks
        self._trackPrinter = trackPrinter
        self._bestMatchingTrack = bestMatchingTrack
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
        return [
            self._prefix+"TP %d pdgId %d%s%s ev:bx %d:%d pT %f eta %f phi %f" % (tp.index(), tp.pdgId(), genIds, fromB, tp.event(), tp.bunchCrossing(), tp.pt(), tp.eta(), tp.phi()),
            self._prefix+" pixel hits %d strip hits %d dxy %f dz %f" % (tp.nPixel(), tp.nStrip(), tp.pca_dxy(), tp.pca_dz())
        ]
        

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
                tmp = []
                for h in simhit.hits():
                    tmp.append(",".join([str(trk.index()) for trk in h.tracks()]) + ":%d"%h.index())
                if len(tmp) == 0:
                    matched = "not matched to any Track/RecHit"
                else:
                    matched = "matched to Tracks:RecHits "+";".join(tmp)

                lst.append(self._prefix+"  %s %d pdgId %d process %d detId %d %s x,y,z %f,%f,%f %s" % (simhit.layerStr(), simhit.index(), simhit.particle(), simhit.process(), detId.detid, str(detId), simhit.x(), simhit.y(), simhit.z(), matched))
        return lst

    def _printMatchedTracksHeader(self):
        return [self._prefix+" matched to tracks"]

    def _printMatchedTracks(self, tracks, header=None, useTrackPrinter=True):
        lst = []
        if header is not None:
            lst.append(self._prefix+" "+header)
        else:
            lst.extend(self._printMatchedTracksHeader())
        if self._trackPrinter is None or not useTrackPrinter:
            lst[-1] += " "+",".join([str(track.index()) for track in tracks])
        else:
            self._trackPrinter.indent(2)
            for track in tracks:
                lst.extend(self._trackPrinter.printTrack(track))
                self._trackPrinter.restoreIndent()
        return lst

    def printMatchedTracks(self, tp, useTrackPrinter=True):
        lst = []
        if tp.nMatchedTracks() == 0:
            header = "not matched to any track"
            lst.append(self._prefix+" "+header)
            if self._bestMatchingTrack:
                bestTrack = tp.bestMatchingTrack()
                if bestTrack is not None:
                    lst.pop()
                    lst.extend(self._printMatchedTracks([bestTrack], header+", but a following track with >= 3 matched hits is found", useTrackPrinter=useTrackPrinter))
        else:
            lst.extend(self._printMatchedTracks([trkInfo.track() for trkInfo in tp.matchedTrackInfos()], useTrackPrinter=useTrackPrinter))
        return lst

    def diffMatchedTracks(self, tp1, tp2):
        ntrk1 = tp1.nMatchedTracks()
        ntrk2 = tp2.nMatchedTracks()

        if ntrk1 == 0 or ntrk2 == 0 or self._trackPrinter is None:
            return _makediff(self.printMatchedTracks(tp1), self.printMatchedTracks(tp2))

        self._trackPrinter.indent(2)

        diff = _makediff(self._printMatchedTracksHeader(), self._printMatchedTracksHeader())
        trks1 = [trkInfo1.track() for trkInfo1 in tp1.matchedTrackInfos()]
        trks2 = [trkInfo2.track() for trkInfo2 in tp2.matchedTrackInfos()]
        #for trkInfo1 in tp1.matchedTrackInfos():
        #    trk1 = trkInfo1.track()
        #    matchedTrk2 = _matchTracksByHits(trk1, trks2)
        #
        #    if matchedTrk2 is None: # no more tracks in tp2
        #        diff.extend(_makediff(self._trackPrinter.printTrack(trk1), []))
        #    else: # diff trk1 to best-matching track from tp2
        #        trks2.remove(matchedTrk2)
        #        diff.extend(self._trackPrinter.diff(trk1, matchedTrk2))
        #
        #for trk2 in trks2: # remaining tracks in tp2
        #    diff.extend(_makediff([], self._trackPrinter.printTrack(trk2)))
        diff.extend(diffTrackListsFromSameTrackingParticle(self._trackPrinter, trks1, trks2))

        self._trackPrinter.restoreIndent()
        return diff

    def _printMatchedSeeds0(self):
        return [self._prefix+ " not matched to any seed"]

    def _printMatchedSeedsHeader(self):
        return [self._prefix+" matched to seeds"]

    def printMatchedSeeds(self, tp):
        lst = []
        if self._seedPrinter:
            if tp.nMatchedSeeds() == 0:
                lst.extend(self._printMatchedSeeds0())
            else:
                lst.extend(self._printMatchedSeedsHeader())
                self._seedPrinter.setIndentFrom(self, adjust=2)
                for seedInfo in tp.matchedSeedInfos():
                    lst.extend(self._seedPrinter.printSeed(seedInfo.seed()))
                self._seedPrinter.restoreIndent()
        return lst

    def diffMatchedSeeds(self, tp1, tp2):
        if not self._seedPrinter:
            return []

        nseed1 = tp1.nMatchedSeeds()
        nseed2 = tp2.nMatchedSeeds()
        if nseed1 == 0 or nseed2 == 0:
            return _makediff(self.printMatchedSeeds(tp1), self.printMatchedSeeds(tp2))

        self._seedPrinter.setIndentFrom(self, adjust=2)

        diff = _makediff(self._printMatchedSeedsHeader(), self._printMatchedSeedsHeader())
        seeds2 = [seedInfo2.seed() for seedInfo2 in tp2.matchedSeedInfos()]
        for seedInfo1 in tp1.matchedSeedInfos():
            seed1 = seedInfo1.seed()
            matchedSeed2 = _matchTracksByHits(seed1, seeds2)[0]

            if matchedSeed2 is None: # no more seeds in tp2
                diff.extend(_makediff(self._seedPrinter.printSeed(seed1), []))
            else: # diff seed1 to best-matching seed from tp2
                seeds2.remove(matchedSeed2)
                diff.extend(_makediff(self._seedPrinter.printSeed(seed1), self._seedPrinter.printSeed(matchedSeed2)))

        for seed2 in seeds2: # remiaining seeds in tp2
            diff.extend(_makediff([], self._seedPrinter.printSeed(seed2)))

        self._seedPrinter.restoreIndent()

        return diff

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
        ret = _DiffResult()
        ret.extend(_mapdiff(self.printTrackingParticle, tp1, tp2))
        ret.extend(_mapdiff(self.printHits, tp1, tp2))
        ret.extend(self.diffMatchedTracks(tp1, tp2))
        ret.extend(self.diffMatchedSeeds(tp1, tp2))
        return ret

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

    def bestMatchingTrackingParticle(self):
        """Returns best-matching TrackingParticle, even for fake tracks, or None if there is no best-matching TrackingParticle.

        Best-matching is defined as the one with largest number of
        hits matched to the hits of a track (>= 3). If there are many
        fulfilling the same number of hits, the one inducing the
        innermost hit of the track is chosen.
        """
        self._checkIsValid()
        if self._nMatchedTrackingParticles() == 1:
            return next(self.matchedTrackingParticleInfos()).trackingParticle()

        tps = collections.OrderedDict()
        for hit in self.hits():
            if not isinstance(hit, _SimHitAdaptor):
                continue
            for simHit in hit.simHits():
                tp = simHit.trackingParticle()
                if tp.index() in tps:
                    tps[tp.index()] += 1
                else:
                    tps[tp.index()] = 1

        best = (None, 2)
        for tpIndex, nhits in tps.iteritems():
            if nhits > best[1]:
                best = (tpIndex, nhits)
        if best[0] is None:
            return None
        return TrackingParticles(self._tree)[best[0]]

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

    def file(self):
        return self._file

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

    def getEvent(self, index):
        """Returns Event for a given index"""
        ientry = self._tree.LoadTree(index)
        if ientry < 0: return None
        nb = self._tree.GetEntry(ientry) # ientry or jentry?
        if nb <= 0: None

        return Event(self._tree, ientry) # ientry of jentry?

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

    def bestMatchingTrack(self):
        """Returns best-matching track, even for non-reconstructed TrackingParticles, or None, if there is no best-matching track.

        Best-matching is defined as the one with largest number of
        hits matched to the hits of a TrackingParticle (>= 3). If
        there are many fulfilling the same number of hits, the one
        inducing the innermost hit of the TrackingParticle is chosen.
        """
        self._checkIsValid()
        if self._nMatchedTracks() == 1:
            return next(self.matchedTrackInfos()).track()

        tracks = collections.OrderedDict()
        for hit in self.simHits():
            for recHit in hit.hits():
                for track in recHit.tracks():
                    if track.index() in tracks:
                        tracks[track.index()] += 1
                    else:
                        tracks[track.index()] = 1

        best = (None, 2)
        for trackIndex, nhits in tracks.iteritems():
            if nhits > best[1]:
                best = (trackIndex, nhits)
        if best[0] is None:
            return None
        return Tracks(self._tree)[best[0]]



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
