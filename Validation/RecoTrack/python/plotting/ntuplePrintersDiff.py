import re
import sys
import math
import difflib
import itertools
import collections

from operator import itemgetter, methodcaller

from Validation.RecoTrack.plotting.ntupleDataFormat import *

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
        return (None, 0)

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

class _TracksByHitsMatcher(object):
    def __init__(self, trklist):
        super(_TracksByHitsMatcher, self).__init__()
        self._hitsToTracks = collections.defaultdict(list)
        for trk in trklist:
            for hit in trk.hits():
                if hit.isValidHit():
                    self._hitsToTracks[ (type(hit), hit.index()) ].append(trk)

    def match(self, trk):
        tracks = collections.defaultdict(int)

        for hit in trk.hits():
            if not hit.isValidHit(): continue

            idx = (type(hit), hit.index())
            try:
                otherTracks = self._hitsToTracks[idx]
            except KeyError:
                continue

            for ot in otherTracks:
                tracks[ot] += 1

        best = (None, 0)
        for t, ncommon in tracks.iteritems():
            if ncommon > best[1]:
                best = (t, ncommon)
        return best


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

    def _highlightLine(self, line, plus, minus):
        char = " "
        if line[0] == "+":
            if plus: char = "+"
        elif line[0] == "-":
            if minus: char = "-"
        elif line[0] == "?":
            char = "?"
        return line[0]+char+line[1:]

    def highlight(self, plus=False, minus=False):
        if not (plus or minus):
            return

        for i, line in enumerate(self._diff):
            if isinstance(line, _DiffResult):
                line.highlight(plus, minus)
            else:
                self._diff[i] = self._highlightLine(line, plus, minus)

    def highlightLines(self, plusregexs=[], minusregexs=[]):
        if len(plusregexs) == 0 and len(minusregexs) == 0:
            return

        for i, line in enumerate(self._diff):
            if isinstance(line, _DiffResult):
                raise Exception("highlightLines() is currently allowed only for text-only _DiffResult objects")
            plus = False
            minus = False
            for p in plusregexs:
                if p.search(line):
                    plus = True
                    break
            for m in minusregexs:
                if m.search(line):
                    plus = True
                    break
            self._diff[i] = self._highlightLine(line, plus, minus)

    def lines(self):
        for line in self._diff:
            if isinstance(line, _DiffResult):
                for l in line.lines():
                    yield l
            else:
                yield line

    def __str__(self):
        return "\n".join([s for s in (str(item) for item in self._diff) if s != ""])

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
    return _makediff(lst1, lst2)

def _areSameTracks(trk1, trk2):
    ncommon = _commonHits(trk1, trk2)

    # if tracks have same hits, consider their reco as identical
    if not (ncommon == trk1.nValid() and ncommon == trk2.nValid()):
        return False

    # although if there is any change in their iterations, mark them different
    if not (trk1.algoMask() == trk2.algoMask() and trk1.algo() == trk2.algo() and trk1.originalAlgo() == trk2.originalAlgo()):
        return False

    # if reco is the same, check if they are matched to the
    # same TPs (in case the track-TP matching is modified
    # between ntuples)
    if trk1.nMatchedTrackingParticles() != trk2.nMatchedTrackingParticles():
        return False

    for tpInfo1, tpInfo2 in itertools.izip(trk1.matchedTrackingParticleInfos(), trk2.matchedTrackingParticleInfos()):
        if tpInfo1.trackingParticle().index() != tpInfo2.trackingParticle().index():
            return False

    return True

def diffTrackListsFromSameTrackingParticle(trackPrinter, lst1, lst2, lst1extra=[], lst2extra=[], diffByHitsOnly=False):
    """lst1 and lst2 are the main lists to make the diff from.

    lst1extra and lst2extra are optional to provide suplementary
    tracks. Use case: lst1 and lst2 are subset of full tracks,
    lst1extra and lst2extra contain tracks matched to the same
    TrackingParticle but are outside of the selection of lst1/lst2.
    """

    diff = _DiffResult()

    _trks1extra = list(lst1extra)
    _trks2extra = list(lst2extra)

    trks1 = list(lst1)+_trks1extra
    trks2 = list(lst2)+_trks2extra # make copy because it is modified

    trks1extra = set([t.index() for t in _trks1extra])
    trks2extra = set([t.index() for t in _trks2extra])

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
            if trk1.index() in trks1extra:
                raise Exception("Track %d was found in trks1extra but matchedTrk2 is None, this should not happen" % trk1.index())
            diff.extend(_makediff(trackPrinter.printTrack(trk1), []))
        else: # diff trk1 to best-matching track from trks2
            someTrk2 = matchedTrk2
            trks2.remove(matchedTrk2)
            tmp = trackPrinter.diff(trk1, matchedTrk2, diffTrackingParticles=False)
            if diffByHitsOnly and _areSameTracks(trk1, matchedTrk2):
                tmp.setDifference(False)
            tmp.highlight(plus=(matchedTrk2.index() in trks2extra), minus=(trk1.index() in trks1extra))
            diff.extend(tmp)

    for trk2 in trks2: # remaining tracks in trks2
        if trk2.index() in trks2extra:
            raise Exception("Track %d was found in trks2extra, but without matching track in trks1, this should not happen" % trk2.index())
        diff.extend(_makediff([], trackPrinter.printTrack(trk2)))

    # finally add information of the trackingParticle
    # easiest is to pass a track matched to the TP
    tmp = _mapdiff(trackPrinter.printMatchedTrackingParticles, someTrk1, someTrk2)
    tmp.setDifference(False) # we know the TP is the same, even if the "track match" information will differ
    def _makere(lst):
        r = []
        for i in lst: r.extend([re.compile("Tracks:.*%d:"%i), re.compile("matched to tracks.*%d"%i)])
        return r
    plusre = _makere(trks2extra)
    minusre = _makere(trks1extra)
    tmp.highlightLines(plusre, minusre)
    diff.extend(tmp)

    return diff

class _TrackAssociation(object):
    def __init__(self):
        super(_TrackAssociation, self).__init__()
        self._trks1 = []
        self._trks2 = []
        self._trks1OutsideList = []
        self._trks2OutsideList = []

        self._trks1Ind = set()
        self._trks2Ind = set()
        self._trks1OutsideListInd = set()
        self._trks2OutsideListInd = set()

    def _extend(self, trks, name):
        lst = getattr(self, name)
        ind = getattr(self, name+"Ind")
        for t in trks:
            if not t.index() in ind:
                lst.append(t)
                ind.add(t.index())

    def extend(self, trks1=[], trks2=[], trks1OutsideList=[], trks2OutsideList=[]):
        self.extendTrks1(trks1)
        self.extendTrks2(trks2)
        self.extendTrks1OutsideList(trks1OutsideList)
        self.extendTrks2OutsideList(trks2OutsideList)

    def extendTrks1(self, trks):
        self._extend(trks, "_trks1")

    def extendTrks2(self, trks):
        self._extend(trks, "_trks2")

    def extendTrks1OutsideList(self, trks):
        self._extend(trks, "_trks1OutsideList")

    def extendTrks2OutsideList(self, trks):
        self._extend(trks, "_trks2OutsideList")

    def trks1(self): return self._trks1
    def trks2(self): return self._trks2
    def trks1OutsideList(self): return self._trks1OutsideList
    def trks2OutsideList(self): return self._trks2OutsideList

    def hasCommonTrackingParticle(self):
        trkGen = itertools.chain(self._trks1, self._trks2)
        try:
            first = next(trkGen)
        except StopIteration:
            return False
        if first.nMatchedTrackingParticles() != 1:
            return False

        tpIndex = next(first.matchedTrackingParticleInfos()).trackingParticle().index()

        for t in trkGen:
            if t.nMatchedTrackingParticles() != 1:
                return False
            if next(t.matchedTrackingParticleInfos()).trackingParticle().index() != tpIndex:
                return False
        return True


    def merge(self, other):
        self.extendTrks1(other._trks1)
        self.extendTrks2(other._trks2)
        self.extendTrks1OutsideList(other._trks1OutsideList)
        self.extendTrks2OutsideList(other._trks2OutsideList)

    def minEta(self):
        _min = lambda lst: min([t.eta() for t in lst])

        if len(self._trks1) > 0:
            return _min(self._trks1)
        if len(self._trks1OutsideList) > 0:
            return _min(self._trks1OutsideList)
        if len(self._trks2) > 0:
            return _min(self._trks2)
        if len(self._trks2_outsideList) > 0:
            return _min(self._trks2OutsideList)
        raise Exception("This _TrackAssociation is empty, minEta() makes no sense")

    def __str__(self):
        s = lambda l: str([t.index() for t in l])
        return s(self._trks1)+" "+s(self._trks2)+" "+s(self._trks1OutsideList)+" "+s(self._trks2OutsideList)

def _associateTracksByTrackingParticlesAndHits(lst1, lst2):
    trks1 = list(lst1)
    trks2 = list(lst2)

    trks1Matcher = _TracksByHitsMatcher(trks1)
    trks2Matcher = _TracksByHitsMatcher(trks2)

    # Used to have exactly the same Track objects for the same index
    trks1Dict = {t.index(): t for t in trks1}
    trks2Dict = {t.index(): t for t in trks2}

    # Bit of a hack...
    tps1 = None
    tps2 = None
    if len(trks1) > 0:
        tps1 = TrackingParticles(trks1[0]._tree)
    if len(trks2) > 0:
        tps2 = TrackingParticles(trks2[0]._tree)

    trkAssoc1 = {}
    trkAssoc2 = {}

    def _getOrCreateAssoc(trk, d, **kwargs):
        if trk.index() in d:
            a = d[trk.index()]
        else:
            a = _TrackAssociation()
            d[trk.index()] = a
        a.extend(**kwargs)
        return a

    while len(trks1) > 0:
        trk1 = trks1.pop(0)
        assoc1 = _getOrCreateAssoc(trk1, trkAssoc1, trks1=[trk1])

        # First associate via TP
        if trk1.nMatchedTrackingParticles() > 0 and tps2:
            matched = False

            for tpInfo1 in trk1.matchedTrackingParticleInfos():
                tp1 = tpInfo1.trackingParticle()

                # Find possible duplicates within trks1
                for trkInfo1 in tp1.matchedTrackInfos():
                    t1 = trkInfo1.track()
                    t1Index = t1.index()
                    if t1Index != trk1.index():
                        if t1Index in trks1Dict:
                            assoc1.extend(trks1=[t1]) # trk1 -> t1
                            _getOrCreateAssoc(t1, trkAssoc1, trks1=[t1, trk1]) # t1 -> trk1
                            #print "trk1 %d <-> t1 %d (TP)" % (trk1.index(), t1.index())
                            trks1.remove(trks1Dict[t1Index])
                        else:
                            #print "trk1 %d -> t1 %d (TP, not in list)" % (trk1.index(), t1.index())
                            assoc1.extend(trks1OutsideList=[t1]) # trk1 -> t1, if t1 is not in trks1

                # Then look for the same tp in trks2
                tp2 = tps2[tp1.index()]
                for trkInfo2 in tp2.matchedTrackInfos():
                    matched = True
                    t2 = trkInfo2.track()
                    t2Index = t2.index()
                    if t2Index in trks2Dict:
                        assoc1.extend(trks2=[t2]) # trk1 -> t2
                        _getOrCreateAssoc(t2, trkAssoc2, trks1=[trk1], trks2=[t2]) # t2 -> trk1
                        #print "trk1 %d <-> t2 %d (TP)" % (trk1.index(), t2.index())
                        try:
                            trks2.remove(trks2Dict[t2Index]) # can fail if t2 has already been matched via hits
                        except ValueError:
                            pass
                    else:
                        #print "trk1 %d -> t2 %d (TP, not in list)" % (trk1.index(), t2.index())
                        assoc1.extend(trks2OutsideList=[t2]) # trk1 -> t2, if t2 is not in trks2

            if matched:
                continue

        # If no matching tracks in trks2 via TrackingParticles, then
        # proceed finding the best match via hits
        (matchedTrk2, ncommon) = trks2Matcher.match(trk1)
        if matchedTrk2 is not None and ncommon >= 3:
            assoc1.extend(trks2=[matchedTrk2])
            assoc2 = _getOrCreateAssoc(matchedTrk2, trkAssoc2, trks1=[trk1], trks2=[matchedTrk2])
            #print "trk1 %d <-> t2 %d (hits)" % (trk1.index(), matchedTrk2.index())
            try:
                trks2.remove(matchedTrk2) # can fail if matchedTrk2 has already been matched via TP
            except ValueError:
                pass

            (matchedTrk1, ncommon1) = trks1Matcher.match(matchedTrk2)
            # if matchedTrk1 has TP, the link from matchedTrk1 -> matchedTrk2 will be created later
            if (matchedTrk1.nMatchedTrackingParticles() == 0 or not tps2) and matchedTrk1.index() != trk1.index():
                assoc2.extend(trks1=[matchedTrk1])
                _getOrCreateAssoc(matchedTrk1, trkAssoc1, trks1=[matchedTrk1], trks2=[matchedTrk2])
                #print "trk1 %d <-> t2 %d (hits, via t2)" % (matchedTrk1.index(), matchedTrk2.index())

        # no match

    # remaining tracks in trks2
    for trk2 in trks2:
        assoc2 = _getOrCreateAssoc(trk2, trkAssoc2, trks2=[trk2])
        # collect duplicates
        if trk2.nMatchedTrackingParticles() > 0:
            for tpInfo2 in trk2.matchedTrackingParticleInfos():
                tp2 = tpInfo2.trackingParticle()
                for trkInfo2 in tp2.matchedTrackInfos():
                    t2 = trkInfo2.track()
                    t2Index = t2.index()
                    if t2Index in trks2Dict:
                        assoc2.extend(trks2=[t2])
                        #print "trk2 %d -> t2 %d (TP)" % (trk2.index(), t2.index())
                    else:
                        assoc2.extend(trks2OutsideList=[t2])
                        #print "trk2 %d -> t2 %d (TP, not in list)" % (trk2.index(), t2.index())

    # merge results
    # any good way to avoid copy-past?
    for ind, assoc in trkAssoc1.iteritems():
        for t1 in assoc.trks1():
            a = trkAssoc1[t1.index()]
            assoc.merge(a)
            a.merge(assoc)
        for t2 in assoc.trks2():
            a = trkAssoc2[t2.index()]
            assoc.merge(a)
            a.merge(assoc)
    for ind, assoc in trkAssoc2.iteritems():
        for t2 in assoc.trks2():
            a = trkAssoc2[t2.index()]
            assoc.merge(a)
            a.merge(assoc)
        for t1 in assoc.trks1():
            a = trkAssoc1[t1.index()]
            assoc.merge(a)
            a.merge(assoc)

    for ind, assoc in itertools.chain(trkAssoc1.iteritems(), trkAssoc2.iteritems()):
        #if ind in [437, 1101]:
        #    print "----"
        #    print ind, [t.index() for t in assoc.trks1()], [t.index() for t in assoc.trks2()]
        for t1 in assoc.trks1():
            a = trkAssoc1[t1.index()]
            assoc.merge(a)
            a.merge(assoc)

        #if ind in [437, 1101]:
        #    print ind, [t.index() for t in assoc.trks1()], [t.index() for t in assoc.trks2()]

        for t2 in assoc.trks2():
            a = trkAssoc2[t2.index()]
            assoc.merge(a)
            a.merge(assoc)
        #if ind in [437, 1101]:
        #    print ind, [t.index() for t in assoc.trks1()], [t.index() for t in assoc.trks2()]
        #    print "####"

    # collapse to a single collection of associations
    allAssocs = []
    while len(trkAssoc1) > 0:
        (t1Index, assoc) = trkAssoc1.popitem()

        #if t1Index == 1299:
        #    print t1Index, [t.index() for t in assoc.trks2()]
        for t1 in assoc.trks1():
            if t1.index() == t1Index: continue
            trkAssoc1.pop(t1.index())
        for t2 in assoc.trks2():
            trkAssoc2.pop(t2.index())
        allAssocs.append(assoc)
    while len(trkAssoc2) > 0:
        (t2Index, assoc) = trkAssoc2.popitem()
        if len(assoc.trks1()) > 0:
            raise Exception("len(assoc.trks1()) %d != 0 !!! %s for t2 %d" % (len(assoc.trks1()), str([t.index() for t in assoc.trks1()]), t2Index))
        for t2 in assoc.trks2():
            if t2.index() == t2Index: continue
            trkAssoc2.pop(t2.index())
        allAssocs.append(assoc)

    return allAssocs

def diffTrackListsGeneric(trackPrinter, lst1, lst2, ignoreAdditionalLst2=False):
    associations = _associateTracksByTrackingParticlesAndHits(lst1, lst2)

    # sort in eta
    associations.sort(key=methodcaller("minEta"))

    diff = _DiffResult()
    for assoc in associations:
        if assoc.hasCommonTrackingParticle():
            if len(assoc.trks1()) == 0 and ignoreAdditionalLst2:
                continue

            tmp = diffTrackListsFromSameTrackingParticle(trackPrinter, assoc.trks1(), assoc.trks2(), lst1extra=assoc.trks1OutsideList(), lst2extra=assoc.trks2OutsideList(), diffByHitsOnly=True)
            if tmp.hasDifference():
                diff.extend(tmp)
                diff.extend([" "])
        elif len(assoc.trks1()) == 1 and len(assoc.trks2()) == 1:
            trk1 = assoc.trks1()[0]
            trk2 = assoc.trks2()[0]

            if not _areSameTracks(trk1, trk2):
                diff.extend(trackPrinter.diff(trk1, trk2))
                diff.extend([" "])
        elif len(assoc.trks2()) == 0:
            for t in assoc.trks1():
                diff.extend(trackPrinter.diff(t, None))
                diff.extend([" "])
        elif len(assoc.trks1()) == 0:
            if ignoreAdditionalLst2:
                continue
            for t in assoc.trks1():
                diff.extend(trackPrinter.diff(None, t))
                diff.extend([" "])
        else:
            # needs to be rather generic, let's start by sorting by the innermost hit
            trks1 = list(assoc.trks1())
            trks2 = list(assoc.trks2())
            trks1.sort(key=lambda t: next(t.hits()).r())
            trks2.sort(key=lambda t: next(t.hits()).r())

            # then calculate number of shared hits for each pair
            ncommon = []
            for i1, t1 in enumerate(trks1):
                for i2, t2 in enumerate(trks2):
                    ncommon.append( (i1, i2, _commonHits(t1, t2)) )

            # sort that by number of common hits, descending order
            ncommon.sort(key=itemgetter(2), reverse=True)

            # then make the diffs starting from the pair with largest number of common hits
            pairs = [None]*len(trks1)
            usedT2 = [False]*len(trks2)
            for i1, i2, ncom in ncommon:
                if pairs[i1] is None:
                    pairs[i1] = i2
                    usedT2[i2] = True

            for i1, i2 in enumerate(pairs):
                t1 = trks1[i1]
                t2 = trks2[i2]
                diff.extend(trackPrinter.diff(t1, t2))
            for i2, used in enumerate(usedT2):
                if not used:
                    diff.extend(trackPrinter.diff(None, trks2[i2]))
            diff.extend([" "])

    return diff

def _formatHitDiffForTwiki(diffHits, prefix):
    line_re = re.compile("(?P<sign>[ \-+])\s+(?P<det>[a-zA-Z]+)(?P<lay>\d+)\D*?(\((?P<missing>missing|inactive)\))?\s+\d+")

    summary = []
    prevdet = ""
    prevsign = " "
    diffLines = diffHits.lines()

    # skip anything before the first line with "hits"
    for line in diffLines:
        if "hits" in line:
            break

    header = True
    for line in diffLines:
        # skip multiple occurrances of "hits" line, but only until
        # first line without "hits" is encountered
        if header:
            if "hits" in line:
                continue
            else:
                header = False

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
def _hitPatternSummary(hits):
    summary = ""

    prevdet = 0
    for hit in hits:
        det = hit.subdet()
        lay = hit.layer()

        if det != prevdet:
            summary += " "+SubDet.toString(det)
            prevdet = det

        summary += str(lay)
        if isinstance(hit, InvalidHit):
            summary += "(%s)"%InvalidHit.Type.toString(hit.type())[0]

    return summary

class _IndentPrinter(object):
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

class _RecHitPrinter(_IndentPrinter):
    def __init__(self, indent=0):
        super(_RecHitPrinter, self).__init__(indent)

    def _printHits(self, hits):
        lst = []
        for hit in hits:
            matched = ""
            glued = ""
            coord = ""
            if hit.isValidHit():
                if hasattr(hit, "matchedSimHitInfos"):
                    matched = " from %s " % HitSimType.toString(hit.simType())
                    matches = []
                    hasChargeFraction = False
                    for shInfo in hit.matchedSimHitInfos():
                        m = "%d:%d" % (shInfo.simHit().trackingParticle().index(), shInfo.simHit().index())
                        if hasattr(shInfo, "chargeFraction"):
                            m += "(%.1f)"%(shInfo.chargeFraction()*100)
                            hasChargeFraction = True
                        matches.append(m)
                    if len(matches) == 0:
                        matched += "not matched to any TP/SimHit"
                    else:
                        matched += "matched to TP:SimHit"
                        if hasChargeFraction:
                            matched += "(%)"
                        matched += " "+",".join(matches)

                coord = "x,y,z %f,%f,%f" % (hit.x(), hit.y(), hit.z())
                if isinstance(hit, GluedHit):
                    glued = "monoHit %d stereoHit %d " % (hit.monoHit().index(), hit.stereoHit().index())

            lst.append(self._prefix+"{layer} {hit} detid {detid} {detidStr} {glued}{coord}{matched}".format(layer=hit.layerStr(), hit=hit.index(),
                                                                                                               detid=hit.detId(), detidStr=hit.detIdStr(),
                                                                                                               glued=glued, coord=coord, matched=matched))
        return lst

class _TrackingParticleMatchPrinter(object):
    def __init__(self, trackingParticles, trackingParticlePrinter, bestMatchingTrackingParticle):
        self._trackingParticles = trackingParticles
        self._trackingParticlePrinter = trackingParticlePrinter
        self._bestMatchingTrackingParticle = bestMatchingTrackingParticle

    def bestMatchingTrackingParticle(self):
        return self._bestMatchingTrackingParticle

    def _printTrackingParticles(self, prefix, tps, header):
        lst = []
        if self._trackingParticlePrinter is None:
            lst.append(prefix+header+" "+",".join([str(tp.index()) for tp in tps]))
        else:
            lst.append(prefix+header)
            for tp in tps:
                lst.extend(self._trackingParticlePrinter.printTrackingParticle(tp))
                lst.extend(self._trackingParticlePrinter.printHits(tp))
                lst.extend(self._trackingParticlePrinter.printMatchedTracks(tp, useTrackPrinter=False))
        return lst

    def printMatchedTrackingParticles(self, prefix, track):
        lst = []
        if not self._trackingParticles:
            return lst

        pfx = prefix+" "
        if self._trackingParticlePrinter is not None:
            self._trackingParticlePrinter.indent(len(pfx)+1)

        if track.nMatchedTrackingParticles() == 0:
            if self._bestMatchingTrackingParticle:
                bestTP = track.bestMatchingTrackingParticle()
                if bestTP is not None:
                    lst.extend(self._printTrackingParticles(pfx, [bestTP], "not matched to any TP, but a following TP with >= 3 matched hits is found (shared hit fraction %.2f)" % track.bestMatchingTrackingParticleShareFrac()))
                else:
                    lst.append(prefix+"not matched to any TP")
            else:
                lst.append(prefix+"not matched to any TP")
        else:
            lst.extend(self._printTrackingParticles(pfx, [tpInfo.trackingParticle() for tpInfo in track.matchedTrackingParticleInfos()], "matched to TPs"))

        if self._trackingParticlePrinter is not None:
            self._trackingParticlePrinter.restoreIndent()

        return lst

class SeedPrinter(_RecHitPrinter):
    def __init__(self, indent=0, hits=True, trackingParticles=False, trackingParticlePrinter=None, bestMatchingTrackingParticle=True):
        super(SeedPrinter, self).__init__(indent)
        self._hits = hits
        self._trackingParticleMatchPrinter = _TrackingParticleMatchPrinter(trackingParticles, trackingParticlePrinter, bestMatchingTrackingParticle)

    def printHeader(self, seed):
        lst = []
        track = seed.track()
        if track.isValid():
            madeTrack = "made track %d" % track.index()
        else:
            madeTrack = "did not make a track, stopReason %s" % SeedStopReason.toString(seed.stopReason())
            if seed.stopReason() == SeedStopReason.NOT_STOPPED:
                madeTrack += " (usually this means that the track was reconstructed, but rejected by the track selection)"

        lst.append(self._prefix+"Seed %d algo %s %s" % (seed.indexWithinAlgo(), Algo.toString(seed.algo()), madeTrack))
        lst.append(self._prefix+" starting state: pT %f local pos x,y %f,%f mom x,y,z %f,%f,%f" % (seed.statePt(), seed.stateTrajX(), seed.stateTrajY(), seed.stateTrajPx(), seed.stateTrajPy(), seed.stateTrajPz()))
        return lst

    def printHits(self, seed):
        lst = []
        if self._hits:
            lst.append(self._prefix+" hits"+_hitPatternSummary(seed.hits()))
            self.indent(2)
            lst.extend(self._printHits(seed.hits()))
            self.restoreIndent()
        return lst

    def printMatchedTrackingParticles(self, seed):
        return self._trackingParticleMatchPrinter.printMatchedTrackingParticles(self._prefix, seed)

    def printSeed(self, seed):
        lst = []
        lst.extend(self.printHeader(seed))
        lst.extend(self.printHits(seed))
        lst.extend(self.printMatchedTrackingParticles(seed))
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

    def diff(self, seed1, seed2, diffForTwiki=False, diffTrackingParticles=False):
        if seed1 is None:
            return _makediff([], self.printSeed(seed2))
        if seed2 is None:
            return _makediff(self.printSeed(seed1), [])

        ret = _DiffResult()
        ret.extend(_mapdiff(self.printHeader, seed1, seed2))
        diffHits = _mapdiff(self.printHits, seed1, seed2)
        ret.extend(diffHits)
        if diffForTwiki:
            ret.extend(_formatHitDiffForTwiki(diffHits, self._prefix))
        if diffTrackingParticles:
            ret.extend(_mapdiff(self.printMatchedTrackingParticles, seed1, seed2))
        return ret

class TrackPrinter(_RecHitPrinter):
    def __init__(self, indent=0, hits=True, seedPrinter=SeedPrinter(), trackingParticles=True, trackingParticlePrinter=None, bestMatchingTrackingParticle=True, diffForTwiki=False):
        super(TrackPrinter, self).__init__(indent)
        self._hits = hits
        self._seedPrinter = seedPrinter
        self._trackingParticleMatchPrinter = _TrackingParticleMatchPrinter(trackingParticles, trackingParticlePrinter, bestMatchingTrackingParticle)
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
        if self._trackingParticleMatchPrinter.bestMatchingTrackingParticle():
            bestTP = track.bestMatchingTrackingParticle()
            if bestTP:
                lst.append(self._prefix+" best-matching TP %d" % bestTP.index())
                lst.append(self._prefix+"  shared hits %d reco denom %.3f sim denom %.3f sim cluster denom %.3f" % (track.bestMatchingTrackingParticleShareFrac()*track.nValid(), track.bestMatchingTrackingParticleShareFrac(), track.bestMatchingTrackingParticleShareFracSimDenom(), track.bestMatchingTrackingParticleShareFracSimClusterDenom()))
                lst.append(self._prefix+"  matching chi2/ndof %f" % track.bestMatchingTrackingParticleNormalizedChi2())
                lst.append(self._prefix+"  pulls pt %f theta %f phi %f dxy %f dz %f" % (track.ptPull(), track.thetaPull(), track.phiPull(), track.dxyPull(), track.dzPull()))
        return lst

    def printHits(self, track):
        lst = []
        if self._hits:
            lst.append(self._prefix+" hits"+_hitPatternSummary(track.hits()))
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
            ret.extend(self._seedPrinter.diff(track1.seed(), track2.seed(), self._diffForTwiki))
            self._seedPrinter.restoreIndent()
        return ret

    def printTrack(self, track):
        lst = self.printHeader(track)
        lst.extend(self.printHits(track))
        lst.extend(self.printSeed(track))
        return lst

    def printMatchedTrackingParticles(self, track):
        return self._trackingParticleMatchPrinter.printMatchedTrackingParticles(self._prefix, track)

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
            lst = self.printTrack(track2) + self.printMatchedTrackingParticles(track2)
            return _makediff([], lst)
        if track2 is None:
            lst = self.printTrack(track1) + self.printMatchedTrackingParticles(track1)
            return _makediff(lst, [])

        ret = _DiffResult()
        ret.extend(_mapdiff(self.printHeader, track1, track2))
        if self._diffForTwiki:
            trk1TPs = [tpInfo.trackingParticle() for tpInfo in track1.matchedTrackingParticleInfos()]
            trk2TPs = [tpInfo.trackingParticle() for tpInfo in track2.matchedTrackingParticleInfos()]

            pt_pull1 = "None"
            pt_pull2 = "None"
            dxy_pull1 = "None"
            dxy_pull2 = "None"
            dz_pull1 = "None"
            dz_pull2 = "None"

            ptPull1 = track1.ptPull()
            ptPull2 = track2.ptPull()
            if ptPull1 is not None and ptPull2 is not None:
                fmt = "{pull:.3g}"
                pt_pull1 = fmt.format(pull=ptPull1)
                pt_pull2 = fmt.format(pull=ptPull2)
                dxy_pull1 = fmt.format(pull=track1.dxyPull())
                dxy_pull2 = fmt.format(pull=track2.dxyPull())
                dz_pull1 = fmt.format(pull=track1.dzPull())
                dz_pull2 = fmt.format(pull=track2.dzPull())

            lst = [
                self._prefix+" parameters",
                self._prefix+"  pt %RED%{pt1:.3g}%ENDCOLOR% %GREEN%{pt2:.3g}%ENDCOLOR%".format(pt1=track1.pt(), pt2=track2.pt()),
            ]
            if pt_pull1 != "None":
                lst.append(self._prefix+"   pull %RED%{pull1}%ENDCOLOR% %GREEN%{pull2}%ENDCOLOR%".format(pull1=pt_pull1, pull2=pt_pull2))
            lst.extend([
                self._prefix+"  eta %RED%{eta1:.3g}%ENDCOLOR% %GREEN%{eta2:.3g}%ENDCOLOR%".format(eta1=track1.eta(), eta2=track2.eta()),
                self._prefix+"  phi %RED%{phi1:.3g}%ENDCOLOR% %GREEN%{phi2:.3g}%ENDCOLOR%".format(phi1=track1.phi(), phi2=track2.phi()),
                self._prefix+"  dxy %RED%{dxy1:.3g}%ENDCOLOR% %GREEN%{dxy2:.3g}%ENDCOLOR% ({dxy1rel:.2f}*err1, {dxy2rel:.2f}*err2)".format(dxy1=track1.dxy(), dxy2=track2.dxy(), dxy1rel=(track2.dxy()-track1.dxy())/track1.dxyErr(), dxy2rel=(track2.dxy()-track1.dxy())/track2.dxyErr()),
            ])
            if dxy_pull1 != "None":
                lst.append(self._prefix+"   pull %RED%{pull1}%ENDCOLOR% %GREEN%{pull2}%ENDCOLOR%".format(pull1=dxy_pull1, pull2=dxy_pull2))
            lst.extend([
                self._prefix+"  dz %RED%{dz1:.3g}%ENDCOLOR% %GREEN%{dz2:.3g}%ENDCOLOR% ({dz1rel:.2f}*err1, {dz2rel:.2f}*err2)".format(dz1=track1.dz(), dz2=track2.dz(), dz1rel=(track2.dz()-track1.dz())/track1.dzErr(), dz2rel=(track2.dz()-track1.dz())/track2.dzErr()),
            ])
            if dz_pull1 != "None":
                lst.append(self._prefix+"   pull %RED%{pull1}%ENDCOLOR% %GREEN%{pull2}%ENDCOLOR%".format(pull1=dz_pull1, pull2=dz_pull2))
            lst.extend([
                self._prefix+"  chi2/ndof %RED%{chi1:.3g}%ENDCOLOR% %GREEN%{chi2:.3g}%ENDCOLOR%".format(chi1=track1.nChi2(), chi2=track2.nChi2()),
            ])
            ret.extend(_makediff(lst, lst, equalPrefix="?"))

        diffHits = _mapdiff(self.printHits, track1, track2)
        ret.extend(diffHits)
        if self._hits and self._diffForTwiki:
            ret.extend(_formatHitDiffForTwiki(diffHits, self._prefix))

        ret.extend(self.diffSeeds(track1, track2))
        if diffTrackingParticles:
            ret.extend(_mapdiff(self.printMatchedTrackingParticles, track1, track2))
        return ret

class TrackingParticlePrinter(_IndentPrinter):
    def __init__(self, indent=0, parentage=True, hits=True, tracks=True, trackPrinter=None, bestMatchingTrack=True, seedPrinter=SeedPrinter()):
        super(TrackingParticlePrinter, self).__init__(indent)
        self._parentage = parentage
        self._hits = hits
        self._tracks = tracks
        self._trackPrinter = trackPrinter
        self._bestMatchingTrack = bestMatchingTrack
        self._seedPrinter = seedPrinter

    def _printTP(self, tp):
        genIds = ""
        if len(tp.genPdgIds()) > 0:
            genIds = " genPdgIds "+",".join([str(pdgId) for pdgId in tp.genPdgIds()])
        fromB = ""
        if tp.isFromBHadron():
            fromB = " from B hadron"
        return [
            self._prefix+"TP %d pdgId %d%s%s ev:bx %d:%d pT %f eta %f phi %f" % (tp.index(), tp.pdgId(), genIds, fromB, tp.event(), tp.bunchCrossing(), tp.pt(), tp.eta(), tp.phi()),
            self._prefix+" pixel hits %d strip hits %d numberOfTrackerHits() %d associated reco clusters %d dxy %f dz %f" % (tp.nPixel(), tp.nStrip(), tp.nTrackerHits(), tp.nRecoClusters(), tp.pca_dxy(), tp.pca_dz())
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
            lst.append(self._prefix+" sim hits"+_hitPatternSummary(tp.simHits()))
            for simhit in tp.simHits():
                tmp = []
                for h in simhit.hits():
                    tmp.append(",".join([str(trk.index()) for trk in h.tracks()]) + ":%d"%h.index())
                if len(tmp) == 0:
                    matched = "not matched to any Track/RecHit"
                else:
                    matched = "matched to Tracks:RecHits "+";".join(tmp)

                lst.append(self._prefix+"  %s %d pdgId %d process %d detId %d %s x,y,z %f,%f,%f %s" % (simhit.layerStr(), simhit.index(), simhit.particle(), simhit.process(), simhit.detId(), simhit.detIdStr(), simhit.x(), simhit.y(), simhit.z(), matched))
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

