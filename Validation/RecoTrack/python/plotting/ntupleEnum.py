# Poor-man enum class with string conversion
class _Enum:
    def __init__(self, **values):
        self._reverse = {}
        for key, value in values.items():
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
    displacedGeneralStep = 25, 
    displacedRegionalStep = 26,
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

# Need to be kept consistent with
# DataFormats/TrackReco/interface/SeedStopReason.h
SeedStopReason = _Enum(
  UNINITIALIZED = 0,
  NOT_STOPPED = 1,
  SEED_CLEANING = 2,
  NO_TRAJECTORY = 3,
  SEED_REGION_REBUILD = 4,
  FINAL_CLEAN = 5,
  SMOOTHING_FAILED = 6,
  SIZE = 7
)

# to be kept is synch with enum HitSimType in TrackingNtuple.py
HitSimType = _Enum(
    Signal = 0,
    ITPileup = 1,
    OOTPileup = 2,
    Noise = 3,
    Unknown = 99
)

