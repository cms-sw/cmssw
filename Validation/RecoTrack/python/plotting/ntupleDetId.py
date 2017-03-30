from Validation.RecoTrack.plotting.ntupleEnum import *

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
