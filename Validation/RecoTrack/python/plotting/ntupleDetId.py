import collections

from Validation.RecoTrack.plotting.plotting import _th1ToOrderedDict
from Validation.RecoTrack.plotting.ntupleEnum import *

class DetId:
    def __init__(self, detid, contentDict):
        self._contentDict = contentDict

        self.detid = detid
        for name, value in contentDict.iteritems():
            setattr(self, name, value)

        # exclude these from printout
        del self._contentDict["det"]
        del self._contentDict["subdet"]

    def __str__(self):
        return " ".join(["%s %d" % tpl for tpl in self._contentDict.iteritems()])

class _SubDetIdParser:
    def __init__(self, th1, entries):
        d = _th1ToOrderedDict(th1)
        self._entries = entries
        for e in entries:
            setattr(self, "_%sStartBit"%e, int(d[e+"StartBit"][0]))
            setattr(self, "_%sMask"%e, int(d[e+"Mask"][0]))

    def parse(self, detid, det, subdet):
        d = collections.OrderedDict()

        d["det"] = det
        d["subdet"] = subdet

        for e in self._entries:
            startBit = getattr(self, "_%sStartBit" % e)
            mask = getattr(self, "_%sMask" % e)
            d[e] = (detid >> startBit) & mask

        return DetId(detid, d)

class DetIdParser:
    def __init__(self, tdirectory):
        self._parsers = {
            SubDet.BPix: _SubDetIdParser(tdirectory.Get("pbVals"), ["layer", "ladder", "module"]),
            SubDet.FPix: _SubDetIdParser(tdirectory.Get("pfVals"), ["side", "disk", "blade", "panel", "module"]),
            SubDet.TIB: _SubDetIdParser(tdirectory.Get("tibVals"), ["layer", "str_fw_bw", "str_int_ext", "str", "module", "ster"]),
            SubDet.TID: _SubDetIdParser(tdirectory.Get("tidVals"), ["side", "wheel", "ring", "module_fw_bw", "module", "ster"]),
            SubDet.TOB: _SubDetIdParser(tdirectory.Get("tobVals"), ["layer", "rod_fw_bw", "rod", "module", "ster"]),
            SubDet.TEC: _SubDetIdParser(tdirectory.Get("tecVals"), ["side", "wheel", "petal_fw_bw", "petal", "ring", "module", "ster"]),
        }

    def parse(self, detid):
        det = (detid >> 28) & 0xF
        subdet = (detid >> 25) & 0x7

        try:
            parser = self._parsers[subdet]
        except KeyError:
            raise Exception("Got unknown subdet %d" % subdet)
        return parser.parse(detid, det, subdet)
