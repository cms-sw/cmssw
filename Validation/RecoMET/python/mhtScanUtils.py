# Helpers shared by the HLT and offline MHT scans (addMHTScanHLT, addMHTScanOffline).

def cutTag(minPt, maxEta):
    """Folder name of a jet selection: `minPt<pt>_maxEta<eta>`, with 'p' as decimal separator."""
    fmt = lambda v: ('%g' % v).replace('.', 'p')
    return f"minPt{fmt(minPt)}_maxEta{fmt(maxEta)}"

def moduleTag(tag):
    """Module labels cannot contain underscores."""
    return tag.replace('_m', 'M').replace('minPt', 'MinPt')

def scanCuts(caller, minPtJetMht, maxEtaJetMht):
    """Returns the equal-length lists of cuts; `maxEtaJetMht` defaults to 5 for all `minPtJetMht` values."""
    minPtJetMht = list(minPtJetMht)
    maxEtaJetMht = [5.]*len(minPtJetMht) if maxEtaJetMht is None else list(maxEtaJetMht)
    if len(minPtJetMht) != len(maxEtaJetMht):
        raise ValueError(f"{caller}: lists must have the same length (minPtJetMht={len(minPtJetMht)}, "
                         f"maxEtaJetMht={len(maxEtaJetMht)})")
    return minPtJetMht, maxEtaJetMht
