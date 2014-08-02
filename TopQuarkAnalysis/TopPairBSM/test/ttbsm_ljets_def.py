def isTightMu(lep):
    isPF      =  lep.isPFMuon()
    isGlob    =  lep.isGlobalMuon()
    if isPF and isGlob:
        normChi2  =  lep.globalTrack().normalizedChi2()
        trkLayers =  lep.track().hitPattern().trackerLayersWithMeasurement()
        mVMuHits  =  lep.globalTrack().hitPattern().numberOfValidMuonHits()
        dB        =  fabs( lep.dB() )
        #diffVz    =  fabs( lep.vertex().z() - PVz )
        mPixHits  =  lep.innerTrack().hitPattern().numberOfValidPixelHits()
        matchStat =  lep.numberOfMatchedStations()

    #if(isPF and isGlob and normChi2<10 and trkLayers>5 and mVMuHits>0 and dB<0.2 and diffVz<0.5 and mPixHits>0 and matchStat >1):
    if(isPF and isGlob and normChi2<10 and trkLayers>5 and mVMuHits>0 and dB<0.2 and mPixHits>0 and matchStat >1):    
        return True
    else:
        return False

def isLooseMu(lep):
    isPF      =  lep.isPFMuon()
    isGlob    =  lep.isGlobalMuon()
    isTrack   =  lep.isTrackerMuon()
    if isPF and (isGlob or isTrack):
        return True
    else:
        return False
