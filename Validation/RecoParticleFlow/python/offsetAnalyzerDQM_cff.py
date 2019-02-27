import FWCore.ParameterSet.Config as cms
import  Validation.RecoParticleFlow.defaults_cfi as default

def plotPSet(name, title, dir, nx, x0, x1, ny=0, y0=0, y1=0, vx=[0], vy=[0]):
    return cms.PSet(
        name = cms.string(name),
        title = cms.string(title),
        dir = cms.string(dir),
        nx = cms.uint32(nx),
        x0 = cms.double(x0),
        x1 = cms.double(x1),
        ny = cms.uint32(ny),
        y0 = cms.double(y0),
        y1 = cms.double(y1),
        vx = cms.vdouble(vx),
        vy = cms.vdouble(vy)
    )

def createOffsetVPSet():
    plots = []
    murange = range( default.muLowOffset, default.muHighOffset )
    npvrange = range( default.npvLowOffset, default.npvHighOffset )

    for pftype in default.candidateType :
        for mu in murange :
            name = "{0}_mu{1}_{2}".format( default.offsetPlotBaseName, mu, pftype )
            plots += [ plotPSet(
                name,
                name+";#eta;Offset Energy_{T} [GeV]",
                default.offsetDir + "muPlots/" + pftype,
                #variable xbinning
                0, 0, 0, default.eBinsOffset, default.eLowOffset, default.eHighOffset,
                default.etaBinsOffset
            )]

        for npv in npvrange :
            name = "{0}_npv{1}_{2}".format( default.offsetPlotBaseName, npv, pftype )
            plots += [ plotPSet(
                name,
                name+";#eta;<Offset Energy_{T}> [GeV]",
                default.offsetDir + "npvPlots/" + pftype,
                #variable xbinning
                0, 0, 0, default.eBinsOffset, default.eLowOffset, default.eHighOffset,
                default.etaBinsOffset
            )]
    return plots

def createTH1DVPSet():
    plots = []
    #hname, title, xmax
    toplot = ( ("mu", "#mu", default.muHighOffset), ("npv", "N_{PV}", default.npvHighOffset) )

    for tup in toplot :
        plots += [ plotPSet(
                tup[0],
                tup[0] + ";" + tup[1],
                default.offsetDir,
                tup[2]*2, 0, tup[2]
            )]
    return plots

offsetAnalyzerDQM = cms.EDProducer("OffsetAnalyzerDQM",

    pvTag = cms.InputTag('offlineSlimmedPrimaryVertices'),
    muTag = cms.InputTag('slimmedAddPileupInfo'),
    pfTag = cms.InputTag('packedPFCandidates'),

    pdgKeys = cms.vuint32( default.candidateDict.keys() ),
    pdgStrs = cms.vstring( default.candidateDict.values() ),

    offsetPlotBaseName = cms.string(default.offsetPlotBaseName),
    offsetPlots = cms.VPSet( createOffsetVPSet() ),
    th1dPlots = cms.VPSet( createTH1DVPSet() ),
)

#print( offsetAnalyzerDQM.offsetPlots[455].parameters_  )
#for i in range(0, len(offsetAnalyzerDQM.th1dPlots)) :
#    print( offsetAnalyzerDQM.th1dPlots[i].parameters_  )

offsetDQM = cms.Sequence(
    offsetAnalyzerDQM
)
