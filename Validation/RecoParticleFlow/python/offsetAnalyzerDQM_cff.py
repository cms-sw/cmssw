import FWCore.ParameterSet.Config as cms
import Validation.RecoParticleFlow.defaults_cfi as default
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

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
            name = default.offset_name( "mu", mu, pftype )
            plots += [ plotPSet(
                name,
                name+";#eta;<Offset Energy_{T}> [GeV]",
                "{0}muPlots/mu{1}".format(default.offsetDir, mu),
                #variable xbinning
                0, 0, 0, default.eBinsOffset, default.eLowOffset, default.eHighOffset,
                default.etaBinsOffset
            )]

        for npv in npvrange :
            name = default.offset_name( "npv", npv, pftype )
            plots += [ plotPSet(
                name,
                name+";#eta;<Offset Energy_{T}> [GeV]",
                "{0}npvPlots/npv{1}".format(default.offsetDir, npv),
                #variable xbinning
                0, 0, 0, default.eBinsOffset, default.eLowOffset, default.eHighOffset,
                default.etaBinsOffset
            )]
    return plots

def createTH1DVPSet():
    plots = []
    #hname, title, xmax
    toplot = ( ("mu", "#mu", default.muHighOffset), ("npv", "N_{PV}", default.npvHighOffset) )

    for hname, title, xmax in toplot :
        plots += [ plotPSet(
                hname,
                hname + ";" + title,
                default.offsetDir,
                xmax, 0, xmax
            )]
    return plots

offsetAnalyzerDQM = DQMEDAnalyzer("OffsetAnalyzerDQM",
                                      
    pvTag = cms.InputTag('offlineSlimmedPrimaryVertices'),
    muTag = cms.InputTag('slimmedAddPileupInfo'),
    pfTag = cms.InputTag('packedPFCandidates'),

    pdgKeys = cms.vuint32( default.candidateDict.keys() ),
    pdgStrs = cms.vstring( default.candidateDict.values() ),

    offsetPlotBaseName = cms.string(default.offsetPlotBaseName),
    offsetPlots = cms.VPSet( createOffsetVPSet() ),
    th1dPlots = cms.VPSet( createTH1DVPSet() ),

    pftypes = cms.vstring( default.candidateType ),
    etabins = cms.vdouble( default.etaBinsOffset ),
    muHigh = cms.untracked.int32( default.muHighOffset ),
    npvHigh = cms.untracked.int32( default.npvHighOffset )                                        

)

offsetDQMPostProcessor = DQMEDHarvester("OffsetDQMPostProcessor",

    offsetPlotBaseName = cms.string( default.offsetPlotBaseName ),
    offsetDir = cms.string( default.offsetDir ),
    offsetVariableTypes = cms.vstring( default.offsetVariableType ),
    offsetR = cms.untracked.double( default.offsetR ),
    pftypes = cms.vstring( default.candidateType ),                                        
    muHigh = cms.untracked.int32( default.muHighOffset ),
    npvHigh = cms.untracked.int32( default.npvHighOffset )                                        
                                        
)

