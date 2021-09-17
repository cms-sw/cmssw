import FWCore.ParameterSet.Config as cms
import Validation.RecoParticleFlow.defaults_cfi as default
from Validation.RecoParticleFlow.defaults_cfi import ptbins, etabins, response_distribution_name, genjet_distribution_name,jetResponseDir,genjetDir

#----- ----- ----- ----- ----- ----- ----- -----
#
# Auxiliary definitions
#

def make_response_plot_pset(name, title, responseNbins, responseLow, responseHigh, ptBinLow, ptBinHigh, etaBinLow, etaBinHigh):
    return cms.PSet(
        name = cms.string(name),
        title = cms.string(title),
        responseNbins = cms.uint32(responseNbins),
        responseLow = cms.double(responseLow),
        responseHigh = cms.double(responseHigh),
        ptBinLow = cms.double(ptBinLow),
        ptBinHigh = cms.double(ptBinHigh),
        etaBinLow = cms.double(etaBinLow),
        etaBinHigh = cms.double(etaBinHigh),
    )

#Jet response is plotted in histograms which can be subdivided by pt and |eta| of the genjet.
#To minimize the amount of logic on the C++ side, we define all response plots here.
#Each plot has low and high pt and |eta| edges, the plot is filled only if the genjet
#is in the bin defined by the edges.
#It is your job here to make sure you define the bins in a non-overlapping way if
#you want to emulate a 2D map over (pT, |eta|) of 1D histograms.
def createResponsePlots(ptbins, etabins):
    response_plots = []
    #we always use a range [ibin, ibin+1)
    for ietabin in range(len(etabins)-1):
        for iptbin in range(len(ptbins)-1):

            response_plots += [make_response_plot_pset(
                response_distribution_name(iptbin, ietabin),
                "Jet response (pT/pTgen) in {0} <= pt < {1}, {2} <= |eta| < {3}".format(ptbins[iptbin], ptbins[iptbin+1], etabins[ietabin], etabins[ietabin+1]),
                100, 0.0, 3.0, ptbins[iptbin], ptbins[iptbin+1], etabins[ietabin], etabins[ietabin+1]
            )]
    return response_plots

def createGenJetPlots(ptbins, etabins):
    plots = []
    for ietabin in range(len(etabins)-1):
        eta_low = etabins[ietabin]
        eta_high = etabins[ietabin + 1]
        plots += [
            cms.PSet(
            name = cms.string(genjet_distribution_name(ietabin)),
            title = cms.string("GenJet pT ({0} <= |eta| <= {1}".format(eta_low, eta_high)),
            ptBins = cms.vdouble(ptbins),
            etaBinLow = cms.double(eta_low),
            etaBinHigh = cms.double(eta_high),
        )]
    return plots

#----- ----- ----- ----- ----- ----- ----- -----
#
# Config for analyzer and postprocessor
#

name = "genjet_pt"
title = "genjet pt"
pfJetAnalyzerDQM = cms.EDProducer("PFJetAnalyzerDQM",

    #match these reco-jets to the gen-jets and compute jet response
    recoJetCollection = cms.InputTag('slimmedJets'),
    genJetCollection = cms.InputTag('slimmedGenJets'),
    jetDeltaR = cms.double(0.2),

    # turn gen jets on or off
    genJetsOn = cms.bool(True),

    responsePlots = cms.VPSet(createResponsePlots(ptbins, etabins)),
    genJetPlots = cms.VPSet(createGenJetPlots(ptbins, etabins))

)

pfPuppiJetAnalyzerDQM = pfJetAnalyzerDQM.clone(
    recoJetCollection = 'slimmedJetsPuppi',
    genJetsOn = False
)

vjetResponseDir = [jetResponseDir + "slimmedJets/JEC/",
                   jetResponseDir + "slimmedJets/noJEC/",
                   jetResponseDir + "slimmedJetsPuppi/JEC/",
                   jetResponseDir + "slimmedJetsPuppi/noJEC/"]

pfJetDQMPostProcessor = cms.EDProducer("PFJetDQMPostProcessor",

    jetResponseDir = cms.vstring( vjetResponseDir ),
    genjetDir = cms.string( genjetDir ),
    ptBins = cms.vdouble( ptbins ),
    etaBins = cms.vdouble( etabins ),
    recoPtCut = cms.double( 15. )

)


# PFCandidates
PFCandAnalyzerDQM = cms.EDProducer("PFCandidateAnalyzerDQM",
    PFCandType = cms.InputTag("packedPFCandidates"),
    etabins = cms.vdouble( default.etaBinsOffset ),
    pdgKeys = cms.vuint32( default.pdgIDDict.keys() ),
    pdgStrs = cms.vstring( default.pdgIDDict.values() )
)


#----- ----- ----- ----- ----- ----- ----- -----
