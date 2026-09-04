import FWCore.ParameterSet.Config as cms

# Inject the HLT ParticleFlow DQM into a re-HLT + @hltValidation job.
#
# @hltValidation already provides hltHgcalValidator (LC / trackster /
# TICLCandidate) and HLTJetMETValSeq (hltAK4PF{,CHS,Puppi}Jets + MET).  What it
# does not provide is any HLT ParticleFlow monitoring, so the three analyzers
# from DQMForPF_HLT_cff are appended here:
#
#   PFCandAnalyzerDQMHLT  candidate composition,        hltParticleFlowTmp
#   offsetAnalyzerDQMHLT  offset energy vs eta vs mu,   hltParticleFlowTmp
#   pfAnalyzerHLT         DQMOffline PFAnalyzer,        hltParticleFlowTmp + hltAK4PFPuppiJets
#
# pfAnalyzerHLT writes to the folder "ParticleFlow" (hard-coded as m_directory
# in DQMOffline/ParticleFlow/plugins/PFAnalyzer.cc), which is exactly the path
# jroloff's pfmonitoringplots expects after harvesting.


def addHLTPFDQM(process):
    process.load('Validation.RecoParticleFlow.DQMForPF_HLT_cff')

    seq = process.DQMHLTPF_withPFAnalyzer

    # The analyzers are DQM producers.  Attach to the validation Path when there
    # is one (the HLT-only workflow has VALIDATION but no DQM step); fall back to
    # a DQM path otherwise.  Use += rather than *, since an EndPath has no __mul__.
    for stepName in ('validation_step', 'dqmoffline_step', 'dqmofflineOnPAT_step'):
        if hasattr(process, stepName):
            step = getattr(process, stepName)
            step += seq
            print('[customiseHLTPFDQM] appended HLT PF DQM to %s (%s)'
                  % (stepName, type(step).__name__))
            break
    else:
        raise RuntimeError('customiseHLTPFDQM: no DQM or validation path to attach to')

    return process


def addHLTPFHarvesting(process):
    """Harvesting counterpart: the offset post-processor."""
    process.load('Validation.RecoParticleFlow.DQMForPF_HLT_cff')
    for stepName in ('dqmHarvesting', 'alcaHarvesting', 'dqmsave_step'):
        if hasattr(process, stepName):
            step = getattr(process, stepName)
            step += process.DQMHarvestHLTPF
            print('[customiseHLTPFDQM] appended HLT PF harvesting to %s (%s)'
                  % (stepName, type(step).__name__))
            break
    else:
        raise RuntimeError('customiseHLTPFDQM: no harvesting path to attach to')
    return process
