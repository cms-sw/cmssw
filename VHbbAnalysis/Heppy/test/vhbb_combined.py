#! /usr/bin/env python

from PhysicsTools.Heppy.utils.cmsswPreprocessor import CmsswPreprocessor

# If vhbb_combined is imported from vhbb_combined_data than the next
# line will have no effect (as vhbb is already imported there)
from vhbb import *

from VHbbAnalysis.Heppy.AdditionalBTag import AdditionalBTag
from VHbbAnalysis.Heppy.AdditionalBoost import AdditionalBoost
from VHbbAnalysis.Heppy.GenHFHadronMatcher import GenHFHadronMatcher


# Add Boosted Information
boostana=cfg.Analyzer(
    verbose=False,
    class_object=AdditionalBoost,
)

#boostana.GT = "Fall15_25nsV2_DATA" 
boostana.GT = "Spring16_25nsV6_DATA" # we do L2L3 for MC and L2L3Res for data. Can therefor use data GT for both
boostana.jecPath = os.environ['CMSSW_BASE']+"/src/VHbbAnalysis/Heppy/data/jec"
boostana.isMC = sample.isMC
boostana.skip_ca15 = False
sequence.insert(sequence.index(VHbb),boostana)

#used freshly computed MVA ID variables
LepAna.updateEleMVA = True

#Use PUID from 76X
JetAna.externalPuId="pileupJetIdUpdated:fullId"

genhfana=cfg.Analyzer(
    verbose=False,
    class_object=GenHFHadronMatcher,
)
sequence.insert(sequence.index(VHbb),genhfana)


treeProducer.collections["ak08"] = NTupleCollection("FatjetAK08ungroomed",  ak8FatjetType,  10,
                                                    help = "AK, R=0.8, pT > 200 GeV, no grooming, calibrated")

treeProducer.collections["ak08softdropsubjets"] = NTupleCollection("SubjetAK08softdrop",
                                                                 patSubjetType,
                                                                 10,
                                                                 help="Subjets of AK, R=0.8 softdrop")

if not boostana.skip_ca15:
    treeProducer.collections["ca15ungroomed"] = NTupleCollection("FatjetCA15ungroomed",  fatjetType,  10,
                                                                 help = "CA, R=1.5, pT > 200 GeV, no grooming")

    treeProducer.collections["ca15softdrop"] = NTupleCollection("FatjetCA15softdrop",
                                                                fourVectorType,
                                                                10,
                                                                help="CA, R=1.5, pT > 200 GeV, softdrop zcut=0.1, beta=0")

    # four-vector + n-subjettiness
    treeProducer.collections["ca15softdropz2b1"] = NTupleCollection("FatjetCA15softdropz2b1",
                                                                    fatjetTauType,
                                                                    10,
                                                                    help="CA, R=1.5, pT > 200 GeV, softdrop zcut=0.2, beta=1")

    treeProducer.collections["ca15softdropfilt"] = NTupleCollection("FatjetCA15softdropfilt",
                                                                fourVectorType,
                                                                10,
                                                                help="CA, R=1.5, pT > 200 GeV, softdrop zcut=0.1, beta=0 + Filtering")

    treeProducer.collections["ca15softdropz2b1filt"] = NTupleCollection("FatjetCA15softdropz2b1filt",
                                                                fourVectorType,
                                                                10,
                                                                help="CA, R=1.5, pT > 200 GeV, softdrop zcut=0.2, beta=1 + Filtering")


    treeProducer.collections["ca15trimmed"] = NTupleCollection("FatjetCA15trimmed",
                                                                fourVectorType,
                                                                10,
                                                                help="CA, R=1.5, pT > 200 GeV, trimmed r=0.2, f=0.06")

    treeProducer.collections["ca15pruned"] = NTupleCollection("FatjetCA15pruned",
                                                                fourVectorType,
                                                                10,
                                                                help="CA, R=1.5, pT > 200 GeV, pruned zcut=0.1, rcut=0.5, n=2")

    treeProducer.collections["ca15subjetfiltered"] = NTupleCollection("FatjetCA15subjetfiltered",
                                                                        fourVectorType,
                                                                        10,
                                                                        help="CA, R=1.5, pT > 200 GeV, BDRS via SubjetFilterJetProducer")

    treeProducer.collections["ca15prunedsubjets"] = NTupleCollection("SubjetCA15pruned",
                                                                     subjetType,
                                                                     10,
                                                                     help="Subjets of CA, R=1.5, pT > 200 GeV, pruned zcut=0.1, rcut=0.5, n=2")

    treeProducer.collections["ca15softdropsubjets"] = NTupleCollection("SubjetCA15softdrop",
                                                                     subjetType,
                                                                     10,
                                                                     help="Subjets of CA, R=1.5, pT > 200 GeV, softdrop z=0.1, beta=0")

    treeProducer.collections["ca15softdropz2b1subjets"] = NTupleCollection("SubjetCA15softdropz2b1",
                                                                     subjetType,
                                                                     10,
                                                                     help="Subjets of CA, R=1.5, pT > 200 GeV, softdrop z=0.2, beta=1")

    treeProducer.collections["ca15softdropfiltsubjets"] = NTupleCollection("SubjetCA15softdropfilt",
                                                                           subjetType,
                                                                           10,
                                                                           help="Subjets of CA, R=1.5, pT > 200 GeV, softdrop z=0.1, beta=0 + Filtering")

    treeProducer.collections["ca15softdropz2b1filtsubjets"] = NTupleCollection("SubjetCA15softdropz2b1filt",
                                                                               subjetType,
                                                                               10,
                                                                               help="Subjets of CA, R=1.5, pT > 200 GeV, softdrop z=0.2, beta=1 + Filtering")


    treeProducer.collections["ca15subjetfilteredsubjets"] = NTupleCollection("SubjetCA15subjetfiltered",
                                                                             subjetType,
                                                                             30,
                                                                             help="Subjets of CA, R=1.5, pT > 200 GeV, BDRS, filterjets")

    treeProducer.collections["httCandidates"] = NTupleCollection("httCandidates",
                                                                 httType,
                                                                 10,
                                                                 help="OptimalR HEPTopTagger Candidates")


# Add b-Tagging Information
btagana=cfg.Analyzer(
    verbose=False,
    class_object=AdditionalBTag,
)
sequence.insert(sequence.index(VHbb),btagana)
VHbb.btagDiscriminator=lambda x: x.btagHip

# Add Information on generator level hadronic tau decays
if sample.isMC:   
    from VHbbAnalysis.Heppy.TauGenJetAnalyzer import TauGenJetAnalyzer
    TauGenJet = cfg.Analyzer(
        verbose = False,
        class_object = TauGenJetAnalyzer,
    )
    sequence.insert(sequence.index(VHbb),TauGenJet)

    treeProducer.collections["tauGenJets"] = NTupleCollection("GenHadTaus", genTauJetType, 15, help="Generator level hadronic tau decays")

# Switch MET inputs to newly created slimmedMETs collection with PFMET significance matrix added
for ic in range(len(config.sequence)):
    obj = config.sequence[ic]
        
    if obj.class_object.__name__ == "METAnalyzer" and obj.instance_label == "METAna":
        obj.metCollection = "slimmedMETs::EX"

# Run Everything
preprocessor = CmsswPreprocessor("combined_cmssw.py", options = {"isMC":sample.isMC})
config.preprocessor=preprocessor
if __name__ == '__main__':
    from PhysicsTools.HeppyCore.framework.looper import Looper 
    looper = Looper( 'Loop', config, nPrint = 1, nEvents = 100)
    import time
    import cProfile
    p = cProfile.Profile(time.clock)
    p.runcall(looper.loop)
    p.print_stats()
    looper.write()
