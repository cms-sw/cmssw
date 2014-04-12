#import FWCore.ParameterSet.Config as cms
import os

runOnTheGrid = False

def customise(process):
        process._Process__name="SECONDHLT"
        #process.Tracer = cms.Service("Tracer")
        #process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )
        
        module_name="overlay"
        process_name="OVERLAY"
        process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

        #for i in [process.HLT_QuadJet30, process.HLT_MET35, process.HLT_L1MuOpen, process.HLT_BTagIP_DoubleJet120, process.HLT_BTagIP_DoubleJet60_Relaxed, process.HLT_BTagIP_TripleJet70, process.HLT_BTagIP_TripleJet40_Relaxed, process.HLT_BTagIP_QuadJet40, process.HLT_BTagIP_QuadJet30_Relaxed,process.HLT_BTagIP_HT470,process.HLT_BTagIP_HT320_Relaxed, process.HLT_IsoEle5_TripleJet30, process.HLT_Mu5_TripleJet30, process.HLT_ZeroBias, process.HLT_MinBiasHcal, process.HLT_MinBiasEcal, process.HLT_MinBiasPixel, process.HLT_MinBiasPixel_Trk5, process.HLT_CSCBeamHalo, process.HLT_CSCBeamHaloOverlapRing1,process.HLT_CSCBeamHaloOverlapRing2,process.HLT_CSCBeamHaloRing2or3,process.AlCa_EcalPhiSym,process.AlCa_HcalPhiSym,process.AlCa_EcalPi0]:
        #       process.schedule.remove(i)
        
        if runOnTheGrid:
                process.source.fileNames=cms.untracked.vstring(__FILE_NAMES__)
                process.source.skipEvents=cms.untracked.uint32(__SKIP_EVENTS__)
                process.maxEvents.input = cms.untracked.int32(__MAX_EVENTS__)
                process.output.fileName = cms.untracked.string("output.root")
                  
        process.output.outputCommands = cms.untracked.vstring("keep *_*_*_*")
                                  
        return (process)        


