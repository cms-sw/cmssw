import os, sys, glob
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from Configuration.ProcessModifiers.gpu_cff import gpu
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

def getHeterogeneousRecHitsSource(pu):
    indir = '/eos/user/b/bfontana/Samples/' #indir = '/home/bfontana/'
    filename_suff = 'step3_ttbar_PU' + str(pu) #filename_suff = 'hadd_out_PU' + str(pu)
    fNames = [ 'file:' + x for x in glob.glob(os.path.join(indir, filename_suff + '*.root')) ]
    print(indir, filename_suff, pu, fNames)
    for _ in range(4):
        fNames.extend(fNames)
    if len(fNames)==0:
        print('Used globbing: ', glob.glob(os.path.join(indir, filename_suff + '*.root')))
        raise ValueError('No input files!')

    keep = 'keep *'
    drop1 = 'drop CSCDetIdCSCALCTPreTriggerDigiMuonDigiCollection_simCscTriggerPrimitiveDigis__HLT'
    drop2 = 'drop HGCRecHitsSorted_HGCalRecHit_HGC*E*RecHits_*'
    return cms.Source("PoolSource",
                      fileNames = cms.untracked.vstring(fNames),
                      inputCommands = cms.untracked.vstring(keep, drop1, drop2),
                      duplicateCheckMode = cms.untracked.string("noDuplicateCheck"))

#arguments parsing
from FWCore.ParameterSet.VarParsing import VarParsing
F = VarParsing('analysis')
F.register('PU',
           1,
           F.multiplicity.singleton,
           F.varType.int,
           "Pileup to consider.")
F.parseArguments()

#package loading
process = cms.Process("gpuValidation", gpu) 
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi')
process.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

#TFileService

dirName = '/eos/user/b/bfontana/Samples/'
fileName = 'validation' + str(F.PU) + '.root'
process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string( os.path.join(dirName,fileName) ),
                                   closeFileFast = cms.untracked.bool(True)
                               )

process.source = getHeterogeneousRecHitsSource(F.PU)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )) #add option for edmStreams

process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEERecHitGPU_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEERecHitGPUtoSoA_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEERecHitFromSoA_cfi')

process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEFRecHitGPU_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEFRecHitGPUtoSoA_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEFRecHitFromSoA_cfi')

process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEBRecHitGPU_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEBRecHitGPUtoSoA_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEBRecHitFromSoA_cfi')

process.HGCalRecHits = HGCalRecHit.clone()

process.valid = cms.EDAnalyzer( 'HeterogeneousHGCalRecHitsValidator',
                                cpuRecHitsEEToken = cms.InputTag('HGCalRecHits', 'HGCEERecHits'),
                                gpuRecHitsEEToken = cms.InputTag('EERecHitFromSoAProd'),
                                cpuRecHitsHSiToken = cms.InputTag('HGCalRecHits', 'HGCHEFRecHits'),
                                gpuRecHitsHSiToken = cms.InputTag('HEFRecHitFromSoAProd'),
                                cpuRecHitsHSciToken = cms.InputTag('HGCalRecHits', 'HGCHEBRecHits'),
                                gpuRecHitsHSciToken = cms.InputTag('HEBRecHitFromSoAProd')
)

process.ee_t = cms.Task( process.EERecHitGPUProd, process.EERecHitGPUtoSoAProd, process.EERecHitFromSoAProd )
process.hef_t = cms.Task( process.HEFRecHitGPUProd, process.HEFRecHitGPUtoSoAProd, process.HEFRecHitFromSoAProd )
process.heb_t = cms.Task( process.HEBRecHitGPUProd, process.HEBRecHitGPUtoSoAProd, process.HEBRecHitFromSoAProd )
process.gpu_t = cms.Task( process.ee_t, process.hef_t, process.heb_t )
process.cpu_t = cms.Task( process.HGCalRecHits )
process.path = cms.Path( process.valid, process.gpu_t, process.cpu_t )


process.out = cms.OutputModule( "PoolOutputModule", 
                                fileName = cms.untracked.string( os.path.join(dirName, 'out.root') ),
                                outputCommands = cms.untracked.vstring('drop *') )

process.outpath = cms.EndPath(process.out)
