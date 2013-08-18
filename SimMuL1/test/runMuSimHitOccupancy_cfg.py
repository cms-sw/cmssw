# files concatenator cmsRun config
import FWCore.ParameterSet.Config as cms
import os

# events to process
nevt = 200000

# are we running on regular MB instead of neutron simhits sample?
isMB = False
isMB = True

# where the neutron hits input files are stored
inputDir = '/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_3_6_3_n/src/out_n'
if isMB: inputDir = '/uscmst1b_scratch/lpc1/lpctau/khotilov/condor/minbias_3_6_2_me42'

#inputDir = '/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_3_6_3_ye4/src/out_n_58k'

ls = os.listdir(inputDir)
input_names = ['file:'+inputDir+"/"+x for x in ls if x.endswith('root')]


ff = open('filelist_minbias_29M.txt', "r")
input_names = ff.read().split('\n')
ff.close()
input_names = filter(lambda x: x.endswith('.root'),  input_names)
input_names = input_names[:1000]


theHistoFileName = "shtree100K.root"
if isMB: theHistoFileName = "shtreeMB.root"

#theHistoFileName = "shtree_hp.root"
#theHistoFileName = "shtreec_hp.root"
#theHistoFileName = "shtreen_hp.root"

#theHistoFileName = "shtree_hp_eml.root"
#theHistoFileName = "shtree_emlsn.root"
#theHistoFileName = "shtree_eml.root"

#inputDir = '/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_3_6_3_ye4/src/crab_0_100826_002606/res'
#theHistoFileName = "shtreem_hp.root"



process = cms.Process('NEUTRON')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(nevt) )

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.EventContent.EventContent_cff')

### GEM Geometry ###

process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
#process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr08v01XML_cfi')
#process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr10v01XML_cfi')
process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
process.load('Geometry.GEMGeometry.gemGeometry_cfi')


### GlobalTag ###

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'POSTLS161_V12::All'


process.options = cms.untracked.PSet(
  makeTriggerResults = cms.untracked.bool(False),
  wantSummary = cms.untracked.bool(True)
)


process.source = cms.Source("PoolSource",
  duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
  fileNames = cms.untracked.vstring(
    #"file:/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_3_6_3_ye4/src/condor_hp/zomg.root"
    #"file:/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_3_9_0_pre2/src/condor_hp_eml/zomg.root"
    #"file:/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_3_9_0_pre2/src/condor_emlsn/zomg.root"
    #"file:/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_3_9_0_pre2/src/condor_eml/zomg.root"
    #['file:'+inputDir+"/"+x for x in ls if x.endswith('root')]
    #"file:/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_3_6_3_n/src/out_std/zomg.root"
    *input_names
  )
)


process.TFileService = cms.Service("TFileService",
    fileName = cms.string(theHistoFileName)
)


process.neutronAna = cms.EDAnalyzer("MuSimHitOccupancy",
    inputIsNeutrons = cms.untracked.bool(not isMB),
    inputTagCSC = cms.untracked.InputTag("g4SimHits","MuonCSCHits"),
    inputTagGEM = cms.untracked.InputTag("g4SimHits","MuonGEMHits"),
    inputTagRPC = cms.untracked.InputTag("g4SimHits","MuonRPCHits"),
    inputTagDT  = cms.untracked.InputTag("g4SimHits","MuonDTHits")
    #inputTagCSC = cms.untracked.InputTag("g4SimHitsNeutrons","MuonCSCHits"),
    #inputTagGEM = cms.untracked.InputTag("g4SimHitsNeutrons","MuonGEMHits"),
    #inputTagRPC = cms.untracked.InputTag("g4SimHitsNeutrons","MuonRPCHits"),
    #inputTagDT  = cms.untracked.InputTag("g4SimHitsNeutrons","MuonDTHits")
)


#ff = open("files.txt", "r")
#files = ff.read().split('\n')
#ff.close()
#
#vstri=[]
#for i in range (0,len(files)):
#    if len(files[i])==0: continue
#    if files[i].find('pnfs') > -1:
#        vstri.append('dcap:'+ files[i])
#    elif files[i].find('castor') == 1:
#        vstri.append('rfio:'+ files[i])
#    else:
#        vstri.append('file:'+ files[i])
#process.source.fileNames = vstri


#process.Timing = cms.Service("Timing")
#process.Tracer = cms.Service("Tracer")

process.neutron_step     = cms.Path(process.neutronAna)
process.endjob_step     = cms.Path(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(
    process.neutron_step,
    process.endjob_step,
#    process.out_step
)

