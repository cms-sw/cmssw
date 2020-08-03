# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleElectronPt10_pythia8_cfi.py -s GEN,SIM,DIGI --conditions auto:mc --datatier GEN-SIM-RAW --eventcontent RECOSIM -n 10 --no_exec --python_filename SingleElectronPt10_cfi_py_GEN_IDEAL.py
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('DIGI',eras.phase2_common)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimCalorimetry.EcalSimProducers.esCATIAGainProducer_cfi')
process.load('SimCalorimetry.EcalSimProducers.esEcalLiteDTUPedestalsProducer_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_Ph2_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('Calibration.EcalCalibAlgos.ecalPedestalPCLHarvester_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_Ph2_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(3)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
        SkipEvent = cms.untracked.vstring('ProductNotFound')
)


# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('SingleElectronPt10_pythia8_cfi.py nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
            dataTier = cms.untracked.string('GEN-SIM-RAW'),
            filterName = cms.untracked.string('')            
    ),
    fileName = cms.untracked.string('SingleElectronPt10_pythia8_cfi_py_GEN_SIM_DIGI_Pt10.root'),
#    outputCommands = process.RECOSIMEventContent.outputCommands,
    outputCommands = cms.untracked.vstring('keep *',
                        'drop *_mix_*_*'),
    splitLevel = cms.untracked.int32(1)
)

#process.RECOSIMoutput.outputCommands.append('keep EBDigiCollection_ecalDigis_*_*')


    
# Additional output definition


# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

process.generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        AddAntiParticle = cms.bool(True),
            MaxEta = cms.double(1.43),
        MaxPhi = cms.double(3.14159265359),
        #MaxPt = cms.double(300.01),
	MaxPt = cms.double(10.01),
            MinEta = cms.double(1.42),
        MinPhi = cms.double(-3.14159265359),
        #MinPt = cms.double(299.99),
	MinPt = cms.double(9.99),
        ParticleID = cms.vint32(11)
    ),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single electron pt 10')
)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)



#CondCore.CondDB.CondDB_cfi
from CondCore.DBCommon.CondDBSetup_cfi import *
#from CondCore.CondDB.CondDB_cfi import *
process.ecalConditions = cms.ESSource("PoolDBESSource", CondDBSetup,
      #connect = cms.string('frontier://FrontierProd/CMS_COND_31X_ECAL'),
      #connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_ECAL'),
      #authpath = cms.string('/afs/cern.ch/cms/DB/conddb'),
      connect = cms.string('sqlite_file:SimCalorimetry/EcalSimProducers/test/simPulseShapePhaseII.db'),

      toGet = cms.VPSet(         # overide Global Tag use EcalTBWeights_EBEE_offline
                  cms.PSet(
                      record = cms.string('EcalSimPulseShapeRcd') ,
                      tag = cms.string('EcalSimPulseShape_default_mc')
                  )
              )
)
process.es_prefer_ecalPulseShape = cms.ESPrefer("PoolDBESSource","ecalConditions")

process.EcalCATIAGainRatiosESProducer = cms.ESProducer(
	"EcalCATIAGainRatiosESProducer",
	ComponentName = cms.string('testGainProducer')
)

process.EcalLiteDTUPedestalsESProducer = cms.ESProducer(
	"EcalLiteDTUPedestalsESProducer",
	ComponentName = cms.string('testPedestalProducer')
)

#LOGGER:
process.MessageLogger.cout = cms.untracked.PSet(
	threshold = cms.untracked.string("DEBUG"),
	default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
	FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
)


#process.es_prefer_EcalCATIAGainRatioESProducer = cms.ESPrefer("EcalCATIAGainRatioESProducer","EcalCATIAGainRatioESProducer")

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.endjob_step,process.RECOSIMoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion







