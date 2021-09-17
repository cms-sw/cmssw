import FWCore.ParameterSet.Config as cms

global_tag='START53_V29B::All'

#customize tag for SR. Empty string to use Global tag default:
#use cmscond_list_iov -c frontier://FrontierProd/CMS_COND_34X_ECAL -P/afs/cern.ch/cms/DB/conddb -a | grep EcalSRSettings
#and cmscond_list_iov -c frontier://FrontierPrep/CMS_COND_ECAL -P/afs/cern.ch/cms/DB/conddb -a | grep EcalSRSettings
#to list available tags. connect string in process.GlobalTag.toGet accordingly to the prod/prep database.
#sr_tag = ''                                    #takes setting from global tag
#sr_tag = 'EcalSRSettings_beam2010_v01_mc'      #beam09/beam10 settings
#sr_tag = 'EcalSRSettings_beam2010_v01_offline' #same as EcalSRSettings_beam2010_v01_mc 
#sr_tag = 'EcalSRSettings_fullreadout_v01_mc'   #full readout / 2010 heavy ion setting
#sr_tag  = 'EcalSRSettings_beam7TeV_v01_mc'      #thresholds of beam09/beam10 but with "optimized" weights
#sr_tag = 'EcalSRSettings_lumi1e33_v01_mc'      #setting used in MC before June 2010 (settings estimated for 2.e33cm-2s-1)
#sr_tag = 'EcalSRSettings_beam7TeV_v02_mc'       #optimized weights with 300MeV threshold in EE, 80MeV in EB. Candidate for beam11 run
sr_tag = 'EcalSRSettings_beam2012_option1_v00_mc' #optimized weights with 360MeV threshold in EE, 96.25MeV in EB. Candidate for beam11 run

process = cms.Process("ProcessOne")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'sqlite_file:' + tag + '.db'
#process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

# Conditions
#process.GlobalTag.globaltag = global_tag
from Configuration.AlCa.GlobalTag import *
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup')


if sr_tag != '' :
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("EcalSRSettingsRcd"),
             tag = cms.string(sr_tag),
             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
#              connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
#             connect = cms.untracked.string('sqlite_file:' + sr_tag + '.db')
    ))



## process.PoolDBESSource = cms.ESSource("PoolDBESSource",
##                                       process.CondDBCommon,
##                                       toGet = cms.VPSet(cms.PSet(
##     record = cms.string('EcalSRSettingsRcd'),
##     tag = cms.string(sr_tag)
##     )))

  

process.readFromDB = cms.EDAnalyzer("EcalSRCondTools",
    mode = cms.string("read")
)

process.p = cms.Path(process.readFromDB)

