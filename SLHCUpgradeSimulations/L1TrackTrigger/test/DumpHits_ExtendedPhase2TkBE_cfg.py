####################################################
# SLHCUpgradeSimulations                           #
# Configuration file for Full Workflow             #
# Step 2 (again)                                   #
# Understand if everything is fine with            #
# L1TkCluster e L1TkStub                           #
####################################################
# Nicola Pozzobon                                  #
# CERN, August 2012                                #
####################################################

#################################################################################################
# import of general framework
#################################################################################################
import FWCore.ParameterSet.Config as cms
import os
process = cms.Process('DumpHits')

#################################################################################################
# global tag
#################################################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

#################################################################################################
# load the specific tracker geometry
#################################################################################################
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBEReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

#process.TrackerDigiGeometryESModule.applyAlignment = False
#process.TrackerDigiGeometryESModule.fromDDD = cms.bool(True)
#process.TrackerGeometricDetESModule.fromDDD = cms.bool(True)

#################################################################################################
# load the magnetic field
#################################################################################################
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
#process.load('Configuration.StandardSequences.MagneticField_40T_cff')
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

#################################################################################################
# define the source and maximum number of events to generate and simulate
#################################################################################################
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
     fileNames = cms.untracked.vstring(
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/0004FBE6-A3C8-E211-854E-002618943939.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/008BB498-52C8-E211-B232-0025905964B2.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/04597B78-52C8-E211-BF3D-00261894386C.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/0A3C9EDC-50C8-E211-8A8A-0026189438ED.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/0A5FAAC6-50C8-E211-B1B2-003048FFD7A2.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/0E2A769D-52C8-E211-953A-00259059391E.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/129682C8-50C8-E211-AFA2-0026189438ED.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/12FAF76A-52C8-E211-A01D-002618943898.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/14ECCC1A-52C8-E211-9F8D-002618943932.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/16255034-52C8-E211-8287-0026189437ED.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/185CF61D-52C8-E211-BA67-0026189438A0.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/1A1FD42A-66C8-E211-B38A-002618943969.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/1A59F7EC-50C8-E211-A93C-002618943949.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/1C5967EE-50C8-E211-A585-002618943949.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/20BAA1BD-50C8-E211-BE09-00261894385D.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/225553E2-50C8-E211-B8EE-002618943894.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/262FE453-52C8-E211-95CE-002618FDA287.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/268BA41E-52C8-E211-87D5-002618943880.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/2896B2D2-50C8-E211-8D89-002618943854.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/28B90777-52C8-E211-9DA1-00261894386C.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/2A60F4FE-52C8-E211-ACCC-003048FFD744.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/2A6AB827-66C8-E211-A360-002618943972.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/2CAE08D4-50C8-E211-908C-002618943854.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/2E73C152-52C8-E211-B06A-002618FDA287.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/302A6659-52C8-E211-AD74-00248C55CC3C.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/324C5F5F-52C8-E211-8646-00248C55CC3C.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/32BAED1C-52C8-E211-BC95-00261894383F.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/34155E1D-52C8-E211-8EFF-00261894383F.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/342A5797-51C8-E211-95BE-0026189437FA.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/36DDF4A7-52C8-E211-8FB4-00261894387D.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/38B07C69-52C8-E211-87D8-002618943926.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/3AA40B80-52C8-E211-A266-002618943860.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/3E2EAF71-52C8-E211-9C8E-00261894383E.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/402613C4-51C8-E211-A7CA-002618FDA250.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/403E9B1E-52C8-E211-9E90-002618943932.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/40C76EA3-52C8-E211-94DD-0025905964B2.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/42053A70-52C8-E211-9C34-002618943860.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/42E04971-52C8-E211-AAAE-00248C55CC3C.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/4802EFDC-50C8-E211-A704-0026189438ED.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/4A66969C-51C8-E211-B2E3-002618FDA250.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/4C9EB81C-52C8-E211-A00A-002618943932.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/4E737ED1-50C8-E211-A105-002618943854.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/4E751BE2-50C8-E211-8120-00259059391E.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/500FBE47-52C8-E211-886C-0026189437F5.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/523A751D-52C8-E211-A929-0026189438A0.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/52DEAD72-52C8-E211-B1D9-002618943898.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/54EDD47F-A3C8-E211-A6A0-00261894386F.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/562D60F9-52C8-E211-BB0E-003048FFD744.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/56AB026B-52C8-E211-BBFE-00261894383E.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/58A9A300-53C8-E211-AC64-003048FFD744.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/5A2A3CF6-50C8-E211-AD84-002618943894.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/5A6E9DF6-50C8-E211-9D6F-002618943894.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/5C2B82E9-50C8-E211-846A-002618943949.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/60339C6C-52C8-E211-B26A-00261894383E.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/6036E4C3-50C8-E211-B9A0-003048FFCB96.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/6062E39B-52C8-E211-A0CA-002618943854.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/60A8D2C7-50C8-E211-919E-0026189438ED.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/62DD4C74-52C8-E211-9DA1-002618943898.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/667C5226-52C8-E211-AFDC-002618943880.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/68778C1C-52C8-E211-BBDD-00261894383F.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/6CF70C1D-52C8-E211-8B7D-002618943880.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/6CFC8C92-52C8-E211-BAC7-003048FFCC18.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/6E36AB29-52C8-E211-9E10-002618943880.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/6E6A60C9-50C8-E211-8080-0025905964BA.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/6E99FF1C-52C8-E211-84F8-0026189438A0.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/6ED24BFB-52C8-E211-A78A-003048FFD744.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/7075F4D2-50C8-E211-BDFF-003048FFD720.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/70D97A0F-52C8-E211-9E0C-0026189438A0.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/7A209D0F-52C8-E211-9744-0026189438A0.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/7A7D2C61-AFC8-E211-8297-0026189438F4.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/7E01828C-51C8-E211-B2A1-002618FDA250.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/7E6123F8-52C8-E211-B470-003048FFD744.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/7EA330F6-50C8-E211-B108-002618943894.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/8094AA69-52C8-E211-81FE-00248C55CC3C.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/80D722CC-52C8-E211-9C16-003048FFD76E.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/80E52FA7-52C8-E211-B82C-0025905938A8.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/82F5B25D-52C8-E211-8434-002618943926.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/84FBCD1C-52C8-E211-9419-00261894383F.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/86686DF5-51C8-E211-AEC5-002618FDA250.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/8E8D36B3-52C8-E211-BF7B-0025905964BC.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/90BE29A1-52C8-E211-AFBD-00259059391E.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/9881436C-52C8-E211-B996-002618943926.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/9A368C1C-52C8-E211-BFD8-0026189438A0.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/9C6D94DC-50C8-E211-9A84-0026189438ED.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/9E62B4CA-50C8-E211-8F48-003048FFD732.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/9EE299D8-50C8-E211-B100-00259059642A.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/A06F4178-52C8-E211-8674-00261894386C.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/A2A6C31C-52C8-E211-A8E0-00261894383F.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/A2C9094F-52C8-E211-BB70-0026189437F5.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/A41095C7-50C8-E211-BE4B-0026189438ED.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/A48B31F6-50C8-E211-9AAA-002618943894.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/AA11E1C9-50C8-E211-B8E2-003048FFD732.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/AC0AB403-53C8-E211-9F5E-003048FFD744.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/AE3E2864-52C8-E211-9A02-00248C55CC3C.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/B09E30F6-50C8-E211-B4F5-002618943894.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/B27076EB-50C8-E211-8980-002618943949.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/B8007A78-52C8-E211-991C-002618943860.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/BA4564CB-50C8-E211-A2CA-00248C55CC97.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/C005798A-52C8-E211-93A2-003048FFCC18.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/C23E2DFA-52C8-E211-9744-003048FFD744.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/C24B1B23-52C8-E211-A8F4-002618943880.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/C862BEA3-52C8-E211-91BC-00259059391E.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/CA0E93C3-50C8-E211-8A01-00248C55CC3C.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/CA89C39D-52C8-E211-A2BC-002590593878.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/CEA32F6E-52C8-E211-ABBE-002618943898.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/CEE577EA-50C8-E211-956B-002618943949.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/D0FC161D-52C8-E211-877F-002618943932.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/D2C0896B-52C8-E211-82C2-002618943926.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/D64729CD-52C8-E211-875B-0025905822B6.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/D6D5D3AD-52C8-E211-B90B-002618943854.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/DA6B9F3E-52C8-E211-8C10-0026189437ED.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/DA7D6A48-52C8-E211-8A00-0026189437F5.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/DC1EB541-52C8-E211-ADA0-0025905964B6.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/DE1FF1A7-52C8-E211-BDA0-003048FFD7A2.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/E0BB9326-52C8-E211-94F3-0026189437FA.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/E29A7CA7-52C8-E211-ADB8-00261894387D.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/E601A39D-52C8-E211-84E4-002590593878.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/E657C9F5-50C8-E211-B992-002618943894.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/E8BECDC9-50C8-E211-9EC1-00248C55CC97.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/EAA5DE19-52C8-E211-A70E-002618943932.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/EAAF37DA-51C8-E211-BFD7-002618FDA250.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/EC3A5C82-52C8-E211-9B16-002618943860.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/EC3FED1E-52C8-E211-BAD8-002618943932.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/EC620B2E-52C8-E211-A70C-0026189437FA.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/EEF945CA-50C8-E211-B7FB-00248C55CC97.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/F49F08D4-50C8-E211-9384-002618943854.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/F4B0411A-52C8-E211-9FB7-0026189438A0.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/F68045FD-52C8-E211-AE9C-003048FFD744.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/F682F3A7-52C8-E211-93E5-00261894387D.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/F69FEBDA-50C8-E211-B576-0025905938A8.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/F881289A-52C8-E211-89CE-002590596490.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/FC359F5C-52C8-E211-8ED7-002618943926.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/FCF2F877-52C8-E211-AA2A-00261894386C.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/FE640D1D-52C8-E211-9C2C-002618943880.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/FE67C9AD-52C8-E211-A367-002618943854.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/FECA43FD-52C8-E211-ABD1-003048FFD744.root',
   'file:/afs/cern.ch/user/p/pozzo/eos/cms/store/mc/Summer13/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradePhase2BE_2013_DR61SLHCx_PU140Bx25_POSTLS261_V2-v1//10000/FEF1D4F4-52C8-E211-B711-003048FFD744.root',
       )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(10)
)

#################################################################################################
# load the analyzer
#################################################################################################
process.DumpHits = cms.EDAnalyzer("DumpHits",
    TextOutput = cms.string('DumpHits_MinBias140PU_ExtendedPhase2TkBE.log'),
    DebugMode = cms.bool(True)
)

#################################################################################################
# define output file and message logger
#################################################################################################
process.TFileService = cms.Service("TFileService",
  fileName = cms.string('file:DumpHits_ExtendedPhase2TkBE.root')
)

#################################################################################################
# define the final path to be fed to cmsRun
#################################################################################################
process.p = cms.Path( process.DumpHits )

