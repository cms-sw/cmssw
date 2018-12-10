import FWCore.ParameterSet.Config as cms

process = cms.Process("MULTITRACKVALIDATOR")

# message logger
process.MessageLogger = cms.Service("MessageLogger",
     default = cms.untracked.PSet( limit = cms.untracked.int32(400) )
)


#Adding SimpleMemoryCheck service:
process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                   ignoreTotal=cms.untracked.int32(1),
                                   oncePerEventMode=cms.untracked.bool(True)
)

process.Timing = cms.Service("Timing"
    ,summaryOnly = cms.untracked.bool(True)
)

# source
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/F011E1AB-68A8-FE46-99F4-D2C6CD8D5C21.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/E1066B8F-692E-4E49-B0C4-57096AC363CD.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/E0625BF3-C6A4-EC4A-A155-68AC2F8CC70C.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/DFD80AE2-308A-D749-88C0-6B66683BA0E9.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/DEB98825-786C-1C45-BA43-1E9D0B60845F.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/DE8C309C-5EFA-364E-BC8A-61532813B5ED.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/D9C19382-77E8-7F4A-824E-B959F9D50CCC.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/D9565594-5A08-7345-A52C-C31CE19A456D.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/D4C5801D-BAA4-B142-BFDE-F972B360C3D9.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/D2DF1CAB-3BC1-EA48-9424-259BF9C074B4.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/CB49E391-375E-1F40-8A23-89C4C4676079.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/C7AD3D88-9661-6841-9695-C5727E3166D8.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/C54C7D23-4FE8-4F45-BF1D-5BDFE7502D9D.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/C1A1B1CD-3A3F-6F4B-8225-9D66BB7CB57F.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/C0DB4FB8-C416-0942-AE3E-EA7B4CD48067.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/BEC37FE3-748F-A64A-95FD-EECDACD23B5E.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/BD4C76D6-0B16-3C4C-B257-65836A1BB892.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/B4AC90AA-877E-8240-B101-8FC4D7882815.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/B2027A64-492A-4348-B5DB-656ADC1317BF.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/AF54CD7D-DD76-E343-9309-5C4D9C69E860.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/AE2645BA-2CCD-1944-95AD-1DDF7E68EA66.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/A958084C-0F3F-8048-8196-310FA615A9BE.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/A7AFFAA1-D6F6-184F-B49C-A3A56FE20DBD.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/A45A6F77-2D27-7541-9226-9BC0BDFDD949.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/A3E7A69C-542F-DE49-AA2E-9339A6FC5128.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/A0EAE37E-B50E-EC40-BEBE-0CFB6A35537A.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/9F5BD363-43A3-C644-A05E-22EA6F494738.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/9C494ED2-27AE-3842-8A28-9DFC1B0C1380.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/99876B71-1C89-494F-BC04-A24D576EA752.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/8D43ED60-182A-A344-A6EF-8C81091D403A.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/8771CFDC-2BB1-164A-9ED4-1F1FF0FA0C50.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/7C521EAD-8597-F845-9508-1E18F3595CFA.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/67B242FD-47C4-5247-93D6-FB502A800948.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/612B9F52-4260-C14A-BC01-EF87EF735FC5.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/5D7E24D8-E14A-2244-B297-A6C163AA4AB7.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/5D74A7D1-C09F-BD4A-AB73-D42FB09BF1DB.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/4C83F5A1-9E8C-5944-A4D5-722BDD625ADF.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/4A911BA2-ABDA-3F4C-8934-01C8090A98E4.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/44EECF6D-A886-0A42-9455-40DD59684D6A.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/3ED8CDC8-2C1B-E246-B366-186F46D35373.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/3307D940-4906-2240-8263-8A8B80975BAE.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/31EC0614-E0D0-D949-BB67-ECE2EC61F4F5.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/2C2703FE-8539-1E44-B449-BBB2868B7A3E.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/26556179-4297-9248-A271-FE1BB2ACB785.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/1F4D5D16-FF73-5942-B00B-FBF3FBF922FA.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/183E2526-056C-6B40-81C1-A467D9D3AEF8.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/180740AC-83FC-8D41-9121-6A4740CAB08C.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/15DD77BC-7D7E-814A-8A78-917FD7B78107.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/15A3695E-04BA-8C48-8D63-3397B4E272A5.root',
# 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/15906C9A-6C70-D64F-9E64-4AAE5BBFE668.root'




                  ] )


secFiles.extend( [
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/FE46D080-D937-1C44-B961-E30362738D82.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/FB5D4F84-BD31-1244-9BD4-0474A466980D.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/F908E4C6-59FB-5A45-A957-2E325B7DC9D7.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/F4944D82-0E2B-A24A-A6FD-C90FF77080FC.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/DF21F6BB-F53E-D248-A648-6B5748A76D79.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/D77110B0-D224-044C-83F5-0317C0FBB396.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/D6F7F1E1-E5D3-E04C-8C09-19E0B44A4F2E.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/D51B1B03-7E43-C349-B7BE-EE2E03C40E21.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/CDAC15C1-5208-1146-B433-8C152132ED67.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/CA20BCC1-4D8B-FC46-B1BD-85EC23E878D3.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/C9BE82A0-81A0-114B-9945-85EC602BFE3E.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/C143CE5F-111D-6149-A2B2-E4D6E2254EDA.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/BF162C8F-DD99-FE4F-B4EA-1DC483DD2620.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/BE989D50-2234-CB45-B41F-8B324A1C4E9D.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/BC98D2F5-2336-DE49-B6C8-AE89D32A8CC2.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/BA2E7CB3-8687-BE4F-93D9-6207D75171C8.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/B614E036-0BDB-4448-9771-36357D54D30B.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/A5F7029B-2340-1E46-A0BC-9ECC1936AE9B.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/A5EFA211-2B9E-8941-9E11-F596083A2896.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/9EE4445D-365E-8C4F-9C57-EB7E7575E019.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/9D706F2E-D6F2-B246-BA73-5942E375BA4C.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/9CC306A0-1FFC-3749-9C6A-E6447C3A1564.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/94E39EEA-1E7A-7B4B-A29B-6ABEB365EE07.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/85B9FD62-BC0A-8940-BBC9-2EEDAC731366.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/7C1072FF-26A9-2D4C-8873-8E39BEBC6528.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/72B1AE9E-FE83-FE40-AFB3-7E9829591E7B.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/716C7757-0D7F-EE4D-887B-6B90D5F3FE9C.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/6FAB8F96-E7D2-D347-BEC7-80E611A8C3DC.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/66A43CD2-3111-BD4A-96A0-C1DD480AA0E3.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/5D22E35C-C0F6-084B-B6CF-7B2DD0AC10EA.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/5AF55B25-EC41-6941-8871-A4CF469E090A.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/54CC4A77-675E-0B4D-8A82-8CA7BB68679F.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/5340479F-DAA4-B14F-AC13-8A8235E9B07D.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/4B26E38A-B418-EE4A-8066-6AD326E384B1.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/4988E6BB-7FCC-5A49-85DA-244D36DC3376.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/402A6D66-6023-194F-A566-B606FDE2F3EC.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/3F25F659-DF08-114A-AB2B-AFC384081BAE.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/3D75A076-9C6B-4E42-BD25-B65D81DD11B4.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/3CF9272C-7D2E-C04A-AE46-F8AC2B782395.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/3A8C2BA7-EBC9-934B-B286-58CCE1086963.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/34EF9399-E327-CA43-B793-279DFDF1538B.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/3470B1E7-1233-424F-9E5E-E219FA59D30A.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/2CE11135-F98E-0444-97E1-8D91E514DF03.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/25D3B0B5-863C-F345-B0B1-6563DF7B14BA.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/13EAC214-EE88-9E4F-8370-82ADD0B9428E.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/0FBC086E-0502-DA40-8583-D960D9A7D304.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/0D6D3AB9-CE48-1047-AFDA-8FFE1988FF1C.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/0BB947B2-0C02-5D4B-969F-A427E058C765.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/0931C133-848D-EA40-BDC6-7CAB661B4D89.root',
                    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_3/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-DIGI-RAW/PU25ns_102X_upgrade2018_realistic_v12-v1/20000/017E3E8D-5BC7-314F-87ED-12654D0D5B30.root',
                    ] )

process.source = source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(400) )

### conditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')


### validation-specific includes
#process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("Validation.Configuration.postValidation_cff")
process.quickTrackAssociatorByHits.SimToRecoDenominator = 'reco'




########### configuration MultiTrackValidator ########
process.multiTrackValidator.associators = ['quickTrackAssociatorByHits']
#process.cutsRecoTracks.quality = ['','highPurity']
#process.cutsRecoTracks.quality = ['']
process.multiTrackValidator.label = ['cutsRecoTracks']
process.multiTrackValidator.histoProducerAlgoBlock.useLogPt = True
process.multiTrackValidator.histoProducerAlgoBlock.minPt = 0.1
process.multiTrackValidator.histoProducerAlgoBlock.maxPt = 3000.0
process.multiTrackValidator.histoProducerAlgoBlock.nintPt = 40
process.multiTrackValidator.UseAssociators = True
# process.multiTrackValidator.cores = cms.InputTag("ak4CaloJets")
# process.multiTrackValidator.ptMinJet= 500


#process.load("Validation.RecoTrack.cuts_cff")
#process.cutsRecoTracks.quality = ['highPurity']
#process.cutsRecoTracks.ptMin    = 0.5
#process.cutsRecoTracks.minHit   = 10
#process.cutsRecoTracks.minRapidity  = -1.0
#process.cutsRecoTracks.maxRapidity  = 1.0

process.quickTrackAssociatorByHits.useClusterTPAssociation = True
process.load("SimTracker.TrackerHitAssociation.tpClusterProducer_cfi")

process.validation = cms.Sequence(
    process.tpClusterProducer *
    process.quickTrackAssociatorByHits *
    process.multiTrackValidator
)

# paths
process.val = cms.Path(
      process.cutsRecoTracks
    * process.validation
)

# Output definition
process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('file:MTV_inDQM_400.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQM')
    )
)

process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)


process.schedule = cms.Schedule(
      process.val,process.endjob_step,process.DQMoutput_step
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(8),
    numberOfStreams = cms.untracked.uint32(8),
    wantSummary = cms.untracked.bool(True)
)
