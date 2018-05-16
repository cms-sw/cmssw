import FWCore.ParameterSet.Config as cms

process = cms.Process("EBTPGTest")

process.load('Configuration.StandardSequences.Services_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.MessageLogger.categories = cms.untracked.vstring('EBPhaseIITPStudies', 'FwkReport')
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
   reportEvery = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

process.source = cms.Source("PoolSource",
                            fileNames= cms.untracked.vstring(
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/0035FBA5-FF77-E711-822C-7CD30AC036FE.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/006AA928-FE77-E711-BDFB-008CFA105EFC.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/00CD24A2-0578-E711-9C7C-0017A477106C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/0257446B-0478-E711-B38B-0CC47A5FC495.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/063E2BF1-FC77-E711-8819-002481DE47D0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/089FA56E-0178-E711-BE35-0CC47A5FC2A5.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/08B38309-0B78-E711-A412-002481CFDE08.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/08C06D58-ED77-E711-ADF7-0025B3E00C96.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/0A3D325E-F477-E711-9D87-A0369F836338.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/0CD24609-F877-E711-8E0A-00266CFFCD00.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/0E55EA85-0778-E711-B05E-0017A477104C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/100D82E8-FD77-E711-9D87-00266CFFC76C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/122E68C6-F377-E711-9B57-7CD30AC0372C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/144A2587-0278-E711-953F-C4346BBCB408.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/148540E7-F677-E711-A828-0017A477106C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/149DEC7B-FD77-E711-9C6C-C4346BC85718.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/168297D7-F677-E711-B2C7-0CC47A5FBE21.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/184A0F0C-0378-E711-B786-6CC2173D44D0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/18830496-0478-E711-B22E-002481CFE672.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/18930A34-0F78-E711-A5C8-ECB1D7B67E10.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/1A97784A-EE77-E711-9DAD-0017A4770C74.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/1A9F562E-ED77-E711-B351-0017A4770460.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/1CA5B2C3-FD77-E711-B98E-C4346BC7EDD8.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/1EDAD890-0578-E711-A5C8-6CC2173C3DD0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/20734259-ED77-E711-92B1-002481CFE25E.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/2092B6EF-FF77-E711-B3D0-6CC2173C9150.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/225EC0CD-F477-E711-8062-6CC2173C9150.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/24446CC1-0878-E711-9A7D-0CC47A5FBE25.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/2473B1A3-FD77-E711-91E7-00266CFFC948.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/2473B3FB-FD77-E711-8BB2-008CFAF28DCE.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/249998B6-F177-E711-A8CE-0025B3E01E66.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/24F6B958-0D78-E711-A0A2-A0369F836316.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/26C6AC49-F877-E711-94BE-6CC2173C39E0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/28046123-F877-E711-A0AC-A0369F8363F2.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/284E5F4F-0178-E711-84D4-C4346BBC1498.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/28559751-ED77-E711-B0B1-6CC2173C4580.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/28599585-FD77-E711-AFA6-002481DE4A28.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/28B8FAD9-F877-E711-A2F3-A0369F83627E.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/2A5C3FEE-EF77-E711-961F-002481CFC92C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/2ADBEBB9-0278-E711-A38F-C4346BC7EDD8.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/2CD904E7-FD77-E711-A722-A0369F8362E6.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/2CE7971F-FC77-E711-9ADD-0017A4770454.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/2EB4963E-EC77-E711-B65E-0017A4770468.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/2ECFC3DD-FF77-E711-BEF6-008CFAF28F22.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3029ECB8-FA77-E711-8F02-7CD30AC036FE.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/30A4B75F-0D78-E711-8D90-A0369F836372.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/323BDB6B-EF77-E711-A76B-002481DE4938.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3261BD28-F877-E711-87E8-A0369F836372.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/326D15C8-F977-E711-AFC2-002481CFE834.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/32CDC7E0-FB77-E711-A4E3-0CC47A5FA3B9.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/347BE7F0-FD77-E711-9E23-A0369F83642A.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/349BB9FF-0178-E711-B28A-00266CFFCA1C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/36A00229-F777-E711-B5E8-6CC2173D6E60.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3882EAD2-FE77-E711-9CFE-00266CFFC9C4.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3ACC3242-FB77-E711-A273-C4346BC70EC8.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3C40D485-EB77-E711-9177-0017A4770C64.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3C8C3684-FD77-E711-A076-0025B3E025B6.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3CAAD0D7-FB77-E711-BCA5-00266CFFC9EC.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3CDEDF6D-F077-E711-8361-90B11CBD0004.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3CED26F6-FD77-E711-B6B2-00266CFFBF84.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3E792C42-EC77-E711-96CA-0017A4770C70.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3E9740FD-FE77-E711-A4C9-C4346BC75558.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3EAE0403-0A78-E711-86A5-0017A4770C64.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/3EBAED36-0E78-E711-81AA-C4346BC08440.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/40FD2D65-FB77-E711-988D-0017A477045C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/420F3792-ED77-E711-BCDA-0017A4770454.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/42637D49-0578-E711-A869-6CC2173D5F20.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/426533C7-0678-E711-ABAA-008CFAF28DCE.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/42FA3705-0278-E711-8748-6CC2173D44D0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/446B9C9F-FE77-E711-A458-008CFAF28F0E.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/44B52865-FE77-E711-AAF2-008CFAEEABF8.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/46A64F92-FD77-E711-A73E-0017A477107C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/46A6525A-F677-E711-B718-6CC2173D44D0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/46BEE057-FD77-E711-94ED-90B11CBCFFF7.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/4869BE55-F777-E711-BAC1-A0369F8363F0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/48A4DC7F-0078-E711-89D7-00266CFFBC3C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/4A2E1043-FB77-E711-B989-6CC2173D4980.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/4A62764F-F577-E711-9E98-6CC2173C3E80.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/4A9B7983-0278-E711-A9E3-002481CFE80A.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/4C3DAC2F-FD77-E711-BDC4-A0369F83630C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/4C5F8ECF-FE77-E711-A29E-00266CFF0AF4.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/4CB89B7A-FC77-E711-879F-AC162DACB208.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/4EA23A83-FF77-E711-B83C-00266CFFCAC0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/5007477D-EC77-E711-97BC-0017A477045C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/54B2A055-0678-E711-875E-0017A4771040.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/56236C8A-ED77-E711-8BD2-002481DE4CC2.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/5688C4D8-FF77-E711-ADD8-A0369F83639C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/56B80B48-0F78-E711-816F-008CFAF28F0C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/5AD340C4-1178-E711-A457-00266CFFBF64.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/5C4DA7EB-0278-E711-8ED0-0017A4770C7C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/5C507D85-F977-E711-B4D6-6CC2173C3DD0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/5CFDBC51-0478-E711-A912-6CC2173C39E0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/5E1B4277-0278-E711-AF22-90B11CBCFFD0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/603D5037-FC77-E711-B7DD-78E7D1E4B772.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/604532EA-0478-E711-B237-00266CFFCCB4.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/60561EF2-FA77-E711-BC12-0017A4770470.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/605F3432-F177-E711-8FA7-0025B3E025B6.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/60752127-FE77-E711-B804-A0369F83641E.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/6437F03E-FC77-E711-BF13-7CD30AC03712.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/647FE9B3-EB77-E711-A244-002481CFE672.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/6A481C8D-ED77-E711-83B5-002481D2495A.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/6ABD9D7A-F077-E711-B0F8-002481DE47D0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/6E77176D-0278-E711-9B96-7CD30AC03722.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/6E8F1F1E-FF77-E711-9DE9-0CC47A5FC2A5.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/6EC3656A-FC77-E711-99FC-A0369F836288.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/72189D82-FE77-E711-93B4-AC162DACC3E8.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/723ADC86-FD77-E711-8072-00266CFEFE08.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/72560E27-0078-E711-8CC4-6CC2173C3E80.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/72D7A3B8-FD77-E711-A535-0025B3E01E66.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/72D7FB7F-FC77-E711-850E-002481CFE43E.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/740B6AC3-0578-E711-B954-6CC2173C4580.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/74BDF168-0878-E711-BDE7-0017A4771058.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/76450079-0078-E711-8616-008CFAEEAD4C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/7662F8E3-F877-E711-86DD-7CD30AC03722.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/784D6E70-F077-E711-BA88-90B11CBCFFD0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/789B7B7B-FD77-E711-B09C-008CFAF28F22.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/7A70DEB4-FD77-E711-9DD9-0025B3268672.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/7A9690BF-0678-E711-8E56-002481DE4CC2.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/7AF86A0E-FE77-E711-8DC1-008CFAF28F0C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/7C07F449-F677-E711-8A5C-008CFAF28DCE.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/7C8ACFC6-FC77-E711-AA4F-A0369F83639C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/7CE61601-0B78-E711-B2F3-0CC47A5FA3B9.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/7EC23DF3-F077-E711-B5D9-002481CFE648.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/80479469-0678-E711-9BED-002481CFE708.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/804D317E-FE77-E711-9980-A0369F8363BE.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/8051AECA-F977-E711-8075-002481DE4CC2.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/80E17369-EC77-E711-8B93-0CC47A5FC2A1.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/80F99156-FB77-E711-9241-0017A4770C70.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/829B17FF-EC77-E711-90F8-0CC47A5FA215.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/82DBBEB3-0078-E711-9E63-6CC2173C9150.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/8470724A-0678-E711-8270-1CC1DE046F00.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/8499DF40-0178-E711-A6AB-008CFAEEAD4C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/84B85C3E-F977-E711-8821-6CC2173D5F20.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/86107AD3-0578-E711-857F-A0369F836430.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/864620B1-EC77-E711-9D1B-90B11CBCFF5B.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/866582F7-FC77-E711-8129-00266CFFB7D0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/86F4B6A2-0678-E711-80FA-0017A4771054.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/8864E7FA-F777-E711-90AB-A0369F8363C2.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/887C3D34-F177-E711-B7BA-002481CFE26A.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/8AE874CE-F777-E711-A733-00266CFEFE70.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/8C99D24D-FE77-E711-863B-0CC47A5FBE35.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/8CAC830A-F677-E711-93FB-C4346BC8E730.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/8E2F6930-F277-E711-8DBB-90B11CBCFFA9.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/902928B7-F177-E711-AF72-0025B3E01F20.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9071372D-F277-E711-BBAA-90B11CBCFF75.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/90BFCD22-F077-E711-BEE5-0CC47A5FC619.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9282928A-FE77-E711-8183-A0369F8362E2.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9284293C-FF77-E711-AD98-008CFAF28F22.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/928B82F8-0578-E711-BE39-0CC47A5FC2A5.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/92F1240E-0178-E711-A966-90B11CBCFF8F.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9419FA5D-F877-E711-972F-ECB1D7B67E10.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/943EF8FB-0978-E711-B6B9-0CC47A5FC619.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/945A1708-F577-E711-9FF9-6CC2173D44D0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/94E8BADE-F677-E711-B103-0CC47A5FBDC1.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/961FFCF0-FB77-E711-8F23-0017A4770460.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/96226262-0878-E711-BC7B-0017A4771050.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/96401524-0278-E711-B86C-90B11CBCFF75.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/96717832-0F78-E711-8A2B-0CC47A5FC679.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/987A5355-F777-E711-A540-90B11CBCFF68.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9A46E2B0-F877-E711-B68F-0CC47A5FA215.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9A649573-F177-E711-96F5-002481DE4A28.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9A8D7B26-FD77-E711-B25F-0017A4770C74.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9C7A5322-0878-E711-8421-6CC2173D4980.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9C88039F-FF77-E711-9FB1-002481D2C9DE.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9C988C74-F077-E711-8502-0017A4770C7C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9CB1ABED-FD77-E711-9167-A0369F836316.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/9E3514F0-FD77-E711-88CC-A0369F83637E.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A07CFC25-0F78-E711-ABAB-0CC47A5FC61D.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A08D0186-FD77-E711-B732-002481CFE26A.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A0AAFCBD-FD77-E711-9B1A-00266CFFBE88.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A21AAEF5-F277-E711-9E2D-90B11CBCFF75.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A25626A9-EF77-E711-A152-002481DE485A.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A28BA427-FF77-E711-B245-1CC1DE0503C0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A290B32B-0F78-E711-A4CE-C4346BC70EC8.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A2AB0C15-F877-E711-97A8-90B11CBCFF41.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A2D803EC-F277-E711-AA5D-0CC47A5FC679.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A2DFA650-FD77-E711-ABE1-00266CFFC13C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A2E9CB63-ED77-E711-A834-0017A4771060.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A40CCDC0-F677-E711-9765-7CD30AC036FE.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A62E976A-EB77-E711-B66D-6CC2173D6E60.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A652961B-FE77-E711-A8DB-008CFAF5550C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A81D9BB5-F177-E711-BDE2-0025B3268672.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A861B1AE-FD77-E711-9D55-001E67E5E8B6.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/A8D7C4E8-FD77-E711-8B47-7CD30AC03722.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/AA0780BC-EC77-E711-9B32-0017A4770478.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/AAD7709A-F377-E711-BD68-A0369F836334.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/AAFA2FE5-FD77-E711-BBBA-AC162DACC328.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/AC09F7BB-0878-E711-A55E-0CC47A5FC2A1.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/AC19FFAC-EB77-E711-8545-7CD30AC03722.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/AC3FF006-F877-E711-A3D6-0CC47A5FC495.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/ACE1489A-FB77-E711-AD43-008CFAF28F0E.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/B061B811-0478-E711-B5CA-008CFA105EFC.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/B09AB70B-FF77-E711-9569-A0369F8363EE.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/B22D848C-1778-E711-BA51-A0369F83641E.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/B2C5D518-FC77-E711-89A0-0025B3268576.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/B2DB40A7-EC77-E711-BF07-ECB1D7B67E10.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/B471FD71-FA77-E711-8619-002481D2495A.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/B4E17B6B-0378-E711-827D-008CFAF28E5C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/B6F6B471-FD77-E711-84FD-008CFAEEACDC.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/BAA7BFB0-FE77-E711-9E9E-008CFAEEAD4C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/BADAE4EA-FD77-E711-AB51-A0369F83633E.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/BC21D265-0B78-E711-BC4F-0CC47A5FBE35.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/BCBD562E-F077-E711-B629-0025B3268576.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/BEBF279E-FB77-E711-98EA-00266CFFCB28.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/BEEBA5F4-F077-E711-B5B5-002481CFDE08.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C0B89CC2-FD77-E711-97FE-00266CFFCC18.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C0D896B4-EC77-E711-B0C0-0017A477047C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C2842DD1-FE77-E711-BD1A-00266CFFC948.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C45916AF-FC77-E711-AA16-1CC1DE1CE56C.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C46DC1E6-F677-E711-9D0F-C4346BC8C638.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C4AE867D-0478-E711-8428-ECB1D7B67E10.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C4D3BA9C-FA77-E711-8BFE-0017A4770C64.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C62A3D59-F477-E711-8EC6-6CC2173C9150.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C62B475B-EC77-E711-AB33-0017A4770470.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C62DD483-FD77-E711-919C-002481CFE648.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C6A8053B-FE77-E711-A8EF-7CD30AC0370E.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/C8B585CC-0F78-E711-BD8F-7CD30AC036FE.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/CA0D2B35-F877-E711-9BBC-008CFAF554D2.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/CA121A3C-F877-E711-AC58-00266CFFCAC0.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/CC02726F-0278-E711-81E0-0CC47A5FC679.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/CE703BBB-F077-E711-9B3E-90B11CBCFFEA.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/CE94731F-F977-E711-A9A9-6CC2173C4580.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/CEA1225D-FF77-E711-AF31-A0369F83627A.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/CEEA2A4C-0578-E711-8A96-0CC47A5FA215.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/D03266B0-FD77-E711-AE37-1CC1DE1D2004.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/D0BE513A-F277-E711-B815-90B11CBCFF8F.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/D218236C-F277-E711-A98A-002481D2C9DE.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/D291A80D-0778-E711-B194-AC162DA8E1E8.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/D2FDAABC-0778-E711-B2F7-0017A4771044.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/D42F3D6F-0278-E711-ABD0-00266CFFC980.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/D48E1690-0378-E711-B929-ECB1D79E5C40.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/D60B8E9E-EF77-E711-9E2B-0CC47A5FBE25.root',
'root://cms-xrd-global.cern.ch//store/mc/PhaseIITDRSpring17DR/SingleNeutrino/GEN-SIM-DIGI-RAW/PU200BX12_91X_upgrade2023_realistic_v3-v1/50000/D6359FFF-FD77-E711-B32E-00266CFEFC38.root'

)
)






# All this stuff just runs the various EG algorithms that we are studying
                         
# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '91X_upgrade2023_realistic_v3', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Choose a 2030 geometry!
#process.load('Configuration.Geometry.GeometryExtended2023simReco_cff') # Has CaloTopology, but no ECal endcap, don't use!
## Not existing in cmssw_8_1_0_pre16 process.load('Configuration.Geometry.GeometryExtended2023GRecoReco_cff') # using this geometry because I'm not sure if the tilted geometry is vetted yet
#process.load('Configuration.Geometry.GeometryExtended2023tiltedReco_cff') # this one good?

process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')

#process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
#XXX process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
#XXX process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')
#XXX process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC_cfi')
#XXX process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC_cfi')

#XXX process.load('Configuration/StandardSequences/L1HwVal_cff')
#XXX process.load('Configuration.StandardSequences.RawToDigi_cff')
#XXX #XXX process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")
#XXX 
#XXX 
#XXX process.slhccalo = cms.Path( process.RawToDigi)
#XXX 
#XXX 
#XXX # run L1Reco to produce the L1EG objects corresponding
#XXX # to the current trigger
#XXX process.load('Configuration.StandardSequences.L1Reco_cff')
#XXX process.L1Reco = cms.Path( process.l1extraParticles )
#XXX 
#XXX 
#XXX 
#XXX # --------------------------------------------------------------------------------------------
#XXX #
#XXX # ----    Produce the L1EGCrystal clusters (code of Sasha Savin)
#XXX 
#XXX # first you need the ECAL RecHIts :
#XXX process.load('Configuration.StandardSequences.Reconstruction_cff')
#XXX #process.bunchSpacingProducer = cms.EDProducer("BunchSpacingProducer")
#XXX #process.bsProd = cms.Path( process.bunchSpacingProducer )
#XXX #process.reconstruction_step = cms.Path( process.bunchSpacingProducer + process.hbheprereco + process.calolocalreco )
#XXX process.reconstruction_step = cms.Path( process.bunchSpacingProducer + process.hbheUpgradeReco + process.calolocalreco )




process.simEcalEBTriggerPrimitiveDigis = cms.EDProducer("EcalEBTrigPrimProducer",
#process.EcalEBTrigPrimProducer = cms.EDProducer("EcalEBTrigPrimProducer",
    BarrelOnly = cms.bool(True),
#    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis","","HLT"),
    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis","ebDigis"),
#    barrelEcalDigis = cms.InputTag("simEcalDigis","ebDigis"),
#    barrelEcalDigis = cms.InputTag("selectDigi","selectedEcalEBDigiCollection"),
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False),
    Famos = cms.bool(False),
    nOfSamples = cms.int32(1),
    binOfMaximum = cms.int32(6) ## optional from release 200 on, from 1-10

)


process.simEcalEBClusterTriggerPrimitiveDigis = cms.EDProducer("EcalEBCluTrigPrimProducer",
#process.EcalEBTrigPrimProducer = cms.EDProducer("EcalEBTrigPrimProducer",
    BarrelOnly = cms.bool(True),
    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis","","HLT"),
#    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis","ebDigis"),
#    barrelEcalDigis = cms.InputTag("simEcalDigis","ebDigis"),
#    barrelEcalDigis = cms.InputTag("selectDigi","selectedEcalEBDigiCollection"),
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False),
    Famos = cms.bool(False),
    nOfSamples = cms.int32(1),
    binOfMaximum = cms.int32(6), ## optional from release 200 on, from 1-10
    etaSize = cms.int32(2), # to build the 3x3 or 3x5 or whatever. the int is always size-1. So fra a 3x3, one needs to input 2,2 
    phiSize = cms.int32(2),
    hitNoiseCut = cms.double(0.175),
    etCutOnSeed = cms.double(0.4375) # 2.5x0.175 see Sasha slides
)


#process.pNancy = cms.Path( process.EcalEBTrigPrimProducer )
#process.pNancy = cms.Path( process.simEcalEBTriggerPrimitiveDigis*process.simEcalEBClusterTriggerPrimitiveDigis )
process.pNancy = cms.Path(process.simEcalEBClusterTriggerPrimitiveDigis )



process.Out = cms.OutputModule( "PoolOutputModule",
#    fileName = cms.untracked.string( "EBTP_PhaseII_RelVal2017.root" ),
    fileName = cms.untracked.string( "EBTP_PhaseII_SingleNeu_Clu3x3_PU200_unsupDigis_updatedNoiseCuts_10K.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring("keep *_EcalEBTrigPrimProducer_*_*",
                                           "keep *_TriggerResults_*_*",
                                           "keep *_ecalRecHit_EcalRecHitsEB_*",
                                           "keep *_simEcalDigis_ebDigis_*",
                                           "keep *_selectDigi_selectedEcalEBDigiCollection_*",
                                           "keep *_g4SimHits_EcalHitsEB_*",
                                           "keep *_addPileupInfo_*_*",
                                           "keep *_simEcalEBTriggerPrimitiveDigis_*_*",
                                           "keep *_simEcalEBClusterTriggerPrimitiveDigis_*_*",
                                           "keep *_genParticles_*_*",
                                           "keep *_gedGsfElectrons_*_*",
                                           "keep *_gedGsfElectronCores_*_*",
                                           "keep *_gedPhotons_*_*",
                                           "keep *_gedPhotonCore_*_*",
                                           "keep recoCaloClusters_particleFlowEGamma_EBEEClusters_*",
                                           "keep recoSuperClusters_particleFlowEGamma_*_*",
                                           "keep SimTracks_g4SimHits_*_*",
                                           "keep SimVertexs_g4SimHits_*_*"

  )
)

process.end = cms.EndPath( process.Out )



#print process.dumpPython()
#dump_file = open("dump_file.py", "w")
#


