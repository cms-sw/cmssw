import FWCore.ParameterSet.Config as cms

process = cms.Process("DEDX")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100000) )

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

#process.GlobalTag.globaltag = 'START36_V9::All'
process.GlobalTag.globaltag = 'GR_P_V32::All'

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/028B3511-1EE9-E111-B583-BCAEC518FF6E.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/1207C121-1EE9-E111-8ADF-BCAEC518FF40.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/44D8F7C7-15E9-E111-B065-5404A63886B7.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/52722FD6-18E9-E111-AEA5-003048F117B6.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/821783CE-15E9-E111-A9BF-003048D2C108.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/881EF45F-1FE9-E111-811E-5404A63886D2.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/92AD89B7-16E9-E111-8F10-5404A63886AD.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/98761DD1-17E9-E111-8FB9-5404A63886D2.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/AC40A08E-1CE9-E111-8B6C-0025901D5D78.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/ACC4CFD2-1FE9-E111-8A26-001D09F24682.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/BA646C0C-23E9-E111-9028-BCAEC518FF65.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/D03C11D8-1EE9-E111-BF84-BCAEC532971C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/992/DE1AD6C6-15E9-E111-91F4-BCAEC532971B.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/083D56A4-6BE9-E111-97DA-BCAEC518FF8A.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/125A1F17-5EE9-E111-B7B3-485B3962633D.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/2E5FB03A-79E9-E111-B822-5404A63886A2.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/36C24BA1-7AE9-E111-A657-5404A63886EC.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/3A112009-69E9-E111-808A-0025901D5DF4.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/462A9301-92E9-E111-BDD4-001D09F24664.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/4AD893BC-88E9-E111-952A-003048D2BEA8.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/5240A04D-6CE9-E111-A04A-0025901D629C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/5AA0D0E5-72E9-E111-8AC6-5404A63886EC.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/5C7225E8-79E9-E111-BBAC-0025901D5D78.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/6C5A2D27-54E9-E111-AA1D-5404A63886A0.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/76D11743-36E9-E111-9F4A-0025901D629C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/76F1E4C0-57E9-E111-9A77-5404A63886AE.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/7AA6C062-7BE9-E111-B2D3-5404A6388698.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/7C45FFCF-83E9-E111-AADD-001D09F29169.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/82D25DE5-72E9-E111-A21D-5404A63886C0.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/90555F2F-9BE9-E111-AA94-003048D3C980.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/94E2151B-5EE9-E111-851E-E0CB4E5536AE.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/9C824037-60E9-E111-BE7E-003048CF99BA.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/9C898D7C-30E9-E111-A8FF-003048CF94A6.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/AC9B9261-42E9-E111-9CB1-0025901D6272.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/ACACE426-74E9-E111-B2EC-003048F1C420.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/B066A5DE-6FE9-E111-9B4C-BCAEC5364CFB.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/B4A4BD46-A3E9-E111-A427-001D09F23174.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/B60E7E5E-82E9-E111-B23E-001D09F25041.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/B6F81B3F-79E9-E111-A7BE-0025901D5E10.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/BC93AC60-7BE9-E111-94C3-5404A6388699.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/CED8E874-42E9-E111-BEB0-0025901D624A.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/D4093704-8DE9-E111-B82D-001D09F24DA8.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/D8DE7159-51E9-E111-A051-BCAEC5329716.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/DE01A433-8AE9-E111-9333-001D09F23C73.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/E87AFDF3-60E9-E111-987A-0025901D624E.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/FA9C966E-7DE9-E111-85C9-5404A6388697.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/991/FCF7B46F-56E9-E111-A10B-0025901D631E.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/990/1811AE76-2BE9-E111-8503-0025B3203898.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/990/1C28F563-22E9-E111-AC57-003048D2C01A.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/990/344BF455-28E9-E111-A41D-0025901D624A.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/990/7ABBE2EA-34E9-E111-BFC1-00237DDBE49C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/981/9E12677A-CEE8-E111-9949-003048CF9B28.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/978/3AACA5FA-CFE8-E111-BA58-BCAEC532971C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/976/9039F3D9-1EE9-E111-B86D-0025901D631E.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/976/908C0544-2CE9-E111-8821-003048D2BD66.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/976/BEDE514F-2CE9-E111-8F51-001D09F244DE.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/971/0E26C367-CAE8-E111-8887-5404A63886B1.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/967/CE6B52D4-C8E8-E111-B466-003048D2BA82.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/964/0E765D6C-CCE8-E111-BC3F-BCAEC532971B.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/961/AC7F9EE5-ECE8-E111-9369-0030486780AC.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/960/F8521E79-CEE8-E111-92A6-00237DDBE41A.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/943/3822CCD6-CCE8-E111-AACC-002481E0D73C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/941/3227205E-CFE8-E111-9FB2-5404A63886AB.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/939/4247FBAA-CCE8-E111-A6A0-0025901D5C86.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/937/38795FAB-CCE8-E111-A9A9-003048F1C424.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/935/CAD08147-CFE8-E111-A78D-E0CB4E553651.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/934/5A937961-CAE8-E111-B233-BCAEC5364C62.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/930/EC995EE7-D0E8-E111-8C6B-003048D2BB58.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/925/AA34E769-CCE8-E111-A9F3-003048F1110E.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/903/144363D4-CCE8-E111-98BA-003048D3750A.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/878/68961C90-C4E8-E111-B3DB-5404A638869C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/867/54F3F175-CAE8-E111-82A1-5404A63886AD.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/866/24A405F3-C2E8-E111-A7B7-BCAEC518FF8D.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/857/6822D842-C5E8-E111-BFB0-BCAEC518FF54.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/856/0E398701-C3E8-E111-9858-BCAEC518FF8A.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/852/4074A011-C3E8-E111-AF1E-5404A638869B.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/850/DE302AC8-C3E8-E111-A25E-003048D37580.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/849/584ED1CA-C3E8-E111-93E7-5404A640A648.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/849/8C12B918-C3E8-E111-840E-003048D2C020.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/849/903186C8-C3E8-E111-824E-0025901D624E.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/849/96249219-C3E8-E111-8BA9-002481E0D7D8.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/842/54FED8C7-C2E8-E111-B016-003048F117B6.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/840/D22AE5B5-BEE8-E111-8488-BCAEC532970D.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/839/A67AD167-BBE8-E111-8FE7-0025901D5D90.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/837/BA1C7B98-BCE8-E111-AB0A-003048D2C0F4.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/836/80BDBD6D-BBE8-E111-B3B2-5404A63886A0.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/834/447377A2-BCE8-E111-8F34-003048F024DE.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/832/EE446697-BCE8-E111-9A0D-5404A63886A2.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/830/96BA7390-BBE8-E111-90DA-BCAEC518FF54.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/829/82BEC17B-BAE8-E111-BE05-5404A63886CE.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/827/3E654DEA-B7E8-E111-83D4-BCAEC518FF6E.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/826/18C43693-BAE8-E111-A461-BCAEC518FF8E.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/823/843A4FFE-BDE8-E111-AC85-BCAEC518FF54.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/816/DEBBF544-B9E8-E111-941C-003048F118C6.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/812/5E4A1CFA-B5E8-E111-A5FB-BCAEC518FF65.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/807/40AE8993-C3E8-E111-A2EC-5404A63886AD.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/806/84A58DEC-B7E8-E111-92FF-BCAEC53296FB.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/803/42B2E045-B9E8-E111-B715-5404A63886CC.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/801/A02C0BE6-B6E8-E111-937E-5404A63886C0.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/800/323321BF-E9E7-E111-AF17-0025901D5E10.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/798/0444EA38-5AE8-E111-A03D-002481E0D7C0.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/798/6C0D601F-68E8-E111-8804-0025901D5D9A.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/798/9ED4EF1C-5AE8-E111-9DCE-001D09F242EF.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/798/A8C06BFD-6BE8-E111-8A43-BCAEC53296F4.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/798/CCC4E5B0-66E8-E111-B410-BCAEC518FF62.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/798/F23F0DAF-72E8-E111-96C3-BCAEC518FF7A.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/795/A470D761-71E9-E111-9B36-5404A63886AE.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/795/AC379701-C0E9-E111-BC06-0030486780EC.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/0CECCE3A-40E9-E111-BCCD-5404A63886C0.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/0E06CE5D-74EA-E111-982E-001D09F34488.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/229BA480-1EE9-E111-8A0B-0025901D5DF4.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/26762512-9DEC-E111-9444-001D09F2AD84.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/2C05EFF9-2CE9-E111-AF0F-003048F1C58C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/300B6CB3-41E9-E111-8D10-001D09F248F8.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/369466EC-A7E8-E111-A740-0025B324400C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/36A0F9FC-60E9-E111-A0A3-BCAEC5364CED.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/48FF7482-4AE8-E111-9D39-001D09F2B2CF.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/506BC1F7-55E8-E111-8196-003048F11942.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/529D5952-52E8-E111-80CB-001D09F2525D.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/5A0F3D75-96E7-E111-84ED-003048F11C58.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/5EF6DD16-94E9-E111-B787-003048F024F6.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/60D4A04A-36E9-E111-BA74-5404A640A642.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/742ADC82-95E8-E111-8A28-0030486780E6.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/86518682-84E9-E111-A0CD-001D09F242EF.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/88F30331-ECE8-E111-82E6-001D09F2841C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/8A9F9773-90E8-E111-9925-5404A63886AB.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/8AC46FF2-40E9-E111-A779-0025901D5DF4.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/9C7830F3-C8E8-E111-979F-BCAEC532971C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/9EABE1B6-F2E7-E111-8E32-003048D2C108.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/A2E1AA8A-30E7-E111-9CE5-002481E0DEC6.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/B204304A-5EE7-E111-A84B-0015C5FDE067.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/BE6894B8-32E9-E111-B362-001D09F2B2CF.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/DEE8C0E7-18E9-E111-A683-0025901D5DB8.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/E000341E-1EE9-E111-9684-003048F117B4.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/ECD42F01-06E8-E111-B5D6-003048F11112.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/F056819B-30E7-E111-9A8A-00237DDBE49C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/786/F6935249-85E9-E111-8C70-001D09F27067.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/779/3E274123-62E6-E111-BCF7-003048D2C174.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/777/14BB5EEB-53E6-E111-A972-001D09F29533.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/775/EE874D12-51E6-E111-ADB6-0015C5FDE067.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/773/4437F930-40E6-E111-9836-001D09F248F8.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/770/967BF9CC-2BE6-E111-A9DA-BCAEC518FF8D.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/767/C4E5BFE7-34E6-E111-AFB1-001D09F2960F.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/766/C64CFCDE-24E6-E111-B8E3-001D09F292D1.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/764/1A703CFE-2FE6-E111-ACB4-001D09F2B2CF.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/763/F61ADB98-2EE6-E111-8D89-001D09F24763.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/762/8EEB58FA-3BE6-E111-9B6B-001D09F27003.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/759/EA984082-2CE6-E111-8AAD-003048D37538.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/758/7670815E-1AE6-E111-81C4-5404A640A642.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/757/06BBD4FD-10E6-E111-A68F-001D09F2B30B.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/756/1E6EEFA2-08E6-E111-8435-001D09F2841C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/743/E4D4C72D-FBE5-E111-AEE4-0019B9F72D71.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/742/4A3E1DAD-F7E5-E111-9DC8-001D09F24FBA.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/739/964DC8DB-F4E5-E111-A6FC-001D09F23A20.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/738/1CDD8824-F4E5-E111-850C-0015C5FDE067.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/737/A817CDC0-F2E5-E111-B8C7-001D09F291D2.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/735/7475DCA2-F0E5-E111-9BD9-001D09F253C0.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/733/328DE5C1-F2E5-E111-BA2F-001D09F2305C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/732/B288A5A3-94E5-E111-84EE-001D09F290CE.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/726/E8687D8C-54E5-E111-B0D5-5404A63886A5.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/721/3E5C80DA-55E5-E111-A2AD-001D09F2A690.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/718/A63CA0C9-58E5-E111-8AD1-0025B32036D2.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/714/CAD99A71-54E5-E111-A25C-003048D2BA82.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/713/0CF81686-54E5-E111-9B41-003048F117F6.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/712/D4D7C47B-54E5-E111-8239-001D09F291D7.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/703/DC29BB8A-54E5-E111-B8C4-0025901D623C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/702/900D37C8-53E5-E111-B076-BCAEC518FF65.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/700/D48DF998-56E5-E111-9181-001D09F2983F.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/698/0CA4F43F-5CE5-E111-BDF1-5404A63886BB.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/696/9E91658C-56E5-E111-8852-003048D2BBF0.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/695/A43DA7A6-52E5-E111-932E-003048D2BD66.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/691/38750477-54E5-E111-8F17-003048F0258C.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/691/9616FF47-57E5-E111-870C-001D09F295FB.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/690/5AB21F4C-50E5-E111-A713-BCAEC518FF8F.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/689/2E4E5B2B-52E5-E111-8595-5404A63886EF.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/688/7CB81895-4FE5-E111-9FE1-5404A63886A8.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/687/6E95B8A1-4CE5-E111-83DD-5404A63886C3.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/679/40F0A87A-51E5-E111-9675-002481E0CC00.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/675/14A9731F-4BE5-E111-9ED3-E0CB4E55365D.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/666/6AC98975-4AE5-E111-AF87-00215AEDFD74.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/663/C420269B-4AE5-E111-BF2A-001D09F29146.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/660/5CE80C69-4AE5-E111-9E08-003048D2BEAA.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/658/109BC412-4BE5-E111-AA17-002481E0DEC6.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/602/7025C809-C1E4-E111-8A78-E0CB4E4408E3.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/601/9054FBF3-F4E4-E111-A317-5404A6388694.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/0EBEA811-2EE5-E111-BEB0-BCAEC518FF52.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/1CD52C2B-39E5-E111-BE53-485B39897227.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/221D8F92-3FE5-E111-B0F7-003048D2BEAA.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/22B30917-64E5-E111-BBB4-003048D2BBF0.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/22CD7C78-FDE4-E111-8AD9-E0CB4E55367F.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/28B2A0F6-42E5-E111-B8DC-003048D2BC30.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/32BCBDF3-5FE5-E111-82AA-001D09F292D1.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/348287A3-3CE5-E111-8942-5404A63886C4.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/38D6BC8E-78E5-E111-9435-001D09F2447F.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/44D680D4-58E5-E111-95B3-001D09F24763.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/4A9C725E-EEE4-E111-A2B8-BCAEC518FF74.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/4E6DB47D-71E5-E111-B6A9-001D09F24FEC.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/500634C3-3EE5-E111-A294-0025901D5DB2.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/68632CA0-67E5-E111-8B9B-001D09F2A690.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/769DD50C-37E5-E111-9574-00237DDC5CB0.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/7C931089-51E5-E111-9FB7-5404A63886A0.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/7E75B0E8-72E5-E111-ADC9-002481E0DEC6.root",
   "/store/data/Run2012C/MinimumBias/RECO/PromptReco-v2/000/200/600/8006FC84-84E5-E111-8526-002481E0DEC6.root",
   )
)

#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange(
#        '167898:108-167898:619',
#        '167898:621-167898:995',
#        '167898:1001-167898:1010',
#        '167898:1013-167898:1053',
#        '167898:1057-167898:1295',
#        '167898:1298-167898:1762',
#)

####################################################################################
#   BEAMSPOT + TRAJECTORY BUILDERS
####################################################################################



process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.TrackRefitter.src = 'generalTracks'


####################################################################################
#   DEDX ESTIMATORS 
####################################################################################

process.dedxHarm2 = cms.EDProducer("DeDxEstimatorProducer",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),

    estimator      = cms.string('generic'),
    exponent       = cms.double(-2.0),

    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(True),
    MeVperADCStrip = cms.double(3.61e-06*265),
    MeVperADCPixel = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration  = cms.bool(True),
    calibrationPath = cms.string("/afs/cern.ch/user/q/querten/workspace/public/dEdx/CMSSW_5_2_4/src/dEdx/ppGridProject/Gains.root"),
    ShapeTest       = cms.bool(False),
)

process.dedxTru40 = cms.EDProducer("DeDxEstimatorProducer",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),

    estimator      = cms.string('truncated'),
    fraction       = cms.double(0.4),

    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(True),
    MeVperADCStrip = cms.double(3.61e-06*265),
    MeVperADCPixel = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration  = cms.bool(True),
    calibrationPath = cms.string("/afs/cern.ch/user/q/querten/workspace/public/dEdx/CMSSW_5_2_4/src/dEdx/ppGridProject/Gains.root"),
    ShapeTest       = cms.bool(False),
)


process.dedxSPHarm2                   = process.dedxHarm2.clone()
process.dedxSPTru40                   = process.dedxTru40.clone()

process.dedxSTSPHarm2                 = process.dedxSPHarm2.clone()
process.dedxSTSPHarm2.ShapeTest       = cms.bool(True)

process.dedxSTSPTru40                 = process.dedxSPTru40.clone()
process.dedxSTSPTru40.ShapeTest       = cms.bool(True)


process.dedxSTNPHarm2                  = process.dedxHarm2.clone()
process.dedxSTNPHarm2.UsePixel         = cms.bool(False)

process.dedxSTNPTru40                  = process.dedxTru40.clone()
process.dedxSTNPTru40.UsePixel         = cms.bool(False)

process.dedxNPHarm2                = process.dedxSTNPHarm2.clone()
process.dedxNPHarm2.ShapeTest      = cms.bool(True)

process.dedxNPTru40                = process.dedxSTNPTru40.clone()
process.dedxNPTru40.ShapeTest      = cms.bool(True)

process.dedxPOHarm2                     = process.dedxHarm2.clone()
process.dedxPOHarm2.UseStrip            = cms.bool(False)

process.dedxPOTru40                     = process.dedxTru40.clone()
process.dedxPOTru40.UseStrip            = cms.bool(False)



####################################################################################
#   DEDX DISCRIMINATORS 
####################################################################################

process.dedxProd               = cms.EDProducer("DeDxDiscriminatorProducer",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),

    Reccord            = cms.untracked.string("SiStripDeDxMip_3D_Rcd"),
    Formula            = cms.untracked.uint32(0),
#    ProbabilityMode    = cms.untracked.string("Integral"),
    ProbabilityMode    = cms.untracked.string("Accumulation"),


    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*265),
    MeVperADCPixel     = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration     = cms.bool(True),
    calibrationPath    = cms.string("file:/afs/cern.ch/user/q/querten/workspace/public/dEdx/CMSSW_5_2_4/src/dEdx/ppGridProject/Gains.root"),
    ShapeTest          = cms.bool(False),

    MaxNrStrips        = cms.untracked.uint32(255)
)

#process.dedxSmi          = process.dedxProd.clone()
#process.dedxSmi.Formula  = cms.untracked.uint32(2)

process.dedxASmi         = process.dedxProd.clone()
process.dedxASmi.Formula = cms.untracked.uint32(3)


process.dedxSTProd                  = process.dedxProd.clone()
process.dedxSTProd.ShapeTest        = cms.bool(True)

#process.dedxSTSmi                   = process.dedxSmi.clone()
#process.dedxSTSmi.ShapeTest         = cms.bool(True)

process.dedxSTASmi                  = process.dedxASmi.clone()
process.dedxSTASmi.ShapeTest        = cms.bool(True)


process.dedxSTProdOld                  = process.dedxProd.clone()
process.dedxSTProdOld.ShapeTest        = cms.bool(True)
process.dedxSTProdOld.Reccord          = cms.untracked.string("SiStripDeDxProton_3D_Rcd")

process.dedxSTASmiOld                  = process.dedxASmi.clone()
process.dedxSTASmiOld.ShapeTest        = cms.bool(True)
process.dedxSTASmiOld.Reccord          = cms.untracked.string("SiStripDeDxProton_3D_Rcd")



process.dedxHitInfo               = cms.EDProducer("HSCPDeDxInfoProducer",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),

    Reccord            = cms.untracked.string("SiStripDeDxMip_3D_Rcd"),
    Formula            = cms.untracked.uint32(0),
#    ProbabilityMode    = cms.untracked.string("Integral"),
    ProbabilityMode    = cms.untracked.string("Accumulation"),

    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*265),
    MeVperADCPixel     = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string("file:Gains.root"),
    ShapeTest          = cms.bool(True),

    MaxNrStrips        = cms.untracked.uint32(255)
)



########################################################################

process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load("HLTrigger.HLTfilters.hltLevel1GTSeed_cfi")
process.bptxAnd = process.hltLevel1GTSeed.clone(L1TechTriggerSeeding = cms.bool(True), L1SeedsLogicalExpression = cms.string('0'))

process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'

process.hltcollision = cms.EDFilter("HLTHighLevel",
   TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
   HLTPaths = cms.vstring('HLT_L1Tech_BSC_minBias'), # provide list of HLT paths (or patterns) you want
#   HLTPaths = cms.vstring('HLT_Mu40_eta2p1*'), # provide list of HLT paths (or patterns) you want
   eventSetupPathsKey = cms.string(''),
   andOr              = cms.bool(True),
   throw              = cms.bool(False) 
)

process.oneGoodVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof >= 5 && abs(z) <= 15 && position.Rho <= 2"),  # old cut
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)

process.noScraping= cms.EDFilter("FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.2)  # to be updated
)



process.load("DPGAnalysis.SiStripTools.largesipixelclusterevents_cfi")
process.largeSiPixelClusterEvents.moduleThreshold = cms.untracked.int32(3500)
process.load("DPGAnalysis.SiStripTools.largesistripclusterevents_cfi")
process.largeSiStripClusterEvents.moduleThreshold = cms.untracked.int32(20000)


#import HLTrigger.HLTfilters.hltHighLevel_cfi
#process.Minbias = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#   andOr = True, ## choose logical OR between Triggerbits
#   eventSetupPathsKey = 'HLT_HIMinBiasBSC',
#   throw = False # tolerate triggers stated above, but not available
#)

#process.MinbiasFilter = cms.EDFilter( "TriggerResultsFilter",
#    triggerConditions = cms.vstring(
#      'HLT_L1Tech_BSC_minBias',
#    ),
#    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
#    l1tResults = cms.InputTag( "" ),
#    l1tIgnoreMask = cms.bool( False ),
#    l1techIgnorePrescales = cms.bool( False ),
#    daqPartitions = cms.uint32( 1 ),
#    throw = cms.bool( True )
#)

########################################################################

process.OUT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
#         "keep *_offlinePrimaryVertices_*_*",
#         "keep SiStripClusteredmNewDetSetVector_*_*_*",
#         "keep SiPixelClusteredmNewDetSetVector_*_*_*",
         "keep recoTracks_TrackRefitter_*_*",
#         "keep L1GlobalTriggerReadoutRecord_gtDigis_*_*",
#         "keep edmTriggerResults_TriggerResults_*_*",
#         "keep recoDeDxDataedmValueMap_*_*_DEDX",
         "keep *_dedxHitInfo_*_*" 
    ),
    fileName = cms.untracked.string('/tmp/querten/dedx.root'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('p')
    ),
)

########################################################################


#process.p = cms.Path(process.bptxAnd * process.oneGoodVertexFilter * process.noScraping * process.largeSiPixelClusterEvents * ~process.largeSiStripClusterEvents * process.offlineBeamSpot * process.TrackRefitter * process.dedxSPHarm2 * process.dedxSPTru40 * process.dedxSTSPHarm2 * process.dedxSTSPTru40 * process.dedxSTNPHarm2 * process.dedxSTNPTru40 * process.dedxProd * process.dedxSmi * process.dedxASmi *  process.dedxSTCNPHarm2 * process.dedxSTCNPTru40 * process.dedxSTProd * process.dedxSTSmi * process.dedxSTASmi * process.dedxPOHarm2 * process.dedxPOTru40)


#process.p = cms.Path(process.bptxAnd * process.hltPhysicsDeclared * process.hltcollision * process.oneGoodVertexFilter * process.noScraping * process.offlineBeamSpot * process.TrackRefitter * process.dedxSPHarm2 * process.dedxSPTru40 * process.dedxSTSPHarm2 * process.dedxSTSPTru40 * process.dedxSTNPHarm2 * process.dedxSTNPTru40 * process.dedxProd * process.dedxSmi * process.dedxASmi *  process.dedxNPHarm2 * process.dedxNPTru40 * process.dedxSTProd * process.dedxSTSmi * process.dedxSTASmi * process.dedxPOHarm2 * process.dedxPOTru40 * process.dedxPONoCS1Harm2 * process.dedxPONoCS1Tru40)

process.p = cms.Path(process.offlineBeamSpot * process.TrackRefitter * process.dedxHitInfo)
#process.p = cms.Path(process.offlineBeamSpot * process.TrackRefitter * process.dedxHitInfo)

process.outpath  = cms.EndPath(process.OUT)
process.schedule = cms.Schedule(process.p, process.outpath)


 
