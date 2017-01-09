import sys,os

# miniaod_version = "RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9*"
# filename = 'datasets_MCRUN2_25ns.txt'

# miniaod_version = "RunIISpring15MiniAODv2-74X_mcRun2_asymptotic_v2-v*"
# filename = 'datasets_MCRUN2_25ns_RunIISpring15MiniAODv2.txt'

# miniaod_version = "RunIIFall15MiniAODv1-PU25nsData2015v1_76X_mcRun2_asymptotic_v12-v*"
# filename = 'datasets_MCRUN2_25ns_RunIIFall15MiniAODv1-PU25nsData2015v1_76X.txt'

# miniaod_version = "RunIIFall15MiniAODv2-PU25nsData2015v1_76X_mcRun2_asymptotic_v12-v*"
# filename = 'datasets_MCRUN2_25ns_RunIIFall15MiniAODv2-PU25nsData2015v1_76X.txt'

# # - FullSim, Without trigger information at all
# miniaod_version = "RunIISpring16MiniAODv2-PUSpring16_80X_mcRun2_asymptotic_2016_miniAODv2*"
# filename = 'datasets_MCRUN2_25ns_RunIISpring16MiniAODv2-PUSpring16_80X_noHLT.txt'

# # - FullSim, With trigger information
# miniaod_version = "RunIISpring16MiniAODv2-PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v*"
# filename = 'datasets_MCRUN2_25ns_RunIISpring16MiniAODv2-PUSpring16_80X_reHLT.txt'
# # These above datasets replace
# # /*/RunIISpring16MiniAODv2-PUSpring16RAWAODSIM_80X_mcRun2_asymptotic_2016_miniAODv2_v0-v*/MINIAODSIM 

# - FullSim, Without trigger information at all
miniaod_version = "*RunIISummer16MiniAODv2*-PUMoriond17_80X*"
filename = 'RunIISummer16MiniAODv2-PUMoriond17_80X.txt'

# REMOVE DATASET NAMES CONTAINING:

remove = [
          'prime','GravToZZ','GravitonToGluonGluon','GravitonToQuarkQuark','GravToGG','GravToWW','ToHHTo2B',
          'SUSY','QstarTo','RSGluonTo','WRTo','TstarTstar','Unpart','LQTo','BstarTo','WpWpJJ','WZTo3LNu',
          'HToZZ','HToWW','HToG','HToT','/ADD','/GJet','GluGluToZZ','TTbarDM','HToInvisible','WToENu_M','WToMuNu_M','WToTauNu_M',
          'ttHJetToGG','ttHJetToTT','Muminus_Pt','/Muplus','Photon','SinglePion','ZZTo4L','DoubleElectron',
          'SingleEta','tGamma','JPsiToMuMu','JpsiToMuMu','mtop1','BdToJpsiKs','tZq_','GG_M',
          'DYJetsToLL_M-1000to1500','DYJetsToLL_M-100to200','DYJetsToLL_M-10to50','DYJetsToLL_M-1500to2000',
          'DYJetsToLL_M-2000to3000','DYJetsToLL_M-400to500','DYJetsToLL_M-500to700','DYJetsToLL_M-500to700',
          'DYJetsToLL_M-200to400','DYJetsToLL_M-700to800','DYJetsToLL_M-800to1000','BuToJpsiK','GluGluHToZG','ZZTo2L2Nu',
          'GGJets','Monotop_S','TTJets_Mtt-','TT_Mtt-','BBbarDM','DarkMatter','GluGlu_LFV','WW_DoubleScattering','HToMuMu',
          'UpsilonMuMu','BsToJpsiPhi','HToMuTau','HToZG','SingleMuMinusFlatPt','DYJetsToLL_M-5to50','HToETau',
          'BulkGravTohhTohtatahbb_narrow_M-1600','BulkGravTohhTohtatahbb_narrow_M-2000',
          'BulkGravTohhTohtatahbb_narrow_M-2500','BulkGravTohhTohtatahbb_narrow_M-1800',
          'BulkGravTohhTohtatahbb_narrow_M-3000','BulkGravTohhTohtatahbb_narrow_M-3500','BulkGravTohhTohtatahbb_narrow_M-4000',
          'BulkGravTohhTohtatahbb_narrow_M-4500','HToEMu','X53X53_M','LongLivedChi0','SMS-T1','WZTo1L3Nu',
          'ZToEE_NNPDF30_13TeV-powheg_M_6000','ZToMuMu_NNPDF30_13TeV-powheg_M_800','ZToEE_NNPDF30_13TeV-powheg_M_1400',
          'ZToEE_NNPDF30_13TeV-powheg_M_3500','ZToEE_NNPDF30_13TeV-powheg_M_400','ZToMuMu_NNPDF30_13TeV-powheg_M_120',
          'ZToMuMu_NNPDF30_13TeV-powheg_M_4500','ZToMuMu_NNPDF30_13TeV-powheg_M_50','ZToMuMu_NNPDF30_13TeV-powheg_M_6000',
          'scaledown','scaleup','WGToLNuG','RadionToZZ','RadionToWW','ZToEE_NNPDF30_13TeV-powheg_M_200','WWTo2L2Nu',
          'ZToEE_NNPDF30_13TeV-powheg_M_2300','ZToEE_NNPDF30_13TeV-powheg_M_800','ZToMuMu_NNPDF30_13TeV-powheg_M_200',
          'ZToMuMu_NNPDF30_13TeV-powheg_M_3500','ZToMuMu_NNPDF30_13TeV-powheg_M_400','RadionTohhTohVVhbb','ChargedHiggs',
          'RadionTohhTohaahbb','/RadionTohhTohtatahbb','ZToMuMu_NNPDF30_13TeV-powheg_M_1400','TGJets','WWJJToLNuQQ',
          'ZToEE_NNPDF30_13TeV-powheg_M_120','ZToEE_NNPDF30_13TeV-powheg_M_4500','VVTo2L2Nu','ZToMuMu_NNPDF30_13TeV-powheg_M_2300',
          'AToZhToLLTauTau','RPVresonantToEMu','XXTo4J','Taustar_TauG','DM_Pseudoscalar','DM_Scalar','InclusivectoMu',
          'BdToKstarMuMu','Estar_EG','ZGTo2LG','Mustar_MuG','Estar_EG','InclusivebtoMu','GluGluHToEEG','InclusiveBtoJpsi',
          '/DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM',
          'BlackHole_','DMS_','DMV_','/DsTo','AxialMonoW_Mphi','X53X53To2L2Nu','VectorMono','AxialMono','StringBall',
          'top_5f_DS_','ExtendedWeakIsospinModel','SMS-T2tt','SinglePi','SingleNeutrin','H0ToUps','SLQ_Rhanded-MLQ',
          'ContinToZZ','SingleNeutron','BuToJpsiPi','Chib0ToUps','DYToLL_M_1_T','EEG_PTG','GluGluWWTo2E2Nu','HToZATo2L2Tau',
          'MinBias_chMulti85','MuMuG_PTG130To400','SingleK0','WLLJJ','WZJJ','ChMulti85','Chib0ToUps','DYToEE_NNPDF30','ZToEE_NNPDF',
          'GluGluToHiggs0PMToZZ','/TTTT_','WWJJToLNuLNu_EWK_QCD_noTop','SeesawTypeIII','RPVStopStop','gluinoGMSB_M',
          'EWKZ2Jets','GluGluSpin0ToZG','DYJetsToEE_M-50_LTbinned','WGstarToL','WmWmJJ_','NMSSM_HToAATo4Mu_M',
          'RSGravTohhTohVVhbbToVVfullLep_narrow_M','SMS-T2bH_mSbottom','GluGluSpin0ToGG_W','GluGluToPhiToTTBar','rToZZ','ToZZ_',
          'Graviton2PBTo','X53ToTW_M','WZJToLLLNu','ToZZTo','ContinZZTo','barToWW','HTo4L','HPlusPlus','ATo2L2B','ATo2Nu2B',
          'ToTauTau','TTToHplusToWA','VBFDM_M','VectorDiJet','Vector_Mono','HToSSTo','X53TTo','Spin0_ggPh','Scalar_Mono',
          'SMM_Mono','RPVresonantTo','SMS-T2','RSGravTo','Pseudo_Mono','Htautau','NonthDMMonoJet','MonoHaa','MonoHZZ','MonoHWW',
          'Gluino','Branon','Axial_Mono','ZNuNuG','FPortDMMonoJet','G1Jet','HZZ4L','H1H','SMS-','TTGamma','ZPrimeTo',
          'ZnunuGamma_Spin2',
          # '/ttHJetToNonbb_M120_13TeV_amcatnloFXFX_madspin_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM',
          # '/ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM',
          # '/ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9_ext1-v3/MINIAODSIM',
          # '/ttHJetToNonbb_M130_13TeV_amcatnloFXFX_madspin_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM',
          ]

# FILELIST OF AVAILABLE DATASETS ON DAS AS VALID

# das_valid = [line.rstrip('\n').rstrip('\r') for line in open('all_datasets_MCRUN2_25ns.txt')]
# das_valid = os.popen('python ./das_client.py --limit=0 --query="dataset=/*/'+miniaod_version+'/MINIAODSIM"').read().split('\n')
das_valid = os.popen('/cvmfs/cms.cern.ch/common/das_client --limit=0 --query="dataset=/*/'+miniaod_version+'/MINIAODSIM"').read().split('\n')
das_valid = filter(None, das_valid)

if len(das_valid)<1: 
  print "ERROR WHILE EVALUATING VALID DATASETS, EMPTY LIST, ABORTING"
  sys.exit(1)
  
# FILELIST OF AVAILABLE DATASETS ON DAS AS PRODUCTION

das_production = os.popen('python ./das_client.py --limit=0 --query="dataset dataset=/*/'+miniaod_version+'/MINIAODSIM status=PRODUCTION"').read().split('\n')
das_production = filter(None, das_production)

# FILTER DATASETS WITH REMOVE LIST
print '\nretrieved DAS lists contain',len(das_valid),'datasets in VALID state,',len(das_production),'datasets in PRODUCTION state'

for i in remove:
  das_valid = [ x for x in das_valid if i not in x ]
  das_production = [ x for x in das_production if i not in x ]

print '\nWARNING: filtering datasets containing',remove
print '\nfiltered lists contain',len(das_valid),'datasets in VALID state,',len(das_production),'datasets in PRODUCTION state'

# CHECK EXISTING LIST OF DATASETS TO BE PROCESSED

vhbb_all = open(filename).read()

print 'HBB filelist ',filename,'contains',len(filter(None, vhbb_all.split('\n'))),'datasets'

print '\n=======================================================================\n'
print 'DATASETS AVAILABLE ON DAS AS VALID AND NOT (YET) INCLUDED IN THE HBB LIST\n'
for line in das_valid:
  if line not in vhbb_all:
    print line
    
vhbb_prod = []
print '\nDATASETS AVAILABLE ON DAS AS PRODUCTION AND NOT (YET) INCLUDED IN THE HBB LIST\n'
for line in das_production:
  if line not in vhbb_all:
    print line
  if line in vhbb_all:
    vhbb_prod.append(line)

print '\n=========================================================\n'
print 'DATASETS INCLUDED IN THE HBB LIST AND STILL IN PRODUCTION\n'
for line in vhbb_prod:
    print line

print '\n======================================================================================================\n'
print 'DATASETS INCLUDED IN THE HBB LIST NOT IN PRODUCTION NOR IN VALID STATE (i.e. REMOVE FROM THE LIST!!!) \n'
for line in filter(None, vhbb_all.split('\n')):
  if (line not in das_production) and (line not in das_valid):
    print line

print '\n'
