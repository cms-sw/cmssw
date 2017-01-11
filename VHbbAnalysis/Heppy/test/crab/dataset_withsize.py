#!/usr/bin/env python
"""

 DBS 3 Client Example.   This script is called 

./dbs3Example.py '/*/*Fall13-POST*/GEN-SIM'

"""
import re
import operator
import  sys,time
from dbs.apis.dbsClient import DbsApi


if (len(sys.argv) != 2):
   print " Need one argument (the dataset pattern) - like \"/*/HIRun2015*/RAW\""
   sys.exit(2)



#from time import gmtime
url="https://cmsweb.cern.ch/dbs/prod/global/DBSReader"
api=DbsApi(url=url)

unit = 1./1000/1000/1000/1000
unitName = 'TB'

totmap = {}

sizetot=0
nevents=0

dataset = sys.argv[1]

print " Summing dataset sizes for pattern ", dataset
oldd=dataset

ds= api.listDatasets(dataset=dataset)
l=len(ds)
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
          'ToTauTau',
          # '/ttHJetToNonbb_M120_13TeV_amcatnloFXFX_madspin_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM',
          # '/ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM',
          # '/ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9_ext1-v3/MINIAODSIM',
          # '/ttHJetToNonbb_M130_13TeV_amcatnloFXFX_madspin_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM',
          ]

for i,dataset in enumerate(ds):
   name = dataset['dataset']
   if i % 100 == 0 :
     print i,"/",l
   skip=False
   for r in remove :
       if re.match(".*"+r+".*",name) :
 #         print "Skip"
          skip=True
          break
   if skip :
    continue
  # print "Sum"
   size = api.listBlockSummaries(dataset=name)
   sizetot = sizetot+ size[0]['file_size']
   nevents = nevents + size[0]['num_event']
#   print '--- dataset: ',name,size[0]['file_size']*unit, unitName
   totmap[name] = (size[0]['file_size']*unit,size[0]['num_event'])
print "Sum Done"   

sorted_x = sorted(totmap.items(), key=operator.itemgetter(0))

#print sorted_x

print "Dataset , Size ,  unitName, Events "
for a in sorted_x:
#   print " NAME", a, " SIZE",b
#   print "Dataset ", a[0], " Size ", a[1][0], unitName, " Events ", a[1][1],
   print  a[0], " , ", a[1][0],",", unitName,",", a[1][1]

print "============"
print " Summing dataset sizes for pattern ", oldd
print " TOT", sizetot*unit, unitName, ' Events ',nevents, ' Avg Size ', sizetot*1./nevents, ' bytes'



