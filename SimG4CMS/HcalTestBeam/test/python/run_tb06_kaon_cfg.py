#!/usr/bin/env python

import os, sys

#====================== check input ==========================

VERSION = os.environ.get('CMSSW_VERSION')
if VERSION is None:
	print('''No environment CMSSW_VERSION''')
	sys.exit()

if len(sys.argv) < 5:
	print('Usage: python run.py PhysList Particle Fhcal VAR')
	print('Example: python run.py FTFP_BERT kaon- 106.5 RR')
	sys.exit()
else:
	phys = sys.argv[1]
	part = sys.argv[2]
	hcal = sys.argv[3]
	var  = sys.argv[4]

text = 'Start RUN for ' + phys + ' ' + part + ' ' + hcal + ' ' + var
print(text)

cmd = 'mkdir -p ' + VERSION
os.system(cmd)

fname = VERSION + '/' + phys + '_' + part + '_' + var + '_'

#====================== beam choice ==========================

listEkp   = ['2.06','3.08','4.03','5','6','7','8','9']
listPkp   = ['2','3','4','5','6','7','8','9']
listEkn   = ['2.55','3.08','4.03','5','6','7','8','9']
listPkn   = ['2.5','3','4','5','6','7','8','9']


pdg = '321'
geom = 'SimG4CMS.HcalTestBeam.TB2006GeometryXML_cfi'
listE = listEkp
listP = listPkp
ecal  = '1.01'
nE = int(8)
stat = '5000'

if(part == 'kaon-') :
  pdg = '-321'
  listE = listEkn
  listP = listPkn
  nE = int(8)

#====================== event loop ==========================

outf = fname + '.log'
cmd4 = 'rm -f ' + outf
os.system(cmd4)
cmd1 = 'echo "Start loop" > ' + outf
print(cmd1)
os.system(cmd1)

for i in range( nE) :
  fnametmp = fname + '.py'
  cmd3 = 'rm -f ' + fnametmp
  os.system(cmd3)
  pfile = open(fnametmp, "w")
  pfile.write('import FWCore.ParameterSet.Config as cms \n\n' )
  pfile.write('from Configuration.Eras.Modifier_h2tb_cff import h2tb   \n\n' )
  pfile.write('process = cms.Process("PROD", h2tb) \n\n' )
  pfile.write('process.load("' + geom + '") \n\n')
  pfile.write('from SimG4CMS.HcalTestBeam.TB2006Analysis_cfi import * \n')
  pfile.write('process = testbeam2006(process) \n\n')
  pfile.write('process.TFileService = cms.Service("TFileService", \n')
  pfile.write('  fileName = cms.string("' + fname + listP[i] + 'gev.root") \n')
  pfile.write(') \n\n')
  pfile.write('process.common_beam_direction_parameters.MinE = cms.double('+listE[i]+') \n')
  pfile.write('process.common_beam_direction_parameters.MaxE = cms.double('+listE[i]+') \n')
  pfile.write('process.common_beam_direction_parameters.PartID = cms.vint32('+pdg+') \n')
  pfile.write('process.common_beam_position_parameters.BeamPosition = cms.double(0.0) \n')
  pfile.write('process.generator.PGunParameters.MinE = process.common_beam_direction_parameters.MinE \n')
  pfile.write('process.generator.PGunParameters.MaxE = process.common_beam_direction_parameters.MaxE \n')
  pfile.write('process.generator.PGunParameters.PartID = process.common_beam_direction_parameters.PartID \n')
  pfile.write('process.VtxSmeared.MinE = process.common_beam_direction_parameters.MinE \n')
  pfile.write('process.VtxSmeared.MaxE = process.common_beam_direction_parameters.MaxE \n')
  pfile.write('process.VtxSmeared.PartID = process.common_beam_direction_parameters.PartID \n')
  pfile.write('process.VtxSmeared.BeamPosition = process.common_beam_position_parameters.BeamPosition \n')
  pfile.write('process.testbeam.MinE = process.common_beam_direction_parameters.MinE \n')
  pfile.write('process.testbeam.MaxE = process.common_beam_direction_parameters.MaxE \n')
  pfile.write('process.testbeam.PartID = process.common_beam_direction_parameters.PartID \n')
  pfile.write('process.testbeam.TestBeamAnalysis.EcalFactor = cms.double(' + ecal + ') \n')
  pfile.write('process.testbeam.TestBeamAnalysis.HcalFactor = cms.double(' + hcal + ') \n')
  pfile.write('process.maxEvents = cms.untracked.PSet( \n')
  pfile.write('  input = cms.untracked.int32(' + stat + ') \n')
  pfile.write(') \n\n')
  pfile.write('process.g4SimHits.Physics.type = "SimG4Core/Physics/' + phys + '" \n\n')
  pfile.write('process.g4SimHits.OnlySDs = ["CaloTrkProcessing", "EcalTBH4BeamDetector", "HcalTB02SensitiveDetector", "HcalTB06BeamDetector", "EcalSensitiveDetector", "HcalSensitiveDetector"] \n\n')
  pfile.close()
  cmd2 = 'cmsRun ' + fnametmp + ' >> ' + outf 
  os.system(cmd2)

#====================== end ==========================
