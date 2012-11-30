#! /usr/bin/env python

import sys
import os
import re
from ROOT import *
import MultipleCompare as MultipleCompare


__author__  = "Lars Perchalla (lars.perchalla@cern.ch)"
__doc__ = """Script to execute multiple plotting commands via MultipleCompare.py. Switch between massiveMode producing a set of plots comparing each one by one, and defaultMode producing a smaller set of default plot combinations by adding the commandline option massiveMode:\n\n
Usage: SteerMultipleCompare.py -T testFile -R refFile [options] [search strings that you want to apply '*' is supported as special character]
  see MultiCompare.py for details
  """

def StripPath(name):
  path = ''
  plot = ''
  matches = re.match(r'(.*)\/(.*)$', name)
  if matches:
    path = matches.group(1)
    plot = matches.group(2)
  return [path, plot]

def CreateDirectory(dir,addToExisting=False):
  if os.path.exists(dir) and not addToExisting:
    print "Output directory %s already exists!  OK to overwrite?" % dir
    while True:
      input = raw_input("Please enter [y/n] ")
      if (input == 'y'):
        break
      elif (input == 'n'):
        print " ...exiting."
        sys.exit()
  if not os.path.exists(dir):
    os.makedirs(dir)

def CreateBaseDirectory(options):
  if options.out == 'MultipleCompare.png' or options.out.find('.')!=-1:
    #default case, so no directory was given
    #or a filename was given
    outputDirName = 'MultipleCompareOutput'
  else:
    outputDirName = options.out
  outputDir = os.path.join(os.getcwd(), outputDirName)
  CreateDirectory(outputDir)
  return outputDir

def CreateSubDirectory(basedir, path):
  outputDir = os.path.join(basedir, path)
  CreateDirectory(outputDir,True)

def CleanArguments(argv, option):
  #remove existing output arguments
  while argv.count(option) > 0:
    index = argv.index(option)
    if index < len(argv)-1:
      argv.pop(index+1)#drop the corresponding value
    argv.pop(index)#drop the option itself


#execute Multicompare for each plot as a comparison one by one
#argv was modified to contain only one plot each
def plotOneByOne(argv, outputDir, histoList, histoSubNames, paths):
  for hist, name, path in zip(histoList, histoSubNames, paths):
    CreateSubDirectory(outputDir, path)
    #now give modified arguments to MultipleCompare
    tmpArgv = argv[:]
    tmpArgv.append('-o')
    tmpArgv.append(outputDir+'/'+path+'/'+name+'.png')
    tmpArgv.append(hist)
    MultipleCompare.main(tmpArgv)

def plotDefault(argv, outputDir, name, type, plots, addArgv=[]):
  tmpArgv = argv[:]
  tmpArgv.append('-o')  
  tmpArgv.append(outputDir+'/'+name+type)
  tmpArgv.extend(addArgv)
  tmpArgv.extend(plots)
  MultipleCompare.main(tmpArgv)

#make some default plots grouping several histograms
def plotDefaults(argv, options, outputDir):
  name = 'Validation_'
  if options.testLabel != None:
    name += options.testLabel+'_'
  else:
    name += options.test+'_vs_'
  if options.refLabel != None:
    name += options.refLabel+'_'
  else:
    name += options.ref+'_'
  outputType = '.eps'
  additionalArgv = []
  if outputDir.find('QCD')!=-1:
    additionalArgv.append('-f') #fakerate
  plotDefault(argv, outputDir, name, 'LeptonRejectionEffphi'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*Rejection/*Effphi'], additionalArgv)
  plotDefault(argv, outputDir, name, 'LeptonRejectionEffeta'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*Rejection/*Effeta'], additionalArgv)
  plotDefault(argv, outputDir, name, 'LeptonRejectionEffpt'+outputType,  ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*Rejection/*Effpt'], additionalArgv)

  if outputDir.find('QCD')!=-1:
    additionalArgv.append('--logScale')
  plotDefault(argv, outputDir, name, 'Effphi'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*Effphi', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*Effphi'], additionalArgv)
  plotDefault(argv, outputDir, name, 'Effeta'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*Effeta', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*Effeta'], additionalArgv)
  plotDefault(argv, outputDir, name, 'Effpt'+outputType,  ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*Effpt', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*Effpt'], additionalArgv)

  plotDefault(argv, outputDir, name, 'pTRatio_allHadronic'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*_pTRatio_allHadronic', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*_pTRatio_allHadronic'])
  plotDefault(argv, outputDir, name, 'pTRatio_oneProng1Pi0'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*_pTRatio_oneProng1Pi0', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*_pTRatio_oneProng1Pi0'])
  plotDefault(argv, outputDir, name, 'pTRatio_threeProng0Pi0'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*_pTRatio_threeProng0Pi0', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*_pTRatio_threeProng0Pi0'])

  plotDefault(argv, outputDir, name, 'Size_isolationPFChargedHadrCands'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*_Size_isolationPFChargedHadrCands', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*_Size_isolationPFChargedHadrCands'])
  plotDefault(argv, outputDir, name, 'Size_isolationPFNeutrHadrCands'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*_Size_isolationPFNeutrHadrCands', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*_Size_isolationPFNeutrHadrCands'])
  plotDefault(argv, outputDir, name, 'Size_isolationPFGammaCands'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*_Size_isolationPFGammaCands', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*_Size_isolationPFGammaCands'])

  plotDefault(argv, outputDir, name, 'SumPt_isolationPFChargedHadrCands'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*_SumPt_isolationPFChargedHadrCands', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*_SumPt_isolationPFChargedHadrCands'])
  plotDefault(argv, outputDir, name, 'SumPt_isolationPFNeutrHadrCands'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*_SumPt_isolationPFNeutrHadrCands', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*_SumPt_isolationPFNeutrHadrCands'])
  plotDefault(argv, outputDir, name, 'SumPt_isolationPFGammaCands'+outputType, ['DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/*_SumPt_isolationPFGammaCands', 'DQMData/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationBy*CombinedIsolationDBSumPtCorr/*_SumPt_isolationPFGammaCands'])


def main(argv=None):
  if argv is None:
    argv = sys.argv
  
  options, toPlot = MultipleCompare.LoadCommandlineOptions(argv)
      
  gROOT.SetBatch()

  testFile = TFile(options.test)
  refFile = None
  if options.ref != '':
    refFile = TFile(options.ref)

  plotList = []
  MultipleCompare.MapDirStructure( testFile,'',plotList)

  if len(toPlot)<1:
    print '\tSteerMultipleCompare:Error! Please specify at least one histogram. The following ones are available in the root file.'
    print "\n".join(plotList)
    sys.exit()

  histoList = []
  histoSubNames = []
  paths = []
  massiveMode = False
  for plot in toPlot:
    #clean the arguments. toPlot contains the list of positional arguments leftover after parsing options
    argv.remove(plot)
    for path in plotList:
      if MultipleCompare.Match(plot.lower(),path.lower()):
        histoList.append(path)
        strippedPath, strippedPlot = StripPath(path)
        paths.append(strippedPath)
        histoSubNames.append(strippedPlot)
        #print histoSubNames[-1]
      elif plot.find('massiveMode') != -1:
        massiveMode = True

  CleanArguments(argv,'--output')
  CleanArguments(argv,'-o')
          
  outputDir = CreateBaseDirectory(options)

  if massiveMode:
    print "Massive mode: scan all subdirs and make plots comparing each histogram one by one."
    plotOneByOne(argv, outputDir, histoList, histoSubNames, paths)          
  else:
    print "Default mode: Make default plot combinations."
    plotDefaults(argv, options, outputDir)


#only execute main() if manually run
if __name__ == '__main__':
  #main(*sys.argv[1:])
  # the calls to sys.exit(n) inside main() all become return n.
  sys.exit(main())
else:
  print "This is ",__name__

