#!/usr/bin/env python

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

from PerformanceCurvePlotter import *

options = VarParsing.VarParsing ()

options.register( 'signalFiles',
                  [os.cwd()],
                  VarParsing.VarParsing.multiplicity.list,
                  VarParsing.VarParsing.varType.string,
                  "Specify path(s) to signal files(s)"
                 )

options.register( 'backgroundFiles',
                  [],
                  VarParsing.VarParsing.multiplicity.list,
                  VarParsing.VarParsing.varType.string,
                  "Specify paths(s) to background files(s)"
                 )

options.register( 'referenceLabels',
                  [],
                  VarParsing.VarParsing.multiplicity.list,
                  VarParsing.VarParsing.varType.string,
                  "Specify labels for each sig/bkg pair"
                )

options.register( 'ValidationSequence',
                  'RunTancValidation',
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.sring,
                  "Specify the sequence of discriminators to plot.  They are defined \
                   in the NUMERATOR/DENOMINATOR section of RecoTauValidation_cff "
                 )


options.parseArguments()

if not hasattr(Validation.RecoTau.RecoTauValidation_cfi, options.ValidationSequence):
   print "Error: Validation sequence %s is not defined in Validation.RecoTau.RecoTauValidation_cfi!!" % options.ValidationSequence

filesAndLabels = zip(options.signalFiles, options.backgroundFiles, options.referenceLabels)

ReleaseToPlot = []

def checkFile(theFile):
   if not os.path.isfile(theFile):
      print "Can't stat file %s!" % theFile
      sys.exit()

for signalFile, backgroundFile, referenceLabel:
   checkFile(signalFile)
   checkFile(backgroundFile)
   toPlot = TauValidationInfo(signalFiles,
                              backgroundFile,
                              referenceLabels,
                              getattr(Validation.RecoTau.RecoTauValidation_cfi, options.ValidationSequence)
                              )
   ReleaseToPlot.append(toPlot)

if not os.path.exists("SummaryPlots"):
   os.mkdir("SummaryPlots)

myOutputFileName = os.path.join("SummaryPlots", "PerformanceCurve.png")

PlotPerformanceCurves(ReleaseToPlot, myOutputFileName)
