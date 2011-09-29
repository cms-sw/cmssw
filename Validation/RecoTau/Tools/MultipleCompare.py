#! /usr/bin/env python

import FWCore.ParameterSet.Config as cms
import sys
import os
import math
import Validation.RecoTau.RecoTauValidation_cfi as validation
from optparse import OptionParser

__author__  = "Mauro Verzetti (mauro.verzetti@cern.ch)"
__doc__ = """Script to plot the content of a Validation .root file and compare it to a different file:\n\n
Usage: MultipleCompare.py -T testFile [options] [search strings that you want to apply '*' is supported as special character]"""


def GetContent(dir):
    tempList = dir.GetListOfKeys()
    retList = []
    for it in range(0,tempList.GetSize()):
       retList.append(tempList.At(it).ReadObj())
    return retList

def MapDirStructure( directory, dirName, objectList ):
    dirContent = GetContent(directory)
    for entry in dirContent:
        if type(entry) is TDirectory or type(entry) is TDirectoryFile:
            subdirName = os.path.join(dirName,entry.GetName())
            MapDirStructure(entry, subdirName,objectList)
        else:
            pathname = os.path.join(dirName,entry.GetName())
            objectList.append(pathname)

def Match(required, got):
    for part in required.split('*'):
        if got.find(part) == -1:
            return False
    return True

def Divide(hNum,hDen):
    ret = hNum.Clone('Division')
    ret.GetYaxis().SetTitle('Ratio')
    for binI in range(hNum.GetNbinsX()):
        denVal = hDen.GetBinContent(binI)
        denErr = hDen.GetBinError(binI)
        numErr = hNum.GetBinError(binI)
        numVal = hNum.GetBinContent(binI)
        if denVal == 0:
            ret.SetBinContent(binI,0)
            ret.SetBinError(binI,0)
        else:
            ret.SetBinContent(binI,numVal/denVal)
            if numVal==0:
                ret.SetBinError(binI,1)
            else:
                ret.SetBinError(binI,(numVal/denVal)*math.sqrt(math.pow(numErr/numVal,2) + math.pow(denErr/denVal,2) ) )
    return ret

def FindParents(histoPath):
    root = histoPath[:histoPath.find('_')]
    par = histoPath[histoPath.find('Eff')+3:]
    validationPlots = validation.TauEfficiencies.plots._Parameterizable__parameterNames
    found =0
    num = ''
    den = ''
    for efficiency in validationPlots:
        effpset = getattr(validation.TauEfficiencies.plots,efficiency)
        effName = effpset.efficiency.value()
        effNameCut = effName[effName.find('_'):effName.find('#')]
        if effNameCut in histoPath:
            if found == 1:
                print 'More than one pair of parents found for ' + histopath + ':'
                assert(False)
            num = root + effpset.numerator.value()[effName.find('_'):].replace('#PAR#',par)
            den = root + effpset.denominator.value()[effName.find('_'):].replace('#PAR#',par)
            found += 1
    return [num,den]

def Rebin(tfile, histoPath, rebinVal):
    parents = FindParents(histoPath)
    num = tfile.Get(parents[0])
    if type(num) != TH1F:
        print 'Looking for '+num
        print 'Plot now found! What the hell are you doing? Exiting...'
        sys.exit()
    denSingle = tfile.Get(parents[1])
    if type(denSingle) != TH1F:
        print 'Looking for '+denSingle
        print 'Plot now found! What the hell are you doing? Exiting...'
        sys.exit()
    num.Rebin(rebinVal)
    den = denSingle.Rebin(rebinVal,'denClone')
    retVal = num.Clone(histoPath+'Rebin%s'%rebinVal)
    #print 'Num : ' + parents[0]
    #print 'Den : ' +parents[1]
    #print "NumBins: %s DenBins: %s" % (num.GetNbinsX(), den.GetNbinsX() )
    retVal.Divide(num,den,1,1,'B')
    return retVal

parser = OptionParser(description=__doc__)
parser.add_option('--TestFile','-T',metavar='testFile', type=str,help='Sets the test file',dest='test',default = '')
parser.add_option('--RefFile','-R',metavar='refFile', type=str,help='Sets the reference file',dest='ref',default = '')
parser.add_option('--output','-o',metavar='outputFile', type=str,help='Sets the output file',dest='out',default = 'MultipleCompare.png')
parser.add_option('--logScale',action="store_true", dest="logScale", default=False, help="Sets the log scale in the plot")
parser.add_option('--fakeRate','-f',action="store_true", dest="fakeRate", default=False, help="Sets the fake rate options and put the correct label (implies --logScale)")
parser.add_option('--testLabel','-t',metavar='testLabel', type=str,help='Sets the label to put in the plots for test file',dest='testLabel',default = None)
parser.add_option('--refLabel','-r',metavar='refLabel', type=str,help='Sets the label to put in the plots for ref file',dest='refLabel',default = None)
parser.add_option('--maxLog',metavar='number', type=float,help='Sets the maximum of the scale in log scale (requires --logScale or -f to work)',dest='maxLog',default = 3)
parser.add_option('--minDiv',metavar='number', type=float,help='Sets the minimum of the scale in the ratio pad',dest='minDiv',default = 0.001)
parser.add_option('--maxDiv',metavar='number', type=float,help='Sets the maximum of the scale in the ratio pad',dest='maxDiv',default = 2)
parser.add_option('--logDiv',action="store_true", dest="logDiv", default=False, help="Sets the log scale in the plot")
parser.add_option('--rebin', dest="rebin", type=int, default=-1, help="Sets the rebinning scale")
#parser.add_option('--search,-s',metavar='searchStrings', type=str,help='Sets the label to put in the plots for ref file',dest='testLabel',default = None) No idea on how to tell python to use all the strings before a new option, thus moving this from option to argument (but may be empty)

(options,toPlot) = parser.parse_args()

from ROOT import *
gROOT.SetStyle('Plain')
gROOT.SetBatch()
gStyle.SetPalette(1)
gStyle.SetOptStat(0)
gStyle.SetPadGridX(True)
gStyle.SetPadGridY(True)
gStyle.SetOptTitle(0)
gStyle.SetPadTopMargin(0.1)


testFile = TFile(options.test)
refFile = None
if options.ref != None:
    refFile = TFile(options.ref)

ylabel =  'Fake rate' if options.fakeRate else 'Efficiency'

#Takes the position of all plots that were produced
plotList = []
MapDirStructure( testFile,'',plotList)

histoList = []
for plot in toPlot:
    for path in plotList:
        if Match(plot.lower(),path.lower()):
            histoList.append(path)

print histoList
legend = TLegend(0.6,0.83,0.6+0.39,0.83+0.17)
legend.SetFillColor(0)
header = ''
if options.testLabel != None:
    header += 'Dots: '+options.testLabel
if options.refLabel != None:
    header += ' Line: '+options.refLabel
legend.SetHeader(header)
canvas = TCanvas('MultiPlot','MultiPlot',validation.standardDrawingStuff.canvasSizeX.value(),832)
effPad = TPad('effPad','effPad',0,0.25,1.,1.,0,0)
effPad.Draw()
diffPad = TPad('diffPad','diffPad',0.,0.,1,.25,0,0)
diffPad.Draw()
colors = [2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1]
first = True
divHistos = []
testHs = []
refHs = []
for histoPath,color in zip(histoList,colors):
    if(options.rebin == -1):
        testH = testFile.Get(histoPath)
    else:
        testH = Rebin(testFile,histoPath,options.rebin)
    if type(testH) != TH1F:
        print 'Looking for '+histoPath
        print 'Test plot now found! What the hell are you doing? Exiting...'
        sys.exit()
    testHs.append(testH)
    xAx = histoPath[histoPath.find('Eff')+len('Eff'):]
    effPad.cd()
    if hasattr(validation.standardDrawingStuff.xAxes,xAx):
        testH.GetXaxis().SetTitle( getattr(validation.standardDrawingStuff.xAxes,xAx).xAxisTitle.value())
    testH.GetYaxis().SetTitle(ylabel)
    testH.GetYaxis().SetTitleOffset(1.2)
    testH.SetMarkerSize(1)
    testH.SetMarkerStyle(20)
    testH.SetMarkerColor(color)
    legend.AddEntry(testH,histoPath[histoPath.rfind('/')+1:histoPath.find('Eff')],'p')
    if first:
        first = False
        testH.GetYaxis().SetRangeUser(0.0,1.6)
        if options.logScale:
            effPad.SetLogy()
        testH.Draw('ex0')
        if ylabel=='Fake rate':
            testH.GetYaxis().SetRangeUser(0.001,options.maxLog)
            effPad.SetLogy()
            effPad.Update()
    else:
        testH.Draw('same ex0 l')
    if refFile == None:
        continue
    if(options.rebin == -1):
        refH = refFile.Get(histoPath)
    else:
        refH = Rebin(refFile,histoPath,options.rebin)
    if type(refH) != TH1F:
        continue
    refHs.append(refH)
    refH.SetLineColor(color)
    refH.SetLineWidth(1)
    refH.DrawCopy('same hist')
    refH.SetFillColor(color)
    refH.SetFillStyle(3001)
    refH.Draw('same e3')
    divHistos.append(Divide(testH,refH))

firstD = True
if refFile != None:
    for histo,color in zip(divHistos,colors):
        diffPad.cd()
        histo.SetMarkerSize(1)
        histo.SetMarkerStyle(20)
        histo.SetMarkerColor(color)
        histo.GetYaxis().SetLabelSize(0.08)
        histo.GetYaxis().SetTitleOffset(0.6)
        histo.GetYaxis().SetTitleSize(0.08)
        histo.GetXaxis().SetLabelSize(0.)
        histo.GetXaxis().SetTitleSize(0.)
        if firstD:
            histo.GetYaxis().SetRangeUser(options.minDiv,options.maxDiv)
            if options.logDiv:
                diffPad.SetLogy()
            histo.Draw('ex0')
            firstD = False
        else:
            histo.Draw('same ex0')
            diffPad.Update()
    effPad.cd()
    legend.Draw()
    canvas.Print(options.out)



