#! /usr/bin/env python

import FWCore.ParameterSet.Config as cms
import sys
import os
import math
import re
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

def DetermineHistType(name):
  #automatically derive all plot types in the future?
  type = ''
  #assuming plots name like: tauType_plotType_xAxis or tauType_plotType_selection
  matches = re.match(r'(.*)_(.*)_(.*)', name)
  if matches:
    knowntypes = (['pTRatio','SumPt','Size'])
    for knowntype in knowntypes:
      if matches.group(2) == knowntype:
        type = knowntype
    if not type:  #there are plots labelled ..._vs_...
      type = 'Eff'
  else:
    type = 'Eff'

  #print 'type is ' + type
  return [type, matches.group(3)]

def DrawTitle(text):
	title = TLatex()
	title.SetNDC()
	title.SetTextAlign(12)#3*10=right,3*1=top
	title.SetTextSize(.035)	
	leftMargin = gStyle.GetPadLeftMargin()
	topMargin = 1 - 0.5*gStyle.GetPadTopMargin()
	title.DrawLatex(leftMargin, topMargin, text)

def DrawBranding():
  if options.branding != None:
    text = TLatex()
    text.SetNDC();
    text.SetTextAlign(11)#3*10=right,3*1=top
    text.SetTextSize(.025)
    text.SetTextColor(13)
    if options.out.find(".eps")!=-1:
      text.SetTextAngle(-91.0)#eps BUG
    else:
      text.SetTextAngle(-90.0)
    rightMargin = 1 - gStyle.GetPadRightMargin()
    topMargin = 1 - gStyle.GetPadTopMargin()
    text.DrawLatex(rightMargin+.01, topMargin+0.025, options.branding);


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
#not needed as automatically defined# parser.add_option('--ptResolution','-p',action="store_true", dest="ptRes", default=False, help="Sets the pt resolution options and put the correct label (normalizes the plot also)")
parser.add_option('--testLabel','-t',metavar='testLabel', type=str,help='Sets the label to put in the plots for test file',dest='testLabel',default = None)
parser.add_option('--refLabel','-r',metavar='refLabel', type=str,help='Sets the label to put in the plots for ref file',dest='refLabel',default = None)
parser.add_option('--maxLog',metavar='number', type=float,help='Sets the maximum of the scale in log scale (requires --logScale or -f to work)',dest='maxLog',default = 3)
parser.add_option('--minDiv',metavar='number', type=float,help='Sets the minimum of the scale in the ratio pad',dest='minDiv',default = 0.001)
parser.add_option('--maxDiv',metavar='number', type=float,help='Sets the maximum of the scale in the ratio pad',dest='maxDiv',default = 2)
parser.add_option('--logDiv',action="store_true", dest="logDiv", default=False, help="Sets the log scale in the plot")
parser.add_option('--normalize',action="store_true", dest="normalize", default=False, help="plot normalized")
parser.add_option('--maxRange',metavar='number',type=float, dest="maxRange", default=1.6, help="Sets the maximum range in linear plots")
parser.add_option('--rebin', dest="rebin", type=int, default=-1, help="Sets the rebinning scale")
parser.add_option('--branding','-b',metavar='branding', type=str,help='Define a branding to label the plots (in the top right corner)',dest='branding',default = None)
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
gStyle.SetPadBottomMargin(0.1)
gStyle.SetPadLeftMargin(0.13)
gStyle.SetPadRightMargin(0.07)


testFile = TFile(options.test)
refFile = None
if options.ref != None:
    refFile = TFile(options.ref)

#Takes the position of all plots that were produced
plotList = []
MapDirStructure( testFile,'',plotList)

histoList = []
for plot in toPlot:
    for path in plotList:
        if Match(plot.lower(),path.lower()):
            histoList.append(path)

print histoList

if len(histoList)<1:
  print '\tError: Please specify at least one histogram.'
  sys.exit()


#WARNING: For now the hist type is assumed to be constant over all histos.
histType = DetermineHistType(histoList[0])[0]
pTResMode = False
if histType=='pTRatio':
  pTResMode = True

ylabel = 'Efficiency'

if options.fakeRate:
  ylabel = 'Fake rate'
elif pTResMode:
  ylabel = 'a.u.'

#legend = TLegend(0.6,0.83,0.6+0.39,0.83+0.17)
x1 = 0.65
x2 = 1-gStyle.GetPadRightMargin()
y2 = 1-gStyle.GetPadTopMargin()
lineHeight = .025
if len(histoList) == 1:
  lineHeight = .05
y1 = y2 - lineHeight*len(histoList)
legend = TLegend(x1,y1,x2,y2)
legend.SetFillColor(0)
if pTResMode:
  y2 = y1
  y1 = y2 - .05*len(histoList)
  statsBox = TPaveText(x1,y1,x2,y2,"NDC")
  statsBox.SetFillColor(0)
  statsBox.SetTextAlign(12)#3*10=right,3*1=top
  statsBox.SetMargin(0.05)
  statsBox.SetBorderSize(1)

    
canvas = TCanvas('MultiPlot','MultiPlot',validation.standardDrawingStuff.canvasSizeX.value(),832)
effPad = TPad('effPad','effPad',0,0.25,1.,1.,0,0)
effPad.SetBottomMargin(0.1);
effPad.SetTopMargin(0.1);
effPad.SetLeftMargin(0.13);
effPad.SetRightMargin(0.07);
effPad.Draw()
header = ''
if options.testLabel != None:
  header += 'Dots: '+options.testLabel
if options.refLabel != None:
  header += ' Line: '+options.refLabel
DrawTitle(header)
DrawBranding()
#legend.SetHeader(header)
diffPad = TPad('diffPad','diffPad',0.,0.,1,.25,0,0)
diffPad.Draw()
colors = [2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1]
first = True
divHistos = []
statTemplate = '%s Mean: %.3f RMS: %.3f'
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
    if pTResMode:
        xAx = 'PFtau p_{T}/genVis p_{T} '
    effPad.cd()
    if not testH.GetXaxis().GetTitle():  #only overwrite label if none already existing
      if hasattr(validation.standardDrawingStuff.xAxes,xAx):
        testH.GetXaxis().SetTitle( getattr(validation.standardDrawingStuff.xAxes,xAx).xAxisTitle.value())
    if not testH.GetYaxis().GetTitle():  #only overwrite label if none already existing
      testH.GetYaxis().SetTitle(ylabel)
    testH.GetXaxis().SetTitleOffset(1.1)
    testH.GetYaxis().SetTitleOffset(1.5)
    testH.SetMarkerSize(1)
    testH.SetMarkerStyle(20)
    testH.SetMarkerColor(color)
    if histType == 'Eff':
      legend.AddEntry(testH,histoPath[histoPath.rfind('/')+1:histoPath.find(histType)],'p')
    else:
      legend.AddEntry(testH,DetermineHistType(histoPath)[1],'p')
    if pTResMode:
        text = statsBox.AddText(statTemplate % ('Dots',testH.GetMean(), testH.GetRMS()) )
        text.SetTextColor(color)
    if first:
        first = False
        #testH.GetYaxis().SetRangeUser(0.0,options.maxRange)
        if options.logScale:
            effPad.SetLogy()
        testH.Draw('ex0')
        if options.normalize or pTResMode:
          if testH.GetEntries() > 0:
            testH.DrawNormalized('P')
        if ylabel=='Fake rate':
            testH.GetYaxis().SetRangeUser(0.001,options.maxLog)
            effPad.SetLogy()
            effPad.Update()
    else:
        if options.normalize or pTResMode:
          if testH.GetEntries() > 0:
            testH.DrawNormalized('same p')
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
    if options.normalize or pTResMode:
      if testH.GetEntries() > 0:
        refH.DrawNormalized('same hist')
      text = statsBox.AddText(statTemplate % ('Line',refH.GetMean(), refH.GetRMS()) )
      text.SetTextColor(color)
    else:
        refH.DrawCopy('same hist')
    refH.SetFillColor(color)
    refH.SetFillStyle(3001)
    if not (options.normalize or pTResMode):
        refH.Draw('same e3')
        divHistos.append(Divide(testH,refH))
    else:
        entries = testH.GetEntries()
        testH.Sumw2()
        if entries > 0:
          testH.Scale(1./entries)
        entries = refH.GetEntries()
        refH.Sumw2()
        if entries > 0:
          refH.Scale(1./entries)
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
    if pTResMode:
        statsBox.Draw()
    canvas.Print(options.out)



