#! /usr/bin/env python

import FWCore.ParameterSet.Config as cms
import sys
import os
import math
import re
import Validation.RecoTau.RecoTauValidation_cfi as validation
from optparse import OptionParser
from ROOT import *

__author__  = "Mauro Verzetti (mauro.verzetti@cern.ch)"
__doc__ = """Script to plot the content of a Validation .root file and compare it to a different file:\n\n
Usage: MultipleCompare.py -T testFile -R refFile [options] [search strings that you want to apply '*' is supported as special character]"""

def LoadCommandlineOptions(argv):
  sys.argv = argv
  parser = OptionParser(description=__doc__)
  parser.add_option('--myhelp',metavar='', action="store_true",help='prints this output message',dest='help',default = False)
  parser.add_option('--TestFile','-T',metavar='testFile', type=str,help='Sets the test file',dest='test',default = '')
  parser.add_option('--RefFile','-R',metavar='refFile', type=str,help='Sets the reference file',dest='ref',default = None)
  parser.add_option('--output','-o',metavar='outputFile', type=str,help='Sets the output file',dest='out',default = 'MultipleCompare.png')
  parser.add_option('--logScale',action="store_true", dest="logScale", default=False, help="Sets the log scale in the plot")
  parser.add_option('--fakeRate','-f',action="store_true", dest="fakeRate", default=False, help="Sets the fake rate options and put the correct label (implies --logScale)")
  parser.add_option('--testLabel','-t',metavar='testLabel', type=str,help='Sets the label to put in the plots for test file',dest='testLabel',default = None)
  parser.add_option('--refLabel','-r',metavar='refLabel', type=str,help='Sets the label to put in the plots for ref file',dest='refLabel',default = None)
  parser.add_option('--maxX', metavar='number', type=float, help='Sets the maximum of the x scale in the ratio pad',dest='maxX')
  parser.add_option('--minDiv',metavar='number', type=float, help='Sets the minimum of the y scale in the ratio pad',dest='minDiv')
  parser.add_option('--maxDiv',metavar='number', type=float, help='Sets the maximum of the y scale in the ratio pad',dest='maxDiv')
  parser.add_option('--minMain',metavar='number',type=float, help='Sets the minimum of the y scale in the main pad (auto-defined if no user input)', dest='minMain', default=-1)
  parser.add_option('--maxMain',metavar='number', type=float, help='Sets the maximum of the y scale in the main pad in linear scale (auto-defined if no user input)', dest='maxMain', default=-1)
  parser.add_option('--maxLog', metavar='number', type=float, help='Sets the maximum of the y scale in the main pad in log scale (requires --logScale or -f to work)',dest='maxLog')
  parser.add_option('--logDiv',action="store_true", dest="logDiv", default=False, help="Sets the log scale in the plot")
  parser.add_option('--normalize',action="store_true", dest="normalize", default=False, help="plot normalized")
  parser.add_option('--rebin', dest="rebin", type=int, default=-1, help="Sets the rebinning scale")
  parser.add_option('--branding','-b',metavar='branding', type=str,help='Define a branding to label the plots (in the top right corner)',dest='branding',default = None)
  #parser.add_option('--search,-s',metavar='searchStrings', type=str,help='Sets the label to put in the plots for ref file',dest='testLabel',default = None) No idea on how to tell python to use all the strings before a new option, thus moving this from option to argument (but may be empty)  
  
  (options,toPlot) = parser.parse_args()
  if options.help:
    parser.print_help()
    sys.exit(0)
  return [options, toPlot]

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
  label = ''
  prefix = ''
  #assuming plots name like: tauType_plotType_xAxis or tauType_plotType_selection
  matches = re.match(r'.*/(.*)_(.*)_(.*)', name)
  if matches:
    prefix = matches.group(1) 
    label = matches.group(3)
    knowntypes = (['pTRatio','SumPt','Size'])
    for knowntype in knowntypes:
      if matches.group(2) == knowntype:
        type = knowntype
    if not type:  #there are plots labelled ..._vs_...
      type = 'Eff'
  else:
    type = 'Eff'

  prefixParts = prefix.partition('Discrimination')
  if prefixParts[2] != '':
    prefix = prefixParts[2]
    prefixParts = prefix.partition('By')
    if prefixParts[2] != '':
      prefix = prefixParts[2]

  #print 'type is ' + type
  return [type, label, prefix]

def DrawTitle(text):
	title = TLatex()
	title.SetNDC()
	title.SetTextAlign(12)#3*10=right,3*1=top
	title.SetTextSize(.035)	
	leftMargin = gStyle.GetPadLeftMargin()
	topMargin = 1 - 0.5*gStyle.GetPadTopMargin()
	title.DrawLatex(leftMargin, topMargin, text)

def DrawBranding(options, label=''):
  if options.branding != None or label != '':
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
    if label!='':
      label += ': '
    text.DrawLatex(rightMargin+.01, topMargin+0.025, label+options.branding);


def FindParents(histoPath):
    root = histoPath[:histoPath.find('_')]
    par = histoPath[histoPath.find('Eff')+3:]
    validationPlots = validation.proc.efficiencies.plots._Parameterizable__parameterNames
    found =0
    num = ''
    den = ''
    for efficiency in validationPlots:
        effpset = getattr(validation.proc.efficiencies.plots,efficiency)
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

def findRange(hists, min0=-1, max0=-1):
  if len(hists) < 1:
    return
  #auto ranges if no user value provided
  min = min0
  max = max0
  if min0 == -1 or max0 == -1:
    for hist in hists:
      if min0 == -1:
        #Divide() sets bin to zero if division not possible. Ignore these bins.
        minTmp = getMinimumIncludingErrors(hist)
        if minTmp < min or min == -1:
          min = minTmp
      if max0 == -1:
        maxTmp = getMaximumIncludingErrors(hist)
        if maxTmp > max or max == -1:
          max = maxTmp
  return [min, max]

def optimizeRangeMainPad(options, pad, hists):
  pad.Update()
  if pad.GetLogy() and options.maxLog:
    maxLog = options.maxLog
  else:
    maxLog = options.maxMain
  min, max = findRange(hists, options.minMain, maxLog)
  if pad.GetLogy():
    if min == 0:
      min = 0.001
    if max < 2:
      max = 2. #prefere fixed range for easy comparison
  else:
    if min < 0.7:
      min = 0. #start from zero if possible
    if max <= 1.1 and max > 0.7:
      max = 1.2 #prefere fixed range for easy comparison
  hists[0].SetAxisRange(min, max, "Y")
  if options.maxX:
    hists[0].SetAxisRange(hists[0].GetXaxis().GetXmin(), options.maxX, "X")

def optimizeRangeSubPad(options, hists):
  min = -1
  max = -1
  if options.minDiv:
    min = options.minDiv
  if options.maxDiv:
    max = options.maxDiv
  min, max = findRange(hists, min, max)
  if max > 2:
    max = 2 #maximal bound
  hists[0].SetAxisRange(min, max, "Y")
  if options.maxX:
    hists[0].SetAxisRange(hists[0].GetXaxis().GetXmin(), options.maxX, "X")


def getMaximumIncludingErrors(hist):
#find maximum considering also the errors
  distance = 1.
  max = -1
  pos = 0
  for i in range(1, hist.GetNbinsX()):
    if hist.GetBinContent(i) > max:#ignore errors here
      max = hist.GetBinContent(i)
      pos = i
  return max + distance*hist.GetBinError(pos)

def getMinimumIncludingErrors(hist):
  #find minimum considering also the errors
  #ignoring zero bins
  distance = 1.
  min = -1
  pos = 0
  for i in range(1, hist.GetNbinsX()):
    if hist.GetBinContent(i)<=0.:#ignore errors here
      continue
    if hist.GetBinContent(i) < min or min==-1:
      min = hist.GetBinContent(i)
      pos = i
      if min < 0:
        min = 0  
  return min - distance*hist.GetBinError(pos)

def setupROOT():
  """ setup ROOT parameters """
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

def drawPlots(testFile, refFile, matchedHists, matchedHistDict, commonLabelDict, options):
  """ do the actual drawing """
  #WARNING: For now the hist type is assumed to be constant over all histos.
  histType, label, prefix = DetermineHistType(matchedHists[0])
  #decide whether or not to scale to an integral of 1
  #should usually not be used most plot types. they are already normalized.
  scaleToIntegral = False
  if options.normalize:
    scaleToIntegral = True

  ylabel = 'Efficiency'
  if options.fakeRate:
    ylabel = 'Fake rate'
  drawStats = False
  if histType=='pTRatio' and len(matchedHists)<3:
    drawStats = True

  #legend = TLegend(0.6,0.83,0.6+0.39,0.83+0.17)
  x1 = 0.3
  x2 = 1-gStyle.GetPadRightMargin()
  y2 = 1-gStyle.GetPadTopMargin()
  lineHeight = .04
  y1 = y2 - lineHeight*len(matchedHists)
  legend = TLegend(x1,y1,x2,y2)
  #legend.SetHeader(label)
  legend.SetFillColor(0)
  if drawStats:
    y2 = y1
    y1 = y2 - .07*len(matchedHists)
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
  DrawBranding(options)
  diffPad = TPad('diffPad','diffPad',0.,0.,1,.25,0,0)
  diffPad.Draw()
  colors = [2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1]
  first = True
  divHistos = []
  statTemplate = '%s Mean: %.3f RMS: %.3f'
  testHs = []
  refHs = []
  for histoPath,color in zip(matchedHists, colors):
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
    if not testH.GetXaxis().GetTitle():  #only overwrite label if none already existing
      if hasattr(validation.standardDrawingStuff.xAxes,xAx):
        testH.GetXaxis().SetTitle( getattr(validation.standardDrawingStuff.xAxes,xAx).xAxisTitle.value())
    if not testH.GetYaxis().GetTitle():  #only overwrite label if none already existing
      testH.GetYaxis().SetTitle(ylabel)
    if label!='':
      testH.GetXaxis().SetTitle(label+': '+testH.GetXaxis().GetTitle())
    testH.GetXaxis().SetTitleOffset(1.1)
    testH.GetYaxis().SetTitleOffset(1.5)
    testH.SetMarkerSize(1)
    testH.SetMarkerStyle(20)
    testH.SetMarkerColor(color)
    legendLabel = ""
    if histoPath in commonLabelDict.keys():
      legendLabel = commonLabelDict[histoPath]
    else:
      if histType == 'Eff':
        legendLabel = histoPath[histoPath.rfind('/')+1:histoPath.find(histType)]
      else:
        legendLabel = DetermineHistType(histoPath)[2]
    legend.AddEntry(testH, legendLabel, 'p')
    if drawStats:
        text = statsBox.AddText(statTemplate % ('Dots',testH.GetMean(), testH.GetRMS()) )
        text.SetTextColor(color)
    if first:
        first = False
        if options.logScale:
            effPad.SetLogy()
        if scaleToIntegral:
          if testH.GetEntries() > 0:
            if not testH.GetSumw2N():
              testH.Sumw2()
              testH.DrawNormalized('ex0 P')
            else:
              print "--> Warning! You tried to normalize a histogram which seems to be already scaled properly. Draw it unscaled."
              scaleToIntegral = False
              testH.Draw('ex0')
        else:
          testH.Draw('ex0')
    else:
        if scaleToIntegral:
          if testH.GetEntries() > 0:
            testH.DrawNormalized('same p')
        else:
            testH.Draw('same ex0 l')
    if refFile == None:
        continue
    refPath = histoPath
    if histoPath in matchedHistDict.keys():
      refPath = matchedHistDict[histoPath]
    if(options.rebin == -1):
        refH = refFile.Get(refPath)
    else:
        refH = Rebin(refFile, refPath, options.rebin)
    if type(refH) != TH1F:
        continue
    refHs.append(refH)
    refH.SetLineColor(color)
    refH.SetLineWidth(1)
    if scaleToIntegral:
      if testH.GetEntries() > 0:
        refH.DrawNormalized('same hist')
    else:
        refH.DrawCopy('same hist')
    if drawStats:
      text = statsBox.AddText(statTemplate % ('Line',refH.GetMean(), refH.GetRMS()) )
      text.SetTextColor(color)
    refH.SetFillColor(color)
    refH.SetFillStyle(3001)
    if scaleToIntegral:
        entries = testH.GetEntries()
        if entries > 0:
          testH.Scale(1./entries)
        entries = refH.GetEntries()
        refH.Sumw2()
        if entries > 0:
          refH.Scale(1./entries)
    #refH.Draw('same e3')
    refH.Draw('same')
    divHistos.append(Divide(testH,refH))

  tmpHists = []
  tmpHists.extend(testHs)
  tmpHists.extend(refHs)
  optimizeRangeMainPad(options, effPad, tmpHists)
  
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
            if options.logDiv:
                diffPad.SetLogy()
            histo.Draw('ex0')
            firstD = False
        else:
            histo.Draw('same ex0')
            diffPad.Update()
    optimizeRangeSubPad(options, divHistos)

    effPad.cd()
  legend.Draw()
  if drawStats:
    statsBox.Draw()
  canvas.Print(options.out)

def determineHistoList(testFile, refFile, plotPattern):
  """scan all histograms of testFile. returns dictionary of histograms that match to plotPattern.
     usually key=value, but using plotPattern like [[ref][test]] allows to define different histogram names for ref and test files"""
  #determin pattern

  plotListTest = [] # all hists contained in test file
  plotListRef = []  # all hists contained in ref file (only tested if files differ and different pattern provided)
  MapDirStructure(testFile,'',plotListTest)
  parseRefFile = False
  if testFile.GetPath()==refFile.GetPath():
    #no need to parse the file twice
    plotListRef = plotListTest # identical lists
    #print "same file at", testFile.GetPath()
  else:
    parseRefFile = True
    #print "two different files"

  matchedHists = [] # list of all hists that could be matched to pattern (i.e. test pattern)
  matchedHistDict = {} # dictionary of only these hists (test: ref) that have been matched to individual pattern for test and ref
  commonLabelDict = {} # dictionary of only these labels (test: common label) from hists that have been matched to individual pattern for test and ref
  for pattern in plotPattern:
    match = re.match(r'(.*)\[\[(.*)\]\[(.*)\]\](.*)', pattern)
    if not match:
      # common histogram names for ref and test files
      print "Apply common pattern to both files:", pattern
      countMatches = len(matchedHists)
      for path in plotListTest:
        if Match(pattern.lower(), path.lower()):
          matchedHists.append(path)
      print "...added", len(matchedHists)-countMatches, "histogram(s)"
    else:
      # different histogram names for ref and test files
      if parseRefFile:
        MapDirStructure(refFile,'',plotListRef)
        parseRefFile = False # only once
      commonPatternLeft = match.group(1) # left common part
      rPattern = match.group(2)
      tPattern = match.group(3)
      commonPatternRight = match.group(4) # right common part
      print "Apply individual pattern:", pattern
      countMatches = len(matchedHists)
      for path in plotListTest:
        if Match((commonPatternLeft+tPattern+commonPatternRight).lower(), path.lower()):
          matchedHists.append(path)
          #find corresponding match in refFile to current explicit pattern
          refPattern = path.replace(tPattern, rPattern)
          for pathRef in plotListRef:
            if pathRef == refPattern:
              matchedHistDict[path] = pathRef
              commonLabelDict[path] = commonPatternLeft.strip('*')
              #print "insert label", commonLabelDict[path]
              break
          if not path in matchedHistDict.keys():
            print "ERROR in determineHistoList! Could not find the corresponding reference plot\n", refPattern
            print "test plot is\n", path
      print "...added", len(matchedHists)-countMatches, "histogram(s)"
  #print "matchedHists", matchedHists
  #print "matchedHistDict", matchedHistDict

  return [matchedHists, matchedHistDict, commonLabelDict]

def main(argv=None):
  if argv is None:
    argv = sys.argv

  options, toPlot = LoadCommandlineOptions(argv)
  setupROOT()

  testFile = TFile(options.test)
  refFile = None
  if options.ref != None:
    refFile = TFile(options.ref)

  matchedHists, matchedHistDict, commonLabelDict = determineHistoList(testFile, refFile, toPlot)
  drawPlots(testFile, refFile, matchedHists, matchedHistDict, commonLabelDict, options)

if __name__ == '__main__':
  sys.exit(main())
