#! /usr/bin/env python

import FWCore.ParameterSet.Config as cms
import sys
import os
import math
import re
import Validation.RecoTau.RecoTauValidation_cfi as validation
from optparse import OptionParser
from ROOT import *

<<<<<<< MultipleCompare.py
__author__  = "Mauro Verzetti (mauro.verzetti@cern.ch) and Lucia Perrini (lucia.perrini@cern.ch)"
=======
__author__  = "Mauro Verzetti (mauro.verzetti@cern.ch), updated by Guler Karapinar  11-June-2012"
>>>>>>> 1.28
__doc__ = """Script to plot the content of a Validation .root file and compare it to a different file:\n\n
Usage: MultipleCompare.py -T testFile -R refFile [options] [search strings that you want to apply '*' is supported as special character]"""

def LoadCommandlineOptions(argv):
  sys.argv = argv
  parser = OptionParser(description=__doc__)
  parser.add_option('--myhelp',metavar='', action="store_true",help='prints this output message',dest='help',default = False)
  parser.add_option('--TestFile','-T',metavar='testFile', type=str,help='Sets the test file',dest='test',default = '')
  parser.add_option('--RefFile','-R',metavar='refFile', type=str,help='Sets the reference file',dest='ref',default = None)
  parser.add_option('--output','-o',metavar='outputFile', type=str,help='Sets the output file',dest='out',default = 'MultipleCompare.png')
  parser.add_option('--logScaleY',action="store_true", dest="logScaleY", default=False, help="Sets the log scale in the plot (Y axis)")
  parser.add_option('--logScaleX',action="store_true", dest="logScaleX", default=False, help="Sets the log scale in the plot (X axis)")
  parser.add_option('--fakeRate','-f',action="store_true", dest="fakeRate", default=False, help="Sets the fake rate options and put the correct label (implies --logScale)")
  parser.add_option('--testLabel','-t',metavar='testLabel', type=str,help='Sets the label to put in the plots for test file',dest='testLabel',default = None)
  parser.add_option('--refLabel','-r',metavar='refLabel', type=str,help='Sets the label to put in the plots for ref file',dest='refLabel',default = None)
  parser.add_option('--maxLogX',metavar='number', type=float,help='Sets the maximum of the scale in log scale both in the main and in the sub pad (requires --logScale or -f to work)',dest='maxLogX',default = 100)
  parser.add_option('--minLogX',metavar='number', type=float,help='Sets the minimum of the scale in log scale (requires --logScale or -f to work)',dest='minLogX',default = 0.001)
  parser.add_option('--minLogY',metavar='number', type=float,help='Sets the minimum of the scale in log scale (requires --logScale or -f to work)',dest='minLogY',default = 0.0001)
  parser.add_option('--maxLogY',metavar='number', type=float,help='Sets the maximum of the scale in log scale (requires --logScale or -f to work)',dest='maxLogY',default = 3)
  parser.add_option('--minYR',metavar='number', type=float,help='Sets the minimum of the scale in sub pad',dest='minYR',default = 0)
  parser.add_option('--maxYR',metavar='number', type=float,help='Sets the maximum of the scale in sub pad',dest='maxYR',default = 1.2)
#  parser.add_option('--minDivY',metavar='number', type=float,help='Sets the minimum of the scale in the ratio pad',dest='minDivY',default = 0.)
#  parser.add_option('--maxDivY',metavar='number', type=float,help='Sets the maximum of the scale in the ratio pad',dest='maxDivY',default = 2)
#  parser.add_option('--minDivX',metavar='number', type=float,help='Sets the minimum of the scale in the ratio pad',dest='minDivX',default = 0.)
#  parser.add_option('--maxDivX',metavar='number', type=float,help='Sets the maximum of the scale in the ratio pad',dest='maxDivX',default = 2)
  parser.add_option('--logDiv',action="store_true", dest="logDiv", default=False, help="Sets the log scale in the plot")
  parser.add_option('--normalize',action="store_true", dest="normalize", default=False, help="plot normalized")
<<<<<<< MultipleCompare.py
  parser.add_option('--maxRange',metavar='number',type=float, dest="maxRange", default=1.6, help="Sets the maximum range in linear plots")
  parser.add_option('--maxXaxis',metavar='number',type=float, dest="maxXaxis", default=800, help="Sets the maximum range on x axis in the main pad")
  parser.add_option('--minXaxis',metavar='number',type=float, dest="minXaxis", default=0, help="Sets the minimum range on x axis in the main pad")
  parser.add_option('--maxYaxis',metavar='number',type=float, dest="maxYaxis", default=2, help="Sets the maximum range on Y axis in the main pad")
  parser.add_option('--minYaxis',metavar='number',type=float, dest="minYaxis", default=0, help="Sets the minimum range on Y axis in the main pad")
=======
  parser.add_option('--maxRange',metavar='number',type=float, dest="maxRange", default=2.0, help="Sets the maximum range in linear plots")
>>>>>>> 1.28
  parser.add_option('--rebin', dest="rebin", type=int, default=-1, help="Sets the rebinning scale")
  parser.add_option('--branding','-b',metavar='branding', type=str,help='Define a branding to label the plots (in the top right corner)',dest='branding',default = None)

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
<<<<<<< MultipleCompare.py
    ret.GetYaxis().SetTitle('Ratio')
    for binI in range(hNum.GetNbinsX()+1):
=======
    ret.GetYaxis().SetTitle('Ratio  (Dots/Line)')
    for binI in range(hNum.GetNbinsX()+1):
>>>>>>> 1.28
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

  prefixParts = prefix.partition('Discr')
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
	title.SetTextAlign(12)
        title.SetTextSize(.03)
        title.SetTextColor(1)
        title.SetLineWidth (1)
        title.SetTextFont(42)
        leftMargin = gStyle.GetPadLeftMargin()+0.25
	topMargin = 1 - 0.5*(gStyle.GetPadTopMargin()+0.5)
        title.DrawLatex(leftMargin, topMargin, text)



def FindParents(histoPath):
    root = histoPath[:histoPath.find('_')]
    par = histoPath[histoPath.find('Eff')+3:]
    validationPlots = validation.efficiencies.plots._Parameterizable__parameterNames
    found =0
    num = ''
    den = ''
    for efficiency in validationPlots:
        effpset = getattr(validation.efficiencies.plots,efficiency)
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
        print 'Plot now found! '
        sys.exit()
    denSingle = tfile.Get(parents[1])
    if type(denSingle) != TH1F:
        print 'Looking for '+denSingle
        print 'Plot now found! '
        sys.exit()
    num.Rebin(rebinVal)
    den = denSingle.Rebin(rebinVal,'denClone')
    retVal = num.Clone(histoPath+'Rebin%s'%rebinVal)
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

def optimizeRangeMainPad(argv, pad, hists, maxLogX_, minX_, maxX_, maxLogY_, minY_, maxY_):
  pad.Update()
  if pad.GetLogy():
    if maxLogY_ > 0:
      maxLogY = maxLogY_
    else:
      maxLogY = -1
    minY, maxY = findRange(hists, -1, maxLogY)
  else:
    minY, maxY = findRange(hists, minY_, maxY_)

  if pad.GetLogy():
<<<<<<< MultipleCompare.py
    if minY == 0:
      minY = 0.001
  else:
    if minY < 0.7:
      minY = 0 #start from zero if possible
    if maxY <= 1.1 and maxY > 0.7:
      maxY = 1.2 #prefere fixed range for easy comparison
  hists[0].SetAxisRange(minY, maxY, "Y")

  if pad.GetLogx():
    if maxLogX_ > 0:
      maxLogX = maxLogX_
    else:
      maxLogX = -1
    minX, maxX = findRange(hists, -1, maxLogX)
  else:
    minX, maxX = findRange(hists, minX_, maxX_)
    
  if pad.GetLogx():
    if minX == 0:
      minX = 0.001
  else:
    if minX < 0.7:
      minX = 0 #start from zero if possible
    if maxX <= 1.1 and maxX > 0.7:
      maxX = 1.2 #prefere fixed range for easy comparison
  hists[0].SetAxisRange(minX, maxX, "X")

def optimizeRangeSubPad(argv, pad, hists, maxLogX_, minX_, maxX_, minYRatio_, maxYRatio_):
  pad.Update()
  if pad.GetLogx():
    if maxLogX_ > 0:
      maxLogX = maxLogX_
    else:
      maxLogX = -1
    minX, maxX = findRange(hists, -1, maxLogX)
  else:
    minX, maxX = findRange(hists, minX_, maxX_)
  if pad.GetLogx():
    if minX == 0:
      minX = 0.001
=======
    if min == 0:
      min = 0.001
    if max <1:
      max = 1#prefere fixed range for easy comparison
>>>>>>> 1.28
  else:
<<<<<<< MultipleCompare.py
    if minX < 0.7:
      minX = 0 #start from zero if possible
    if maxX <= 1.1 and maxX > 0.7:
      maxX = 1.2 #prefere fixed range for easy comparison
  hists[0].SetAxisRange(minX, maxX, "X")
=======
    if min < 0.7:
      min = 0. #start from zero if possible
    if max <= 1.1 and max > 0.7:
      max = 1.6 #prefere fixed range for easy comparison
  hists[0].SetAxisRange(min, max, "Y")
>>>>>>> 1.28

  min = -1
  max = -1
  if minYRatio_ > 0:
    min = minYRatio_
  if maxYRatio_ > 0:
    max = maxYRatio_
  min, max = findRange(hists, min, max)
  if max > 2:
    max = 2 #maximal bound
<<<<<<< MultipleCompare.py
  hists[0].SetAxisRange(min, max, "Y")                                     
=======
  hists[0].SetAxisRange(min, max, "Y")
  

>>>>>>> 1.28

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


def main(argv=None):
  if argv is None:
    argv = sys.argv

  options, toPlot = LoadCommandlineOptions(argv)



  gROOT.SetStyle('Plain')
  gROOT.SetBatch()
  gStyle.SetPalette(1)
  gStyle.SetOptStat(0)
  gStyle.SetOptTitle(0)
  gStyle.SetPadTopMargin(0.1)
  gStyle.SetPadBottomMargin(0.1)
  gStyle.SetPadLeftMargin(0.13)
  gStyle.SetPadRightMargin(0.07)
  gStyle.SetErrorX(0)
  

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
    if len(toPlot)>0:
      print 'Check your plot list:', toPlot
    sys.exit()


  #WARNING: For now the hist type is assumed to be constant over all histos.
  histType, label, prefix = DetermineHistType(histoList[0])
  #decide whether or not to scale to an integral of 1
  #should usually not be used most plot types. they are already normalized.
  scaleToIntegral = False
  if options.normalize:
    scaleToIntegral = True

  ylabel = 'Efficiency'

  if options.fakeRate:
    ylabel = 'Fake Rate'

  drawStats = False
  if histType=='pTRatio' and len(histoList)<3:
    drawStats = True
<<<<<<< MultipleCompare.py

  #legend = TLegend(0.50,0.73,0.50+0.37,1)
  x1 = 0.33
  x2 = 1-gStyle.GetPadRightMargin()
  y2 = 1-gStyle.GetPadTopMargin()
  lineHeight = .055
=======
  
  legend = TLegend(0.6,0.83,0.6+0.35,0.83+0.15)
  x1 = 0.25
  x2 = 1-(gStyle.GetPadRightMargin()-0.04)
  y2 = 1-(gStyle.GetPadTopMargin()-0.11)
  lineHeight = .05
>>>>>>> 1.28
  if len(histoList) == 1:
    lineHeight = .075
  y1 = y2 - lineHeight*len(histoList)
  legend = TLegend(x1,y1,x2,y2)
<<<<<<< MultipleCompare.py
  legend.SetHeader(label)
=======
  #legend.SetHeader("hpsPFTauDiscriminators")       
  legend.SetTextSize(.025)
  legend.SetTextColor(1)
>>>>>>> 1.28
  legend.SetFillColor(0)
<<<<<<< MultipleCompare.py
  legend.SetTextSize(0.032)
=======
  legend.SetTextFont(42)
  
>>>>>>> 1.28
  if drawStats:
    y2 = y1
    y1 = y2 - 3.0*len(histoList)
    statsBox = TPaveText(x1,y1,x2,y2,"NDC")
    statsBox.SetFillColor(0)
    statsBox.SetTextAlign(12)
    statsBox.SetBorderSize(1)
    statsBox.SetBorderMode(0) 





    
  canvas = TCanvas('MultiPlot','MultiPlot',validation.standardDrawingStuff.canvasSizeX.value(),832)
<<<<<<< MultipleCompare.py
  effPad = TPad('effPad','effPad',0.01,0.35,0.99,0.99)#0,0.25,1.,1.,0,0)
  effPad.SetBottomMargin(0.0)#0.1)
  #effPad.SetTopMargin(0.1)
  #effPad.SetLeftMargin(0.13)
  #effPad.SetRightMargin(0.07)
=======
  effPad = TPad("effPad", "effPad",0,0.25,1,1)   
  effPad.SetFillColor(0)
  effPad.SetBorderMode(0)
  effPad.SetBorderSize(2)
  effPad.SetLeftMargin(0.13)
  effPad.SetRightMargin(0.07)
  effPad.SetFrameBorderMode(0)
  effPad.SetTitle("Tau Release Validation")
>>>>>>> 1.28
  effPad.Draw()
  header = ''
  if options.branding != None:
    header += ' Sample: '+options.branding
  if options.testLabel != None:
    header += ' Dots: '+options.testLabel
  if options.refLabel != None:
    header += ' Line: '+options.refLabel
<<<<<<< MultipleCompare.py
  DrawTitle(header)
  DrawBranding(options)
  diffPad = TPad('diffPad','diffPad',0.01,0.01,0.99,0.32)#0.,0.,1,.25,0,0)
  diffPad.SetTopMargin(0.00);
  diffPad.SetBottomMargin(0.30);
=======
  #DrawTitle(header)
  
  diffPad = TPad("diffPad", "diffPad",0,0,1,0.25)
  diffPad.Range(0,0,2,2)
  diffPad.SetLeftMargin(0.13)
  diffPad.SetTopMargin(0.05)
  diffPad.SetBottomMargin(0.13)
  diffPad.SetRightMargin(0.07)
  diffPad.SetFrameBorderMode(0)
  diffPad.SetGridy() 
>>>>>>> 1.28
  diffPad.Draw()
  colors = [2,1,3,7,4,6]
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
        print 'Test plot now found! '
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
<<<<<<< MultipleCompare.py
    testH.GetXaxis().SetTitleOffset(1.1)
    testH.GetXaxis().SetRangeUser(options.minXaxis,options.maxXaxis)
    testH.GetYaxis().SetTitleOffset(1.5)
    #testH.GetYaxis().SetTitleSize(0.08)
    #testH.GetYaxis().CenterTitle()
=======
    testH.GetXaxis().SetTitleOffset(0.85)
    testH.GetYaxis().SetTitleOffset(0.9)
    #testH.GetXaxis().SetTitleOffset(1.1)
    #testH.GetYaxis().SetTitleOffset(1.1)
>>>>>>> 1.28
    testH.SetMarkerSize(1)
    testH.SetMarkerStyle(21)
    testH.SetMarkerColor(color)
    testH.GetYaxis().SetLabelFont(42)
    testH.GetXaxis().SetLabelFont(42)
    if histType == 'Eff':
      legend.AddEntry(testH,histoPath[histoPath.rfind('/')+1:histoPath.find(histType)],'p')
    else:
      legend.AddEntry(testH,DetermineHistType(histoPath)[2],'p')
    if drawStats:
        text = statsBox.AddText(statTemplate % ('Dots',testH.GetMean(), testH.GetRMS()) )
        text.SetTextColor(color)
    if first:
        first = False
        if options.logScaleY:
            effPad.SetLogy()
        if options.logScaleX:
            effPad.SetLogx()
            diffPad.SetLogx()
        if scaleToIntegral:
          if testH.GetEntries() > 0:
            if not testH.GetSumw2N():
              testH.Sumw2()
              testH.DrawNormalized('ex0 HIST')
            else:
              print "--> Warning! You tried to normalize a histogram which seems to be already scaled properly. Draw it unscaled."
              scaleToIntegral = False
              testH.Draw('ex0')
        else:
          testH.Draw('ex0')
    else:
        if scaleToIntegral:
          if testH.GetEntries() > 0:
            testH.DrawNormalized('same HIST')
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
    if scaleToIntegral:
      if testH.GetEntries() > 0:
        refH.DrawNormalized('same HIST')
    else:
        refH.DrawCopy('same HIST')
    if drawStats:
      text = statsBox.AddText(statTemplate % ('Line',refH.GetMean(), refH.GetRMS()) )
      text.SetTextColor(color)
    #uncommment the following two lines only for filled option
    #refH.SetFillColor(color) 
    #refH.SetFillStyle(3001)  
    if scaleToIntegral:
        entries = testH.GetEntries()
        if entries > 0:
          testH.Scale(1./entries)
        entries = refH.GetEntries()
        refH.Sumw2()
        if entries > 0:
          refH.Scale(1./entries)
    refH.Draw('same HIST')
    divHistos.append(Divide(testH,refH))
    

    if options.maxLogY > 0:
       maxlY=options.maxLogY
    if options.maxLogX > 0:
       maxlX=options.maxLogX

  tmpHists = []
  tmpHists.extend(testHs)
  tmpHists.extend(refHs)
  optimizeRangeMainPad(argv, effPad, tmpHists, maxlX, options.minXaxis, options.maxXaxis, maxlY, options.minYaxis, options.maxYaxis)
  
  
  firstD = True
  if refFile != None:
    for histo,color in zip(divHistos,colors):
<<<<<<< MultipleCompare.py
      diffPad.cd()
      histo.SetMarkerSize(1)
      histo.SetMarkerStyle(20)
      histo.SetMarkerColor(color)
      histo.GetYaxis().SetLabelSize(0.07)
      histo.GetYaxis().SetTitleOffset(0.75)
      histo.GetYaxis().SetTitleSize(0.08)
      histo.GetXaxis().SetLabelSize(0.08)
      histo.GetXaxis().SetTitleSize(0.08)
      #histo.GetYaxis().CenterTitle()
                                         

      if firstD:
        histo.Draw('ex0')
        firstD = False
      else:
        histo.Draw('same ex0')
        diffPad.Update()
        
    if options.maxLogX > 0:
      maxlX=options.maxLogX
    optimizeRangeSubPad(argv, diffPad, divHistos, maxlX, options.minXaxis, options.maxXaxis, options.minYR, options.maxYR)
=======
        diffPad.cd()
        histo.SetMarkerSize(1)
        histo.SetMarkerStyle(21)
        histo.SetMarkerColor(color)
        histo.GetYaxis().SetLabelSize(0.08)
        histo.GetYaxis().SetTitleOffset(0.6)
        histo.GetYaxis().SetTitleSize(0.08)
        histo.GetYaxis().SetLabelSize(0.09)
        histo.GetYaxis().SetLabelFont(42)
        histo.GetYaxis().SetTitleFont(42)
        histo.GetXaxis().SetTitleSize(0.08)
        histo.GetXaxis().SetLabelSize(0.09)
        histo.GetXaxis().SetLabelFont(42)
        histo.GetYaxis().SetTitleFont(42)
        histo.GetXaxis().SetTitle('')
        divHistos[0].SetAxisRange(0, 2, "Y") 
        histo.SetFillColor(0)
        #t = TLine(-3,1.,3,1.)
        #t.SetLineColor(1)
        #t.SetLineStyle(6)  
        #t.Draw()      
        if firstD:
            if options.logDiv:
                diffPad.SetLogy()
            histo.Draw('ex0')
            firstD = False
        else:
            histo.Draw('same')
            diffPad.Update()
            
    #optimizeRangeSubPad(argv, divHistos)
>>>>>>> 1.28

<<<<<<< MultipleCompare.py
  effPad.cd()
  legend.Draw()

=======
    effPad.cd()
  #legend.Draw()
>>>>>>> 1.28
  if drawStats:
    statsBox.Draw()
<<<<<<< MultipleCompare.py
  
=======
    
>>>>>>> 1.28
  canvas.Print(options.out)


if __name__ == '__main__':
  sys.exit(main())

