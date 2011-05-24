import FWCore.ParameterSet.Config as cms
import sys
import os
import math
from ROOT import *
import Validation.RecoTau.RecoTauValidation_cfi as validation

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

gROOT.SetStyle('Plain')
gROOT.SetBatch()
gStyle.SetPalette(1)
gStyle.SetOptStat(0)
gStyle.SetPadGridX(True)
gStyle.SetPadGridY(True)
gStyle.SetOptTitle(0)
gStyle.SetPadTopMargin(0.1)

test = ''
ref = ''
ylabel = ''
testLabel = ''
refLabel = None
toPlot = []
maxLog = 3
minDiv = 0.001
maxDiv = 2
logDiv = False
for option in sys.argv:
    if(option.find('maxLog=') != -1):
        maxLog= float(option[option.find('=')+1:])
    elif(option.find('minDiv=') != -1):
        minDiv= float(option[option.find('=')+1:])
    elif(option.find('maxDiv=') != -1):
        maxDiv= float(option[option.find('=')+1:])
    elif(option.find('logDiv=') != -1):
        logDiv= option[option.find('=')+1:] == 'True'
    elif option.find('testLabel=') != -1:
        testLabel = option[option.find('=')+1:]
    elif option.find('refLabel=') != -1:
        refLabel = option[option.find('=')+1:]
    elif option.find('test=') != -1:
        test = option[option.find('=')+1:]
    elif option.find('ref=') != -1:
        ref = option[option.find('=')+1:]
    elif option.find('label=') != -1:
        ylabel = option[option.find('=')+1:]
    else:
        toPlot.append(option)

testFile = TFile(test)
refFile = None
if ref != '':
    refFile = TFile(ref)

if ylabel != 'Efficiency' and ylabel != 'Fake rate' and ylabel != 'Significance':
    print 'Please specify in the label arg: "Efficiency" or "Fake rate" or "Significance". Exiting...'
    sys.exit()


#Takes the position of all plots that were produced
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
if testLabel != '':
    header += 'Dots: '+testLabel
if refLabel != '':
    header += ' Line: '+refLabel
legend.SetHeader(header)
canvas = TCanvas('MultiPlot','MultiPlot',validation.standardDrawingStuff.canvasSizeX.value(),832)
effPad = TPad('effPad','effPad',0,0.25,1.,1.,0,0)
effPad.Draw()
diffPad = TPad('diffPad','diffPad',0.,0.,1,.25,0,0)
diffPad.Draw()
colors = [2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1]
first = True
divHistos = []
for histoPath,color in zip(histoList,colors):
    testH = testFile.Get(histoPath)
    if type(testH) != TH1F:
        print 'Looking for '+histoPath
        print 'Test plot now found! What the hell are you doing? Exiting...'
        sys.exit()
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
        testH.Draw('ex0')
        if ylabel=='Fake rate':
            testH.GetYaxis().SetRangeUser(0.001,maxLog)
            effPad.SetLogy()
            effPad.Update()
    else:
        testH.Draw('same ex0 l')
    if refFile == None:
        continue
    refH = refFile.Get(histoPath)
    if type(refH) != TH1F:
        print 'Ref plot not found! It will not be plotted!'
        continue
    refH.SetLineColor(color)
    refH.SetLineWidth(1)
    refH.DrawCopy('same hist')
    refH.SetFillColor(color)
    refH.SetFillStyle(3001)
    refH.Draw('same e3')
    divHistos.append(Divide(testH,refH))

firstD = True
if ylabel == 'significance':
    diffPad = TCanvas('Efficiency','Efficiency',validation.standardDrawingStuff.canvasSizeX.value(),validation.standardDrawingStuff.canvasSizeY.value())

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
            histo.GetYaxis().SetRangeUser(minDiv,maxDiv)
            if ylabel == 'significance':
                histo.GetYaxis().SetTitle('Signal eff. / Background eff.')
            if logDiv:
                diffPad.SetLogy()
            histo.Draw('ex0')
            firstD = False
        else:
            histo.Draw('same ex0')
            diffPad.Update()
    if ylabel == 'significance':
        diffPad.cd()
        legend.Draw()
        diffPad.Print('MultipleCompare.png')
    else:
        effPad.cd()
        legend.Draw()
        canvas.Print('MultipleCompare.png')



