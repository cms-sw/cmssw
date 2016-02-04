import FWCore.ParameterSet.Config as cms
import sys
import os
from ROOT import gROOT, TCanvas, TFile, gStyle, TLegend, TH1F
import Validation.RecoTau.RecoTauValidation_cfi as validation

def Match(required, got):
    for part in required.split('*'):
        if got.find(part) == -1:
            return False
    return True

gROOT.SetStyle('Plain')
gStyle.SetPalette(1)
gStyle.SetOptStat(0)
gStyle.SetPadGridX(True)
gStyle.SetPadGridY(True)
gStyle.SetOptTitle(0)

testFile = TFile(sys.argv[1])
refFile = TFile(sys.argv[2])
ylabel = sys.argv[3]
if ylabel != 'Efficiency' and ylabel != 'Fake rate':
    print 'Please specify in the third arg "Efficiency" or "Fake rate". Exiting...'
    sys.exit()
toPlot = sys.argv[4:]

maxLog = 3
for option in toPlot:
    if(option.find('maxLog=') != -1):
        maxLog= float(option[option.find('=')+1:])
        toPlot.remove(option)

#Takes the position of all plots that were produced
plotList = []
parList = ['pt', 'eta', 'phi', 'energy']
for attr in dir(validation.TauEfficiencies.plots):
    if type(getattr(validation.TauEfficiencies.plots,attr)) is cms.PSet:
        pset = getattr(validation.TauEfficiencies.plots,attr)
        effPlot = pset.efficiency.value()
        for par in parList:
            plotList.append('DQMData/'+effPlot.replace('#PAR#',par))

histoList = []
for plot in toPlot:
    for path in plotList:
        if Match(plot.lower(),path.lower()):
            histoList.append(path)

print histoList
legend = TLegend(0.6,0.82,0.6+0.39,0.82+0.17)
legend.SetHeader('')
canvas = TCanvas('MultiPlot','MultiPlot',validation.standardDrawingStuff.canvasSizeX.value(),validation.standardDrawingStuff.canvasSizeY.value())
colors = [2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1,2,3,4,6,5,7,28,1]
first = True
for histoPath,color in zip(histoList,colors):
    testH = testFile.Get(histoPath)
    if type(testH) != TH1F:
        print 'Looking for '+histoPath
        print 'Test plot now found! What the hell are you doing? Exiting...'
        sys.exit()
    xAx = histoPath[histoPath.find('Eff')+len('Eff'):]
    if hasattr(validation.standardDrawingStuff.xAxes,xAx):
        testH.GetXaxis().SetTitle( getattr(validation.standardDrawingStuff.xAxes,xAx).xAxisTitle.value())
    testH.GetYaxis().SetTitle(ylabel)
    testH.SetMarkerSize(1)
    testH.SetMarkerStyle(20)
    testH.SetMarkerColor(color)
    legend.AddEntry(testH,histoPath[histoPath.rfind('/')+1:histoPath.find('Eff')],'p')
    if first:
        first = False
        testH.Draw('ex0')
        if ylabel=='Fake rate':
            testH.GetYaxis().SetRangeUser(0.001,maxLog)
            canvas.SetLogy()
            canvas.Update()
    else:
        testH.Draw('same ex0 l')
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
legend.Draw()
canvas.Print('MultipleCompare.pdf')


