from __future__ import print_function
from ROOT import *
import re

files = ['DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root']

folders = [
    'DQMData/Run 1/Vertexing/Run summary/PrimaryVertexV/selectedOfflinePrimaryVertices',
    'DQMData/Run 1/Vertexing/Run summary/PrimaryVertexV/selectedPixelVertices'
]

histo_names = [
    {'name': 'RecoVtx_vs_GenVtx', 'o': '', 'xAxis': 'Pileup Interactions', 'yAxis': 'Reco Vertices', 'yMax': '90', 'yMin': ''},
    {'name': 'MatchedRecoVtx_vs_GenVtx', 'o': '', 'xAxis': 'Pileup Interactions', 'yAxis': 'Matched Reco Vertices', 'yMax': '90', 'yMin': ''},
    {'name': 'RecoAllAssoc2GenProperties', 'o': '', 'xAxis': 'Kind of Reco Vertex', 'yAxis': '', 'yMax': '', 'yMin': ''},
    {'name': 'RecoAllAssoc2Gen_PairDistanceZ', 'o': '', 'xAxis': 'Reco Vertex, Pair Distance', 'yAxis': '', 'yMax': '', 'yMin': ''},
    {'name': 'globalEfficiencies', 'o': '', 'xAxis': '', 'yAxis': '', 'yMax': '', 'yMin': ''},
    {'name': 'KindOfSignalPV', 'o': '', 'xAxis': 'Type Of Signal VTX', 'yAxis': '', 'yMax': '', 'yMin': '0'},
    {'name': 'MisTagRate', 'o': '', 'xAxis': 'Misidentification', 'yAxis': '', 'yMax': '', 'yMin': '0'},
    {'name': 'MisTagRate_vs_PU', 'o': '', 'xAxis': 'Pileup Interactions', 'yAxis': 'Misidentification', 'yMax': '', 'yMin': '0'},
    {'name': 'MisTagRate_vs_sum-pt2', 'o': 'logx', 'xAxis': '#sum_{pt^{2}}', 'yAxis': 'Misidentification', 'yMax': '', 'yMin': '0'},
    {'name': 'MisTagRate_vs_Z', 'o': '', 'xAxis': 'Z', 'yAxis': 'Misidentification', 'yMax': '', 'yMin': '0'},
    {'name': 'MisTagRate_vs_R', 'o': '', 'xAxis': 'R', 'yAxis': 'Misidentification', 'yMax': '', 'yMin': '0'},
    {'name': 'MisTagRate_vs_NumTracks', 'o': '', 'xAxis': 'Number of Tracks in Vertex', 'yAxis': 'Misidentification', 'yMax': '', 'yMin': '0'},
    {'name': 'TruePVLocationIndex', 'o': '', 'xAxis': 'True PV index in RecoVtx collection', 'yAxis': '', 'yMax': '', 'yMin': ''},
    {'name': 'effic_vs_NumVertices', 'o': '', 'xAxis': 'Number of Vertices', 'yAxis': 'Efficiency', 'yMax': '', 'yMin': ''},
    {'name': 'effic_vs_NumTracks', 'o': '', 'xAxis': 'Number of Tracks in Vertex', 'yAxis': 'Efficiency', 'yMax': '', 'yMin': ''},
    {'name': 'effic_vs_ClosestVertexInZ', 'o': 'logx', 'xAxis': 'Closest Distance in Z', 'yAxis': 'Efficiency', 'yMax': '', 'yMin': ''},
    {'name': 'effic_vs_Pt2', 'o': 'logx', 'xAxis': '#sum_{pt^{2}}', 'yAxis': 'Efficiency', 'yMax': '', 'yMin': ''},
    {'name': 'effic_vs_Z', 'o': '', 'xAxis': 'Z', 'yAxis': 'Efficiency', 'yMax': '', 'yMin': ''},
    {'name': 'gen_duplicate_vs_NumVertices', 'o': '', 'xAxis': 'Number of Vertices', 'yAxis': 'GenLevel Duplicates', 'yMax': '', 'yMin': ''},
    {'name': 'gen_duplicate_vs_NumTracks', 'o': '', 'xAxis': 'Number of Tracks in Vertex', 'yAxis': 'GenLevel Duplicates', 'yMax': '', 'yMin': ''},
    {'name': 'gen_duplicate_vs_ClosestVertexInZ', 'o': 'logx', 'xAxis': 'Closest Distance in Z', 'yAxis': 'GenLevel Duplicates', 'yMax': '', 'yMin': ''},
    {'name': 'gen_duplicate_vs_Pt2', 'o': 'logx', 'xAxis': '#sum_{pt^{2}}', 'yAxis': 'GenLevel Duplicates', 'yMax': '', 'yMin': ''},
    {'name': 'gen_duplicate_vs_Z', 'o': '', 'xAxis': 'Z', 'yAxis': 'GenLevel Duplicates', 'yMax': '', 'yMin': ''},
    {'name': 'fakerate_vs_NumVertices', 'o': '', 'xAxis': 'Number of Vertices', 'yAxis': 'Fake Rate', 'yMax': '', 'yMin': ''},
    {'name': 'fakerate_vs_PU', 'o': '', 'xAxis': 'Pileup Interactions', 'yAxis': 'Fake Rate', 'yMax': '', 'yMin': ''},
    {'name': 'fakerate_vs_Ndof', 'o': '', 'xAxis': 'Vertex DOF', 'yAxis': 'Fake Rate', 'yMax': '', 'yMin': ''},
    {'name': 'fakerate_vs_NumTracks', 'o': '', 'xAxis': 'Number of Tracks in Vertex', 'yAxis': 'Fake Rate', 'yMax': '', 'yMin': ''},
    {'name': 'fakerate_vs_ClosestVertexInZ', 'o': 'logx', 'xAxis': 'Closest Distance in Z', 'yAxis': 'Fake Rate', 'yMax': '', 'yMin': ''},
    {'name': 'fakerate_vs_Pt2', 'o': 'logx', 'xAxis': '#sum_{pt^{2}}', 'yAxis': 'Fake Rate', 'yMax': '', 'yMin': ''},
    {'name': 'fakerate_vs_Z', 'o': '', 'xAxis': 'Z', 'yAxis': 'Fake Rate', 'yMax': '', 'yMin': ''},
    {'name': 'duplicate_vs_NumVertices', 'o': '', 'xAxis': 'Number of Vertices', 'yAxis': 'Duplicate Rate', 'yMax': '', 'yMin': ''},
    {'name': 'duplicate_vs_PU', 'o': '', 'xAxis': 'Pileup Interactions', 'yAxis': 'Duplicate Rate', 'yMax': '', 'yMin': ''},
    {'name': 'duplicate_vs_NumTracks', 'o': '', 'xAxis': 'Number of Tracks in Vertex', 'yAxis': 'Duplicate Rate', 'yMax': '', 'yMin': ''},
    {'name': 'duplicate_vs_ClosestVertexInZ', 'o': 'logx', 'xAxis': 'Closest Distance in Z', 'yAxis': 'Duplicate Rate', 'yMax': '', 'yMin': ''},
    {'name': 'duplicate_vs_Pt2', 'o': 'logx', 'xAxis': '#sum_{pt^{2}}', 'yAxis': 'Duplicate Rate', 'yMax': '', 'yMin': ''},
    {'name': 'duplicate_vs_Z', 'o': '', 'xAxis': 'Z', 'yAxis': 'Duplicate Rate', 'yMax': '', 'yMin': ''},
    {'name': 'merged_vs_NumVertices', 'o': '', 'xAxis': 'Number of Vertices', 'yAxis': 'Merge Rate', 'yMax': '', 'yMin': ''},
    {'name': 'merged_vs_PU', 'o': '', 'xAxis': 'Pileup Interactions', 'yAxis': 'Merge Rate', 'yMax': '', 'yMin': ''},
    {'name': 'merged_vs_NumTracks', 'o': '', 'xAxis': 'Number of Tracks in Vertex', 'yAxis': 'Merge Rate', 'yMax': '', 'yMin': ''},
    {'name': 'merged_vs_ClosestVertexInZ', 'o': 'logx', 'xAxis': 'Closest Distance in Z', 'yAxis': 'Merge Rate', 'yMax': '', 'yMin': ''},
    {'name': 'merged_vs_Pt2', 'o': 'logx', 'xAxis': '#sum_{pt^{2}}', 'yAxis': 'Merge Rate', 'yMax': '', 'yMin': ''},
    {'name': 'merged_vs_Z', 'o': '', 'xAxis': 'Z', 'yAxis': 'Merge Rate', 'yMax': '', 'yMin': ''}
]

histograms = []
# Build full list of histograms
for h in histo_names:
    for f in folders:
        histograms.append({'name': '%s/%s' % (f, h['name']), 'o': h['o'],
                           'xAxis': h['xAxis'], 'yAxis': h['yAxis'],
                           'yMax': h['yMax'], 'yMin': h['yMin']})

file_handles = []

def prepareFileHandles():
    for file in files:
        file_handles.append(TFile(file))

def cleanOptions():
    gPad.SetLogx(0)
    gPad.SetLogy(0)

def setTextProperties(obj, label=False, title=False):
    textFont = 42
    textSize = 0.027
    titleOffset = 1.25
    labelOffset = 0.002
    if not label and not title:
        obj.SetTextFont(textFont)
        obj.SetTextSize(textSize)
    if label:
        obj.SetLabelOffset(labelOffset)
        obj.SetLabelFont(textFont)
        obj.SetLabelSize(textSize)
    if title:
        gStyle.SetTitleX(0.5)
        gStyle.SetTitleAlign(23)
        obj.SetTitleFont(textFont)
        obj.SetTitleSize(textSize)
        obj.SetTitleOffset(titleOffset)
    return obj

def producePlots():
    gStyle.SetOptStat(0)
    c = TCanvas('c', 'c', 1024, 1024)
    histo = {}
    for h in histograms:
        counter = 0
        draw_options = ''
        for f in file_handles:
            histo = f.Get(h['name'])
            if not histo:
                print('Failed to get histograms %s', h)
            else:
                if counter == 0:
                    counter += 1
                else:
                    draw_options += 'SAME'
                cleanOptions()
                if h['o'] != '':
                    if h['o'] == 'logx':
                        gPad.SetLogx()
                if h['yMax'] != '':
                    histo.SetMaximum(float(h['yMax']))
                if h['yMin'] != '':
                    histo.SetMinimum(float(h['yMin']))
                histo.GetXaxis().SetTitle(h['xAxis'])
                histo.GetYaxis().SetTitle(h['yAxis'])
                setTextProperties(histo.GetXaxis(), title=True)
                setTextProperties(histo.GetYaxis(), title=True)
                setTextProperties(histo.GetXaxis(), label=True)
                setTextProperties(histo.GetYaxis(), label=True)
                histo.SetMarkerStyle(20)
                histo.SetMarkerColor(kAzure)
                histo.SetMarkerSize(1.2)
                histo.Draw(draw_options)
                c.Update()
                c.SaveAs('%s' % ("_".join(h['name'].split('/')[-2:]) + ".png"))

if __name__ == '__main__':
    prepareFileHandles()
    producePlots()
