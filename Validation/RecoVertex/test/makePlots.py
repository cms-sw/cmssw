from ROOT import *
import re

files = ['DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root']

folders = [
    'DQMData/Run 1/Validation/Run summary/Vertices/selectedOfflinePrimaryVertices',
    'DQMData/Run 1/Validation/Run summary/Vertices/selectedPixelVertices'
]

histo_names = [
    {'name': 'globalEfficiencies', 'o': ''},
    {'name': 'effic_vs_NumVertices', 'o': ''},
    {'name': 'effic_vs_NumTracks', 'o': ''},
    {'name': 'effic_vs_ClosestVertexInZ', 'o': 'logx'},
    {'name': 'effic_vs_Pt2', 'o': 'logx'},
    {'name': 'effic_vs_Z', 'o': ''},
    {'name': 'gen_duplicate_vs_NumVertices', 'o': ''},
    {'name': 'gen_duplicate_vs_NumTracks', 'o': ''},
    {'name': 'gen_duplicate_vs_ClosestVertexInZ', 'o': 'logx'},
    {'name': 'gen_duplicate_vs_Pt2', 'o': 'logx'},
    {'name': 'gen_duplicate_vs_Z', 'o': ''},
    {'name': 'fakerate_vs_NumVertices', 'o': ''},
    {'name': 'fakerate_vs_Ndof', 'o': ''},
    {'name': 'fakerate_vs_NumTracks', 'o': ''},
    {'name': 'fakerate_vs_ClosestVertexInZ', 'o': 'logx'},
    {'name': 'fakerate_vs_Pt2', 'o': 'logx'},
    {'name': 'fakerate_vs_Z', 'o': ''},
    {'name': 'duplicate_vs_NumVertices', 'o': ''},
    {'name': 'duplicate_vs_NumTracks', 'o': ''},
    {'name': 'duplicate_vs_ClosestVertexInZ', 'o': 'logx'},
    {'name': 'duplicate_vs_Pt2', 'o': 'logx'},
    {'name': 'duplicate_vs_Z', 'o': ''},
    {'name': 'merged_vs_NumVertices', 'o': ''},
    {'name': 'merged_vs_NumTracks', 'o': ''},
    {'name': 'merged_vs_ClosestVertexInZ', 'o': 'logx'},
    {'name': 'merged_vs_Pt2', 'o': 'logx'},
    {'name': 'merged_vs_Z', 'o': ''}
]

histograms = []
# Build full list of histograms
for h in histo_names:
    for f in folders:
        histograms.append({'name': '%s/%s' % (f, h['name']), 'o': h['o']})

file_handles = []

def prepareFileHandles():
    for file in files:
        file_handles.append(TFile(file))

def cleanOptions():
    gPad.SetLogx(0)
    gPad.SetLogy(0)

def producePlots():
    c = TCanvas('c', 'c', 1024, 1024)
    histo = {}
    for h in histograms:
        counter = 0
        draw_options = ''
        for f in file_handles:
            histo = f.Get(h['name'])
            if not histo:
                print 'Failed to get histograms %s', h
            else:
                if counter == 0:
                    counter += 1
                else:
                    draw_options += 'SAME'
                cleanOptions()
                if h['o'] != '':
                    if h['o'] == 'logx':
                        gPad.SetLogx()
                histo.SetMarkerStyle(20)
                histo.SetMarkerColor(kAzure)
                histo.SetMarkerSize(1.2)
                histo.Draw(draw_options)
                c.Update()
                c.SaveAs('%s' % ("_".join(h['name'].split('/')[-2:]) + ".png"))

if __name__ == '__main__':
    prepareFileHandles()
    producePlots()
