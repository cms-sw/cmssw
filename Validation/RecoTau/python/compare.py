from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from .officialStyle import officialStyle
from array import array
from ROOT import gROOT, gStyle, TH1F, TH1D, TF1, TFile, TCanvas, TH2F, TLegend, TGraphAsymmErrors, Double, TLatex
import os, copy, sys

gROOT.SetBatch(True)
officialStyle(gStyle)
gStyle.SetOptTitle(0)
#set_palette("color")
#gStyle.SetPaintTextFormat("2.0f")


argvs = sys.argv
argc = len(argvs)

if argc != 2:
    print('Please specify the runtype : python3 tauPOGplot.py <ZTT, ZEE, ZMM, QCD>')
    sys.exit(0)

runtype = argvs[1]
print('You selected', runtype)


tlabel = 'Z #rightarrow #tau#tau'
xlabel = 'gen. tau p_{T}^{vis} (GeV)'

if runtype == 'QCD':
    tlabel = 'QCD'
    xlabel = 'jet p_{T} (GeV)'
elif runtype == 'ZEE':
    tlabel = 'Z #rightarrow ee'
    xlabel = 'electron p_{T} (GeV)'
elif runtype == 'ZMM':
    tlabel = 'Z #rightarrow #mu#mu'
    xlabel = 'muon p_{T} (GeV)'




def ensureDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save(canvas, name):
    ensureDir('compare_' + runtype)
    canvas.SaveAs(name.replace(' ','').replace('&&','')+'.pdf')
    canvas.SaveAs(name.replace(' ','').replace('&&','')+'.gif')


def LegendSettings(leg, ncolumn):
    leg.SetNColumns(ncolumn)
    leg.SetBorderSize(0)
    leg.SetFillColor(10)
    leg.SetLineColor(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetTextFont(42)



def makeCompareVars(tree, var, sel, leglist, nbin, xmin, xmax, xtitle, ytitle, scale, name):
   
#    print leglist

    c = TCanvas()

    hists = []
    col = [1,2,4,8,6]
    ymax = 0

    for ii, isel in enumerate(sel):
        hist = TH1F('h_' + str(ii), 'h_' + str(ii), nbin, xmin, xmax)
        hist.GetXaxis().SetTitle(xtitle)
        hist.GetYaxis().SetTitle(ytitle)
        hist.GetYaxis().SetNdivisions(507)
        hist.SetLineColor(col[ii])
        hist.SetLineWidth(len(sel)+1-ii)
        hist.SetLineStyle(1)
        hist.SetMarkerSize(0)
        hist.SetMinimum(0)
        hist.Sumw2()

#        print hist.GetName(), var, isel
        tree.Project(hist.GetName(), var, isel)
        hist.Scale(1./hist.GetEntries())

        if ymax < hist.GetMaximum():
            ymax = hist.GetMaximum()

        hists.append(hist)


    leg = TLegend(0.6,0.65,0.91,0.9)
    LegendSettings(leg,1)

    for ii, ihist in enumerate(hists):
        ihist.SetMaximum(ymax*1.2)
        ihist.SetMinimum(0.)

        if ii ==0:
            ihist.Draw('h')
        else:
            ihist.Draw('hsame')

        if leglist[ii] != 'None':
            leg.AddEntry(ihist, leglist[ii], "l")


    if leglist[0]!='None':
        leg.Draw()


#    save(c, 'compare_' + runtype + '/compare_' + name)




    
def overlay(hists, ytitle, header, addon):

    print('number of histograms = ', len(hists))

    canvas = TCanvas()
    leg = TLegend(0.2,0.7,0.5,0.9)
    LegendSettings(leg, 1)

    col = [1,2,4,6,8,9,12]

    ymax = -1
    ymin = 100

    for ii, hist in enumerate(hists):
        hist.GetYaxis().SetTitle('efficiency')
        hist.SetLineColor(col[ii])
        hist.SetMarkerColor(col[ii])
        hist.SetLineWidth(2)
        hist.SetMarkerSize(1)


        for ip in range(hist.GetN()):
            x = Double(-1)
            y = Double(-1)
            hist.GetPoint(ip, x, y)

            if ymin > y:
                ymin = y
            if ymax < y:
                ymax = y

#
#        if ymax < hist.GetMaximum():
#            ymax = hist.GetMaximum()
#        if ymin > hist.GetMinimum():
#            ymin = hist.GetMinimum()

        if ii==0:
            hist.Draw("Azp")
        else:
            
            hist.Draw("pzsame")
 
#        print hist.GetName(), hist.GetTitle()
        legname = hist.GetName()

        leg.AddEntry(hist, legname, 'lep')


    for hist in hists:
        hist.SetMaximum(ymax*2)
#        hist.SetMinimum(ymin*0.5)

    leg.Draw()

    tex = TLatex( hists[-1].GetXaxis().GetXmin() + 0.01*(hists[-1].GetXaxis().GetXmax() - hists[-1].GetXaxis().GetXmin()), ymax*2.1, addon.replace('tau_',''))

    tex.SetTextFont(42)
    tex.SetTextSize(0.03)
    tex.Draw()

    tex2 = TLatex( hists[-1].GetXaxis().GetXmin() + 0.87*(hists[-1].GetXaxis().GetXmax() - hists[-1].GetXaxis().GetXmin()), ymax*2.1, tlabel)

    tex2.SetTextFont(42)
    tex2.SetTextSize(0.03)
    tex2.Draw()



    save(canvas, 'compare_' + runtype + '/' + header)



def hoverlay(hists, xtitle, ytitle, name):
   
    c = TCanvas()

    ymax = 0
    for hist in hists:
        if ymax < hist.GetMaximum():
            ymax = hist.GetMaximum()


    leg = TLegend(0.6,0.65,0.91,0.9)
    LegendSettings(leg,1)

    for ii, ihist in enumerate(hists):
        ihist.SetMaximum(ymax*1.2)
        ihist.SetMinimum(0.)
        ihist.SetMarkerSize(0.)
        ihist.GetXaxis().SetTitle(xtitle)
        ihist.GetYaxis().SetTitle(ytitle)

        if ii ==0:
            ihist.Draw('h')
        else:
            ihist.Draw('hsame')

        leg.AddEntry(ihist, ihist.GetName(), "l")


    leg.Draw()


    save(c, 'compare_' + runtype + '/hist_' + name)




def makeEffPlotsVars(tree, varx, vary, sel, nbinx, xmin, xmax, nbiny, ymin, ymax, xtitle, ytitle, leglabel = None, header='', addon='', option='pt', marker=20):
   
    binning = [20,30,40,50,60,70,80,100,150,200]

    c = TCanvas()

    if option=='pt':
        _hist_ = TH1F('h_effp_' + addon, 'h_effp' + addon, len(binning)-1, array('d',binning))
        _ahist_ = TH1F('ah_effp_' + addon, 'ah_effp' + addon, len(binning)-1, array('d',binning))
    elif option=='eta':
        _hist_ = TH1F('h_effp_' + addon, 'h_effp' + addon, nbinx, xmin, xmax)
        _ahist_ = TH1F('ah_effp_' + addon, 'ah_effp' + addon, nbinx, xmin, xmax)
    elif option=='nvtx':
        _hist_ = TH1F('h_effp_' + addon, 'h_effp' + addon, len(vbinning)-1, array('d',vbinning))
        _ahist_ = TH1F('ah_effp_' + addon, 'ah_effp' + addon, len(vbinning)-1, array('d',vbinning))


    tree.Draw(varx + ' >> ' + _hist_.GetName(), sel)
    tree.Draw(varx + ' >> ' + _ahist_.GetName(), sel + ' && ' + vary)
    
    g_efficiency = TGraphAsymmErrors()
    g_efficiency.BayesDivide(_ahist_, _hist_)
    g_efficiency.GetXaxis().SetTitle(xtitle)
    g_efficiency.GetYaxis().SetTitle('efficiency')
    g_efficiency.GetYaxis().SetNdivisions(507)
    g_efficiency.SetLineWidth(3)
    g_efficiency.SetName(header)    
    g_efficiency.SetMinimum(0.)
    g_efficiency.GetYaxis().SetTitleOffset(1.3)
    g_efficiency.SetMarkerStyle(marker)
    g_efficiency.SetMarkerSize(1)
    g_efficiency.Draw('ap')

#    save(c, 'plots/' + addon)

    return copy.deepcopy(g_efficiency)





if __name__ == '__main__':

    
    vardict = {
        'againstMuonLoose3':{'var':'tau_againstMuonLoose3 > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'againstMuonLoose3'},
        'againstMuonTight3':{'var':'tau_againstMuonTight3 > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'againstMuonTight3'},

        'againstElectronVLooseMVA5':{'var':'tau_againstElectronVLooseMVA5 > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'againstElectronVLooseMVA5'},
        'againstElectronLooseMVA5':{'var':'tau_againstElectronLooseMVA5 > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'againstElectronLooseMVA5'},
        'againstElectronMediumMVA5':{'var':'tau_againstElectronMediumMVA5 > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'againstElectronMediumMVA5'},

        'againstElectronVLooseMVA6':{'var':'tau_againstElectronVLooseMVA6 > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'againstElectronVLooseMVA6'},
        'againstElectronLooseMVA6':{'var':'tau_againstElectronLooseMVA6 > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'againstElectronLooseMVA6'},
        'againstElectronMediumMVA6':{'var':'tau_againstElectronMediumMVA6 > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'againstElectronMediumMVA6'},

        'byLoosePileupWeightedIsolation3Hits':{'var':'tau_byLoosePileupWeightedIsolation3Hits > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byLoosePileupWeightedIsolation3Hits'},
        'byMediumPileupWeightedIsolation3Hits':{'var':'tau_byMediumPileupWeightedIsolation3Hits > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byMediumPileupWeightedIsolation3Hits'},
        'byTightPileupWeightedIsolation3Hits':{'var':'tau_byTightPileupWeightedIsolation3Hits > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byTightPileupWeightedIsolation3Hits'},

        'byLooseCombinedIsolationDeltaBetaCorr3Hits':{'var':'tau_byLooseCombinedIsolationDeltaBetaCorr3Hits > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byLooseCombinedIsolationDeltaBetaCorr3Hits'},
        'byMediumCombinedIsolationDeltaBetaCorr3Hits':{'var':'tau_byMediumCombinedIsolationDeltaBetaCorr3Hits > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byMediumCombinedIsolationDeltaBetaCorr3Hits'},
        'byTightCombinedIsolationDeltaBetaCorr3Hits':{'var':'tau_byTightCombinedIsolationDeltaBetaCorr3Hits > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byTightCombinedIsolationDeltaBetaCorr3Hits'},

        'byLooseIsolationMVA3oldDMwLT':{'var':'tau_byLooseIsolationMVA3oldDMwLT > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byLooseIsolationMVA3oldDMwLT'},
        'byMediumIsolationMVA3oldDMwLT':{'var':'tau_byMediumIsolationMVA3oldDMwLT > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byMediumIsolationMVA3oldDMwLT'},
        'byTightIsolationMVA3oldDMwLT':{'var':'tau_byTightIsolationMVA3oldDMwLT > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byTightIsolationMVA3oldDMwLT'},

        'byLooseIsolationMVArun2v1DBoldDMwLT':{'var':'tau_byLooseIsolationMVArun2v1DBoldDMwLT > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byLooseIsolationMVArun2v1DBoldDMwLT'},
        'byMediumIsolationMVArun2v1DBoldDMwLT':{'var':'tau_byMediumIsolationMVArun2v1DBoldDMwLT > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byMediumIsolationMVArun2v1DBoldDMwLT'},
        'byTightIsolationMVArun2v1DBoldDMwLT':{'var':'tau_byTightIsolationMVArun2v1DBoldDMwLT > 0.5 && tau_decayModeFindingOldDMs > 0.5', 'nbin':2, 'min':-0.5, 'max':1.5, 'title':'byTightIsolationMVArun2v1DBoldDMwLT'},

        }


    reco_cut = 'tau_pt > 20 && abs(tau_eta) < 2.3'
    loose_id = 'tau_decayModeFindingOldDMs > 0.5 && tau_byLooseCombinedIsolationDeltaBetaCorr3Hits > 0.5'

    sampledict = {
#        '7_6_0_pre7':{'file':'Myroot_7_6_0_pre7_' + runtype + '.root', 'col':2, 'marker':20, 'width':3},
#        '7_6_0':{'file':'Myroot_7_6_0_' + runtype + '.root', 'col':1, 'marker':21, 'width':4},
#        '7_6_1':{'file':'Myroot_7_6_1_' + runtype + '.root', 'col':4, 'marker':22, 'width':2},
        '7_6_1_v3':{'file':'Myroot_7_6_1_v3_' + runtype + '.root', 'col':3, 'marker':23, 'width':1},
        }

    for hname, hdict in sorted(vardict.items()):        

        hists = []

        for rel, rdict in sorted(sampledict.items()):

            if rel.find('7_6_1')==-1 and (hname.find('MVA6')!=-1 or hname.find('MVArun2')!=-1): continue
            
            tfile = TFile(rdict['file'])
            tree = tfile.Get('per_tau')

            num_sel = reco_cut
            den_sel = '1'

            if hname.find('against')!=-1: 
                num_sel = '1'
                den_sel = reco_cut + ' && ' + loose_id

            hists.append(makeEffPlotsVars(tree, 'tau_genpt', num_sel + '&&' + hdict['var'], den_sel, 30, 0, 300, hdict['nbin'], hdict['min'], hdict['max'], xlabel, hdict['title'], '', rel, rel, 'pt', rdict['marker']))


#            if rel=='7_6_1' and (hname.find('MVA5')!=-1 or hname.find('IsolationMVA3')!=-1):
#                xvar = hdict['var'].replace('IsolationMVA3', 'IsolationMVArun2v1DB').replace('MVA5','MVA6')
#                print 'adding', xvar
#
#                hists.append(makeEffPlotsVars(tree, 'tau_genpt', num_sel + '&&' + xvar, den_sel, 30, 0, 300, hdict['nbin'], hdict['min'], hdict['max'], xlabel, hdict['title'], '', rel + '(' + xvar.replace('tau_','').replace('> 0.5','').replace(' && decayModeFindingOldDMs ','') + ')', rel, 'pt', rdict['marker']))



        overlay(hists, hname, hname, hdict['title'])




    hvardict = {
        'tau_dm':{'var':'tau_dm', 'nbin':12, 'min':0., 'max':12, 'title':'decay Mode', 'sel':'1'},
        'tau_mass_1prong':{'var':'tau_mass', 'nbin':30, 'min':0., 'max':2.5, 'title':'Tau mass, 1prong', 'sel':'tau_dm==0'},
        'tau_mass_1prongp0':{'var':'tau_mass', 'nbin':30, 'min':0., 'max':2.5, 'title':'Tau mass, 1prong+#pi^{0}', 'sel':'tau_dm==1'},
        'tau_mass_2prong':{'var':'tau_mass', 'nbin':30, 'min':0., 'max':2.5, 'title':'Tau mass, 2prong', 'sel':'(tau_dm==5 || tau_dm==6)'},
        'tau_mass_3prong':{'var':'tau_mass', 'nbin':30, 'min':0., 'max':2.5, 'title':'Tau mass, 3prong (+#pi^{0})', 'sel':'(tau_dm==10 || tau_dm==11)'},
        'pt_resolution_1prong':{'var':'(tau_genpt-tau_pt)/(tau_genpt)', 'nbin':30, 'min':-1., 'max':1., 'title':'pT resolution, 1prong', 'sel':'tau_dm==0'},
        'pt_resolution_1prongp0':{'var':'(tau_genpt-tau_pt)/(tau_genpt)', 'nbin':30, 'min':-1., 'max':1., 'title':'pT resolution, 1prong+#pi^{0}', 'sel':'tau_dm==1'},
        'pt_resolution_2prong':{'var':'(tau_genpt-tau_pt)/(tau_genpt)', 'nbin':30, 'min':-1., 'max':1., 'title':'pT resolution, 2prong', 'sel':'(tau_dm==5 || tau_dm==6)'},
        'pt_resolution_3prong':{'var':'(tau_genpt-tau_pt)/(tau_genpt)', 'nbin':30, 'min':-1., 'max':1., 'title':'pT resolution, 3prong (+#pi^{0})', 'sel':'(tau_dm==10 || tau_dm==11)'},
        }


    for hname, hdict in sorted(hvardict.items()):        

        hists = []

        if runtype != 'ZTT' and hname.find('pt_resolution')!=-1: continue
        

        for rel, rdict in sorted(sampledict.items()):

            tfile = TFile(rdict['file'])
            tree = tfile.Get('per_tau')

            hist = TH1F('h_' + hname + '_' + rel, 'h_' + hname + '_' + rel, hdict['nbin'], hdict['min'], hdict['max'])
            hist.GetYaxis().SetNdivisions(507)
            hist.SetLineColor(rdict['col'])
            hist.SetLineWidth(rdict['width'])
            hist.SetMinimum(0)
            hist.SetName(rel)
            hist.Sumw2()

            tree.Project(hist.GetName(), hdict['var'], hdict['sel'])        
            hist.Scale(1./hist.GetEntries())

            hists.append(copy.deepcopy(hist))

        hoverlay(hists, hdict['title'], 'a.u.', hname)
