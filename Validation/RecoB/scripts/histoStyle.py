
####### 

#  automatized plots generator for b-tagging performances
#  Adrien Caudron, 2013, UCL

#######

#do all import
import os, sys

try:
    import ROOT
except:
    print "\nCannot load PYROOT, make sure you have setup ROOT in the path"
    print "and pyroot library is also defined in the variable PYTHONPATH, try:\n"
    if (os.getenv("PYTHONPATH")):
        print " setenv PYTHONPATH ${PYTHONPATH}:$ROOTSYS/lib\n"
    else:
        print " setenv PYTHONPATH $ROOTSYS/lib\n"
        sys.exit()

from ROOT import TFile
from ROOT import TCanvas
from ROOT import TPad
from ROOT import TLegend
from ROOT import TLatex
from ROOT import TH1F
from ROOT import TF1
from ROOT import TVectorD
from ROOT import TGraphErrors
from ROOT import Double

import Style
from listHistos import *

#define the input root files
#fileVal = TFile("BTagRelVal_TTbar_Startup_14_612SLHC2.root","READ")
#fileRef = TFile("BTagRelVal_TTbar_Startup_612SLHC1_14.root","READ")
fileNameVal = "BTagRelVal_TTbar_Startup_14_612SLHC2.root"
fileNameRef = "BTagRelVal_TTbar_Startup_612SLHC1_14.root"
#define the val/ref labels
ValRel = "612SLHC2_14"
RefRel = "612SLHC1_14"
#define the sample labels
ValSample = "TTbar_FullSim"
RefSample = "TTbar_FullSim"
#define different settings
batch = False #run on batch mode ?
weight = 1. #rescale the histos according to this weight
drawLegend = False #draw legend ?
printBanner = False #draw text Banner on top of the histos
Banner = "CMS Preliminary"
doRatio = True #plot the ratios
drawOption = "" # "" or "HIST"
#path in the file
pathInFile = "/DQMData/Run 1/Btag/Run summary/"
#ETA/PT bins, GLOBAL ?
EtaPtBin =[
    "GLOBAL",
    #"ETA_0-1v4",
    #"ETA_1v4-2v4",
    #"PT_50-80",
    #"PT_80-120",
    ]
#list of taggers to look at
listTag = [
    "CSV",
    "CSVMVA",
    "JP",
    "JBP",
    "TCHE",
    "TCHP",
    "SSVHE",
    "SSVHP",
    "SMT",
    #"SMTIP3d",
    #"SMTPt",
    "SET",
    ]
#list of flavors to look at
listFlavors = [
        #"ALL",
        "B",
        "C",
        #"G",
        #"DUS",
        "DUSG",
        #"NI",
        ]
#map for marker color for flav-col and tag-col
mapColor = {
    "ALL"  : 4 ,
    "B"    : 3 ,
    "C"    : 1 ,
    "G"    : 2 ,
    "DUS"  : 2 ,
    "DUSG" : 2 ,
    "NI"   : 5 ,
    "CSV"       : 5 ,
    "CSVMVA"   : 6 ,
    "JP"        : 3 ,
    "JBP"       : 9 ,
    "TCHE"      : 1,
    "TCHP"      : 2,
    "SSVHE"     : 4,
    "SSVHP"     : 7,
    "SMT"       : 8 ,
    "SMTIP3d" : 11 ,
    "SMTPt"   : 12
    }
#marker style map for Val/Ref
mapMarker = {
    "Val" : 22,
    "Ref" :  8
    }
mapLineWidth = {
    "Val" : 3,
    "Ref" : 2
    }
mapLineStyle = {
    "Val" : 2,
    "Ref" : 1
    }
#choose the formats to save the plots 
listFromats = [
    "gif",
    ]
#unity function
unity = TF1("unity","1",-1000,1000)
unity.SetLineColor(8)
unity.SetLineWidth(1)
unity.SetLineStyle(1)
#list of histos to plots
listHistos = [
    jetPt,
    jetEta,
    discr,
    effVsDiscrCut_discr,
    FlavEffVsBEff_discr,
    performance,
    #performanceC,

    #IP,
    #IPe,
    #IPs,
    #NTracks,
    #decayLength,
    #distToJetAxis,
    #NHits,
    #NPixelHits,
    #NormChi2,
    #trackPt,


    #flightDist3Dval,
    #flightDist3Dsig,
    #jetNSecondaryVertices,
    
    #vertexMass,
    #vertexNTracks,
    #vertexJetDeltaR,
    #vertexEnergyRatio,

    #vertexCategory,
    #trackSip3dVal,
    #trackSip3dSig,
    #trackSip3dSigAboveCharm,
    #trackDeltaR,
    #trackEtaRel,
    #trackDecayLenVal,
    #trackSumJetDeltaR,
    #trackJetDist,
    #trackSumJetEtRatio,
    #trackPtRel,
    #trackPtRatio,
    #trackMomentum,
    #trackPPar,
    #trackPParRatio,
    ]

#methode to do a plot from histos       
def histoProducer(plot,histos,keys,isVal=True):
    if histos is None : return
    if isVal : sample = "Val"
    else : sample = "Ref"
    outhistos = []
    minY=9999.
    maxY=0.
    for k in keys :
        #Binning
        if plot.binning and len(plot.binning)==3 :
            histos[k].SetBins(plot.binning[0],plot.binning[1],plot.binning[2])
        elif plot.binning and len(plot.binning)==2 :
            nbins=plot.binning[1]+1-plot.binning[0]
            xmin=histos[k].GetBinLowEdge(plot.binning[0])
            xmax=histos[k].GetBinLowEdge(plot.binning[1]+1)
            valtmp=TH1F(histos[k].GetName(),histos[k].GetTitle(),nbins,xmin,xmax)
            i=1
            for bin in range(plot.binning[0],plot.binning[1]+1) :
                valtmp.SetBinContent(i,histos[k].GetBinContent(bin))
                i+=1
            histos[k]=valtmp
        if plot.Rebin and plot.Rebin > 0 :
            histos[k].Rebin(plot.Rebin)
        #Style
        histos[k].SetLineColor(mapColor[k])
        histos[k].SetMarkerColor(mapColor[k])
        histos[k].SetMarkerStyle(mapMarker[sample])
        if drawOption == "HIST" :
            histos[k].SetLineWidth(mapLineWidth[sample])
            histos[k].SetLineStyle(mapLineStyle[sample])
        #compute errors
        histos[k].Sumw2()
        #do the norm
        if plot.doNormalization :
            histos[k].Scale(1./histos[k].Integral())
        elif weight!=1 :
            histos[k].Scale(weight)
        #get Y min
        if histos[k].GetMinimum(0.) < minY :
            minY = histos[k].GetMinimum(0.)
        #get Y max
        if histos[k].GetBinContent(histos[k].GetMaximumBin()) > maxY :
            maxY = histos[k].GetBinContent(histos[k].GetMaximumBin())+histos[k].GetBinError(histos[k].GetMaximumBin())
        #Axis
        if plot.Xlabel :
            histos[k].SetXTitle(plot.Xlabel)
        if plot.Ylabel :
            histos[k].SetYTitle(plot.Ylabel)
        outhistos.append(histos[k])    
    #Range
    if not plot.logY : outhistos[0].GetYaxis().SetRangeUser(0,1.1*maxY)
    #else : outhistos[0].GetYaxis().SetRangeUser(0.0001,1.05)
    else : outhistos[0].GetYaxis().SetRangeUser(max(0.0001,0.5*minY),1.1*maxY)
    return outhistos        

#method to do a plot from a graph
def graphProducer(plot,histos,tagFlav="B",mistagFlav=["C","DUSG"],isVal=True):
    if histos is None : return
    if isVal : sample = "Val"
    else : sample = "Ref"
    #define graphs
    g = {}
    g_out = []
    if tagFlav not in listFlavors :
        return
    if plot.tagFlavor and plot.mistagFlavor :
        tagFlav = plot.tagFlavor
        mistagFlav = plot.mistagFlavor
    for f in listFlavors :
        #compute errors, in case not already done
        histos[f].Sumw2()
    #efficiency lists
    Eff = {}
    EffErr = {}
    for f in listFlavors :
        Eff[f] = []
        EffErr[f] = []
    #define mapping points for the histos
    maxnpoints = histos[tagFlav].GetNbinsX()
    for f in listFlavors :
        Eff[f].append(histos[f].GetBinContent(1))
        EffErr[f].append(histos[f].GetBinError(1))
    for bin in range(2,maxnpoints+1) :
        #check if we add the point to the graph for Val sample
        if len(Eff[tagFlav])>0 :
            delta = Eff[tagFlav][-1]-histos[tagFlav].GetBinContent(bin)
            if delta>max(0.005,EffErr[tagFlav][-1]) :
                #get efficiencies
                for f in listFlavors :
                    Eff[f].append(histos[f].GetBinContent(bin))
                    EffErr[f].append(histos[f].GetBinError(bin))
    #create TVector
    len_ = len(Eff[tagFlav])
    TVec_Eff = {}
    TVec_EffErr = {}
    for f in listFlavors :
        TVec_Eff[f] = TVectorD(len_)
        TVec_EffErr[f] = TVectorD(len_)
    #Fill the vector
    for j in range(0,len_) :
        for f in listFlavors :
            TVec_Eff[f][j] = Eff[f][j]
            TVec_EffErr[f][j] = EffErr[f][j]
    #fill TGraph
    for mis in mistagFlav :
        g[tagFlav+mis]=TGraphErrors(TVec_Eff[tagFlav],TVec_Eff[mis],TVec_EffErr[tagFlav],TVec_EffErr[mis])
    #style
    for f in listFlavors :
        if f not in mistagFlav : continue
        g[tagFlav+f].SetLineColor(mapColor[f])
        g[tagFlav+f].SetMarkerStyle(mapMarker[sample])
        g[tagFlav+f].SetMarkerColor(mapColor[f])
        g_out.append(g[tagFlav+f])
    index = -1     
    for g_i in g_out :
        index+=1
        if g_i is not None : break
    #Axis
    g_out[index].GetXaxis().SetRangeUser(0,1)
    g_out[index].GetYaxis().SetRangeUser(0.0001,1)
    if plot.Xlabel :
        g_out[index].GetXaxis().SetTitle(plot.Xlabel)
    if plot.Ylabel :
        g_out[index].GetYaxis().SetTitle(plot.Ylabel)
    #add in the list None for element in listFlavors for which no TGraph is computed
    for index,f in enumerate(listFlavors) :
        if f not in mistagFlav : g_out.insert(index,None)
    return g_out   

#method to draw the plot and save it
def savePlots(title,saveName,listFromats,plot,Histos,keyHisto,listLegend,options,ratios=None,legendName="") :
    #create canvas
    c = {}
    pads = {}
    if options.doRatio :
        c[keyHisto] = TCanvas(saveName,keyHisto+plot.title,700,700+24*len(listFlavors))
        pads["hist"] = TPad("hist", saveName+plot.title,0,0.11*len(listFlavors),1.0,1.0)    
    else :
        c[keyHisto] = TCanvas(keyHisto,saveName+plot.title,700,700)
        pads["hist"] = TPad("hist", saveName+plot.title,0,0.,1.0,1.0)
    pads["hist"].Draw()
    if ratios :
        for r in range(0,len(ratios)) :
            pads["ratio_"+str(r)] = TPad("ratio_"+str(r), saveName+plot.title+str(r),0,0.11*r,1.0,0.11*(r+1))
            pads["ratio_"+str(r)].Draw()
    pads["hist"].cd()
    #canvas style                                                                                                                                                                          
    if plot.logY : pads["hist"].SetLogy()
    if plot.grid : pads["hist"].SetGrid()
    #legend                                                                                                                                                                                
    leg = TLegend(0.6,0.4,0.8,0.6)
    leg.SetMargin(0.12)
    leg.SetTextSize(0.035)
    leg.SetFillColor(10)
    leg.SetBorderSize(0)
    #draw histos                                                                                                                                                                           
    first = True
    option = drawOption
    optionSame = drawOption+"same"
    if plot.doPerformance :
        option = "AP"
        optionSame = "sameP"
    for i in range(0,len(Histos)) :
        if Histos[i] is None : continue
        if first :
            if not plot.doPerformance : Histos[i].GetPainter().PaintStat(ROOT.gStyle.GetOptStat(),0)
            Histos[i].SetTitle(title)
            Histos[i].Draw(option)
            first = False
        else : Histos[i].Draw(optionSame)
        #Fill legend                                                                                                                                                                       
        if plot.legend and len(Histos)%len(listLegend)==0:
            r=len(Histos)/len(listLegend)
            index=i-r*len(listLegend)
            while(index<0):
                index+=len(listLegend)
            legName = legendName.replace("KEY",listLegend[index])
            if i<len(listLegend) : legName = legName.replace("isVAL",options.ValRel)
            else : legName = legName.replace("isVAL",options.RefRel)
            if drawOption=="HIST" :
                leg.AddEntry(Histos[i],legName,"L")
            else :
                leg.AddEntry(Histos[i],legName,"P")
    #Draw legend                                                                                                                                                                           
    if plot.legend and options.drawLegend : leg.Draw("same")
    tex = None
    if options.printBanner :
        print type(options.printBanner)
        tex = TLatex(0.55,0.75,options.Banner)
        tex.SetNDC()
        tex.SetTextSize(0.05)
        tex.Draw()
    #save canvas
    if ratios :
        for r in range(0,len(ratios)) :
            pads["ratio_"+str(r)].cd()
            if ratios[r] is None : continue
            pads["ratio_"+str(r)].SetGrid()
            ratios[r].GetYaxis().SetTitle(listLegend[r]+"-jets")
            ratios[r].GetYaxis().SetTitleSize(0.15)
            ratios[r].GetYaxis().SetTitleOffset(0.2)
            ratios[r].GetYaxis().SetNdivisions(3,3,2)
            ratios[r].Draw("")
            unity.Draw("same")
    for format in listFromats :
        save = saveName+"."+format
        c[keyHisto].Print(save)
    return [c,leg,tex,pads]    
#to create ratio plots from histograms
def createRatio(hVal,hRef):
    ratio = []
    for h_i in range(0,len(hVal)): 
        if hVal[h_i] is None : continue
        r = TH1F(hVal[h_i].GetName()+"ratio","ratio "+hVal[h_i].GetTitle(),hVal[h_i].GetNbinsX(),hVal[h_i].GetXaxis().GetXmin(),hVal[h_i].GetXaxis().GetXmax())
        r.Add(hVal[h_i])
        r.Divide(hRef[h_i])
        r.GetYaxis().SetRangeUser(0.25,1.75)
        r.SetMarkerColor(hVal[h_i].GetMarkerColor())
        r.SetLineColor(hVal[h_i].GetLineColor())
        r.GetYaxis().SetLabelSize(0.15)
        r.GetXaxis().SetLabelSize(0.15)
        ratio.append(r)
    return ratio
#to create ratio plots from TGraphErrors
def createRatioFromGraph(hVal,hRef):
    ratio = []
    for g_i in range(0,len(hVal)):
        if hVal[g_i] is None :
            ratio.append(None)
            continue
        tmp = hVal[g_i].GetHistogram()
        histVal = TH1F(hVal[g_i].GetName()+"_ratio",hVal[g_i].GetTitle()+"_ratio",tmp.GetNbinsX(),tmp.GetXaxis().GetXmin(),tmp.GetXaxis().GetXmax())
        histRef = TH1F(hRef[g_i].GetName()+"_ratio",hRef[g_i].GetTitle()+"_ratio",histVal.GetNbinsX(),histVal.GetXaxis().GetXmin(),histVal.GetXaxis().GetXmax())
        #loop over the N points
        for p in range(0,hVal[g_i].GetN()-1):
            #get point p
            x = Double(0)
            y = Double(0)
            hVal[g_i].GetPoint(p,x,y)
            xerr = hVal[g_i].GetErrorX(p)
            yerr = hVal[g_i].GetErrorY(p)
            bin_p = histVal.FindBin(x)
            xHist = histVal.GetBinCenter(bin_p)
            #get the other point as xHist in [x,xbis]
            xbis = Double(0)
            ybis = Double(0)
            #points are odered from high x to low x
            if xHist>x : 
                if p==0 : continue
                xbiserr = hVal[g_i].GetErrorX(p-1)
                ybiserr = hVal[g_i].GetErrorY(p-1)
                hVal[g_i].GetPoint(p-1,xbis,ybis)
            else :
                xbiserr = hVal[g_i].GetErrorX(p+1)
                ybiserr = hVal[g_i].GetErrorY(p+1)
                hVal[g_i].GetPoint(p+1,xbis,ybis)
            if ybis==y : 
                #just take y at x
                bin_p_valContent = y
                bin_p_valContent_errP = y+yerr
                bin_p_valContent_errM = y-yerr
            else :
                #do a linear extrapolation (equivalent to do Eval(xHist))
                a=(ybis-y)/(xbis-x)
                b=y-a*x
                bin_p_valContent = a*xHist+b
                #extrapolate the error
                aerrP = ( (ybis+ybiserr)-(y+yerr) ) / (xbis-x)
                berrP = (y+yerr)-aerrP*x
                bin_p_valContent_errP = aerrP*xHist+berrP
                aerrM = ( (ybis-ybiserr)-(y-yerr) ) / (xbis-x)
                berrM = (y-yerr)-aerrM*x
                bin_p_valContent_errM = aerrM*xHist+berrM
            #fill val hist
            histVal.SetBinContent(bin_p,bin_p_valContent)
            histVal.SetBinError(bin_p,(bin_p_valContent_errP-bin_p_valContent_errM)/2)
            #loop over the reference TGraph to get the corresponding point
            for pRef in range(0,hRef[g_i].GetN()):
                #get point pRef
                xRef = Double(0)
                yRef = Double(0)
                hRef[g_i].GetPoint(pRef,xRef,yRef)
                #take the first point as xRef < xHist
                if xRef > xHist : continue
                xReferr = hRef[g_i].GetErrorX(pRef)
                yReferr = hRef[g_i].GetErrorY(pRef)
                #get the other point as xHist in [xRef,xRefbis]
                xRefbis = Double(0)
                yRefbis = Double(0)
                xRefbiserr = hRef[g_i].GetErrorX(pRef+1)
                yRefbiserr = hRef[g_i].GetErrorY(pRef+1)
                hRef[g_i].GetPoint(pRef+1,xRefbis,yRefbis)
                if yRefbis==yRef :
                    #just take yRef at xRef
                    bin_p_refContent = yRef
                    bin_p_refContent_errP = yRef+yReferr
                    bin_p_refContent_errM = yRef-yReferr
                else :
                    #do a linear extrapolation (equivalent to do Eval(xHist))
                    aRef=(ybis-y)/(xbis-x)
                    bRef=yRef-aRef*xRef
                    bin_p_refContent = aRef*xHist+bRef
                    #extrapolate the error
                    aReferrP = ((yRefbis+yRefbiserr)-(yRef+yReferr))/((xRefbis)-(xRef))
                    bReferrP = (yRef+yReferr)-aReferrP*(xRef-xReferr)
                    bin_p_refContent_errP = aReferrP*xHist+bReferrP
                    aReferrM = ((yRefbis-yRefbiserr)-(yRef-yReferr))/((xRefbis)-(xRef))
                    bReferrM = (yRef-yReferr)-aReferrM*(xRef+xReferr)
                    bin_p_refContent_errM = aReferrM*xHist+bReferrM
                break
            #fill ref hist
            histRef.SetBinContent(bin_p,bin_p_refContent)
            histRef.SetBinError(bin_p,(bin_p_refContent_errP-bin_p_refContent_errM)/2)
        #do the ratio
        histVal.Sumw2()
        histRef.Sumw2()
        histVal.Divide(histRef)
        #ratio style
        histVal.GetXaxis().SetRangeUser(0.,1.)
        #histRef.GetXaxis().SetRangeUser(0.,1.)
        histVal.GetYaxis().SetRangeUser(0.25,1.75)
        histVal.SetMarkerColor(hVal[g_i].GetMarkerColor())
        histVal.SetLineColor(hVal[g_i].GetLineColor())
        histVal.GetYaxis().SetLabelSize(0.15)
        histVal.GetXaxis().SetLabelSize(0.15)
        ratio.append(histVal)
    return ratio
