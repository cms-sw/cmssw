#!/usr/bin/env python

#######

#  automatized plots generator for b-tagging performances
#  Adrien Caudron, 2013, UCL

#######

#import all what is needed
from histoStyle import *
#parser options
from optparse import OptionParser
usage="""%prog [options]"""
description="""A simple script to generate validation plots"""
epilog="""Example:
plotFactory.py -f BTagRelVal_TTbar_Startup_600.root -F BTagRelVal_TTbar_Startup_600gspre3.root -r 600 -R 600gspre3 -s TTbar_Startup -S TTbar_Startup 
"""
parser = OptionParser(usage=usage,add_help_option=True,description=description,epilog=epilog)
parser.add_option("-f", "--valInputFile", dest="valPath", default=fileNameVal,
                  help="Read input file for sample to validated", metavar="VALFILE")
parser.add_option("-F", "--refInputFile", dest="refPath", default=fileNameRef,
                  help="Read input file for reference sample", metavar="RAFFILE")
parser.add_option("-r", "--valReleaseName", dest="ValRel", default=ValRel,
                  help="Name to refer to the release/conditions to validate, ex: 600, GTV18 ...", metavar="VALREL")
parser.add_option("-R", "--refReleaseName", dest="RefRel", default=RefRel,
                  help="Name to refer to the reference release/conditions, ex: 600pre11, GTV16 ...", metavar="REFREL")
parser.add_option("-s", "--valSampleName", dest="ValSample", default=ValSample,
                  help="Name to refer to the sample name to validate, ex: TTbar_FullSim, 2012C ...", metavar="VALSAMPLE")
parser.add_option("-S", "--refSampleName", dest="RefSample", default=RefSample,
                  help="Name to refer to the reference sample name, ex: TTbar_FullSim, 2012C ...", metavar="REFSAMPLE")
parser.add_option("-b", "--batch", dest="batch", default=batch,
                  action="store_true", help="if False, the script will run in batch mode")
parser.add_option("-l", "--drawLegend", dest="drawLegend", default=drawLegend,
                  action="store_true", help="if True the legend will be drawn on top of the plots")
parser.add_option("-p", "--printBanner", dest="printBanner", default=printBanner,
                  action="store_true", help="if True, a banner will be print on top of the plots")
parser.add_option("-B", "--Banner", dest="Banner", default=Banner,
                  help="String to write as banner on top of the plots, option -B should be used")
parser.add_option("-n", "--noRatio", dest="doRatio", default=doRatio,
                  action="store_false", help="if True, ratios plots will be created")
(options, args) = parser.parse_args()
print "file for validation", options.valPath, "file for reference", options.refPath
print "Validation release:", options.ValRel, "Reference release:", options.RefRel
print "Validation sample:", options.ValSample, "Reference sample:", options.RefSample
print "Options : batch mode ?", options.batch, "draw legend ?", options.drawLegend, "print banner ?", options.printBanner, "banner is ", options.Banner, "make ratio plots ?", options.doRatio
#define the input root files                                                                                                                                                                              
if options.valPath and options.refPath :
    fileVal = TFile(options.valPath,"READ")
    fileRef = TFile(options.refPath,"READ") 
#batch mode ?
if options.batch : ROOT.gROOT.SetBatch()
# style
_style = Style.Style()
_style.SetStyle()
#title
if options.ValSample==options.RefSample : title=options.ValRel+"vs"+options.RefRel+" "+options.ValSample+" "
elif options.ValRel==options.RefRel : title=options.ValRel+" "+options.ValSample+"_vs_"+options.RefSample+" "
else : title=options.ValRel+"vs"+options.RefRel+" "+options.ValSample+"_vs_"+options.RefSample+" "
#declaration
c = {}
perfAll_Val = {}
perfAll_Ref = {}
perfAll_keys = []
valHistos = {}
refHistos ={}
Histos = {}
ratios = {} 
#loop over eta an pt bins
for b in EtaPtBin :
    #loop over the histos
    for h in listHistos :
        for f in listFlavors :
            perfAll_Val[f] = {}
            perfAll_Ref[f] = {}
        #loop over the list of taggers
        if h.listTagger is None : h.listTagger=listTag
        for tag in h.listTagger :
            keyHisto = tag+"_"+h.name+"_"+b
            if h.doPerformance :
                keyHisto = tag+"_performance_vs_"+h.tagFlavor
            #loop over the flavours
            h_Val = {}
            h_Ref = {}
            passH = False
            for f in listFlavors :
                path = pathInFile+tag+"_"+b+"/"+h.name+"_"+tag+"_"+b+f
                if "_B_" in path : 
                    path=path.replace("_B_","_"+f+"_")
                    path=path.replace(b+f,b)
                print path
                #get histos
                h_Val[f] = fileVal.Get(path)
                h_Ref[f] = fileRef.Get(path)
                if not h_Val[f] :
                    print "ERROR :", path, "not found in the roofiles, please check the spelling or check if this histogram is present in the rootdile"
                    passH = True
            if passH : continue
            #stop if FlavEffVsBEff_?_discr plot for all the taggers
            if h.name=="FlavEffVsBEff_B_discr" :
                for f in listFlavors :
                    perfAll_Val[f][tag]=h_Val[f]
                    perfAll_Ref[f][tag]=h_Ref[f]
                perfAll_keys.append(tag)
                continue
            #create final histos   
            if h.doPerformance :
                valHistos[keyHisto]=graphProducer(plot=h,histos=h_Val,isVal=True)
                refHistos[keyHisto]=graphProducer(plot=h,histos=h_Ref,isVal=False)
            else :    
                valHistos[keyHisto]=histoProducer(plot=h,histos=h_Val,keys=listFlavors,isVal=True)
                refHistos[keyHisto]=histoProducer(plot=h,histos=h_Ref,keys=listFlavors,isVal=False)
            if valHistos[keyHisto] is None or refHistos[keyHisto] is None : continue
            if len(valHistos[keyHisto])!=len(refHistos[keyHisto]) : print "ERROR"
            #compute ratios 
            if options.doRatio :
                if h.doPerformance:
                    ratiosList = createRatioFromGraph(valHistos[keyHisto],refHistos[keyHisto])
                else :
                    ratiosList = createRatio(valHistos[keyHisto],refHistos[keyHisto])
                ratios[keyHisto] = ratiosList
            else :
                ratiosList = None
            #set name file
            if options.ValSample == options.RefSample : saveName=options.ValRel+"vs"+options.RefRel+"_"+options.ValSample+"_Val_"+keyHisto+"_all"
            elif options.ValRel==options.RefRel : saveName=options.ValRel+"_"+options.ValSample+"_vs_"+options.RefSample+"_Val_"+keyHisto+"_all"
            else : saveName=options.ValRel+"vs"+options.RefRel+"_"+options.ValSample+"_vs_"+options.RefSample+"_Val_"+keyHisto+"_all"
            #save canvas
            c[keyHisto] = savePlots(title=title+tag,saveName=saveName,listFromats=listFromats,plot=h,Histos=valHistos[keyHisto]+refHistos[keyHisto],options=options,ratios=ratiosList,keyHisto=keyHisto,listLegend=listFlavors,legendName=h.legend)
        #for FlavEffVsBEff_B_discr
        if h.name=="FlavEffVsBEff_B_discr" :
            for f in ["C","DUSG"] :
                for isVal in [True,False] :
                    keyHisto=f+str(isVal)
                    #setup the histos
                    if isVal : Histos[keyHisto]=histoProducer(plot=h,histos=perfAll_Val[f],keys=perfAll_keys,isVal=isVal)
                    else : Histos[keyHisto]=histoProducer(plot=h,histos=perfAll_Ref[f],keys=perfAll_keys,isVal=isVal)
                    #set name file    
                    if isVal : saveName=options.ValRel+"_"+options.ValSample+"_performance_Bvs"+f+"_allTaggers"
                    else : saveName=options.RefRel+"_"+options.RefSample+"_performance_Bvs"+f+"_allTaggers"
                    #set title
                    if isVal : titleFlav = options.ValRel+"_"+options.ValSample+"_performance_Bvs"+f+"_allTaggers"
                    else : titleFlav = options.RefRel+"_"+options.RefSample+"_performanceBvs"+f+"_allTaggers"
                    #save canvas
                    c[keyHisto] = savePlots(title=titleFlav,saveName=saveName,listFromats=listFromats,plot=h,Histos=Histos[keyHisto],keyHisto=keyHisto,listLegend=h.listTagger,options=options,legendName=h.legend.replace("FLAV",f))
