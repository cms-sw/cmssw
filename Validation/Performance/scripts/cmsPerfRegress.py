#!/usr/bin/env python

import time, os, sys, math, re, gzip
import tempfile as tmp
import optparse as opt
from cmsPerfCommons import CandFname
#from ROOT import gROOT, TCanvas, TF1
import ROOT
from array import array

_cmsver = os.environ['CMSSW_VERSION']
values_set=('vsize','delta_vsize','rss','delta_rss')

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class TimingParseErr(Error):
    """Exception raised when Could not parse TimingReport Log.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class SimpMemParseErr(Error):
    """Exception raised when Could not parse TimingReport Log.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class EdmSizeErr(Error):
    """Exception raised when Could not parse TimingReport Log.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message            

class PerfReportErr(Error):
    """Exception raised when Could not parse TimingReport Log.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message 

def getParameters():
    parser = opt.OptionParser()
    #
    # Options
    #
    parser.add_option('-n',
                      type="int",
                      help='Number of secs per bin. Default is 1.' ,
                      default=1,
                      dest='startevt')
    parser.add_option('-t',
                      '--report-type',
                      type="choice",
                      choices= ("timing", "simplememory","edmsize","igprof","callgrind",""),
                      #store = "store_choices",#["Timing","SimpleMemory"],
                      help='Type of report to perform regrssion on. Default is TimingReport.' ,
                      default="timing",
                      dest='reporttype')      
    (options,args) = parser.parse_args()
    if not len(args) == 2:
        parser.error("ERROR: Not enough arguments")
        sys.exit()


    path1 = os.path.abspath(args[0])
    path2 = os.path.abspath(args[1])    
    if os.path.exists(path1) and os.path.exists(path2):
        return (path1, path2, options.startevt, options.reporttype)
    else:
        print "Error: one of the paths does not exist"
        sys.exit()

def get_max(data,index=1):
    max_time=-1
    for el in data:
        sec=el[index]
        if max_time<sec:
            max_time=sec
    return max_time

def get_min(data,index=1):
    min_time=1e20
    for el in data:
        sec=el[index]
        if min_time>sec:
            min_time=sec
    return min_time  

def setBatch():
    __argv=sys.argv # trick for a strange behaviour of the TApp..
    sys.argv=sys.argv[:1]
    ROOT.gROOT.SetStyle("Plain") # style paranoia
    sys.argv=__argv
    #Cannot use this option when the logfile includes
    #a large number of events... PyRoot seg-faults.
    #Set ROOT in batch mode to avoid canvases popping up!
    ROOT.gROOT.SetBatch(1)
    #ROOT.gEnv.SetValue("Root.MemStat",0)
    #ROOT.gEnv.SetValue("Root.ObjectStat",0)    

def createROOT(outdir,filename):

    # Save in file
    rootfilename = os.path.join(outdir,filename)
    myfile = None
    exists = os.path.exists(rootfilename)
    if exists:
        myfile=ROOT.TFile(rootfilename,'UPDATE')
    else:
        myfile=ROOT.TFile(rootfilename,'RECREATE')        
    return myfile

def getTimingLogData(logfile_name):
    data=[]
    
    # open file and read it and fill the structure!
    logfile=open(logfile_name,'r')
    logfile_lines=logfile.readlines()
    logfile.close()

    # we get the info we need!
    i=0
    while i < len(logfile_lines):
        line=logfile_lines[i]
        if 'TimeEvent>' in line:
            line=line.strip()
            line_content_list = line.split(' ')[0:]
            event_number = int(line_content_list[1])
            seconds = float(line_content_list[3])
            data.append((event_number,seconds))
        i+=1

    return data

def getSimpleMemLogData(logfile_name,startevt):
    data=[]
    values_set=('vsize','delta_vsize','rss','delta_rss')
    
    # open file and read it and fill the structure!
    logfile=open(logfile_name,'r')
    logfile_lines=logfile.readlines()
    logfile.close()
    
    step = ""
    steps = []
    # we get the info we need!
    i=0
    while i < len(logfile_lines):
        line=logfile_lines[i]
        if "RelValreport" in line and "cmsDriver" in line and "step" in line:
            stepreg = re.compile("--step=([^ ]*)")
            found = stepreg.search(line)
            if found:
                if step == "":
                    step = found.groups()[0]
                else:
                    steps.append((step,data))
                    step = found.groups()[0]
                    data = []        
        elif '%MSG-w MemoryCheck:' in line:
            line=line[:-1] #no \n!
            line_content_list=line.split(' ')
            event_number=int(line_content_list[-1])
            if event_number<startevt:
                i+=1
                continue
            i+=1 # we inspect the following line
            try:
                line=logfile_lines[i]
            except IndexError:
                continue
            line=line[:-1] #no \n!
            line_content_list=line.split(' ')
            vsize=float(line_content_list[4])
            delta_vsize=float(line_content_list[5])
            rss=float(line_content_list[7])
            delta_rss=float(line_content_list[8])
            
            data.append((event_number,{'vsize':vsize,
                                       'delta_vsize':delta_vsize,
                                       'rss':rss,
                                       'delta_rss':delta_rss}))
        i += 1

    if not len(data) == 0:
        steps.append((step,data))

    return steps

def newGraphAndHisto(histoleg,leg,npoints,nbins,min_val,max_val,data,graph_num,prevrev=""):
    
    colors = [2,4]
    prevRevName = "???"
    if not prevrev == "":
        (head, tail) = os.path.split(prevrev)
        prevRevName  = os.path.basename(tail)
    releaseNames = ["Previous (%s)" % prevRevName,_cmsver]

    histo=ROOT.TH1F("Seconds per event (histo: %s)" % graph_num,'Seconds per event',nbins,min_val,max_val)

    graph=ROOT.TGraph(npoints)
        
    evt_counter=0
    peak = data[0][1]
    for evt_num,secs in data:
        if secs > peak:
            peak = secs
        graph.SetPoint(evt_counter,evt_num,secs)
        histo.Fill(secs)        
        evt_counter+=1

    allsecs = []
    map(lambda x: allsecs.append(x[1]),data)
    total = reduce(lambda x,y: x + y,allsecs)
    mean  = total / evt_counter
    rms   = math.sqrt(reduce(lambda x,y: x + y,map(lambda x: x * x,allsecs)) / evt_counter)
                
        
    graph.SetMarkerStyle(8)
    graph.SetMarkerSize(.7)
    graph.SetMarkerColor(1)
    graph.SetLineWidth(3)
    graph.SetLineColor(colors[graph_num]) # for each iterate through colors
    histo.SetLineColor(colors[graph_num])
    histo.SetStats(1)
    histoleg.AddEntry(histo, "%s release" % releaseNames[graph_num], "l")
    leg.AddEntry(graph     , "%s release" % releaseNames[graph_num], "l")
    leg.AddEntry(graph     , "Mean: %s" % str(mean)                , "l")            
    leg.AddEntry(graph     , "RMS : %s" % str(rms)                 , "l")
    leg.AddEntry(graph     , "Peak: %s" % str(peak)                , "l")
    leg.AddEntry(graph     , "Total time: %s" % str(total)         , "l")                    
    if graph_num == 0 :
        histo.SetFillColor(colors[graph_num])

    return (graph,histo,mean)

def getLimits(data,secsperbin):
    min_val=get_min(data,1)
    max_val=get_max(data,1)
    interval=int(max_val-min_val)
    
    min_val=min_val-interval*0.2
    max_val=max_val+interval*0.2
    interval=int(max_val-min_val)
    
    nbins=int(interval/secsperbin)

    npoints=len(data)

    last_event=data[-1][0]

    #print 'last event =',last_event    

    return (min_val,max_val,interval,npoints,last_event)

def setupSuperimpose(graph1,graph2,last_event,max_val,reporttype=0, title = ""):
    name   = ""
    xtitle = ""
    ytitle = ""
    if   reporttype == 0:
        title  = 'Seconds per event'
        name   = 'SecondsPerEvent'
        xtitle = "Event"
        ytitle = "Processing time for each event (s)"
        graph1.GetYaxis().SetRangeUser(0,max_val)
        graph2.GetYaxis().SetRangeUser(0,max_val)            
    elif reporttype == 1:
        name   = "%s_graph" % title
        xtitle = "Event"
        ytitle = "MB"
        
    graph1.SetTitle(title)
    graph1.SetName(name)
    graph1.GetXaxis().SetTitle(xtitle)
    graph1.GetYaxis().SetTitleOffset(1.3)
    graph1.GetYaxis().SetTitle(ytitle)
    graph1.GetXaxis().SetLimits(0,last_event)
    # Do I need to set limits on graph 2? I don't know
    # I'm doing it anyway, can't hurt.
    graph2.GetXaxis().SetLimits(0,last_event)


def getMeanLines(avg,last_event,graph_num):
    colors = [2,4]
    avg_line=ROOT.TLine(1,avg,last_event,avg)
    avg_line.SetLineColor(colors[graph_num])
    avg_line.SetLineWidth(2)

    return avg_line

def getTimingDiff(data1,data2,npoints,last_event,orig_max_val):
    data3 = []
    for x in range(len(data2)):
        try:
            #avgEventNum = (data2[x][0] + data1[x][0]) / 2
            if data2[x][0] == data1[x][0]:
                avgEventNum = data2[x][0]
                diffSecs    = data2[x][1] - data1[x][1]                
                data3.append((avgEventNum,diffSecs))                
        except IndexError:
            pass
        except ValueError:
            pass

    graph=ROOT.TGraph(npoints)

    evt_counter=0
    peak = data3[0][1]
    for evt_num,secs in data3:
        if secs > peak:
            peak = secs
        graph.SetPoint(evt_counter,evt_num,secs)
        evt_counter+=1

    allsecs = []
    map(lambda x: allsecs.append(x[1]),data3)
    total = reduce(lambda x,y: x + y,allsecs)
    mean  = total / evt_counter
    rms   = math.sqrt(reduce(lambda x,y: x + y,map(lambda x: x * x,allsecs)) / evt_counter)
        
    min_val=get_min(data3,1)
    max_val=get_max(data3,1)
    interval=int(max_val-min_val)
    
    min_val=min_val-interval*0.2
    max_val=max_val+interval*0.2
    interval=int(max_val-min_val)
    #print min_val
    # Determine the max value to be something that makes the scale similar to what
    # the original graph had. Unless we can't seem the maximum value.

    new_max = min_val + orig_max_val
    #print "new max", new_max, "min_val", min_val, "max_val", max_val
    if new_max < max_val:
        pass
    else :
        max_val = new_max
    
    graph.SetTitle('Change in processing time for each event between revs')
    graph.SetName('SecondsPerEvent')    
    graph.GetXaxis().SetTitle("Event Number")
    graph.GetYaxis().SetTitle("Change in processing time between revs (s)")    
    graph.GetYaxis().SetTitleOffset(1.3)
    #graph.GetYaxis().SetTitle("s")
    graph.SetLineColor(2)
    graph.SetMarkerStyle(8)
    graph.SetMarkerSize(.7)
    graph.SetMarkerColor(1)
    graph.SetLineWidth(3)        
    graph.GetXaxis().SetLimits(0,last_event)
    graph.GetYaxis().SetRangeUser(min_val,max_val)
    leg = ROOT.TLegend(0.5,0.7,0.89,0.89)
    leg.AddEntry(graph, "Mean: %s" % str(mean), "l")            
    leg.AddEntry(graph, "RMS : %s" % str(rms) , "l")
    leg.AddEntry(graph, "Peak: %s" % str(peak), "l")
    leg.AddEntry(graph, "Total time change: %s" % str(total)  , "l")                    

    return (graph,leg)

def drawGraphs(graph1,graph2,avg1,avg2,leg):
    graph_canvas = ROOT.TCanvas("graph_canvas")
    graph_canvas.cd()
    graph1.Draw("ALP")
    graph2.Draw("LP")
    avg1.Draw("Same")
    avg2.Draw("Same")
#    legPad = ROOT.TPad("legend","Trans pad",0,0,1,1)
#    legPad.cd()
#    legPad.Draw()
    leg.Draw()
    return graph_canvas

def drawHistos(histo_stack,hstleg):
    histo_canvas = ROOT.TCanvas("histo_canvas")
    histo_canvas.cd()
    histo_stack.Draw("nostack")
    hstleg.Draw()
    return histo_canvas

def drawChanges(graph,chgleg):
    graph_canvas = ROOT.TCanvas("change_canvas")
    graph_canvas.cd()
    graph.Draw("ALP")
    chgleg.Draw()
    return graph_canvas    

def getTwoGraphLimits(last_event1,max_val1,last_event2,max_val2,min_val1=-1,min_val2=-1):
    biggestLastEvt = last_event1
    biggestMaxval  = max_val1
    lowest_val     = min_val1
    
    if min_val2 < lowest_val:
        lowest_val = min_val2
    if last_event2 > biggestLastEvt:
        biggestLastEvt = last_event2
    if max_val2 > biggestMaxval:
        biggestMaxval  = max_val2
    return (biggestLastEvt,biggestMaxval,lowest_val)

def getNpoints(data):
    new_data=[]
    try:
        if data[0][0]==data[1][0]:
            print 'Two modules seem to have some output.\nCollapsing ...'
            i=0
            while True:
                dataline1=data[i]
                i+=1
                dataline2=data[i]
                new_eventnumber=dataline1[0]
                new_vsize=dataline2[1]['vsize']
                new_delta_vsize=dataline1[1]['delta_vsize']+dataline2[1]['delta_vsize']
                new_rss=dataline2[1]['rss']
                new_delta_rss=dataline1[1]['delta_rss']+dataline2[1]['delta_rss']

                new_data.append((new_eventnumber,{'vsize':new_vsize,
                                                  'delta_vsize':new_delta_vsize,
                                                  'rss':new_rss,
                                                  'delta_rss':new_delta_rss}))
                i+=1
                if i==len(data): break

            data=new_data
            print 'Collapsing: Done!'        
    except IndexError:
        pass

    return (data,len(data))

def createSimplMemGraphs(data,npoints,graph_num,legs,prevrev=""):
    colors = [2,4]    
    values = ["vsize", "rss"]
    
    prevRevName = "???"
    if not prevrev == "":
        (head, tail) = os.path.split(prevrev)
        prevRevName  = os.path.basename(tail)
        
    releaseNames = ["Previous (%s)" % prevRevName,_cmsver]        
    #rss_graph  = ROOT.TGraph(npoints)
    #newSimplGraph(rss_graph,values[1],graph_num)

    #fill the graphs
    graphs = []
    minim  = -1
    peak   = -1
    peaks  = []
    minims = []
    idx = 0
    for value in values:
        leg = legs[idx]

        graph = ROOT.TGraph(npoints)
        graph.SetTitle(value)
        #graph.SetName('%s_graph' % value)
        graph.SetLineColor(colors[graph_num])
        graph.SetMarkerStyle(8)
        graph.SetMarkerSize(.7)
        graph.SetMarkerColor(1)
        graph.SetLineWidth(3)
        graph.GetXaxis().SetTitle("Event")
        graph.GetYaxis().SetTitleOffset(1.3)
        graph.GetYaxis().SetTitle("MB")

        total = 0
        point_counter=0
        rms   = 0
        first = True
        for event_number,vals_dict in data:
            if first:
                minim = vals_dict[value]
                peak  = vals_dict[value]
                first = False
            if vals_dict[value] > peak:
                peak = vals_dict[value]
            if vals_dict[value] < minim:
                minim = vals_dict[value]
            #if value == values[0]:
            graph.SetPoint(point_counter, event_number, vals_dict[value])
            #else:
            #rss_graph.SetPoint(point_counter, event_number, vals_dict["rss"])
            total += vals_dict[value]
            rms   += vals_dict[value] * vals_dict[value]
            point_counter+=1

        rms  = math.sqrt(rms / float(point_counter))
        mean = total / float(point_counter)
        last_event=data[-1][0]
        peaks.append(peak)
        minims.append(minim)
        graph.GetXaxis().SetRangeUser(0,last_event+1)
        leg.AddEntry(graph     , "%s release" % releaseNames[graph_num], "l")
        leg.AddEntry(graph     , "Mean: %s" % str(mean)                , "l")            
        leg.AddEntry(graph     , "RMS : %s" % str(rms)                 , "l")
        leg.AddEntry(graph     , "Peak: %s" % str(peak)                , "l")
        graphs.append(graph)
        idx += 1
        #rss_graph.GetXaxis().SetRangeUser(0,last_event+1)    

    return (graphs[0] , last_event, peaks[0], minims[0], graphs[1], last_event, peaks[1], minims[1])

def getMemDiff(data1,data2,npoints,last_event,orig_max_val,stepname,rss=False):
    data3 = []
    memtype = "vsize"
    if rss:
        memtype = "rss"

    graph=ROOT.TGraph(npoints)
    total = 0
    rms = 0
    evt_counter=0
    peak  = -1
    minum = -1
    first = True
    for x in range(len(data2)):
        try:
            #eventNum = (data2[x][0] + data1[x][0]) / 2
            (evtnum2,valdict2) = data2[x]
            (evtnum1,valdict1) = data1[x]
            if evtnum2 == evtnum1:
                diffMBytes = valdict2[memtype] - valdict1[memtype]

                if first:
                    peak  = diffMBytes
                    minum = diffMBytes
                    first = False
                if diffMBytes > peak:
                    peak  = diffMBytes
                if diffMBytes < minum:
                    minum = diffMBytes   
                graph.SetPoint(evt_counter,evtnum2,diffMBytes)
                evt_counter+=1
                total += diffMBytes
                rms += (diffMBytes * diffMBytes)
        except IndexError:
            pass
        except ValueError:
            pass
   
    mean  = total / evt_counter
    rms   = math.sqrt(rms / float(evt_counter))
        
    min_val  = minum
    max_val  = peak
    interval = int(max_val-min_val)
    
    min_val=min_val-interval*0.2
    max_val=max_val+interval*0.2
    interval=int(max_val-min_val)
    #print min_val
    # Determine the max value to be something that makes the scale similar to what
    # the original graph had. Unless we can't seem the maximum value.

    new_max = min_val + orig_max_val
    #print "new max", new_max, "min_val", min_val, "max_val", max_val
    if new_max < max_val:
        pass
    else :
        max_val = new_max
    
    graph.SetTitle("Change in %s memory usage for each event between revs for step %s" % (memtype,stepname))
    graph.SetName('MemoryUsagePerEvent')    
    graph.GetXaxis().SetTitle("Event Number")
    graph.GetYaxis().SetTitle("Change in memory usage between revs (MBs)")    
    graph.GetYaxis().SetTitleOffset(1.3)
    graph.SetLineColor(2)
    graph.SetMarkerStyle(8)
    graph.SetMarkerSize(.7)
    graph.SetMarkerColor(1)
    graph.SetLineWidth(3)    
    #graph.GetYaxis().SetTitle("s")
    graph.GetXaxis().SetLimits(0,last_event)
    graph.GetYaxis().SetRangeUser(min_val,max_val)
    leg = ROOT.TLegend(0.5,0.7,0.89,0.89)
    leg.AddEntry(graph, "Mean: %s" % str(mean), "l")            
    leg.AddEntry(graph, "RMS : %s" % str(rms) , "l")
    leg.AddEntry(graph, "Peak: %s" % str(peak), "l")
    leg.AddEntry(graph, "Trough: %s" % str(minum)  , "l")                    

    return (graph,leg)

def drawMemGraphs(graph1,graph2,min_val,max_val,last_event,leg,memtype,stepname):
    graph_canvas=ROOT.TCanvas("%s_%s_canvas" % (memtype,stepname))
    graph_canvas.cd()
    graph1.GetYaxis().SetRangeUser(min_val,max_val)
    graph1.GetXaxis().SetRangeUser(0,last_event)        
    graph1.Draw("ALP")
    graph2.Draw("LP" )
    leg.Draw()    
    graph_canvas.ForceUpdate()
    graph_canvas.Flush()
    return graph_canvas

def drawMemChangeGraphs(graph,leg,memtype,stepname):
    graph_canvas=ROOT.TCanvas("%s_%s_change_canvas" % (memtype,stepname))
    graph_canvas.cd()
    graph.Draw("ALP" )
    leg.Draw()    
    graph_canvas.ForceUpdate()
    graph_canvas.Flush()
    return graph_canvas

def getMemOrigScale(fst_min,snd_min,fst_max,snd_max):
    minim = fst_min
    if snd_min < minim:
        minim = snd_min
    maxim = fst_max
    if snd_max > maxim:
        maxim = snd_max
    return (minim,maxim)
        
def cmpSimpMemReport(rootfilename,outdir,oldLogfile,newLogfile,startevt,batch=True,candle="",prevrev=""):
    if batch:
        setBatch()
    # the fundamental structure: the key is the evt number the value is a list containing
    # VSIZE deltaVSIZE RSS deltaRSS
    try:
        info1 = getSimpleMemLogData(oldLogfile,startevt)
        if len(info1) == 0:
            raise IndexError
    except IndexError:
        raise SimpMemParseErr(oldLogfile)
    except IOError:
        raise SimpMemParseErr(oldLogfile)        
    
    try:
        info2 = getSimpleMemLogData(newLogfile,startevt)
        if len(info2) == 0:
            raise IndexError
    except IndexError:
        raise SimpMemParseErr(newLogfile)
    except IOError:
        raise SimpMemParseErr(newLogfile)        

    canvases = []
    # skim the second entry when the event number is the same BUG!!!!!!!
    # i take elements in couples!

    candreg = re.compile("(.*)(?:\.log)")
    vsize_canvas = None
    rss_canvas = None
    i = 0
    firstRoot = True
    newrootfile = None
    while ( i < len(info1) and i < len(info2)):
        curinfo1 = info1[i]
        curinfo2 = info2[i]
        (stepname1, data1) = curinfo1
        (stepname2, data2) = curinfo2

        if not stepname1 == stepname2:
            print "WARNING: Could not compare %s step and %s step because they are not the same step" % (step1,step2)
            print " Searching for next occurence"
            x = 1
            if not (i + 1) > len(info1):
                found = False
                for info in info1[i + 1:]:
                    (stepname,trash) = info
                    if stepname == stepname2:
                        i += x
                        print " Next occurence found, skipping in-between steps"
                        assert i < len(info1)
                        notFound = True
                        break
                    x += 1
                if found:
                    continue
            print " No more occurences of this step, continuing" 
            i += 1
            continue
                        
        (data1,npoints1) = getNpoints(data1)
        (data2,npoints2) = getNpoints(data2)

        legs = []
        leg      = ROOT.TLegend(0.6,0.99,0.89,0.8)
        legs.append(leg)
        leg      = ROOT.TLegend(0.6,0.99,0.89,0.8)
        legs.append(leg)
        
        try:
            if len(data1) == 0:
                raise IndexError
            (vsize_graph1,
             vsize_lstevt1,
             vsize_peak1,
             vsize_minim1,
             rss_graph1,
             rss_lstevt1,
             rss_peak1,
             rss_minim1) = createSimplMemGraphs(data1,npoints1,0,legs,prevrev=prevrev)
        except IndexError:
            raise SimpMemParseErr(oldLogfile)

        try:
            if len(data2) == 0:
                raise IndexError
            (vsize_graph2,
             vsize_lstevt2,
             vsize_peak2,
             vsize_minim2,
             rss_graph2,
             rss_lstevt2,
             rss_peak2,
             rss_minim2) = createSimplMemGraphs(data2,npoints2,1,legs,prevrev=prevrev)
        except IndexError:
            raise SimpMemParseErr(newLogfile)  


        (vsize_lstevt, vsize_max_val, vsize_min_val) = getTwoGraphLimits(vsize_lstevt1, vsize_peak1, vsize_lstevt2, vsize_peak2, vsize_minim1, vsize_minim2)
        (rss_lstevt  , rss_max_val  , rss_min_val)   = getTwoGraphLimits(rss_lstevt1  , rss_peak1, rss_lstevt2  , rss_peak2, rss_minim1,   rss_minim2)    

        (vsize_min,vsize_max) = getMemOrigScale(vsize_minim1,vsize_minim2,vsize_peak1,vsize_peak2)
        (rss_min  ,rss_max  ) = getMemOrigScale(rss_minim1,rss_minim2,rss_peak1,rss_peak2)

        setupSuperimpose(vsize_graph1,
                         vsize_graph2,
                         vsize_lstevt,
                         0,
                         reporttype = 1,
                         title = "%s_vsize" % stepname1)
        setupSuperimpose(rss_graph1  ,
                         rss_graph2  ,
                         rss_lstevt  ,
                         0,
                         reporttype = 1,
                         title = "%s_rss"  %  stepname2)

        (vsizePerfDiffgraph, vsizeleg) = getMemDiff(data1,data2,npoints2,vsize_lstevt, (vsize_max - vsize_min), stepname1, rss=False)
        (rssPerfDiffgraph, rssleg)     = getMemDiff(data1,data2,npoints2,rss_lstevt  , (rss_max - rss_min)    , stepname1, rss=True )        
        
    ##     avg_line1 = getMeanLines(mean1,last_event1,0)
    ##     avg_line2 = getMeanLines(mean2,last_event2,1)

        vsize_canvas = drawMemGraphs(vsize_graph1, vsize_graph2, vsize_min_val, vsize_max_val, vsize_lstevt, legs[0], "vsize", stepname1)
        rss_canvas   = drawMemGraphs(rss_graph1  , rss_graph2  , rss_min_val, rss_max_val, rss_lstevt, legs[1], "rss"  , stepname1)
        vsize_change_canvas = drawMemChangeGraphs(vsizePerfDiffgraph, vsizeleg, "vsize", stepname1)         
        rss_change_canvas   = drawMemChangeGraphs(rssPerfDiffgraph  , rssleg  , "rss"  , stepname1)         

        if batch:


            logcandle = ""
            candname  = ""            
            found = candreg.search(os.path.basename(newLogfile))
            
            if found:
                logcandle = found.groups()[0]
                
            if   CandFname.has_key(candle):
                candFilename = CandFname[candle]
            elif CandFname.has_key(logcandle):
                candFilename = CandFname[logcandle]
            else:
                candFilename = "Unknown-candle"

            outputdir = "%s_%s_SimpleMemReport" % (candFilename,stepname1)
            outputdir = os.path.join(outdir,outputdir)

            if not os.path.exists(outputdir):
                os.mkdir(outputdir)

            #print the graphs as files :)

            newrootfile = createROOT(outputdir,rootfilename)                
            
            vsize_canvas.Print(       os.path.join(outputdir,"vsize_graphs.gif"), "gif")
            rss_canvas.Print(         os.path.join(outputdir,"rss_graphs.gif"  ), "gif")
            vsize_change_canvas.Print(os.path.join(outputdir,"vsize_change.gif"), "gif")
            rss_change_canvas.Print(  os.path.join(outputdir,"rss_change.gif"  ), "gif")
            # write it on file
            map(lambda x: x.Write(), [vsize_graph1,vsize_graph2, rss_graph1, rss_graph2, vsizePerfDiffgraph, rssPerfDiffgraph])
            map(lambda x: x.Write(), [vsize_canvas,rss_canvas,vsize_change_canvas,rss_change_canvas])
            newrootfile.Close()
        else:
            # we have to do this if we are running the application standalone
            # For some reason it will not draw some graphs at all if there is more than
            # one step.
            # If we wait between iterations of this loop the graphs will be drawn correctly.
            # Perhaps a graphics buffer problem with ROOT?
            # Perhaps a garbage collection problem in python? (Doubt it)
            canvases.append(rss_canvas)
            canvases.append(vsize_canvas)
            canvases.append(vsize_change_canvas)
            canvases.append(rss_change_canvas)            
            time.sleep(5.0)

        i += 1
        
    #
    # Create a one dimensional function and draw it
    #

    if batch:
        pass
    else:
        if len(canvases) > 0:
            while reduce(lambda x,y: x or y ,canvases):
                time.sleep(2.5)
    return 0            
        

def cmpTimingReport(rootfilename,outdir,oldLogfile,newLogfile,secsperbin,batch=True,prevrev=""):
    if batch:
        setBatch()
    
    data1 = getTimingLogData(oldLogfile)
    data2 = getTimingLogData(newLogfile)        
        
    try:
        (min_val1,max_val1,nbins1,npoints1,last_event1) = getLimits(data1,secsperbin)
    except IndexError, detail:
        raise TimingParseErr(oldLogfile)
    
    try:
        (min_val2,max_val2,nbins2,npoints2,last_event2) = getLimits(data2,secsperbin)
    except IndexError, detail:
        raise TimingParseErr(newLogfile)

    hsStack  = ROOT.THStack("hsStack","Histogram Comparison")
    leg      = ROOT.TLegend(0.6,0.99,0.89,0.8)
    histoleg = ROOT.TLegend(0.5,0.8,0.89,0.89)    
    #if not nbins1 == nbins2:
    #    print "ERRORL bin1 %s is not the same size as bin2 %s" % (nbins1,nbins2)

    (graph1,histo1,mean1) = newGraphAndHisto(histoleg,leg,npoints1,nbins1,min_val1,max_val1,data1,0,prevrev)
    hsStack.Add(histo1)
    (graph2,histo2,mean2) = newGraphAndHisto(histoleg,leg,npoints2,nbins2,min_val2,max_val2,data2,1,prevrev)
    hsStack.Add(histo2)

    (biggestLastEvt,biggestMaxval, trashthis) = getTwoGraphLimits(last_event1,max_val1,last_event2,max_val2,min_val1,min_val2)
    
    (changegraph,chgleg) = getTimingDiff(data1,data2,npoints2,biggestLastEvt,biggestMaxval)
    setupSuperimpose(graph1,graph2,biggestLastEvt,biggestMaxval)
    avg_line1 = getMeanLines(mean1,last_event1,0)
    avg_line2 = getMeanLines(mean2,last_event2,1)

    #
    # Create a one dimensional function and draw it
    #
    histo1.GetXaxis().SetTitle("s")            
    graph_canvas   = drawGraphs(graph1,graph2,avg_line1,avg_line2,leg)
    changes_canvas = drawChanges(changegraph,chgleg)
    histo_canvas   = drawHistos(hsStack,histoleg)
    
    newrootfile = None
    if batch:

        newrootfile = createROOT(outdir,rootfilename)

##         (rootfile1, rootfile2) = (oldLogfile, newLogfile)
##         rootreg = re.compile("(.*)_TimingReport.log")
##         found = rootreg.search(oldLogfile)
##         if found:
##             rootfile1 = found.groups()[0] + ".root"
##         found = rootreg.search(newLogfile)
##         if found:
##             rootfile2 = found.groups()[0] + ".root"

##         (fsize1, fsize2) = (0.0, 0.0)
##         if os.path.exists(rootfile1):
##             statinfo = os.stat(rootfile1)
##             fsize1 = statinfo.st_size
##         if os.path.exists(rootfile2):
##             statinfo = os.stat(rootfile2)
##             fsize2 = statinfo.st_size

##         fs_t = ROOT.TTree()
##         fs1 = array("i", [0])
##         fs2 = array("i", [0])
##         fs1[0] = fsize1
##         fs2[0] = fsize2

##         fs_t.Branch("fsize1",fs1,"fsize1/I")
##         fs_t.Branch("fsize2",fs2,"fsize2/I")
##         fs_t.Fill()
##         fs_t.Write("fsize_tuple",ROOT.TObject.kOverwrite)        

        cput = ROOT.TTree()
        #  array(typecode, initializer)
        #  typecode is i for int, f for float etc.
        tot_a1 = array( "f", [ 0 ] )
        tot_a2 = array( "f", [ 0 ] )

        tot_a1[0] = mean1
        tot_a2[0] = mean2

        cput.Branch("total1",tot_a1,"total1/F")
        cput.Branch("total2",tot_a2,"total2/F")
        cput.Fill()
        cput.Write("cpu_time_tuple",ROOT.TObject.kOverwrite)
        
        names = ["graphs.gif","changes.gif","histos.gif"]
        
        graph_canvas.Print(  os.path.join(outdir,names[0]),"gif")
        changes_canvas.Print(os.path.join(outdir,names[1]),"gif")
        histo_canvas.Print(  os.path.join(outdir,names[2]),"gif")
        
        map(lambda x:x.Write(),[graph1,graph2,changegraph,hsStack,histo1,histo2])
        
        graph_canvas.Write()    # to file
        changes_canvas.Write()
        histo_canvas.Write() 
        newrootfile.Close()   

        return names
    else:
        
        while graph_canvas or histo_canvas or changes_canvas:
            time.sleep(2.5)
        return 0
    
def rmtree(path):
    try:
        os.remove(path)
    except OSError, detail:
        if detail.errno == 39:
            try:
                gen = os.walk(path)
                nodes    = gen.next()
                nodes[0] = par
                nodes[1] = dirs
                nodes[2] = files
                for f in files:
                    os.remove(os.path.join(path,f))
                for d in dirs:
                    rmtree(os.path.join(path,d))
            except OSError, detail:
                print detail
            except IOError, detail:
                print detail
            os.remove(path)

def perfreport(perftype,file1,file2,outdir):
    src = ""
    try:
        src = os.environ["CMSSW_SEARCH_PATH"]
    except KeyError , detail:
        print "ERROR: scramv1 environment could not be located", detail 

    vars = src.split(":")
    loc  = vars[0]

    proftype = ""
    if   perftype == 0: # EdmSize
        proftype = "-fe"
    elif perftype == 1: # IgProf
        proftype = "-fi"
    else:               # Callgrind
        proftype = "-ff"

    # -c xml.file
    #

    cmssw_base = ""
    cmssw_data = ""
    try:
        cmssw_base = os.environ['CMSSW_BASE']
        cmssw_data = os.environ['CMSSW_DATA_PATH']
    except KeyError, detail:
        raise PerfReportErr

    xmlfile = os.path.join(cmssw_base,"src","Validation","Performance","doc","regress.xml")

    prRoot = "/afs/cern.ch/user/g/gbenelli/public/PerfReport2/2.0.1"

    # this might be useful at some point
    #cd %s ; eval `scramv1 runtime -csh`  ; source $CMSSW_DATA_PATH/perfreport/2.0.0/etc/profile.d/init.csh; cd - ; %s\"" % (loc,perfcmd)

    # Before adding Danilo's 2.1 we did this
    #perfcmd = "perfreport -tmp %s -i %s -r %s -o %s" % (proftype,file2,file1,outdir)    
    #cmd = "tcsh -c \"source %s/perfreport/2.0.0/etc/profile.d/init.csh; cd - ; %s\"" % (cmssw_data,perfcmd)

    # now we do

    tmpdir  = tmp.mkdtemp(prefix=os.path.join(outdir,"tmp"))

    perfcmd = "%s %s -c %s -t%s -i %s -r %s -o %s" % (os.path.join(prRoot,"bin","perfreport"),proftype,xmlfile,tmpdir,file2,file1,outdir)            
    cmd     = "tcsh -c \"cd %s ; eval `scramv1 runtime -csh` ; cd - ;source %s/etc/profile.d/init.csh ; %s\"" % (loc,prRoot,perfcmd)
    
    process  = os.popen(cmd)
    cmdout   = process.read()
    exitstat = process.close()

    try:
        rmtree(tmpdir)        
        os.rmdir(tmpdir)
    except IOError, detail:
        print "WARNING: Could not remove dir because %s" % detail                
    except OSError, detail:
        print "WARNING: Could not remove dir because %s" % detail                

    if True:
        print cmd
        print cmdout

    if not exitstat == None:
        sig     = exitstat >> 16    # Get the top 16 bits
        xstatus = exitstat & 0xffff # Mask out all bits except the bottom 16
        raise PerfReportErr("ERROR: PerfReport returned a non-zero exit status (%s, SIG_INT = %s) run %s. Dump follows: \n%s" % (perfcmd,xstatus,sig,cmdout))
    
    
def cmpEdmSizeReport(outdir,file1,file2):
    perfreport(0,file1,file2,outdir)

def ungzip(inf,outh):
    gzf = gzip.open(inf,"r")
    print "ungzipping"
    for char in gzf:
        os.write(outh,char)
    os.close(outh)
    print "finish ungzipping"
    gzf.close()

def ungzip2(inf,out):
    os.system("gzip -c -d %s > %s" % (inf,out)) 

def cmpIgProfReport(outdir,file1,file2):
    (tfile1, tfile2) = ("", "")
    try:
        # don't make temp files in /tmp because it's never bloody big enough
        (th1, tfile1) = tmp.mkstemp(prefix=os.path.join(outdir,"igprofRegressRep."))
        (th2, tfile2) = tmp.mkstemp(prefix=os.path.join(outdir,"igprofRegressRep."))
        os.close(th1)
        os.close(th2)
        os.remove(tfile1)
        os.remove(tfile2)
        #ungzip(file1,th1)
        #ungzip(file2,th2)
        ungzip2(file1,tfile1)
        ungzip2(file2,tfile2)        

        perfreport(1,tfile1,tfile2,outdir)

        os.remove(tfile1)
        os.remove(tfile2)
    except OSError, detail:
        raise PerfReportErr("WARNING: The OS returned the following error when comparing %s and %s\n%s" % (file1,file2,str(detail)))
        if os.path.exists(tfile1):
            os.remove(tfile1)
        if os.path.exists(tfile2):
            os.remove(tfile2)
    except IOError, detail:
        raise PerfReportErr("IOError: When comparing %s and %s using temporary files %s and %s. Error message:\n%s" % (file1,file2,tfile1,tfile2,str(detail)))
        if os.path.exists(tfile1):
            os.remove(tfile1)
        if os.path.exists(tfile2):
            os.remove(tfile2)        


def cmpCallgrindReport(outdir,file1,file2):
    perfreport(2,file1,file2,outdir)

def _main():
    outdir = os.getcwd()
    
    (file1,file2,secsperbin,reporttype)  = getParameters()

    try:
        if   reporttype == "timing":
            rootfilename = "timingrep-regression.root"
            cmpTimingReport(rootfilename ,outdir,file1,file2,secsperbin,False)
        elif reporttype == "simplememory":
            rootfilename = "simpmem-regression.root"
            cmpSimpMemReport(rootfilename,outdir,file1,file2,secsperbin,False)
        elif reporttype == "edmsize":
            cmpEdmSizeReport(outdir,file1,file2)
        elif reporttype == "callgrind":
            cmpCallgrindReport(outdir,file1,file2)
        elif reporttype == "igprof":
            cmpIgProfReport(outdir,file1,file2)            
    except TimingParseErr, detail:
        print "WARNING: Could not parse data from Timing report file %s; not performing regression" % detail.message
    except SimpMemParseErr, detail:
        print "WARNING: Could not parse data from Memory report file %s; not performing regression" % detail.message
    except PerfReportErr     , detail:
        print "WARNING: Could not parse data from Edm file %s; not performing regression" % detail.message
    except IOError, detail:
        print detail
    except OSError, detail:
        print detail

if __name__ == "__main__":
    _main()

