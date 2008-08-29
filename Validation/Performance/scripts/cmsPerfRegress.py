#!/usr/bin/env python

import time, os, sys, math
import optparse as opt
#from ROOT import gROOT, TCanvas, TF1
import ROOT

_cmsver = os.environ['CMSSW_VERSION']

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
    (options,args) = parser.parse_args()
    if not len(args) == 2:
        print "ERROR: Not enough arguments"
        sys.exit()

    path1 = os.path.abspath(args[0])
    path2 = os.path.abspath(args[1])    
    if os.path.exists(path1) and os.path.exists(path2):
        return (path1, path2, options.startevt)
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

def createROOT(outdir,filename):
    __argv=sys.argv # trick for a strange behaviour of the TApp..
    sys.argv=sys.argv[:1]
    ROOT.gROOT.SetStyle("Plain") # style paranoia
    sys.argv=__argv
    #Cannot use this option when the logfile includes
    #a large number of events... PyRoot seg-faults.
    #Set ROOT in batch mode to avoid canvases popping up!
    ROOT.gROOT.SetBatch(1)

    # Save in file
    rootfilename='%s/%s' %(outdir,filename)
    myfile=ROOT.TFile(rootfilename,'RECREATE')
    return myfile

def getDataFromTimingLog(logfile_name):
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

def setupSuperimpose(graph1,graph2,last_event,max_val):
    graph1.SetTitle('Seconds per event')
    graph1.SetName('SecondsPerEvent')    
    graph1.GetXaxis().SetTitle("Event")
    graph1.GetYaxis().SetTitleOffset(1.3)
    graph1.GetYaxis().SetTitle("Processing time for each event (s)")
    graph1.GetXaxis().SetLimits(0,last_event)
    graph1.GetYaxis().SetRangeUser(0,max_val)
    # Do I need to set limits on graph 2? I don't know
    # I'm doing it anyway, can't hurt.
    graph2.GetXaxis().SetLimits(0,last_event)    
    graph2.GetYaxis().SetRangeUser(0,max_val)    

def getMeanLines(avg,last_event,graph_num):
    colors = [2,4]
    avg_line=ROOT.TLine(1,avg,last_event,avg)
    avg_line.SetLineColor(colors[graph_num])
    avg_line.SetLineWidth(2)

    return avg_line

def getDiff(data1,data2,npoints,last_event,orig_max_val):
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

def getTwoGraphLimits(last_event1,max_val1,last_event2,max_val2):
    biggestLastEvt = last_event1
    biggestMaxval  = max_val1
    
    if last_event2 > biggestLastEvt:
        biggestLastEvt = last_event2
    if max_val2 > biggestMaxval:
        biggestMaxval  = max_val2
    return (biggestLastEvt,biggestMaxval)

def regressCompare(rootfilename,outdir,oldLogfile,newLogfile,secsperbin,batch=True,prevrev=""):
    
    data1 = getDataFromTimingLog(oldLogfile)
    data2 = getDataFromTimingLog(newLogfile)

    newrootfile = None
    if batch:
        newrootfile = createROOT(outdir,rootfilename)

    (min_val1,max_val1,nbins1,npoints1,last_event1) = getLimits(data1,secsperbin)
    (min_val2,max_val2,nbins2,npoints2,last_event2) = getLimits(data2,secsperbin)

    hsStack = ROOT.THStack("hsStack","Histograms Comparison")
    leg      = ROOT.TLegend(0.6,0.99,0.89,0.8)
    histoleg = ROOT.TLegend(0.7,0.8,0.89,0.89)    
    #if not nbins1 == nbins2:
    #    print "ERRORL bin1 %s is not the same size as bin2 %s" % (nbins1,nbins2)

    (graph1,histo1,mean1) = newGraphAndHisto(histoleg,leg,npoints1,nbins1,min_val1,max_val1,data1,0,prevrev)
    hsStack.Add(histo1)
    (graph2,histo2,mean2) = newGraphAndHisto(histoleg,leg,npoints2,nbins2,min_val2,max_val2,data2,1,prevrev)
    hsStack.Add(histo2)

    (biggestLastEvt,biggestMaxval) = getTwoGraphLimits(last_event1,max_val1,last_event2,max_val2)
    
    (changegraph,chgleg) = getDiff(data1,data2,npoints2,biggestLastEvt,biggestMaxval)
    setupSuperimpose(graph1,graph2,biggestLastEvt,biggestMaxval)
    avg_line1 = getMeanLines(mean1,last_event1,0)
    avg_line2 = getMeanLines(mean2,last_event2,1)


    
    #
    # Create a one dimensional function and draw it
    #

    if batch:
        names = ["graph.gif","changes.gif","histo.gif"]
        #Graphs
        graph_canvas   = drawGraphs(graph1,graph2,avg_line1,avg_line2,leg)
        graph_canvas.Print("%s/%s" % (outdir,names[0]),"gif")
        graph_canvas.Write()    # to file
        #Changes
        changes_canvas = drawChanges(changegraph,chgleg)
        changes_canvas.Print("%s/%s" % (outdir,names[1]),"gif")
        changes_canvas.Write()
        #Histograms
        histo1.GetXaxis().SetTitle("s")        
        histo_canvas   = drawHistos(hsStack,histoleg)        
        histo_canvas.Print("%s/%s" % (outdir,names[2]),"gif")
        histo_canvas.Write() 
        newrootfile.Close()   

        # The html page!------------------------------------------------------------------------------

        titlestring='<b>Report executed with release %s on %s.</b>\n<br>\n<hr>\n'\
                                       %(_cmsver,time.asctime())

        #html_file_name='%s/%s.html' %(outdir,logfile_name[:-4])# a way to say the string until its last but 4th char
        #html_file=open(html_file_name,'w')
        #html_file.write('<html>\n<body>\n'+\
        #                titlestring)
        #return map (lambda x: "%s/%s" % (outdir,x),names)
        return names
    else:
        graph_canvas   = drawGraphs(graph1,graph2,avg_line1,avg_line2,leg)
        changes_canvas = drawChanges(changegraph,chgleg)
        histo1.GetXaxis().SetTitle("s")        
        histo_canvas   = drawHistos(hsStack,histoleg)        
        while graph_canvas or histo_canvas or changes_canvas:
            time.sleep(2.5)
        return 0

def main():
    rootfilename = "regression.root"
    outdir = os.getcwd()
    
    (file1,file2,secsperbin)  = getParameters()    
    regressCompare(rootfilename,outdir,file1,file2,secsperbin,False)

if __name__ == "__main__":
    main()

