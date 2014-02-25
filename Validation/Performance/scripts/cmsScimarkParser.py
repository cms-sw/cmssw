#! /usr/bin/env python
#Script cloned from cmsTiming_parser.py

def get_max(data,index=1):
    max_score=-1
    for el in data:
        sec=el[index]
        if max_score<sec:
            max_score=sec
    return max_score

def get_min(data,index=1):
    min_score=1e20
    for el in data:
        sec=el[index]
        if min_score>sec:
            min_score=sec
    return min_score    
    
def manipulate_log(outdir,logfile_name,secsperbin):

    import time
    import sys
    import ROOT       
    
    # the fundamental structure: the key is the evt number the value is a list containing
    # Composite Score
    data=[]
    
    # open file and read it and fill the structure!
    logfile=open(logfile_name,'r')
    logfile_lines=logfile.readlines()
    if not logfile_lines:
        print "The logfile %s is empty! Exiting now."%logfile_name
        sys.exit()
    logfile.close()

    # we get the info we need!
    i=0
    bench_number=0;
    while i < len(logfile_lines):
        line=logfile_lines[i]
        #if 'TimeEvent>' in line:
        if 'Composite Score:' in line:
            line=line[:-1] #no \n!
            line_content_list=line.split()
            #event_number=int(line_content_list[1])
            #seconds=float(line_content_list[3])
            composite_score=float(line_content_list[2])
            #data.append((event_number,seconds))
            bench_number+=1
            data.append((bench_number,composite_score))
        i+=1
                                              
    # init Graph and histo
    
    # The Graphs 
    __argv=sys.argv # trick for a strange behaviour of the TApp..
    sys.argv=sys.argv[:1]
    ROOT.gROOT.SetStyle("Plain") # style paranoia
    sys.argv=__argv
    #Cannot use this option when the logfile includes ~2000
    #Composite Scores or more... PyRoot seg-faults.
    #Set ROOT in batch mode to avoid canvases popping up!
    #ROOT.gROOT.SetBatch(1)

    # Save in file
    rootfilename='%s/graphs.root' %outdir
    myfile=ROOT.TFile(rootfilename,'RECREATE')   
    
    
    # Set fancy limits
    min_val=get_min(data,1)
    max_val=get_max(data,1)
    interval=int(max_val-min_val)
    
    min_val=min_val-interval*0.2
    max_val=max_val+interval*0.2
    interval=int(max_val-min_val)
    
    nbins=int(interval/secsperbin)
    
    print 'Minval=',min_val,' maxval=',max_val, ' interval=',interval
    
    histo=ROOT.TH1F('Composite Score(Mflops)','Composite Score (Mflops)',nbins,min_val,max_val)
    histo.GetXaxis().SetTitle("Mflops")    
    
    npoints=len(data)   
    
    graph=ROOT.TGraph(npoints)
    graph.SetMarkerStyle(8)
    graph.SetMarkerSize(.7)
    graph.SetMarkerColor(1)
    graph.SetLineWidth(3)
    graph.SetLineColor(2)        
    graph.SetTitle('Composite Score')
    graph.SetName('Composite Score')    
    graph.GetXaxis().SetTitle("Benchmark sequential number")
    
    last_event=data[-1][0]
    print 'last event =',last_event
    graph.GetXaxis().SetLimits(0,last_event)
        
    graph.GetYaxis().SetTitleOffset(1.3)
    graph.GetYaxis().SetTitle("Mflops")
    graph.GetYaxis().SetRangeUser(min_val,max_val)

    
    
    # Fill them
    
    evt_counter=0
    for bench_number,composite_score in data:
        graph.SetPoint(evt_counter,bench_number,composite_score)
        histo.Fill(composite_score)
        evt_counter+=1
                
    #A line which represents the average is drawn in the TGraph
    avg=histo.GetMean()
    avg_line=ROOT.TLine(1,avg,last_event,avg)
    avg_line.SetLineColor(4)
    avg_line.SetLineWidth(2)
        
    # draw and save!
    graph_canvas=ROOT.TCanvas('graph_canvas')
    graph_canvas.cd()
    graph.Draw("ALP")
    avg_line.Draw("Same")
    
    graph_canvas.Print("%s/graph.png" %outdir,"png")
    
    # write it on file
    graph.Write()
    graph_canvas.Write()    

    histo_canvas=ROOT.TCanvas('histo_canvas')
    histo_canvas.cd()
    histo.Draw('')

    histo_canvas.Print("%s/histo.png" %outdir,"png")
    
    # write it on file
    histo.Write()
    histo_canvas.Write() 
    
    myfile.Close()   
                    
    # The html page!------------------------------------------------------------------------------
    
    titlestring='<b>Report executed with release %s on %s.</b>\n<br>\n<hr>\n'\
                                   %(os.environ['CMSSW_VERSION'],time.asctime())
        
    html_file_name='%s/%s.html' %(outdir,logfile_name[:-4])# a way to say the string until its last but 4th char
    html_file=open(html_file_name,'w')
    html_file.write('<html>\n<body>\n'+\
                    titlestring)
    html_file.write('<table>\n'+\
                    '<tr><td><img  src=graph.png></img></td></tr>'+\
                    '<tr><td><img  src=histo.png></img></td></tr>'+\
                    '</table>\n')
    html_file.write('\n</body>\n</html>')
    html_file.close()    
    
    
#################################################################################################    
        
if __name__ == '__main__':
    
    import optparse
    import os
    
    # Here we define an option parser to handle commandline options..
    usage='cmsScimarkParser.py <options>'
    parser = optparse.OptionParser(usage)
    parser.add_option('-i', '--in_  profile',
                      help='The profile to manipulate' ,
                      default='',
                      dest='profile')
                      
    parser.add_option('-o', '--outdir',
                      help='The directory of the output' ,
                      default='',
                      dest='outdir')

    parser.add_option('-n', 
                      help='Number of secs per bin. Default is 1.' ,
                      default='1',
                      dest='startevt')                      
                                            
    (options,args) = parser.parse_args()
    
    # Now some fault control..If an error is found we raise an exception
    if options.profile=='' or\
       options.outdir=='':
        raise('Please select a profile and an output dir!')
    
    if not os.path.exists(options.profile) or\
       not os.path.exists(options.outdir):
        raise ('Outdir or input profile not present!')
    
    try:
        startevt=float(options.startevt)        
    except ValueError:
         print 'Problems in convertng starting event value!'
         
            
    #launch the function!
    manipulate_log(options.outdir,options.profile,startevt)
    
    
