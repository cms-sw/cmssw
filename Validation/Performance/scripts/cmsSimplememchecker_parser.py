#! /usr/bin/env python

    
def manipulate_log(outdir,logfile_name,startevt):

    import time
    import sys
    import ROOT       
    
    os.system('pwd')
    
    # the fundamental structure: the key is the evt number the value is a list containing
    # VSIZE deltaVSIZE RSS deltaRSS
    data=[]
    values_set=('vsize','delta_vsize','rss','delta_rss')
    report=''

    # open file and read it and fill the structure!
    logfile=open(logfile_name,'r')
    logfile_lines=logfile.readlines()
    logfile.close()

    # we get the info we need!
    i=0
    max_rss=(0,0)
    parse_report=True
    while i < len(logfile_lines):
        line=logfile_lines[i]
        if '%MSG-w MemoryCheck:' in line:
            line=line[:-1] #no \n!
            line_content_list=line.split(' ')
            event_number=int(line_content_list[-1])
            if event_number<startevt:
                i+=1
                continue
            i+=1 # we inspect the following line
            line=logfile_lines[i]
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
            # find maximum rss of the job
            if rss > max_rss[1]:
                max_rss=(event_number, rss)

        # include memory report
        elif parse_report and 'MemoryReport' in line:
            while 'TimeReport' not in line:
                report += line.replace('MemoryReport', '')
                i+=1 
                line = logfile_lines[i]
            parse_report=False
        i+=1

    # print maximum rss for this job
    print 'Maximum rss =', max_rss[1]
                                    
    # skim the second entry when the event number is the same BUG!!!!!!!
    # i take elements in couples!
    new_data=[]
    if len(data)>2:
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
        
    npoints=len(data)
    
    print '%s values read and stored ...' %npoints

            
    # The Graphs 
    __argv=sys.argv # trick for a strange behaviour of the TApp..
    sys.argv=sys.argv[:1]
    ROOT.gROOT.SetStyle("Plain") # style paranoia
    sys.argv=__argv

    #Cannot use this option when the logfile includes
    #a large number of events... PyRoot seg-faults.
    #Set ROOT in batch mode to avoid canvases popping up!
    ROOT.gROOT.SetBatch(1)

    # Save in file
    rootfilename='%s/graphs.root' %outdir
    myfile=ROOT.TFile(rootfilename,'RECREATE')    
       
    # dictionary of graphs!
    graph_dict={}
    for value in values_set:
        #graph_dict[value]
        graph=ROOT.TGraph(npoints)
        graph.SetMarkerStyle(8)
        graph.SetMarkerSize(.7)
        graph.SetMarkerColor(1)
        graph.SetLineWidth(3)
        graph.SetLineColor(2)        
        graph.SetTitle(value)
        graph.SetName('%s_graph' %value)
        
    
        #fill the graphs
        point_counter=0
        for event_number,vals_dict in data:
            graph.SetPoint(point_counter,
                                       event_number,
                                       vals_dict[value])
            point_counter+=1
        
        graph.GetXaxis().SetTitle("Event")
        last_event=data[-1][0]
        graph.GetXaxis().SetRangeUser(0,last_event+1)
        graph.GetYaxis().SetTitleOffset(1.3)
        graph.GetYaxis().SetTitle("MB")
                          
        
            
        #print the graphs as files :)
        mycanvas=ROOT.TCanvas('%s_canvas' %value)
        mycanvas.cd()
        graph.Draw("ALP")
    
        mycanvas.Print("%s/%s_graph.png"%(outdir,value),"png")
        
        # write it on file
        graph.Write()
        mycanvas.Write()
        
    myfile.Close() 
        
    os.system('pwd') 
                
    # The html page!------------------------------------------------------------------------------
    
    titlestring='<b>Report executed with release %s on %s.</b>\n<br>\n<hr>\n'\
                                   %(os.environ['CMSSW_VERSION'],time.asctime())
    #Introducing this if to catch the cmsRelvalreport.py use case of "reuse" of TimingReport
    #profile when doing the SimpleMemReport... otherwise the filename for the html
    #would be misleadingly TimingReport...
    if len(logfile_name)>16 and 'TimingReport.log' in logfile_name[-16:]:
        file_name=logfile_name[:-16]+"_SimpleMemReport"
    else:
        file_name=logfile_name[:-4]+"_SimpleMemReport"
    html_file_name='%s/%s.html' %(outdir,file_name)
    html_file=open(html_file_name,'w')
    html_file.write('<html>\n<body>\n'+\
                    titlestring)
    html_file.write('<table>\n'+\
                    '<tr>\n<td><img  src=vsize_graph.png></img></td>\n'+\
                    '<td><img src=rss_graph.png></img></td>\n</tr>\n'+\
                    '<tr>\n<td><img  src=delta_vsize_graph.png></img></td>\n'+\
                    '<td><img  src=delta_rss_graph.png></img></td>\n</tr>\n' +\
                    '</table>\n')
    html_file.write('<hr>\n<h1>Memory Checker Report</h1>\n<pre>\n' + report + '</pre>')
    html_file.write('\n</body>\n</html>')
    html_file.close()    
    
    
#################################################################################################    
        
if __name__ == '__main__':
    
    import optparse
    import os
    
    # Here we define an option parser to handle commandline options..
    usage='simplememchecker_parser.py <options>'
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
                      help='The event number from which we start. Default is 1.' ,
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
        startevt=int(options.startevt)        
    except ValueError:
         print 'Problems in convertng starting event value!'
         
            
    #launch the function!
    manipulate_log(options.outdir,options.profile,startevt)
    
    
