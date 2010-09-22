# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="Aurelija"
__date__ ="$2010-08-25 11.18.53$"

def pathsToRegEx(Paths):
    paths = []
    for path in Paths:
        path = pathToRegEx(path)
        paths.append(path)
    return paths

def pathToRegEx(path):
    path = path.replace("\\", "\\\\")
    path = "\A%s$" %path
    path = path.replace(".", "\.")
    path = path.replace("*",".*")
    return path
