# About
Fitting tool for **Langmuir Probe Array** analysis in NIFS
Visual conformation of the fit quality for each voltage ramp-up phase based on pyqtgraph module.

April 2018
![img](icons/langmuir_tool.png)


# INSTALLATION

## Python version
This programm works with both Python 2.x and 3.x.
To install Python go to https://www.python.org/ and select proper version for your system. Best to use 64 bit version, it has no memory limitations.

However, I recommend to use conda to manage your python envoronments, specifically conda-forge https://github.com/conda-forge/miniforge.

When Python is installed it is important to check if system variable PATH is updated properly.

Following Python folders must be in the PATH variable:
```
a) C:\Python27 
b) C:\Python27\Scripts
c) C:\Python27\Lib\site-packages
```
In this example Python version is 2.7 and it is installed on Windows in C:\Python27

When installation is complete the following Python packages have to be installed:

     1. numpy with MKL - basic mathematical library
     2. scipy - scientific library
     3. matplotlib - plotting library for publication ready plots
     4. pyqtgraph - fast plotting library for Grafical User Interface (GUI)
     5. PyQt4 or PyQt5 - python bindings for Qt application framework (https://en.wikipedia.org/wiki/Qt_(software))
     6. lmfit - fitting package based on scipy optimization tools
     7. labcom - NIFS pakcage
     
Packages could be installed using Python package manager pip with following command:

```
    pip install package_name
```

For numpy and PyQt on Windows pip may not work and precompiled pakcages should be downloaded instead.
Unoficial, but trusted source for the compiled packages is on this page: 
https://www.lfd.uci.edu/~gohlke/pythonlibs/

numpy with MKL *.whl file should be downloaded for the proper python version and 32 or 64 version of OS.
After *.whl file is downloaded, just run the following command:
```
    pip install downloaded_file_name.whl
``` 
Where downloaded_file_name is the name of the downloaded pakcage.

labcom package is a python package to communicate with Retrive programm.
To install it, find an installation file insyde > ..Retrieve\lib\python execute it on Windows or use python setup.py install on unix

After installation confirm that either > ..Retrieve\bin64 or > ..Retrieve\bin folder is in the system variable PATH. Select bin if using 32 bit os and bin64 if using 64 bit OS

When finished, confirm if all neccesary packages are installed by running the program from terminal:
```
    python visual_fit.pyw
```

If there are still some problems with packages it will be written in the Error Messages.

