Data and Code are available at https://github.com/junzhang19/CS7641.git
I use the "PyDev" in Eclipse to build and run my jython programs.
How I run the code:

Step 1: prepare the software and environment
1.install python
2.install jython
3.install Eclipse
4.install "PyDev" module in eclipse

Step 2: build the jython project and import the abagail classes
1.open eclipse, and create a project
2.Setup jython interpreter in Eclipse. Find eclipse->preference->PyDev->Interpreters->Jython Interpreter->browze for jython jar, find the folder where the jython2.7.1 is, select jython.jar, then click OK. Then apply and close.
3.In your eclipse, file->new->Project, select PyDev-PyDev Project, then next. create your project, say "JunABA". Now we have our project.
4.On the left of the workspace, right click "JunABA", PyDev - PYTHONPATH -> external libraries -> add source folder, adding the "Jun-HW2-to-submit/bin" folder to the project. Now the project is able to use the ABAGAIL classes and functions.
5.there may be a pop-up window asking you to config your python interpreter, choose from a list, the system will automatically find the options for you.
6.import all the .py files in the folder "Jun-HW2-to-submit/jython" to the jython project. right click JunABA, -> Import -> General -> File System, then choose those .py files to add.
7.Now we are able to run the jython files. 

Step 3: run the .py files
"continuouspeakITER.py", "knapsackITER.py", "tsmITER.py" runs iterations for the 3 optimization problems. 
"continuouspeakSACooling.py", "continuouspeakSATemperature.py" investigate parameters of simulated annealing.
"knapsackMIMICSamples.py", "knapsackMIMICToKeep.py" investigate parameters of MIMIC.
"tsmGACrossover.py", "tsmGAMutation.py", "tsmGAPopulation.py" investigate parameters of generic algorithm.
"NNCalculation.py" runs optimization algorithms to train the neural network.
"NNBP.py" runs back propagation to train the neural network.
The file "NNBP.py" is run using python interpreter, all the other files are fun using jython interpreter.

Step 4: plot figures
I organized the results and put them in the folder ""Jun-HW2-to-submit/results and figures".
Run the file "plotter.py" using either python or jython, the figures will be generated. 
I placed the figures into different folders.
