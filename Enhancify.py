
from distutils.log import warn
from src.Enhancify_sequential import Enhancify
import getopt
import sys
import glob
import os
import time
import subprocess
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Sequential version of Enhancify
def run(imagePath, folderIn, folderOut, population, generations, selection, cross_rate, mut_rate, pressure, elitism, verbose):

    startAll = time.time()

    toProcess = []

    # Looking for the provided input image
    if folderIn is None:

        ext = imagePath.split(".")[-1].lower()
        listExts = ["tiff", "tif", "png", "jpeg", "jpg"]

        if ext not in listExts:
            print("******************************************************************************************")
            print("Unsupported format. Please provide", listExts, "images")
            print("Warning", imagePath, "has not been processed")
            print("******************************************************************************************")
            exit(-6)

        if not os.path.exists(imagePath):
            print("******************************************************************************************")
            print(imagePath, "does not exists")
            print("Warning", imagePath, "has not been processed") 
            print("******************************************************************************************")
            exit(-7)

        else:
            toProcess.append(imagePath)

    else:
        alreadyprint = False
        # Looking for the images in the provided input folder
        listImages = glob.glob(folderIn+os.sep+"*")
        for imagePath in listImages:
            ext = imagePath.split(".")[-1]
            listExts = ["tiff", "tif", "png", "png", "jpeg", "jpg"]

            # Only tiff, png and jpg images can be analyzed
            if ext not in listExts:
                if not alreadyprint:
                    print("******************************************************************************************")
                print("Unsupported format. Please provide", listExts, "images")
                print("Warning", imagePath, "will be not processed\n")
                alreadyprint = True
                pass

            elif not os.path.exists(imagePath):
                if not alreadyprint:
                    print("******************************************************************************************")
                print(imagePath, "does not exists")
                print("Warning", imagePath, "will be not processed\n")
                alreadyprint = True
                pass

            else:
                toProcess.append(imagePath)

    if not os.path.exists(folderOut):
        os.makedirs(folderOut)

    if len(toProcess) == 0:
        print( "******************************************************************************************")
        exit(-11)

    if verbose:
        print( "******************************************************************************************")
        print( "* Running the sequential version of Enhancify\n")

        print( " * GA settings")
        print( "   -> Number of chromosome: %d" % population)
        print( "   -> Number of elite chromosomes: %d" % elitism)
        print( "   -> Number of generations: %d" % generations)
        print( "   -> Crossover rate: %.2f" % cross_rate)
        print( "   -> Mutation rate:  %.2f" % mut_rate)

        if selection == 'wheel':
            print( "   -> Selection: wheel roulette\n\n")
        elif selection == 'ranking':
            print( "   -> Selection: ranking \n\n")
        else:
            print( "   -> Selection: tournament with %d individuals\n\n" % pressure)

    times = np.zeros(len(toProcess))

    # Processing the images in the input folder
    # The input images are characterized by an undelying bimodal histogram (intensity level distribution)
    # Possibly, previously masked and cropped images (according to a bounding region containing the Region of Interest)
    for i in range(len(toProcess)):

        # Output folders
        string = toProcess[i].split("/")[1:]
        subfolder = string[-1].split(".")[:-1]

        pathOutput = folderOut+os.sep+subfolder[0]

        if not os.path.exists(pathOutput):
            os.makedirs(pathOutput)

        if verbose:
            print( " * Analyzed image %s" % toProcess[i])
            start = time.time()
            
            

        # Enhancify execution on the image to be processed by using the provided GA settings
        enhancify = Enhancify(toProcess[i], pathOutput)
        enhancify.startGA(population, generations, selection,
                      cross_rate, mut_rate, elitism, pressure)

        end = time.time()
        elapsed = end-start
        times[i] = elapsed

        if verbose:
            print( "-> Elapsed time %5.2fs" % (elapsed))

        endAll = time.time()
        elapsedAll = endAll-startAll

    if verbose:
        if len(toProcess) > 1:
            print( "\n * Total elapsed time %5.2fs" % elapsedAll, "for computing", len(toProcess), "images")
            print( " * Mean elapsed time  %5.2fs per image" % np.mean(times))
        print( "******************************************************************************************")


if __name__ == '__main__':

    helpString = """Enhancify.py -h <help>
         -i <image>
         -f <folder>
         -o <output>      (default: output)
         -p <population>  (default: 100)
         -g <generations> (default: 100)
         -s <selection>   (default: tournament)
         -c <cross_rate>  (default: 0.9)
         -m <mut_rate>    (default: 0.01)
         -k <pressure>    (default: 20)
         -e <elitism>     (default: 1)
         -v <verbose>     (default: False)"""

    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:i:f:o:p:g:s:c:m:k:e:v", ["help", "image", "folder", "output",
                                                                               "population", "generations", "selection",
                                                                               "cross_rate", "mut_rate", "pressure",
                                                                               "elitism", "verbose"])
    except:
        print( helpString)
        exit(-1)

    if len(opts) == 0:
        print( helpString)
        exit(-2)

    # default settings of Enhancify
    imagePath = None
    folderIn = None
    folderOut = "output"
    population = 100
    generations = 100
    selection = "tournament"
    cross_rate = 0.9
    mut_rate = 0.01
    pressure = 20
    elitism = 1
    verbose = False
    warning = False
    alreadyprint = False

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print( helpString)
            exit(-3)

        elif opt in ("-i", "--image"):
            imagePath = arg

        elif opt in ("-f", "--folder"):
            folderIn = arg

        elif opt in ("-o", "--output"):
            folderOut = arg

        elif opt in ("-p", "--population"):
            try:
                population = int(arg)
            except:
                if not alreadyprint:
                    print( "******************************************************************************************")
                print( " * Warning, the provided population is not correct. It has been set to 100")
                population = 100
                warning = True
                alreadyprint = True

        elif opt in ("-g", "--generations"):
            try:
                generations = int(arg)
            except:
                if not alreadyprint:
                    print( "******************************************************************************************")
                print( " * Warning, the provided generations is not correct. It has been set to 100")
                generations = 100
                warning = True
                alreadyprint = True

        elif opt in ("-s", "--selection"):
            try:
                selection = arg
            except:
                if not alreadyprint:
                    print( "******************************************************************************************")
                print( " * Warning, the provided selection is not correct. It has been set to tournament")
                selection = selection
                warning = True
                alreadyprint = True

        elif opt in ("-c", "--cross_rate"):
            try:
                cross_rate = float(arg)
            except:
                if not alreadyprint:
                    print( "******************************************************************************************")
                print( " * Warning, the provided cross_rate is not correct. It has been set to 0.9")
                cross_rate = 0.9
                warning = True
                alreadyprint = True

        elif opt in ("-m", "--mut_rate"):
            try:
                mut_rate = float(arg)
            except:
                if not alreadyprint:
                    print( "******************************************************************************************")
                print( " * Warning, the provided mut_rate is not correct. It has been set to 0.01")
                mut_rate = 0.01
                warning = True
                alreadyprint = True

        elif opt in ("-k", "--pressure"):
            try:
                pressure = int(arg)
            except:
                if not alreadyprint:
                    print( "******************************************************************************************")
                print( " * Warning, the provided pressure is not correct. It has been set to 20")
                pressure = 20
                warning = True
                alreadyprint = True

        elif opt in ("-e", "--elitism"):
            try:
                elitism = int(arg)
            except:
                if not alreadyprint:
                    print( "******************************************************************************************")
                print( " * Warning, the provided elitism is not correct. It has been set to 1")
                elitism = 1
                warning = True
                alreadyprint = True

        elif opt in ("-v", "--verbose"):
            verbose = True

    if warning:
        print(warning)

# ************************************************ Checking provided settings ************************************************
    warning = False
    if population <= 0:
        if not alreadyprint:
            print( "******************************************************************************************")
        print( " * Warning, the provided population is %d. It has been set to 100" % population)
        population = 100
        warning = True
        alreadyprint = True

    if generations <= 0:
        if not alreadyprint:
            print( "******************************************************************************************")
        print( " * Warning, the provided generations is %d. It has been set to 100" % generations)
        generations = 100
        warning = True
        alreadyprint = True

    if selection not in ["wheel", "ranking", "tournament"]:
        if not alreadyprint:
            print( "******************************************************************************************")
        print( " * Warning, the provided selection is %s. It has been set to tournament" % selection)
        selection = "tournament"
        warning = True
        alreadyprint = True

    if (cross_rate < 0) or (cross_rate > 1):
        if not alreadyprint:
            print( "******************************************************************************************")
        print( " * Warning, the provided cross_rate is %f. It has been set to 0.9" % cross_rate)
        cross_rate = 0.9
        warning = True
        alreadyprint = True

    if (mut_rate < 0) or (mut_rate > 1):
        if not alreadyprint:
            print( "******************************************************************************************")
        print( " * Warning, the provided mut_rate is %f. It has been set to 0.01" % mut_rate)
        mut_rate = 0.01
        warning = True
        alreadyprint = True

    if (pressure > population) or (pressure <= 0):
        if not alreadyprint:
            print( "******************************************************************************************")
        print( " * Warning, the provided pressure is %d. It has been set to 20" % pressure)
        pressure = 20
        warning = True
        alreadyprint = True

    if (elitism > population) or (elitism < 0):
        if not alreadyprint:
            print( "******************************************************************************************")
        print( " * Warning, the provided elitism is %d. It has been set to 1" % elitism)
        elitism = 1
        warning = True
        alreadyprint = True

    if (imagePath is None) and (folderIn is None):
        if not alreadyprint:
            print( "******************************************************************************************")
        print( " * Please, provide either an image or a folder containing at least an image")
        print( "******************************************************************************************")
        exit(-4)

    if (imagePath is not None) and (folderIn is not None):
        if not alreadyprint:
            print( "******************************************************************************************")
        print( " * Please, provide either an image or a folder containing at least an image")
        print( "******************************************************************************************")
        exit(-5)

    if warning:
        print(warning)

# ************************************************ Running Enhancify ************************************************
    # Run sequential version on either a folder or a single image
    run(imagePath, folderIn, folderOut, population, generations,
        selection, cross_rate, mut_rate, pressure, elitism, verbose)

