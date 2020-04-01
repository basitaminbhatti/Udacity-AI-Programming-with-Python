#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/check_images.py
#
# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: Abdul Basit   
# DATE CREATED: 30-03-20                                 
# REVISED DATE: 
# PURPOSE: Classifies pet images using a pretrained CNN model, compares these
#          classifications to the true identity of the pets in the images, and
#          summarizes how well the CNN performed on the image classification task. 
#          Note that the true identity of the pet (or object) in the image is 
#          indicated by the filename of the image. Therefore, your program must
#          first extract the pet image label from the filename before
#          classifying the images using the pretrained CNN model. With this 
#          program we will be comparing the performance of 3 different CNN model
#          architectures to determine which provides the 'best' classification.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
from time import time, sleep

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Imports functions created for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results

# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg 
    check_command_line_arguments(in_arg)

    
    # Creates Pet Image Labels by creating a dictionary 
    answers_dic = get_pet_labels(in_arg.dir)

    # Function that checks Pet Images Dictionary- answers_dic    
    check_creating_pet_image_labels(answers_dic)

    
    # Creates Classifier Labels with classifier function, Compares Labels, 
    # and creates a results dictionary 
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)

    # Function that checks Results Dictionary - result_dic    
    check_classifying_images(result_dic)    

    
    # Adjusts the results dictionary to determine if classifier correctly 
    # classified images as 'a dog' or 'not a dog'. This demonstrates if 
    # model can correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)

    # Function that checks Results Dictionary for is-a-dog adjustment- result_dic  
    check_classifying_labels_as_dogs(result_dic)

    
    # Calculates results of run and puts statistics in results_stats_dic
    results_stats_dic = calculates_results_stats(result_dic)

    # Function that checks Results Stats Dictionary - results_stats_dic  
    check_calculating_results(result_dic, results_stats_dic)


    # Prints summary results, incorrect classifications of dogs
    # and breeds if requested
    print_results(result_dic, results_stats_dic, in_arg.arch, True, True)
    
    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()