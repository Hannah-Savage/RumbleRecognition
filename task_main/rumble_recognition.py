# Rumble Recognition: Gastric Interoception Task (Stethoscope - Self Audio)
# -----------------------------------
# Code written by Hannah Savage, Novemer 2024

"""
This PsychoPy script implements the Rumble Recognition: Gastric Interoception Task (Stethoscope - Self Audio) as part of the EM-BODY study battery. 
The task assesses participants' ability to discriminate between live or pre-recorded audio from their stomach, as transmitted via digital stethoscope.

Key Components:
- **Integration Options**: The script includes toggles to determine if it is integrated with external hardware (Spike2) or a larger toolkit (EM-BODY).
- **Visual Stimuli**: Employs the PsychoPy `visual` module to present instructions, and collect feedback.
- **Data Logging**: Records participant responses and other relevant data into a TSV file for later analysis.

Script Breakdown:
- **Setup Variables**: 
  - `is_integrated_external`: Toggle for integrating with external devices.
  - `is_integrated_toolbox`: Toggle for integration with the EM-BODY toolkit.
  - `save_dir` and `participant_id`: Specify save path and participant ID if not integrated.
  - `if_debug`: Debug mode toggle for troubleshooting.

- **Environment Setup**: 
  - Imports necessary modules, sets up the window and monitor parameters, and initializes the serial port for external device communication if applicable.

- **Logging Mechanism**: 
  - Opens a CSV file to log the participant's responses and relevant trial information, writing headers for clarity.

- **Experiment Flow**: 
  - Displays instructions and a countdown before starting the trials.
  - Executes a loop through the trials, collecting discrimination judgement and confidence ratings,
  - After 10 trials, participants consume Drink 1 (sparkling water)
  - After 20 trials, participants consume Drink 2 (protein drink)
  - logs data for each trial.

- **Post-Trial Procedures**: 
  - Closes the CSV file and window after completing all trials, ensuring clean exit from the experiment.

Usage Notes:
- **Configuration**: Adjust `save_dir` and `participant_id` as needed if not using the integrated toolbox.
- **Serial Port Settings**: Modify the `port` and `baud_rate` based on your specific hardware setup for external device integration.
- **Debugging**: Enable the `if_debug` toggle to print additional information for troubleshooting.

"""


### ------- SET ENVIRONMENT ----- ###
import os
import csv
import datetime
from pathlib import Path
from psychopy import visual, sound, core, monitors, data
from psychopy.sound import Sound
import logging
import numpy as np
import random
import serial
import sounddevice as sd
import yaml

from rumble_recognition_functions import play_sound_on_click, wait_for_key_press, run_instructions_calibration, run_training, run_discrimination_trial, get_post_task_qs


def run_gastric_stethoscope_self(is_integrated_external = True, is_integrated_toolbox = True, if_debug = False):
    logger = logging.getLogger(name=None)

    if is_integrated_toolbox:
        config_file_path = os.path.join(os.getcwd(), 'config.yaml')[2:]
        # Open and load the participants YAML file
        with open(config_file_path, 'r') as file:
            config_data = yaml.safe_load(file)

        # Access the 'Participant ID' from the loaded YAML data
        participant_id = config_data.get('Participant ID')
        sound_dir = config_data.get('Physio Directory')
        save_dir = config_data.get('Behavioural Directory')
        print(save_dir)
    else:
        save_dir = str(Path.home() / "Desktop")
        participant_id = '999'
        sound_dir = str(Path.home() / "Desktop")
    
    logger.info(f'Participant ID loaded for gastric task: {participant_id}')
    logger.info(f'Saving gastric stethoscope data into: {save_dir}')

    ## ------Set Serial Port Parameters ------ ###
    if is_integrated_external:
        port = 'COM2'  # Adjust this to your specific COM port
        baud_rate = 9600  # Adjust baud rate as per your device's specification
        timeout = 1  # Timeout in seconds for reading the serial port
    # Initialize the serial port
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        if ser.is_open:
            logger.info(f"Successfully connected to port {port}")
            print(f"Successfully connected to port {port}")
        else:
            logger.info(f"Failed to connect to port {port}")
            print(f"Failed to connect to port {port}")
            ser.open()
    else: 
        ser = None  # Dummy object in testing mode

    if is_integrated_external:
        # Send DELAY VALUE  to Spike2
        spike_log = 'START TASK;'
        ser.write(spike_log.encode('utf-8'))
        #print(f"Sent: {delay_command}")

    ### ------ LOGGING ----- ###
    # Open CSV file for writing (in append mode)
    base_name = f"sub-{participant_id}_rumble_recognition"
    base_path = os.path.join(save_dir, base_name)
    #csv_file_name = os.path.join(save_dir, 'sub-' +participant_id + '_rumble_recognition.tsv')

    # Check if the file already exists
    if os.path.exists(base_path + ".tsv"):
        # List all files in the directory
        existing_files = [f for f in os.listdir(save_dir) if f.startswith(base_name)]
        
        # Extract the suffixes to determine the next index
        indices = []
        for file in existing_files:
            if file == base_name + ".tsv":
                continue  # Skip the base file name
            if file.startswith(base_name + "_") and file.endswith(".tsv"):
                suffix = file[len(base_name) + 1 : -4]  # Extract the part after '_'
                if len(suffix) == 1 and suffix.isalpha():
                    indices.append(ord(suffix.lower()) - ord('a'))
        
        # Determine the next suffix (a/b/c...)
        next_index = 0 if not indices else max(indices) + 1
        next_suffix = chr(ord('a') + next_index)

        # Create the new file name
        csv_file_name = f"{base_name}_{next_suffix}.tsv"
        csv_file_name = os.path.join(save_dir, csv_file_name)
    else:
        # If no file exists, use the default name
        csv_file_name = base_name + ".tsv"
        csv_file_name = os.path.join(save_dir, csv_file_name)

    logger.info(f'CSV name: {csv_file_name}')
    print(csv_file_name)
    csv_file = open(csv_file_name, mode='a', newline='')
    csv_writer = csv.writer(csv_file, delimiter = '\t')
    # Write the header
    csv_writer.writerow(['Participant ID', participant_id])
    #csv_writer.writerow(['Date', datetime.now().strftime('%Y-%m-%d')])  
    #csv_writer.writerow(['Time', datetime.now().strftime('%H:%M:%S')]) 
    csv_writer.writerow(['Trial', 'Participant_Sound_Loc', 'Button', 'Response_code', 'Confidence']) #Data columns
    csv_file.close()
    logger.info('Trial \t Participant_Sound_Loc Button \t Response_code \t Confidence')
   

    ### ------ WINDOW ----- ###
    # Set up the window
    screen = monitors.Monitor('testMonitor')
    screen.setSizePix([1680, 1050])
    screen.setWidth(47.475)
    screen.setDistance(57)

    # Create a window
    desired_frame_rate = 60
    #win = visual.Window(size=(800, 600), color='white', units='pix')
    win = visual.Window(fullscr=True, color='white', units='pix')
    visual.text.Font = 'Arial'  

    ### ------ STETHOSCOPE INTEGRATION ----- ###
    # Parameters
    fs = 48000  # Sample rate

    # Target device name
    target_device_name = "Stethoscope (USB Audio Device)"
    # Get the list of available devices
    devices = sd.query_devices()
    if if_debug: 
        print(devices)
    # Find the device number
    input_device_id = None
    for i, device in enumerate(devices):
        if if_debug:
            print(i)
            #print(device)
        if target_device_name in device['name'] and device['default_low_input_latency'] < 0.01:
            #select the device that matches the name and has the lowest latency (Stethoscope (USB Audio Device), Windows WASAPI (1 in, 0 out))
            input_device_id = i
            break
    if input_device_id is not None:
        if if_debug: 
            print(f"Device number for '{target_device_name}' is: {input_device_id}")
        logger.info(f"Stethoscope set up: \t Sample Rate: {fs}. \n \t \t Input device: {input_device_id}")

    else:
        if if_debug: 
            print(f"Device with name '{target_device_name}' not found.")
        logger.info(f"Stethoscope set up: \t Sample Rate: {fs}. \n \t \t Input device: NOT FOUND")

    #output DEFAULT = 4 Speakers (Realtek(R) Audio), MME (0 in, 2 out)
    duration = 15 # duration of stethoscope sounds played/recorded


    ### ------  TRAINING----- ###
    # Wait for Enter press to start   
    # Instructions + 2 recordings
    gastric_sound_dir = os.path.join(sound_dir, 'gastric_sounds')
    # Ensure subdirectory exists
    os.makedirs(gastric_sound_dir, exist_ok=True)

    play_sound_on_click(win, 'Sound Test', r'../task_helpers/chimes.wav')
    
    run_instructions_calibration(win, ser, fs, duration, gastric_sound_dir, is_integrated_external, if_debug)
    logger.info("Calibration complete. Start training trial")

    instruction_text = visual.TextStim(win, text="RESEARCHER:\n Calibration complete \n\n Press ENTER to test sound.", color='black', height=30, wrapWidth=800, pos = (0,0))
    wait_for_key_press(win, instruction_text)
    win.flip()

    play_sound_on_click(win, 'Sound Test', r'../task_helpers/chimes.wav')

    instruction_text = visual.TextStim(win, text="RESEARCHER:\n Press ENTER to continue to training.", color='black', height=30, wrapWidth=800, pos = (0,0))
    wait_for_key_press(win, instruction_text)
    win.flip()

    #Training trial   
    sound_file_play = os.path.join(gastric_sound_dir, 'gastric_rec_training.wav')
    print(sound_file_play)
    logger.info(f"Live sound will be played in position 1.")
    
    button, response_code, confidence = run_training(win, ser, duration, sound_file_play, is_integrated_external, if_debug)

    # Write trial information to the CSV file
    with open(csv_file_name, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['Training', '1',  button, response_code, confidence])
            csv_file.close()
    logger.info(f"[Training, 1, {button}, {response_code}, {confidence}]")


    ### ------ TASK----- ###
    # Wait for Enter press to start
    instruction_text = visual.TextStim(win, text="Training Complete \n RESEARCHER:\nPress ENTER when you are ready to begin task", color='black', height=30, wrapWidth=800, pos = (0,0))
    wait_for_key_press(win, instruction_text)
    win.flip()
    logger.info("Commencing Task")
    core.wait(0.5)
    
    #T1 QUESTIONS
    questions = [
            ('Hunger1', 'I am', ['Not at all hungry', 'Very hungry']),
            ('Thirst1 ', 'I feel', ['Not at all thirsty', 'Very thirsty']),
            ('Nausea1', 'I feel', ['Not at all nauseous', 'Very nauseous']),
            ('Disgust1 ', 'I feel', ['Not at all disgusted', 'Very disgusted'])]
    get_post_task_qs(win, ser, csv_file_name, csv_writer, questions, is_integrated_external, use_mouse=True)
    win.flip()

    # Start a 3-second countdown
    countdown_text = visual.TextStim(win, text='', color='black', height=20, pos=(0, 0))
    countdown_clock = core.Clock()
    while countdown_clock.getTime() < 3:
        countdown_text.setText(f"{3 - int(countdown_clock.getTime())}")
        countdown_text.draw()
        win.flip()
        core.wait(0.5)
        win.flip()  

    # for each trial. 
    trials = 30
    half_trials_per_set = 5  # Half of each 10-trial set
    stim_order = []
    # Create three sets of 10 trials, each with five 1s and five 2s
    for _ in range(trials // 3):
        set_of_10 = [1] * half_trials_per_set + [2] * half_trials_per_set
        random.shuffle(set_of_10)
        stim_order.extend(set_of_10)
        if if_debug:
            print(stim_order)
    logger.info(f"Participant sounds will be played in positions [{stim_order}].")
    
    for t in range(0,trials):

        if if_debug:
            print(t)
        #Set other sound file to play
        if t < 9:
            condition = 'Fasted'
        elif 9 <= t < 19:
            condition = 'Post_Drink_1'
        else: 
            condition = 'Post_Drink_2'

        if is_integrated_external:
            # Send DELAY VALUE  to Spike2
            spike_log = 'TRIAL_' +str(t) +'-COND_'+condition +';'
            ser.write(spike_log.encode('utf-8'))
            
        #Specify the sound file to play
        #t = 0  = sound file created during training
        #else created during questions
        sound_file_play = os.path.join(gastric_sound_dir, 'gastric_rec_' + str(t) +'.wav')
        print(sound_file_play)

        button, response_code, confidence = run_discrimination_trial(win, ser, fs, duration, t, stim_order[t], gastric_sound_dir, sound_file_play, is_integrated_external, if_debug)

        # Write trial information to the CSV file
        with open(csv_file_name, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                csv_writer.writerow([t, stim_order[t],  button, response_code, confidence])
                csv_file.close()
        logger.info(f"[{t}, {stim_order[t]}, {button}, {response_code}, {confidence}]")

        #Additional drink 
        if t == 9: 
            #T2 QUESTIONS
            questions = [
                        ('Hunger2', 'I am', ['Not at all hungry', 'Very hungry']),
                        ('Thirst2 ', 'I feel', ['Not at all thirsty', 'Very thirsty']),
                        ('Nausea2', 'I feel', ['Not at all nauseous', 'Very nauseous']),
                        ('Disgust2 ', 'I feel', ['Not at all disgusted', 'Very disgusted'])]
            get_post_task_qs(win, ser, csv_file_name, csv_writer, questions, is_integrated_external, use_mouse=True)
            win.flip()

            drink_text = visual.TextStim(win, text="Consume Drink 1\n\n RESEARCHER: \n Press ENTER to continue", color='black', height=30, wrapWidth=800, pos = (0,0))
            logger.info("Drink 1 Prompt")
            wait_for_key_press(win, drink_text)
            logger.info("Drink 1 Continue")
        if t == 19: 
            #T3 QUESTIONS
            questions = [
                    ('Hunger3', 'I am', ['Not at all hungry', 'Very hungry']),
                    ('Thirst3 ', 'I feel', ['Not at all thirsty', 'Very thirsty']),
                    ('Nausea3', 'I feel', ['Not at all nauseous', 'Very nauseous']),
                    ('Disgust3 ', 'I feel', ['Not at all disgusted', 'Very disgusted'])]
            get_post_task_qs(win, ser, csv_file_name, csv_writer, questions, is_integrated_external, use_mouse=True)
            win.flip()

            drink_text = visual.TextStim(win, text="Consume Drink 2\n\n RESEARCHER: \nPress ENTER to continue", color='black', height=30, wrapWidth=800, pos = (0,0))
            logger.info("Drink 2 Prompt")
            wait_for_key_press(win, drink_text)
            logger.info("Drink 2 Continue")
        if t == 29:

            #T4 QUESTIONS
            questions = [
                ('Hunger4', 'I am', ['Not at all hungry', 'Very hungry']),
                ('Thirst4 ', 'I feel', ['Not at all thirsty', 'Very thirsty']),
                ('Nausea4', 'I feel', ['Not at all nauseous', 'Very nauseous']),
                ('Disgust4 ', 'I feel', ['Not at all disgusted', 'Very disgusted']),
                ('TaskGeneral', 'I find the task', ['Very unpleasant' , 'Very pleasant']),
                ('Difficulty', 'I find the task', ['Very easy', 'Very hard']),
                ('Gut_Norm ', 'Usually my stomach', ['Is still/\nQuiet', 'Moves a lot/\nIs noisy']),
                ('Drink1_Like ', 'Drink 1 was', ['Very unpleasant' , 'Very pleasant']),
                ('Drink2_Like ', 'Drink 2 was', ['Very unpleasant' , 'Very pleasant'])
            ]

            get_post_task_qs(win, ser, csv_file_name, csv_writer, questions, is_integrated_external, use_mouse=True)
            win.flip()
            complete_text = visual.TextStim(win, text="Task Complete", color='black', height=30, wrapWidth=800, pos = (0,0))
            complete_text.draw()
            win.flip()
            logger.info("Task Complete")
            core.wait(2)
            break

    if is_integrated_external:
        # Send DELAY VALUE  to Spike2
        spike_log = 'END TASK;'
        ser.write(spike_log.encode('utf-8'))
        
         
    # Close the window after the trials are completed
    win.close()
    #core.quit()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='rumble_recognition.log',
                        filemode='a')
    logger = logging.getLogger(__name__)

    config_file_path = os.path.join('config.yaml')
    directory = "DIRECTORY/PATH/"
    #directory = os.getcwd()
    # Format the data as a dictionary for structured output
    data_to_write = {
        'Participant ID': 'sub-999',
        'Home Directory': directory,
        'Behavioural Directory': directory,
        'Log Directory': directory}
    with open(config_file_path , 'w') as file:
        yaml.dump(data_to_write, file, sort_keys=False)

    run_gastric_stethoscope_self(is_integrated_external = False, is_integrated_toolbox = False, if_debug = False)