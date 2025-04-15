import csv
import logging
import numpy as np
import random
import os
import sounddevice as sd
import sys  
import time
import wave
from psychopy import visual, sound, core, monitors, data, event
from psychopy.sound import Sound
from scipy.io import wavfile
from scipy.io.wavfile import write

def play_sound_on_click(win, button_text, sound_file, button_pos=(0, 0), button_size=(200, 100)):
    """
    Displays a button on the screen and plays a sound when the button is clicked.

    Parameters:
    win (visual.Window): The PsychoPy window object.
    button_text (str): The text to display on the button.
    sound_file (str): The file path to the sound to be played.
    button_pos (tuple): The position of the button on the screen.
    button_size (tuple): The size of the button.
    """
    # Create the button
    button = visual.Rect(win, width=button_size[0], height=button_size[1], pos=button_pos, fillColor='lightgray', lineColor='black')
    button_label = visual.TextStim(win, text=button_text, pos=button_pos, color='black')

    # Load the sound
    samplerate, data = wavfile.read(sound_file)

    # Create a mouse object
    mouse = event.Mouse(win=win)

    while True:
        # Draw the button and label
        button.draw()
        button_label.draw()
        win.flip()

        # Check for mouse click
        if mouse.isPressedIn(button):
            sd.play(data, samplerate)
            sd.wait()

        # Check for Enter key press to break the loop
        keys = event.getKeys()
        if 'return' in keys:
            break

        # Check for escape key press to exit
        if 'escape' in keys:
            core.quit() 

def wait_for_key_press(win, instruction_text):
    """
    Presents an instruction on the screen and waits for 'RETURN' key press to proceed

     Parameters
    ----------
    win : visual.Window() object
    
     Examples
    ----------
    win = visual.Window(fullscr=False, color='white', units='pix')
    
    """
    while True:
        instruction_text.draw()
        win.flip()
        # event.waitKeys(keyList=['return'])

        # Check for Enter key press to break the loop
        keys = event.getKeys()
        if 'return' in keys:
            break

        # Check for escape key press to exit
        if 'escape' in keys:
            core.quit() 

def show_instructions(win, instruction_text, color, height, wrapWidth, position):
    """
    Presents an instruction on the screen and waits for 'RETURN' key press to proceed

     Parameters
    ----------
    win : visual.Window() object
    
     Examples
    ----------
    win = visual.Window(fullscr=False, color='white', units='pix')
    
    """
    # Create the instruction text stimulus
    instructions = visual.TextStim(
        win=win,
        text=instruction_text,
        color=color,
        height=height,
        wrapWidth=wrapWidth,
        pos=position
    )
    
    while True:
        instructions.draw()
        win.flip()
    
        keys = event.getKeys()
        if 'escape' in keys:
            core.quit()

        if 'left' in keys:
            return 'previous' # Exit loop if left arrow key is pressed

        if 'return' in keys:
            return 'next' # Exit loop if enter is pressed

def show_instructions_list(instructions_list, logger):
    """
        Displays a list of instruction pages sequentially, moving to the next page when the right arrow key is pressed,
        and to the previous page when the left arrow key is pressed.

        Parameters:
        instructions_list (list): A list of tuples, each containing parameters for the show_instructions function.
        """
    index = 0  # Initialize the index to display the first instruction
    while index < len(instructions_list):
        result = show_instructions(*instructions_list[index])
        logger.info(f"Instructions_{index + 1}")
        if result == 'next':
            index += 1  # Move to the next instruction in the list
        elif result == 'previous' and index > 0:
            index -= 1  # Move to the previous instruction in the list

def get_confidence_mouse(win):
    """
    Collects the participant's confidence rating about their response.

    Parameters
    ----------
    win : visual.Window
        PsychoPy window object used to present stimuli.

    Returns
    -------
    float
        The final position of the slider, representing the participant’s confidence rating on a scale from 0 to 100.
    
    Notes
    -----
    This function displays a text prompt and a slider for the participant to indicate their confidence in their response. 
    The slider has three ticks at 0, 50, and 100, corresponding to "Guess," an empty label, and "Certain," respectively. 
    The initial slider position is randomized between 20% and 80%. 
    Participants can adjust the slider by moving the mouse, with the slider's position constrained between 0 and 100.
    """

    # Create a slider with a marker
    slider = visual.Slider(win, ticks=(0, 100), labels=["Guess", "Certain"], granularity=1,
                           style=['rating'], pos=(0, 0), size=(400, 20),
                           labelHeight=20, color="Black")

    # Randomize initial slider position between 20% and 80%
    initial_position = random.uniform(20, 80)
    slider.markerPos = initial_position
    line = visual.Line(win, start=(-200, -5), end=(200, -5), lineColor=(0.5, 0.5, 0.5), lineWidth=5)
    
    # Instruction text
    instruction = visual.TextStim(win, text="How confident are you? (0-100)", pos=(0, 50), color="Black", height=20)

    # Response collection loop
    response = None
    mouse = event.Mouse(win=win)

    while response is None:
        instruction.draw()
        line.draw() 
        slider.draw()
        win.flip()

        keys = event.getKeys()
        if 'escape' in keys:
            core.quit()

        # Check for mouse click
        if mouse.getPressed()[0]:  # Left mouse click
            response = slider.markerPos
            #print(f"Confidence rating = {response}")
            core.wait(1)  # Wait 1 second before ending
            break

        # Update the slider marker position with mouse movement
        mouse_x, _ = mouse.getPos()
        slider_value = (mouse_x + 300) / 6  # Mapping mouse_x to range 0-100
        if 0 <= slider_value <= 100:  # Ensure the value is within bounds
            slider.markerPos = slider_value

        event.clearEvents(eventType='keyboard')

    return response

def run_instructions_calibration(win, ser, fs, duration, gastric_sound_dir, is_integrated_external, if_debug=True):
    """
    Runs the training phase, playing 20 seconds of a participant's gastric sound while recording it.
    The audio file is saved in the specified subdirectory with a filename format
    that includes the participant ID.

    Parameters
    ----------
    win : visual.Window
        The PsychoPy window used for displaying visual stimuli.
    fs : int
        Sample rate for recording and playback.
    input_device_id : int
        Device ID for capturing audio input.
    output_device_id : int
        Device ID for audio output (None for default).
    duration : int
        Duration of the recording in seconds.
    participant_id : str
        Unique ID for the participant, used in the filename.
    sub_dir : str
        The directory where the recorded audio file will be saved.
    if_debug : bool
        Whether to print debug information.
    
    Returns
    -------
    None
    """

    logger = logging.getLogger(name=None)

    # Ensure subdirectory exists
    os.makedirs(gastric_sound_dir, exist_ok=True)

    # Prepare filename with participant ID
    filename = os.path.join(gastric_sound_dir, "gastric_rec_0.wav")
    if if_debug:
        print(f"Recording will be saved to {filename}")
    logger.info(f"Recording will be saved to {filename}")

    if is_integrated_external:
        # Send DELAY VALUE  to Spike2
        spike_log = 'RECORDING_0;'
        ser.write(spike_log.encode('utf-8'))
        spike_log = 'INSTRUCTIONS;'
        ser.write(spike_log.encode('utf-8'))

    instruction_text1 = ("In this task we are interested to know how well people can detect sensations from their stomach."
                        "\n \n You will hear two sounds - in one of them we will play your stomach sounds to you live, and in the other we will play pre-recorded sounds from your stomach. \n "
                        "\n \n Your task is to figure out which of the sounds, sound 1 or 2, is live.")
    instruction_text2 = ("You will see an icon of a stomach on the screen as your signal that the sounds are about to begin."
                        "\n Once they have finished, you will be asked to submit your answer by clicking the button using the mouse on sound 1 or sound 2."
                        "\n Then, you need to indicate how confident you are in your decision using a sliding scale from Guess to Certain."
                        "\n First move the mouse to move the ball, and then click to lock it into place.")
    instruction_text3 = ("You will do this 10 times, \n then you will drink Drink 1 (fizzy water) and do this another 10 times, \n then you drink Drink 2 (Huel) and do this a final 10 times."
                        "\n The whole task takes about 30 minutes."
                        "\n \n At the end there are some questions about how you found the task, and you answer them using the same type of sliding scale as before.")

    instruction_1 = [win, instruction_text1, 'black', 20, 600, (0,0)]
    instruction_2 = [win, instruction_text2, 'black', 20, 600, (0,0)]
    instruction_3 = [win, instruction_text3, 'black', 20, 600, (0,0)]
    instruction_4 = [win, "RESEARCHER:\n Press enter to start calibration", 'black', 30, 800, (0,0)]

    instructions_list = [instruction_1, instruction_2, instruction_3, instruction_4]

    show_instructions_list(instructions_list=instructions_list, logger=logger)
    
    logger.info("Calibration prompt")

    
    # Set up countdown
    countdown_text = visual.TextStim(win, text='', color='black', height=20, pos=(0, 0))
    chunk_duration = duration  # 15 seconds per chunk

    # Prepare to record #1
    gastric_data_full = np.zeros((int(chunk_duration * fs), 1))  
    countdown_clock = core.Clock()

    # Start recording
    gastric_data = sd.playrec(np.zeros((int(chunk_duration * fs), 1)), samplerate=fs, channels=1)

    # Countdown for 15 seconds, but display values 30 to 15
    countdown_clock.reset()  # Reset the clock
    while countdown_clock.getTime() < chunk_duration:
        elapsed_time = countdown_clock.getTime()
        displayed_value = int(31 - elapsed_time)  # Map time 0-15s to values 30-15
        countdown_text.setText(f"{displayed_value}")
        countdown_text.draw()
        win.flip()
        core.wait(0.5)
        win.flip()

        keys = event.getKeys()
        if 'escape' in keys:
            core.quit()

    logger.info("Finished part 1")

    # Ensure recording for the chunk is finished
    sd.wait()

    # Extract the 15-second chunk from the recording
    gastric_chunk_data = gastric_data[:int(chunk_duration * fs)]

    # Convert to 16-bit integer format
    gastric_chunk_data_int = np.int16(gastric_chunk_data * 32767)  # Scale to 16-bit integer range

    # Save the chunk
    file_path = os.path.join(gastric_sound_dir, "gastric_rec_training.wav")
    write(file_path, fs, gastric_chunk_data_int)

    # Prepare to record #2
    gastric_data_full = np.zeros((int(chunk_duration * fs), 1))  
    countdown_clock = core.Clock()

    # Start recording
    gastric_data = sd.playrec(np.zeros((int(chunk_duration * fs), 1)), samplerate=fs, channels=1)

    # Countdown for 15 seconds, but display values 30 to 15
    countdown_clock.reset()  # Reset the clock
    while countdown_clock.getTime() < chunk_duration:
        elapsed_time = countdown_clock.getTime()
        displayed_value = int(16 - elapsed_time)  # Map time 0-15s to values 30-15
        countdown_text.setText(f"{displayed_value}")
        countdown_text.draw()
        win.flip()
        core.wait(0.5)
        win.flip()

        keys = event.getKeys()
        if 'escape' in keys:
            core.quit()

    logger.info("Finished part 2")

    # Ensure recording for the chunk is finished
    sd.wait()

    # Extract the 15-second chunk from the recording
    gastric_chunk_data = gastric_data[:int(chunk_duration * fs)]

    # Convert to 16-bit integer format
    gastric_chunk_data_int = np.int16(gastric_chunk_data * 32767)  # Scale to 16-bit integer range

    # Save the chunk
    file_path = os.path.join(gastric_sound_dir, "gastric_rec_0.wav")
    write(file_path, fs, gastric_chunk_data_int)

def run_training(win, ser, duration, sound_file_play, is_integrated_external, if_debug):

    logger = logging.getLogger(name=None)

    # Display the gut image
    gut_image = visual.ImageStim(win, image=r'../tasks_helpers/gut_icon.png', pos=(0, 0.5))
    gut_image.draw()    
    win.flip()
    core.wait(0.5)
    
    if if_debug:
            participant_sound = visual.TextStim(win, text='Play live', pos=(0, 100), color='black', height=20) 
            participant_sound.draw()
        
    logger.info(f"LIVE sound")
    sound1 = visual.TextStim(win, text='Sound 1', pos=(0, 0), color='black', height=20) 
    sound1.draw()
    win.flip()
    core.wait(2)
    
    fixation_text = visual.TextStim(win, text='+', pos=(0, 0), color='black', height=40)
    fixation_text.draw()
    win.flip()
    if is_integrated_external:
        #Send log  to Spike2
        delay_command = 'LIVE START;'  #PARTICIPANT START
        ser.write(delay_command.encode('utf-8'))
        if if_debug: 
            print(f"Sent: {delay_command}")
    #PLAY SOUND FROM PARTICIPANT FOR [duration] SECONDS
    #print(sd.query_devices())
    stream = sd.Stream(channels=1, callback=callback)
    with stream:
        sd.sleep(int(duration * 1000))

    sd.sleep(10)  # Small delay to ensure stream resets properly
    
    if is_integrated_external:
        #Send log  to Spike2
        delay_command = 'LIVE END;'  #PARTICIPANT START
        ser.write(delay_command.encode('utf-8'))
        if if_debug: 
            print(f"Sent: {delay_command}")

    win.flip()
    core.wait(2) #2s ISI
    
    #Fake
    if if_debug:
        not_participant_sound = visual.TextStim(win, text='Play pre-recorded ', pos=(0, 100), color='black', height=20) 
        not_participant_sound.draw()
    logger.info(f"REC sound")
    logger.info(f"Sound file playing: {sound_file_play}")
    sound2 = visual.TextStim(win, text='Sound 2', pos=(0, 0), color='black', height=20) 
    sound2.draw()
    win.flip()
    core.wait(2)

    fixation_text = visual.TextStim(win, text='+', pos=(0, 0), color='black', height=40)
    fixation_text.draw()
    win.flip()
    samplerate, data = wavfile.read(sound_file_play)
    sd.play(data, samplerate)
    sd.wait()
    if is_integrated_external:
            #Send log  to Spike2
            delay_command = 'REC END;' #FAKE START
            ser.write(delay_command.encode('utf-8'))
            if if_debug: 
                print(f"Sent: {delay_command}")

     # Define the buttons and their properties
    buttons = ['Sound 1', 'Sound 2']
    button_positions = [(-100, 0), (100, 0)]  # x, y positions for the buttons
    button_colors = [(0.5, 0.5, 0.5)] * len(buttons)  # Initial button colors (light grey)

    # Create button visual components
    button_visuals = []
    for i, button in enumerate(buttons):
        # Rectangle for each button
        rect = visual.Rect(win, width=150, height=50, pos=button_positions[i], fillColor=button_colors[i])
        button_visuals.append(rect)
        # Text for each button
        button_text = visual.TextStim(win, text=button, pos=button_positions[i], color=(-1,-1,-1))
        button_visuals.append(button_text)

    # Create question text
    question = visual.TextStim(win, text="Which sound was the livestream of your stomach?", pos=(0, 100), color=(-1,-1,-1))

    # Function to update button colors
    def update_button_colors(selected_index):
        for i in range(len(buttons)):
            if i == selected_index:
                button_visuals[i * 2].fillColor = (1, 0, 0)  # Red for selected
            else:
                button_visuals[i * 2].fillColor = (0.5, 0.5, 0.5)  # Light grey for unselected

    # Draw the question and buttons
    question.draw()
    for button in button_visuals:
        button.draw()
    win.flip()

    # Record response
    response = None
    mouse = event.Mouse(win=win)
    while response is None:
        pos = mouse.getPos()

        # Check which button is hovered over
        buttons_clicked = [button_rect.contains(pos) for button_rect in button_visuals[::2]]
        
        # Determine selected button
        selected_button_index = None
        for i, clicked in enumerate(buttons_clicked):
            if clicked:
                selected_button_index = i
                break
        # Update button colors
        if selected_button_index is not None:
            update_button_colors(selected_button_index)
        
        # Draw question and buttons again to update colors
        question.draw()
        for button in button_visuals:
            button.draw()
        win.flip()

        keys = event.getKeys()
        if 'escape' in keys:
            core.quit()

        # Check for mouse click
        if mouse.getPressed()[0]:  # Left mouse button
            if selected_button_index is not None:
                response = selected_button_index

    #core.wait(2)

    # Code the response
    if response is not None:
        response_code = [1, 2][response]  # Maps to 1: sound 1; 2: sound 2
        if if_debug:
            print("Participant Response:", response_code, buttons[response])
    
    confidence = get_confidence_mouse(win)

    #Display 'calibrating'
    instruction_text = visual.TextStim(win, text="Calibrating...", color='black', height=30, wrapWidth=800, pos = (0,0))
    instruction_text.draw()
    win.flip()
    core.wait(2)

    if is_integrated_external:
        # Send log  to Spike2
        spike_log = 'RESPONSES: ' + str(buttons[response]) +'_' + str(response_code) +'_' + str(confidence) +';'
        ser.write(spike_log.encode('utf-8'))

    return  buttons[response], response_code, confidence



def callback(indata, outdata, frames, time, status):
    """
    Handles audio input data processing by forwarding the incoming data to output.

    Parameters
    ----------
    indata : numpy.ndarray
        The input audio data received from the audio stream, containing samples for processing.

    outdata : numpy.ndarray
        The output array where processed audio data will be written, allowing it to be sent to the audio output stream.

    frames : int
        The number of audio frames being processed in this callback, representing how many samples are being handled at once.

    time : float
        The timestamp indicating the current time in seconds for the audio processing callback, useful for synchronization.

    status : StreamStatus
        An object containing information about the audio stream status, such as any errors or warnings that may have occurred.

    Returns
    -------
    None
        This function does not return a value but processes the input data and writes it to the output.

    Notes
    -----
    - The function checks if there is any status information available; if so, it prints the status to the console for debugging purposes.
    - The input audio data (`indata`) is directly copied to the output data array (`outdata`), allowing for real-time audio playback or further processing.
    - The function is used as a callback in an audio processing framework, such sounddevice, where it is called automatically during the audio stream.
    """
    if status:
        print(status)
    outdata[:] = indata
    
def run_discrimination_trial(win, ser, fs, duration, trial_num, participant_loc, gastric_sound_dir, sound_file_play, is_integrated_external, if_debug): 
    """
    Conducts a discrimination trial where participants listen to two sounds and indicate which one they believe corresponds to a given stimulus.
    
    Parameters
    ----------
    win : visual.Window
        The PsychoPy window used for displaying visual stimuli and capturing participant input.
    
    ser : serial.Serial
        The serial connection used for sending log commands to an external system (e.g., Spike2).
        
    fs : int
        The sample rate for audio recording and playback.
    
    duration : float
        The duration in seconds for which participant sounds are recorded and played back.
        
    trial_num : int
        The current trial number, used for naming the saved recording file.
        
    participant_loc : int
        An integer indicating the order of sound presentation:
        - 1: Participant sound first, then fake sound.
        - 2: Fake sound first, then participant sound.
        
    gastric_sound_dir : str
        Directory where recorded participant sound files are saved.
        
    sound_file_play : str
        Filepath for the pre-recorded (fake) sound that is played during the trial.
        
    is_integrated_external : bool
        A flag indicating whether the system is integrated with external logging (e.g., Spike2). 
        If True, logs are sent to Spike2 during the trial.
        
    if_debug : bool
        A flag that determines if debug information should be printed to the console for troubleshooting purposes.

    Returns
    -------
    tuple
        A tuple containing:
        - str: The label of the sound selected by the participant ('Sound 1' or 'Sound 2').
        - int: A response code corresponding to the participant's choice:
          - 1: Sound 1
          - 2: Sound 2
        - int: The confidence rating obtained from the participant (e.g., a scale from 1 to 5).

    Notes
    -----
    - The trial begins with a gut image display, followed by a fixation cross.
    - The order of sound playback depends on the `participant_loc` parameter.
    - Each sound is preceded by a text stimulus showing Sound 1 or Sound 2 (and participant or fake for debugging).
    - If `is_integrated_external` is True, commands are sent to an external system (e.g., Spike2) to log events during the trial.
    - After both sounds are played, participants are prompted to select which sound they believe corresponds to their stomach's sound.
    - Two buttons are displayed for the participant's selection, and their interaction is captured via mouse clicks.
    - The button colors change based on mouse hover status for visual feedback.
    - The function includes a response mapping to convert participant choices into a standardized response code.
    - A confidence rating is requested from the participant after they make their choice.
    - The recorded sound data is converted to a 16-bit integer format and saved to the specified file path.
    - PsychoPy resources are managed to ensure proper memory release and program continuity.
   """
   
    logger = logging.getLogger(name=None)

    # Display the gut image
    gut_image = visual.ImageStim(win, image=r'../tasks_helpers/gut_icon.png', pos=(0, 0.5))
    gut_image.draw()    
    win.flip()
    core.wait(0.5)
    
    if participant_loc == 1:
        #Participant sound first, then fake sound 
        #Participant
        if if_debug:
            participant_sound = visual.TextStim(win, text='Play live', pos=(0, 100), color='black', height=20) 
            participant_sound.draw()
        logger.info(f"LIVE sound")
        sound1 = visual.TextStim(win, text='Sound 1', pos=(0, 0), color='black', height=20) 
        sound1.draw()
        win.flip()
        core.wait(2)
        
        fixation_text = visual.TextStim(win, text='+', pos=(0, 0), color='black', height=40)
        fixation_text.draw()
        win.flip()
        if is_integrated_external:
            #Send log  to Spike2
            delay_command = 'LIVE START;'  #PARTICIPANT START
            ser.write(delay_command.encode('utf-8'))
            if if_debug: 
                print(f"Sent: {delay_command}")
        #PLAY SOUND FROM PARTICIPANT FOR [duration] SECONDS
        #print(sd.query_devices())
        stream = sd.Stream(channels=1, callback=callback)
        with stream:
            sd.sleep(int(duration * 1000))
        sd.sleep(10)  # Small delay to ensure stream resets properly
        
        if is_integrated_external:
            #Send log  to Spike2
            delay_command = 'LIVE END;'  #PARTICIPANT START
            ser.write(delay_command.encode('utf-8'))
            if if_debug: 
                print(f"Sent: {delay_command}")

        win.flip()
        core.wait(2) #2s ISI
        
        #Fake
        if if_debug:
            not_participant_sound = visual.TextStim(win, text='Play pre-recorded ', pos=(0, 100), color='black', height=20) 
            not_participant_sound.draw()
        logger.info(f"REC sound")
        logger.info(f"Sound file playing: {sound_file_play}")
        sound2 = visual.TextStim(win, text='Sound 2', pos=(0, 0), color='black', height=20) 
        sound2.draw()
        win.flip()
        core.wait(2)
        
        if is_integrated_external:
            #Send log  to Spike2
            delay_command = 'REC START;' #FAKE START
            ser.write(delay_command.encode('utf-8'))
            if if_debug: 
                print(f"Sent: {delay_command}")

        fixation_text = visual.TextStim(win, text='+', pos=(0, 0), color='black', height=40)
        fixation_text.draw()
        win.flip()
        samplerate, data = wavfile.read(sound_file_play)
        sd.play(data, samplerate)
        sd.wait()
        
        if is_integrated_external:
            #Send log  to Spike2
            delay_command = 'REC END;' #FAKE START
            ser.write(delay_command.encode('utf-8'))
            if if_debug: 
                print(f"Sent: {delay_command}")


    else: 
        #Fake sound first, then participant sound
        #Fake
        if if_debug:
            not_participant_sound = visual.TextStim(win, text='Play pre-recorded', pos=(0, 100), color='black', height=20) 
            not_participant_sound.draw()
        logger.info(f"REC sound")
        logger.info(f"Sound file playing: {sound_file_play}")
        sound1 = visual.TextStim(win, text='Sound 1', pos=(0, 0), color='black', height=20) 
        sound1.draw()
        win.flip()
        core.wait(2)

        if is_integrated_external:
            #Send log  to Spike2
            delay_command = 'REC START;' #FAKE START
            ser.write(delay_command.encode('utf-8'))
            if if_debug: 
                print(f"Sent: {delay_command}")

        fixation_text = visual.TextStim(win, text='+', pos=(0, 0), color='black', height=40)
        fixation_text.draw()
        win.flip()
        samplerate, data = wavfile.read(sound_file_play)
        sd.play(data, samplerate)
        sd.wait()
        if is_integrated_external:
            #Send log  to Spike2
            delay_command = 'REC END;' #FAKE START
            ser.write(delay_command.encode('utf-8'))
            if if_debug: 
                print(f"Sent: {delay_command}")
        
        win.flip()
        core.wait(2) #2s ISI
        
        #Participant
        if if_debug:
            participant_sound = visual.TextStim(win, text='Play live', pos=(0, 100), color='black', height=20) 
            participant_sound.draw()
        logger.info(f"LIVE sound")
        sound2 = visual.TextStim(win, text='Sound 2', pos=(0, 0), color='black', height=20) 
        sound2.draw()
        win.flip()
        core.wait(2)

        fixation_text = visual.TextStim(win, text='+', pos=(0, 0), color='black', height=40)
        fixation_text.draw()
        win.flip()
        if is_integrated_external:
            #Send log  to Spike2
            delay_command = 'LIVE START;'  #PARTICIPANT START
            ser.write(delay_command.encode('utf-8'))
            if if_debug: 
                print(f"Sent: {delay_command}")
        #PLAY SOUND FROM PARTICIPANT FOR [duration] SECONDS
        #print(sd.query_devices())
        stream = sd.Stream(channels=1, callback=callback)
        with stream:
            sd.sleep(int(duration * 1000))
        sd.sleep(10)  # Small delay to ensure stream resets properly

        if is_integrated_external:
            #Send log  to Spike2
            delay_command = 'LIVE END;'  #PARTICIPANT START
            ser.write(delay_command.encode('utf-8'))
            if if_debug: 
                print(f"Sent: {delay_command}")
    
    # START RECORDING AUDIO FOR NEXT TRIAL
    if is_integrated_external:
        # Send DELAY VALUE  to Spike2
        spike_log = 'RECORDING_' +str(trial_num+1) +';'
        ser.write(spike_log.encode('utf-8'))
        spike_log = 'QUESTION;'
        ser.write(spike_log.encode('utf-8'))

    #Prepare filename
    sound_file_rec = os.path.join(gastric_sound_dir, "gastric_rec_" +str(trial_num+1) +".wav")
    if if_debug:
        print(f"Recording will be saved to {sound_file_rec}")
    logger.info(f"Recording will be saved to {sound_file_rec}")

    # Start non-blocking recording
    gastric_data = sd.playrec(np.zeros((int(duration * fs), 1)), samplerate=fs, channels=1)


    ## COLLECT PARTICIPANT RECORDINGS WHILE RECORDING OCCURS
    # Define the buttons and their properties
    buttons = ['Sound 1', 'Sound 2']
    button_positions = [(-100, 0), (100, 0)]  # x, y positions for the buttons
    button_colors = [(0.5, 0.5, 0.5)] * len(buttons)  # Initial button colors (light grey)

    # Create button visual components
    button_visuals = []
    for i, button in enumerate(buttons):
        # Rectangle for each button
        rect = visual.Rect(win, width=150, height=50, pos=button_positions[i], fillColor=button_colors[i])
        button_visuals.append(rect)
        # Text for each button
        button_text = visual.TextStim(win, text=button, pos=button_positions[i], color=(-1,-1,-1))
        button_visuals.append(button_text)

    # Create question text
    question = visual.TextStim(win, text="Which sound was the livestream of your stomach?", pos=(0, 100), color=(-1,-1,-1))

    # Function to update button colors
    def update_button_colors(selected_index):
        for i in range(len(buttons)):
            if i == selected_index:
                button_visuals[i * 2].fillColor = (1, 0, 0)  # Red for selected
            else:
                button_visuals[i * 2].fillColor = (0.5, 0.5, 0.5)  # Light grey for unselected

    # Draw the question and buttons
    question.draw()
    for button in button_visuals:
        button.draw()
    win.flip()

    # Record response
    response = None
    mouse = event.Mouse(win=win)
    while response is None:
        pos = mouse.getPos()

        # Check which button is hovered over
        buttons_clicked = [button_rect.contains(pos) for button_rect in button_visuals[::2]]
        
        # Determine selected button
        selected_button_index = None
        for i, clicked in enumerate(buttons_clicked):
            if clicked:
                selected_button_index = i
                break
        # Update button colors
        if selected_button_index is not None:
            update_button_colors(selected_button_index)
        
        # Draw question and buttons again to update colors
        question.draw()
        for button in button_visuals:
            button.draw()
        win.flip()

        # Check for mouse click
        if mouse.getPressed()[0]:  # Left mouse button
            if selected_button_index is not None:
                response = selected_button_index

        keys = event.getKeys()
        if 'escape' in keys:
            core.quit()

    #core.wait(2)

    # Code the response
    if response is not None:
        response_code = [1, 2][response]  # Maps to 1: sound 1; 2: sound 2
        if if_debug:
            print("Participant Response:", response_code, buttons[response])
    
    confidence = get_confidence_mouse(win)

    #Display 'calibrating'
    instruction_text = visual.TextStim(win, text="Calibrating...", color='black', height=30, wrapWidth=800, pos = (0,0))
    instruction_text.draw()
    win.flip()
    
    # Ensure recording is finished
    sd.wait()
    logger.info("Check recording finished")
    sd.stop()  

    # Convert data to 16-bit integer format
    gastric_data_int = np.int16(gastric_data * 32767)  # Scale to 16-bit integer range

    # Save the recording as an integer-format .wav file
    write(sound_file_rec, fs, gastric_data_int)
    if if_debug:
        print(f"Recording saved as {sound_file_rec}")
        print("Recording complete.")
    logger.info(f"Recording saved as {sound_file_rec}")

    core.wait(0.5)

    if is_integrated_external:
        # Send log  to Spike2
        spike_log = 'RESPONSES: ' + str(buttons[response]) +'_' + str(response_code) +'_' + str(confidence) +';'
        ser.write(spike_log.encode('utf-8'))

    return  buttons[response], response_code, confidence

def get_post_task_qs(win, ser, csv_file_name, csv_writer, questions, is_integrated_external, use_mouse=True):
    """
    Displays a series of questions with sliders for post-task feedback and collects responses using either mouse clicks or arrow keys.

    Parameters
    ----------
    win : visual.Window
        PsychoPy window object used to present stimuli.

    use_mouse : bool
        Flag to determine whether to use mouse click or keyboard for responses.

    Returns
    -------
    list of dict
        A list of dictionaries containing the responses to each question.
    """

    for label, question, scale in questions:
        # Create text stimulus for the question
        question_text = visual.TextStim(win, text=question, pos=(0, 0.5), height=20)
        # Instruction text
        instruction = visual.TextStim(win, text=f"{question}", pos=(0, 100), color="Black", height=20)

         # Create a slider with a marker
        slider = visual.Slider(win, ticks=(0, 50, 100), labels=scale, granularity=1,
                               style=['rating'], pos=(0, 0), size=(600, 20),
                               labelHeight=20, color="Black")
        line = visual.Line(win, start=(-300, -5), end=(300, -5), lineColor=(0.5, 0.5, 0.5), lineWidth=5)

        # Randomize initial slider position between 20% and 80%
        initial_position = random.uniform(20, 80)
        slider.markerPos = initial_position

   
        # Display question and slider
        response = None
        mouse = event.Mouse(win=win)

        while response is None:
            win.flip()
            instruction.draw()
            question_text.draw()
            line.draw()
            slider.draw() 

            keys = event.getKeys()
            if 'escape' in keys:
                core.quit()           

            if use_mouse:
                # Mouse click response
                # Check for mouse click
                if mouse.getPressed()[0]:  # Left mouse click
                    response = slider.markerPos
                    #print(f"Confidence rating = {response}")
                    core.wait(1)  # Wait 1 second before ending
                    break

                # Update the slider marker position with mouse movement
                mouse_x, _ = mouse.getPos()
                slider_value = (mouse_x + 300) / 6  # Mapping mouse_x to range 0-100
                if 0 <= slider_value <= 100:  # Ensure the value is within bounds
                    slider.markerPos = slider_value

                event.clearEvents(eventType='keyboard')
            else:
                # Keyboard response
                keys = event.getKeys()
                if 'left' in keys:
                    response = 0  # Corresponding to the first option in the scale
                elif 'right' in keys:
                    response = len(scale) - 1  # Corresponding to the last option in the scale
                if response is not None:
                    core.wait(1)  # Wait 1 second before ending

            event.clearEvents(eventType='keyboard')
        

        with open(csv_file_name, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['Question', label, response, 'NA', 'NA']) 
            csv_file.close()