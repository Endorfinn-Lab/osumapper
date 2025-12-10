import gradio as gr
import subprocess
import os
import sys
import io
import traceback
import platform
import re
from contextlib import contextmanager
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='librosa')

try:
    from mania_act_data_prep import step1_load_maps
    from act_train_rhythm import step2_build_model, step2_train_model, step2_evaluate, step2_save
    from act_flow_ds import step3_read_maps_pattern, step3_save_pattern_dataset
    from act_timing import get_timed_osu_file
    from act_newmap_prep import step4_read_new_map
    from mania_act_rhythm_calc import (
        step5_load_model,
        step5_load_npz,
        step5_predict_notes,
        step5_build_pattern,
        merge_objects_each_key,
        mania_modding
    )
    from mania_act_final import step8_save_osu_mania_file, step8_clean_up
except ImportError as e:
    print(f"Error importing project files: {e}")
    print("Please ensure all required .py files are in the same directory as app.py.")
    sys.exit(1)

@contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    sys.stdout = new_stdout = io.StringIO()
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout

def update_log(log_state, new_content):
    if not log_state:
        return new_content
    return log_state + "\n" + new_content

def run_maplist_maker_in_new_terminal():
    command = ["node", "gen_maplist.js"]
    system = platform.system()
    try:
        if system == "Windows":
            subprocess.Popen(['start', 'cmd', '/k'] + command, shell=True)
        elif system == "Darwin":
            script = f'tell app "Terminal" to do script "cd {os.getcwd()} && {" ".join(command)}"'
            subprocess.Popen(['osascript', '-e', script])
        elif system == "Linux":
            try:
                subprocess.Popen(['gnome-terminal', '--'] + command)
            except FileNotFoundError:
                try:
                    subprocess.Popen(['konsole', '-e'] + command)
                except FileNotFoundError:
                    subprocess.Popen(['xterm', '-e'] + command)
        else:
            return f"Unsupported OS: {system}. Please run 'node gen_maplist.js' manually."
        return "Maplist Maker launched in a new terminal."
    except Exception as e:
        return f"Failed to launch terminal: {e}\nTry running 'node gen_maplist.js' manually in your terminal."

def handle_train_step1(current_log, progress=gr.Progress()):
    log = update_log(current_log, "### Step 1/3: Preparing Map Data")
    yield log, gr.Button(interactive=False), gr.Button(interactive=False)
    try:
        progress(0.5, desc="Preparing Map Data...")
        with capture_stdout() as captured:
            step1_load_maps()
        log = update_log(log, f"**Success!** Data preparation complete. You can run this step again to replace the files, or proceed to Step 2.\n```\n{captured.getvalue()}```")
        yield log, gr.Button(interactive=True), gr.Button(interactive=True)
    except Exception:
        error_details = traceback.format_exc()
        log = update_log(log, f"**Error:**\n```\n{error_details}```")
        yield log, gr.Button(interactive=True), gr.Button(interactive=False)

def handle_train_step2(current_log, epochs, batch_size, plot_history, progress=gr.Progress()):
    log = update_log(current_log, "### Step 2/3: Training Rhythm Model")
    yield log, gr.Button(interactive=False), gr.Button(interactive=False)
    try:
        with capture_stdout() as captured:
            train_params = {
                "divisor": 4, "train_epochs": epochs, "train_batch_size": batch_size,
                "plot_history": plot_history, "too_many_maps_threshold": 200,
                "train_epochs_many_maps": 6, "data_split_count": 80
            }
            progress(0.1, desc="Building Model...")
            print("Building model...")
            model = step2_build_model()
            progress(0.2, desc="Training Model...")
            print("\nTraining model... (This may take a while)")
            model = step2_train_model(model, train_params)
            if model is None:
                raise ValueError("Model training failed. Please check your data and parameters.")
            progress(0.8, desc="Evaluating Model...")
            print("\nEvaluating model...")
            step2_evaluate(model)
            progress(0.9, desc="Saving Model...")
            step2_save(model)
        log = update_log(log, f"**Success!** Rhythm model training complete. You can run this step again to retrain and replace the model, or proceed to Step 3.\n```\n{captured.getvalue()}```")
        progress(1)
        yield log, gr.Button(interactive=True), gr.Button(interactive=True)
    except ValueError as ve:
        error_message_str = str(ve)
        user_friendly_error = f"**Error during model training:**\n\n_{error_message_str}_\n\n"
        if "incompatible with the layer" in error_message_str and "shape" in error_message_str:
             user_friendly_error += (
                "**This indicates a 'shape mismatch' error during training.**\n\n"
                "**Possible Cause & Solution:**\n"
                "The data prepared in 'Step 1: Prepare Map Data' has a different format (shape) than what the model architecture expects.\n"
                "This could be caused by issues in the data preparation script (`mania_act_data_prep.py`) or the model definition script (`act_train_rhythm.py`).\n\n"
                "**Please check recent code changes and ensure your map data is processed correctly.**"
            )
        log = update_log(log, user_friendly_error)
        yield log, gr.Button(interactive=True), gr.Button(interactive=False)
    except Exception:
        error_details = traceback.format_exc()
        log = update_log(log, f"**An unexpected error occurred:**\n```\n{error_details}```")
        yield log, gr.Button(interactive=True), gr.Button(interactive=False)

def handle_train_step3(current_log, progress=gr.Progress()):
    log = update_log(current_log, "### Step 3/3: Creating Pattern Dataset")
    yield log, gr.Button(interactive=False)
    try:
        progress(0.5, desc="Creating Pattern Dataset...")
        with capture_stdout() as captured:
            data = step3_read_maps_pattern([])
            step3_save_pattern_dataset(data)
        log = update_log(log, f"**Success!** Pattern dataset creation complete. You can run this step again to replace the file.\n```\n{captured.getvalue()}```")
        log = update_log(log, "## All training steps completed successfully!")
        progress(1)
        yield log, gr.Button(interactive=True)
    except Exception:
        error_details = traceback.format_exc()
        log = update_log(log, f"**Error:**\n```\n{error_details}```")
        yield log, gr.Button(interactive=True)

def reset_training_ui():
    return "", gr.Button(interactive=True), gr.Button(interactive=False), gr.Button(interactive=False)

def handle_step1_click(current_log, osu_file_in, osu_audio_in, progress=gr.Progress()):
    log = update_log(current_log, "### Step 1: Preparing Input File")
    yield log, gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False), None, None, None

    try:
        progress(0.5, desc="Preparing input file...")
        with capture_stdout() as captured:
            if osu_file_in is None or osu_audio_in is None:
                raise ValueError("Please upload both the .osu file and its corresponding audio file.")
            
            desired_filename = os.path.basename(osu_file_in.name)

            with open(osu_file_in.name, 'r', encoding='utf-8') as f:
                osu_content = f.read()
            modified_osu_content = re.sub(
                r"^(AudioFilename:).*",
                r"\1 audio.mp3",
                osu_content,
                flags=re.MULTILINE
            )
            with open("timing.osu", 'w', encoding='utf-8') as f:
                f.write(modified_osu_content)
            command = ["ffmpeg", "-y", "-i", osu_audio_in.name, "-vn", "-ar", "44100", "-b:a", "192k", "audio.mp3"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Processed '{os.path.basename(osu_file_in.name)}' and saved as 'timing.osu' with corrected audio path.")
            print(f"Converted '{os.path.basename(osu_audio_in.name)}' to 'audio.mp3'")
            input_osu_path = "timing.osu"
            step4_read_new_map(input_osu_path)
        log = update_log(log, f"**Success!** Input files processed. You can now proceed to Step 2.\n```\n{captured.getvalue()}```")
        yield log, gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False), gr.Button(interactive=False), desired_filename, None, None
    except FileNotFoundError:
        error_message = (
            "**FATAL ERROR: `ffmpeg` not found!**\n\n"
            "`ffmpeg` is a required program for audio conversion, but it was not found on your system's PATH.\n\n"
            "**Please follow these steps to fix the issue:**\n"
            "1. **Download FFmpeg:** Go to [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/) and download the latest `ffmpeg-release-full` version.\n"
            "2. **Extract the files:** Unzip the downloaded file to a permanent location on your computer, for example, `C:\\ffmpeg`.\n"
            "3. **Add to PATH:** You must add the `bin` folder from your FFmpeg installation (e.g., `C:\\ffmpeg\\bin`) to your Windows System PATH environment variables.\n"
            "4. **Restart:** Close this application and the terminal it's running in, then start it again."
        )
        yield update_log(current_log, error_message), gr.Button(interactive=True), gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False), None, None, None
    except Exception:
        error_details = traceback.format_exc()
        log = update_log(log, f"**Error:**\n```\n{error_details}```")
        yield log, gr.Button(interactive=True), gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False), None, None, None

def handle_step2_click(current_log, note_density, hold_favor, divisor_favor_str, hold_max_ticks, hold_min_return, rotate_mode_str, progress=gr.Progress()):
    log = update_log(current_log, "### Step 2: Predicting Rhythm & Building Patterns")
    yield log, gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False), None, None
    
    try:
        progress(0, desc="Loading model and data...")
        with capture_stdout() as captured:
            rotate_mode_map = {"No Rotation": 0, "Random": 1, "Mirror": 2, "Circulate": 3, "Circulate + Mirror": 4}
            df = [float(x.strip()) for x in divisor_favor_str.split(',')]
            rm = rotate_mode_map.get(rotate_mode_str, 4)
            rhythm_params = (note_density, hold_favor, df, hold_max_ticks, hold_min_return, rm)
            
            model = step5_load_model()
            npz = step5_load_npz()
            
            progress(0.3, desc="Predicting rhythm...")
            predictions = step5_predict_notes(model, npz, rhythm_params)
            
            progress(0.7, desc="Building patterns...")
            notes_each_key = step5_build_pattern(predictions, rhythm_params)
            
        log = update_log(log, f"**Success!** Rhythm predicted and patterns built. You can now proceed to Step 3.\n```\n{captured.getvalue()}```")
        progress(1)
        yield log, gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False), notes_each_key, None

    except ValueError as ve:
        error_message_str = str(ve)
        user_friendly_error = f"**Error during rhythm prediction:**\n\n_{error_message_str}_\n\n"
        if "incompatible with the layer" in error_message_str and "shape" in error_message_str:
            user_friendly_error += (
                "**This indicates a 'shape mismatch' error.**\n\n"
                "**Possible Cause & Solution:**\n"
                "The pre-trained model expects data in a different format (shape) than what was generated from your input file.\n"
                "This can happen if the model was trained with an older version of the scripts, and the data processing has since changed.\n\n"
                "**Please try retraining your model using the 'Model Training' tab to resolve this incompatibility.**"
            )
        elif "Input audio processing resulted in no data" in error_message_str or "too short" in error_message_str:
            user_friendly_error += (
                "**This usually means there is a problem with the input audio file.**\n\n"
                "**Possible Causes & Solutions:**\n"
                "1.  **The audio file is too short:** The model needs several seconds of audio to analyze.\n"
                "2.  **The audio file is silent or corrupted.**\n\n"
                "Please try again with a different, valid audio file."
            )
        log = update_log(log, user_friendly_error)
        yield log, gr.Button(interactive=True), gr.Button(interactive=False), gr.Button(interactive=False), None, None
    except Exception:
        error_details = traceback.format_exc()
        log = update_log(log, f"**An unexpected error occurred:**\n```\n{error_details}```")
        yield log, gr.Button(interactive=True), gr.Button(interactive=False), gr.Button(interactive=False), None, None

def handle_step3_click(current_log, notes_each_key_state, key_fix_str, progress=gr.Progress()):
    log = update_log(current_log, "### Step 3: Applying Mods")
    yield log, gr.Button(interactive=False), gr.Button(interactive=False), None

    try:
        progress(0.5, desc="Applying mods...")
        key_fix_map = {"Inactive": 0, "Remove Late Note": 1, "Remove Early Note": 2, "Divert (Recommended)": 3}
        key_fix = key_fix_map.get(key_fix_str, 3)
        with capture_stdout() as captured:
            modding_params = {"key_fix": key_fix}
            notes_each_key = mania_modding(notes_each_key_state, modding_params)
            notes, final_key_count = merge_objects_each_key(notes_each_key)
        log = update_log(log, f"**Success!** Mods applied and notes merged. You can now proceed to Step 4.\n```\n{captured.getvalue()}```")
        yield log, gr.Button(interactive=True), gr.Button(interactive=True), (notes, final_key_count)
    except Exception:
        error_details = traceback.format_exc()
        log = update_log(log, f"**Error:**\n```\n{error_details}```")
        yield log, gr.Button(interactive=True), gr.Button(interactive=False), None

def handle_step4_click(current_log, final_notes_state, desired_filename, progress=gr.Progress()):
    log = update_log(current_log, "### Step 4: Saving Final .osu File")
    yield log, gr.Button(interactive=False), None
    
    try:
        progress(0.5, desc="Saving .osu file...")
        notes, final_key_count = final_notes_state
        with capture_stdout() as captured:
            default_generated_name = step8_save_osu_mania_file(notes, final_key_count)
            
            if desired_filename and os.path.exists(default_generated_name):
                print(f"Renaming '{default_generated_name}' to '{desired_filename}'...")
                if os.path.exists(desired_filename):
                    os.remove(desired_filename)
                os.rename(default_generated_name, desired_filename)
                final_filename = desired_filename
            else:
                final_filename = default_generated_name

        log = update_log(log, f"**Success!** Map saved as `{os.path.basename(final_filename)}`.\n\n## Generation Complete!")
        yield log, gr.Button(interactive=True), final_filename
    except Exception:
        error_details = traceback.format_exc()
        log = update_log(log, f"**Error:**\n```\n{error_details}```")
        yield log, gr.Button(interactive=True), None

def cleanup_and_reset():
    step8_clean_up()
    return (
        "", 
        None, None, None, None,
        gr.Button(interactive=True),
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Button(interactive=False),
    )

try:
    blocks_ctx = gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), title="Osumapper UI")
except TypeError:
    try:
        blocks_ctx = gr.Blocks(title="Osumapper UI")
    except Exception:
        blocks_ctx = gr.Blocks()

with blocks_ctx as demo:
    gr.Markdown("<h1>Osumapper UI</h1>")
    gr.Markdown("A user interface for the osumapper project. Create maplists, train models, and generate osu!mania charts.")

    with gr.Tabs():
        with gr.TabItem("Maplist Maker"):
            gr.Markdown("## Create `maplist.txt`")
            gr.Markdown(
                "Click the button to launch the Maplist Maker. This tool runs in a **new terminal window** "
                "and starts a local web server in your browser to help you select maps for training. "
                "Keep the new terminal open while using the Maplist Maker."
            )
            maplist_button = gr.Button("Launch Maplist Maker")
            maplist_status = gr.Textbox(label="Status", interactive=False)
            maplist_button.click(fn=run_maplist_maker_in_new_terminal, outputs=maplist_status)

        with gr.TabItem("Model Training"):
            gr.Markdown("## Train Your osu!mania Model")
            gr.Markdown("Follow these steps to prepare your data, train the rhythm model, and create a pattern dataset. You can re-run any step to overwrite its previous output. Subsequent steps are unlocked upon successful completion of the previous one.")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Step 1: Prepare Map Data"):
                            gr.Markdown("Reads `maplist.txt` and converts the maps into `.npz` files for training.")
                            train_step1_button = gr.Button("Run Step 1", variant="primary")

                        with gr.TabItem("Step 2: Train Rhythm Model"):
                            gr.Markdown("Train the model using the prepared files. This may take a long time.")
                            train_epochs = gr.Slider(minimum=1, maximum=100, value=32, step=1, label="Training Epochs")
                            train_batch_size = gr.Slider(minimum=8, maximum=256, value=32, step=8, label="Batch Size")
                            plot_history = gr.Checkbox(value=False, label="Plot Training History (Console Only)")
                            train_step2_button = gr.Button("Run Step 2", variant="primary", interactive=False)

                        with gr.TabItem("Step 3: Create Pattern Dataset"):
                            gr.Markdown("Create the dataset for the map flow generator.")
                            train_step3_button = gr.Button("Run Step 3", variant="primary", interactive=False)
                    
                    reset_train_button = gr.Button("Reset Training")

                with gr.Column(scale=3):
                    gr.Markdown("### Training Log")
                    train_log = gr.Markdown(value="*The training process log will appear here.*")

            train_step1_button.click(
                fn=handle_train_step1,
                inputs=[train_log],
                outputs=[train_log, train_step1_button, train_step2_button]
            )
            train_step2_button.click(
                fn=handle_train_step2,
                inputs=[train_log, train_epochs, train_batch_size, plot_history],
                outputs=[train_log, train_step2_button, train_step3_button]
            )
            train_step3_button.click(
                fn=handle_train_step3,
                inputs=[train_log],
                outputs=[train_log, train_step3_button]
            )
            reset_train_button.click(
                fn=reset_training_ui,
                inputs=[],
                outputs=[train_log, train_step1_button, train_step2_button, train_step3_button]
            )

        with gr.TabItem("Map Generation"):
            gr.Markdown("## Generate an osu!mania Map")
            gr.Markdown("Follow the steps below to generate a chart. You can go back to a previous step to change settings; this will clear the results of all subsequent steps.")
            
            notes_each_key_state = gr.State(None)
            final_notes_state = gr.State(None)
            desired_filename_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Step 1: Prepare Input"):
                            gr.Markdown("Upload a timed `.osu` file and its corresponding audio file.")
                            osu_file_input = gr.File(label="Upload .osu File", file_count="single", file_types=[".osu"], visible=True)
                            osu_audio_input = gr.File(label="Upload Corresponding Audio", file_count="single", file_types=["audio"], visible=True)
                            step1_button = gr.Button("Execute Step 1", variant="primary")
                        
                        with gr.TabItem("Step 2: Predict Rhythm"):
                            note_density = gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.01, label="Note Density")
                            hold_favor = gr.Slider(minimum=-1.0, maximum=1.0, value=0.5, step=0.01, label="Hold Favor")
                            divisor_favor_str = gr.Textbox(label="Divisor Favor (e.g., 0,0,0,0)", value="0,0,0,0")
                            hold_max_ticks = gr.Slider(minimum=4, maximum=256, value=100, step=4, label="Max Hold Ticks")
                            hold_min_return = gr.Slider(minimum=1, maximum=50, value=1, step=1, label="Min Hold Return (Pattern DB)")
                            rotate_mode_map = {"No Rotation": 0, "Random": 1, "Mirror": 2, "Circulate": 3, "Circulate + Mirror": 4}
                            rotate_mode = gr.Radio(list(rotate_mode_map.keys()), label="Rotate Mode", value="Circulate + Mirror")
                            step2_button = gr.Button("Execute Step 2", variant="primary", interactive=False)

                        with gr.TabItem("Step 3: Apply Modding"):
                            key_fix_map = {"Inactive": 0, "Remove Late Note": 1, "Remove Early Note": 2, "Divert (Recommended)": 3}
                            key_fix = gr.Radio(list(key_fix_map.keys()), label="Single-Key Fix Mode", value="Divert (Recommended)")
                            step3_button = gr.Button("Execute Step 3", variant="primary", interactive=False)

                        with gr.TabItem("Step 4: Save File"):
                            step4_button = gr.Button("Execute Step 4 & Save", variant="primary", interactive=False)
                            output_file = gr.File(label="Download Generated Map")

                    reset_button = gr.Button("Start Over & Clean Up", variant="stop")

                with gr.Column(scale=3):
                    gr.Markdown("### Console Log")
                    generation_log = gr.Markdown("*The generation process log will appear here.*")

            step1_button.click(
                fn=handle_step1_click, 
                inputs=[generation_log, osu_file_input, osu_audio_input],
                outputs=[generation_log, step1_button, step2_button, step3_button, step4_button, desired_filename_state, notes_each_key_state, final_notes_state]
            )
            
            step2_button.click(
                fn=handle_step2_click, 
                inputs=[generation_log, note_density, hold_favor, divisor_favor_str, hold_max_ticks, hold_min_return, rotate_mode],
                outputs=[generation_log, step2_button, step3_button, step4_button, notes_each_key_state, final_notes_state]
            )

            step3_button.click(
                fn=handle_step3_click, 
                inputs=[generation_log, notes_each_key_state, key_fix],
                outputs=[generation_log, step3_button, step4_button, final_notes_state]
            )
            
            step4_button.click(
                fn=handle_step4_click, 
                inputs=[generation_log, final_notes_state, desired_filename_state],
                outputs=[generation_log, step4_button, output_file]
            )

            reset_button.click(
                fn=cleanup_and_reset, 
                inputs=[],
                outputs=[
                    generation_log, notes_each_key_state, final_notes_state, desired_filename_state, output_file, 
                    step1_button, step2_button, step3_button, step4_button
                ]
            )

if __name__ == "__main__":
    demo.launch()
