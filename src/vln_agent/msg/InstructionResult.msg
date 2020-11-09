Header header
string instr_id
# Reason why the navigation was ended, e.g. predicted stop action, move_base failed, etc
string reason
# Filepaths of the theta images acquired for this navigation
string[] image_filenames
# Timestamps associated with the theta images
time[] image_timestamps
# Timestamp when navigation started for this instruction
time start_time
# Timestamp when navigation ended for this instruction
time end_time
