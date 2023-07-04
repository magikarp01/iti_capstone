import sys

# Specify the file path to log the output
log_file = "ccs_comparison_output.log"
code_to_run = "ccs_comparisonacrossdatasets.py"

# Open the log file in append mode
with open(log_file, "a") as f:
    # Redirect the standard output to the log file
    sys.stdout = f

    # Execute the code in code_to_run.py
    exec(open(code_to_run).read())

    # Reset the standard output to the console
    sys.stdout = sys.__stdout__
