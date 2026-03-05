import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Filepath to your CSV data
filepath = '/Users/javierweddington/SpotDMouse/P2-Terrain_Challenge/mp2/hardware_metrics/sim_motor_5_LF_thigh_amp0.3.csv '

try:
    # Load the data from the CSV file
    df = pd.read_csv(filepath)

    # Get the unique frequencies, sorted
    frequencies = sorted(df['freq_hz'].unique())
    
    delays = []

    # Analyze each frequency separately
    for freq in frequencies:
        # Filter the dataframe for the current frequency
        freq_df = df[df['freq_hz'] == freq].copy()

        # Ensure there's enough data to analyze
        if len(freq_df) < 2:
            continue

        # Calculate the average time step (dt) for this segment
        dt = np.mean(np.diff(freq_df['time_s']))

        # Detrend signals by subtracting the mean to improve cross-correlation
        command_signal = freq_df['command_rad'] - freq_df['command_rad'].mean()
        actual_signal = freq_df['actual_rad'] - freq_df['actual_rad'].mean()

        # Compute the cross-correlation
        correlation = np.correlate(command_signal, actual_signal, mode='full')
        
        # The cross-correlation result has a length of 2*N - 1, where N is the signal length.
        # The lag is the distance from the center of the correlation array.
        lags = np.arange(-len(command_signal) + 1, len(command_signal))
        
        # Find the lag that corresponds to the maximum correlation
        lag_in_samples = lags[np.argmax(correlation)]
        
        # Convert the lag from samples to time
        delay_in_seconds = lag_in_samples * dt
        delays.append(delay_in_seconds * 1000) # Convert to milliseconds

    # --- Plotting the results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a bar chart of the delays
    bars = ax.bar(frequencies, delays, color='skyblue', edgecolor='black', width=0.5)

    # Add labels and title
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Calculated Delay (ms)', fontsize=12)
    ax.set_title('Motor Response Delay vs. Frequency', fontsize=16, fontweight='bold')
    ax.set_xticks(frequencies)
    ax.set_xticklabels([f'{f:.1f}' for f in frequencies])
    
    # Add the delay value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f} ms', ha='center', va='bottom')

    plt.tight_layout()
    
    # Save and show the plot
    output_filename = 'motor_delay_analysis.png'
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")
    plt.show()


except FileNotFoundError:
    print(f"Error: The file was not found at {filepath}")
except Exception as e:
    print(f"An error occurred: {e}")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Filepath to your CSV data
filepath = '/Users/javierweddington/SpotDMouse/P2-Terrain_Challenge/mp2/hardware_metrics/sim_motor_5_LF_thigh_amp0.3.csv'

try:
    # Load the data from the CSV file
    df = pd.read_csv(filepath)

    # Find the unique frequencies in the data
    frequencies = df['freq_hz'].unique()
    frequencies.sort()

    # Determine the number of subplots needed
    num_freqs = len(frequencies)

    # Create a figure and a set of subplots
    # Arrange subplots in a grid, aiming for a roughly square layout
    cols = int(np.ceil(np.sqrt(num_freqs)))
    rows = int(np.ceil(num_freqs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True, sharey=True, squeeze=False)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot the data for each frequency on a separate subplot
    for i, freq in enumerate(frequencies):
        ax = axes[i]
        
        # Filter the dataframe for the current frequency
        freq_df = df[df['freq_hz'] == freq]
        
        # Plot commanded and actual radians vs. time
        ax.plot(freq_df['time_s'], freq_df['command_rad'], label='Commanded Position', linestyle='--')
        ax.plot(freq_df['time_s'], freq_df['actual_rad'], label='Actual Position', alpha=0.8)
        
        ax.set_title(f'Frequency: {freq} Hz')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for i in range(num_freqs, len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout to prevent titles and labels from overlapping
    plt.suptitle('Motor Response: Commanded vs. Actual Position', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    # Save the plot to a file
    output_filename = 'motor_response_subplots.png'
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")
    
    # Display the plot
    plt.show()

except FileNotFoundError:
    print(f"Error: The file was not found at {filepath}")
except Exception as e:
    print(f"An error occurred: {e}")
