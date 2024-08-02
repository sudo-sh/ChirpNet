import pickle

def simulate_adc(data, bits):
    """
    Simulate the effect of an ADC with a given bit precision on the data.
    
    :param data: 4D array of shape (N, 64, 8, 192) representing analog input values.
    :param bits: Bit precision of the ADC.
    :return: 4D array of digitized values.
    """
    max_value = 2**bits - 1
    
    # Normalize data if not already normalized (assuming range is 0 to 1)
    # Adjust this part if your data has a different range
    data_normalized = data / np.max(data)

    # Scale and quantize
    scaled_data = data_normalized * max_value
    quantized_data = np.round(scaled_data).astype(int)

    return quantized_data


def load_pkl_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def merge_label_dicts(dict1, dict2):
    try:
        # Find the maximum frame ID in the first dictionary
        max_frame_id = max(map(int, dict1.keys()))
    except ValueError:
        # If dict1 is empty, start frame IDs from -1 so that dict2 starts from 0
        max_frame_id = -1

    # Increment frame IDs in the second dictionary
    incremented_dict2 = {}
    for frame_id, data in dict2.items():
        new_frame_id = str(int(frame_id) + max_frame_id + 1)
        incremented_dict2[new_frame_id] = data

    # Merge the two dictionaries
    merged_dict = {**dict1, **incremented_dict2}
    return merged_dict

def process_radar_data(radar_data, device):
    """
    Process radar data using FFT and return the concatenated magnitude and phase,
    reshaped to (batch, 8, 128, 192).

    Parameters:
    radar_data (torch.Tensor): Input radar data of shape (batch, 64, 8, 192).

    Returns:
    torch.Tensor: Processed data of shape (batch, 8, 128, 192).
    """

    # Move data to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    radar_data = radar_data.to(device)

    # Perform FFT along the last three dimensions (64, 8, 192)
    fft_data = torch.fft.fftn(radar_data, dim=(1, 2, 3))

    # Get magnitude and phase
    magnitude = torch.abs(fft_data)
    phase = torch.angle(fft_data)

    # Concatenate magnitude and phase along the channel dimension
    processed_data = torch.cat((magnitude, phase), dim=1)

    # Reshape to (batch, 8, 128, 192)
    processed_data = processed_data.permute(0, 2, 1, 3)

    return processed_data