import os
import torch
import numpy as np


def create_class_folder(arg, class_name: str, modality: str):
    """
    Creates a directory for a specific class and modality if it does not already exist.

    Args:
        arg: An object containing feeder arguments, specifically a dictionary with data paths.
        class_name (str): The name of the class for which the folder is to be created.
        modality (str): The modality type (e.g., 'rgb', 'depth') used to determine the data path.

    Returns:
        None
    """
    folder = os.path.join(arg.extractor["data_paths"][f"{modality}_path"], class_name)
    os.makedirs(folder, exist_ok=True)
    print("\tAttempting to create folder:", folder)


def get_incomplete(arg, modality: str):
    """
    Identifies and returns a list of unprocessed videos for each class.
    Args:
        arg: An object containing feeder arguments and data paths.
        modality (str): The modality type (e.g., 'rgb', 'flow', 'pose') to check for unprocessed videos.
    Returns:
        dict: A dictionary where keys are class names and values are lists of unprocessed video names.
    """
    class_annotations = {}
    for key in arg.feeder_args["labels"]:
        class_name, video_name = key.split("/")
        class_annotations.setdefault(class_name, []).append(video_name)

    # Get the unprocessed videos, make sure class folders exist
    incomplete = {}
    for class_name, class_videos in class_annotations.items():
        create_class_folder(arg, class_name, modality)
        processed_videos = {
            i.split(".")[0]
            for i in os.listdir(
                f"{arg.extractor['data_paths'][f'{modality}_path']}/{class_name}"
            )
        }
        incomplete_videos = set(class_videos) - processed_videos
        if incomplete_videos:
            incomplete[class_name] = sorted(list(incomplete_videos))
    
    with open("./data_gen/ucf101/incomplete_classes.txt", "w") as f:
        for class_name, video_names in incomplete.items():
            video_names=','.join(video_names)
            f.write(f"{class_name}:{video_names}\n")

    return incomplete


def get_class_by_index(process_number: int):
    """
    Retrieve the class name and video names by the given process number.
    Args:
        arg: The argument to be passed to the get_incomplete function.
        process_number (int): The index of the process to retrieve.
        modality (str): The modality to be used in the get_incomplete function.
    Returns:
        tuple: A tuple containing the class name and a list of video names.
    Raises:
        IndexError: If the process_number is out of range of the incomplete list.
    """
    with open('./data_gen/ucf101/incomplete_classes.txt', 'r') as f:
        incomplete = {}
        for line in f.readlines():
            line = line.strip()
            class_name, video_names = line.split(":")
            incomplete[class_name] = video_names.split(",")
    if process_number <= len(incomplete):
        class_name, video_names = list(incomplete.items())[process_number]
    else:
        raise IndexError(f"Index {process_number} out of range\nCancelled processing")

    return class_name, video_names


def is_null(data):
    """Check if the data is filled with zeros"""
    if isinstance(data, torch.Tensor):
        return torch.all(data == 0).item()
    if isinstance(data, np.ndarray):
        return np.all(data == 0)
    return False


def log_zero_data(class_name, video_name):
    """Function to log the name of the data to a text file"""
    with open(f"./TMP/zero_data_{class_name}.txt", "a") as f:
        f.write(f"{video_name}\n")  # Write the key to the file


def extract_data(
    arg,
    process_number: int,
    transforms,
    modality: str,
    debug: bool = False
):
    """
    Extracts data for a given class and processes it using specified transforms.
    Args:
        arg: Argument object containing feeder arguments and data paths.
        process_number (int): The process number for parallel processing.
        transforms: A function or callable that applies transformations to the data.
        modality (str): The type of data to process ('flowpose' or other).
        save_as_numpy (bool, optional): Whether to save the processed data as a numpy file. Defaults to True.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
    Returns:
        None
    """
    class_name, video_names = get_class_by_index(process_number=process_number)
    print("Processing:", class_name)
    print("\tTotal videos:", len(video_names))

    for video_name in video_names:
        if modality == "flowpose":
            # TODO: Change this implementation to default to numpy
            poses = np.load(
                f"{arg.extractor['data_paths']['pose_path']}/{class_name}/{video_name}.npy"
            )
            flows = torch.load(
                f"{arg.extractor['data_paths']['flow_path']}/{class_name}/{video_name}.pt",
                map_location="cpu",
            )
            data = transforms(flows, poses)
        else:
            # Get the video path
            video_path = f"{os.path.join(arg.extractor['data_paths']['rgb_path'], class_name, video_name)}.avi"
            # Transform and estimate poses
            data = transforms(video_path)
        if is_null(data):  # Check if the data is all zeros
            log_zero_data(class_name=class_name, video_name=video_name)

        # Get the path to save the estimated poses to
        save_path = os.path.join(
            arg.extractor["data_paths"][f'{modality}_path'],  # Data path
            class_name,  # Class folder
            video_name + ".npy")
        if debug:
            print(f"Modality: {modality}")
            print(f"Data shape: {data.shape}")
            print(f"Data type: {type(data)}")
            print(f"Saving {video_name} to {save_path}")
            break
        # Save the data
        np.save(save_path, data)

    print(f"Processed {class_name} class")


if __name__ == "__main__":
    from config.argclass import ArgClass
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(prog="check_incomplete")
    parser.add_argument(
        "-m", dest="modality", help="Modality to check for incomplete classes"
    )
    parsed = parser.parse_args()

    # Create ArgClass object
    arg = ArgClass(arg="./config/ucf101/train_base.yaml")

    # Ensure the modality folder exists
    os.makedirs(arg.extractor["data_paths"][f"{parsed.modality}_path"], exist_ok=True)

    # Get the incomplete classes (also creates folders if they're empty)
    incomplete = get_incomplete(arg, parsed.modality)