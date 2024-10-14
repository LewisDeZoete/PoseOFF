import cv2
from moviepy.editor import VideoFileClip

def main():
    # Open video file
    # video_path = 'C:/Users/ldezoetegrundy/OneDrive - Swinburne University/Coding/Datasets/Sample gestures/Raw/Forward_full.mp4'  # Change this to your video file path
    video_path = 'C:/Users/ldezoetegrundy/OneDrive - Swinburne University/Coding/Datasets/Sample gestures/Raw/Group_full.mp4'  # Change this to your video file path
    # video_path = 'C:/Users/ldezoetegrundy/OneDrive - Swinburne University/Coding/Datasets/Sample gestures/Raw/Wave_full.mp4'  # Change this to your video file path

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # Display the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        return

    cv2.imshow('Video Player', frame)
    print(f"Displaying frame: {current_frame}")

    while True:
        key = cv2.waitKey(0) & 0xFF  # Wait for a key press and get the key code

        print(f"Key pressed: {key}")  # Debugging output for key codes

        if key == 27:  # ESC key to exit
            break
        elif key == 97:  # Left arrow key
            if current_frame > 10:
                current_frame -= 10
        elif key == 115:
            if current_frame > 0:
                current_frame -= 1
        elif key == 100:  # Right arrow key
            if current_frame < total_frames - 1:
                current_frame += 1
        elif key == 102:  # Right arrow key
            if current_frame < total_frames - 11:
                current_frame += 10
        

        # Move to the new frame and display it
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the frame.")
            break

        cv2.imshow('Video Player', frame)
        print(f"Displaying frame: {current_frame}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def out_path(name, num):
    path = f'C:/Users/ldezoetegrundy/OneDrive - Swinburne University/Coding/Datasets/Sample gestures/Trimmed/{name}_{num}.mp4'  # Change this to your video file path
    return path

def trim_video(input_file, output_file, start_frame, end_frame, frame_rate=30):
    # Load the video clip
    video = VideoFileClip(input_file)
    
    # Calculate start and end times in seconds
    start_time = start_frame / frame_rate
    end_time = end_frame / frame_rate
    
    # Trim the video clip
    trimmed_video = video.subclip(start_time, end_time)
    
    # Write the result to the output file
    trimmed_video.write_videofile(output_file, codec="libx264")

forward_dict = {0: (175, 290),
                1: (370, 490),
                2: (565, 688),
                3: (764, 892),
                4: (944, 1064),
                5: (1129, 1257),
                6: (1314,1453),
                7: (1499,1628)}
group_dict = {0:(130, 219),
            1:(293, 402),
            2:(460, 560),
            3:(628, 723),
            4:(790, 908),
            5:(976,1103),
            6:(1150,1261),
            7:(1329,1457)}
wave_dict = {0:(109, 210),
            1: (285,375),
            2: (465,585),
            3: (654, 755),
            4: (841, 927),
            5: (1020, 1125),
            6: (1197, 1289),
            7: (1375, 1491)}




if __name__ == "__main__":
    main()

    '''
    Here we do the actual cropping, you need to change all the dicts for each video
    '''
    # name = 'Wave'
    # input_file = f'C:/Users/ldezoetegrundy/OneDrive - Swinburne University/Coding/Datasets/Sample gestures/Raw/{name}_full.mp4'  # Change this to your video file path
    # frame_rate = 30

    # for key in wave_dict.keys():
    #     output_file = out_path(name, key)
    #     start_frame = wave_dict[key][0]
    #     end_frame = wave_dict[key][1]

    #     trim_video(input_file, output_file, start_frame, end_frame, frame_rate)


## -----------------------------------------------------------
## Convert annotation file to with or without file extensions
## -----------------------------------------------------------
# import yaml

# with open('../Datasets/UCF-101/ucf101_annotations.yaml', 'r') as read_file:
#     ann_file = yaml.safe_load(read_file)

# new_anns = {k.split('.')[0]:v for (k,v) in ann_file.items()}

# with open('../Datasets/UCF-101/ucf101_annotations.yaml', 'w') as write_file:
#     yaml.dump(new_anns, write_file)

