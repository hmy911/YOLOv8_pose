import cv2
import gradio as gr
from ultralytics import YOLO, solutions
# from ultralytics.utils.plotting import Annotator, colors
# from collections import defaultdict
#import spaces

css = '''
.gradio-container{max-width: 600px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
'''

yolo_model = YOLO("yolov8x.pt")
pose_model = YOLO("yolov8x-pose-p6.pt")
names = yolo_model.model.names

# Initialize Solutions
speed_obj = solutions.SpeedEstimator(
    reg_pts=[(0, 360), (1280, 360)],
    names=names,
    view_img=False,
)

gym_object = solutions.AIGym(
    line_thickness=2,
    view_img=False,
    pose_type="pushup",
    kpts_to_check=[6, 8, 10],
)
# track_history = defaultdict(lambda: [])

#@spaces.GPU()
def process_video(video_path, function):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    output_path = f"{function}_output.avi"
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break        
        if function == "Workout Monitoring":
            results = pose_model.track(im0, verbose=False)
            im0 = gym_object.start_counting(im0, results)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    return output_path

def gradio_interface(video, function):
    processed_video_path = process_video(video, function)
    return processed_video_path

with gr.Blocks(css=css, theme="allenai/gradio-theme") as demo:
    gr.Markdown("# YOLOv8 Ultralytics Solutions")
    with gr.Column():
        with gr.Row():
            with gr.Column():
                video_input = gr.Video()
                function_selector = gr.Dropdown(
                    choices=[
                        "Workout Monitoring"

                    ],
                    label="Select Function",
                    value="Speed Estimation",
                )
                process_button = gr.Button("Process Video & Scroll⬇️")
            with gr.Column():
                gr.Examples(
                    examples=[
                        ["assets/WorkoutMonitoring.mp4", "Workout Monitoring"],
                    ],
                    inputs=[video_input, function_selector]
                )
    
    output_video = gr.Video()
    process_button.click(fn=gradio_interface, inputs=[video_input, function_selector], outputs=output_video)
    gr.Markdown("⚠️ The videos that are 30 seconds or less will avoid time consumption issues and provide better accuracy.")

demo.launch(share=True)
