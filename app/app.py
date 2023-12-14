from fastai.vision.all import *
import gradio as gr
import torch

def greet(name):
    return "Hello " + name + "!!"


# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

model = load_learner('models/veg-recognizer-v1.pkl')
                     
# Load the data loader
dls = torch.load(f'data_loaders/veg_dataloader_v1.pkl', map_location=torch.device('cpu'))


# Access the vocabulary
veg_labels = dls.train.vocab

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    print(pred, probs)
    return dict(zip(veg_labels, map(float, probs)))

image = gr.Image()
label = gr.Label()
examples = [
    '/Users/sadihossain/Desktop/Vegetable_Recognizer/test_images/test_image1.jpeg',
    '/Users/sadihossain/Desktop/Vegetable_Recognizer/test_images/test_image2.jpg',
    '/Users/sadihossain/Desktop/Vegetable_Recognizer/test_images/test_image3.jpg',
    '/Users/sadihossain/Desktop/Vegetable_Recognizer/test_images/test_image4.jpg'
]
iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False,share=True)
